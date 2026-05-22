import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
import os

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (ПАРАМЕТРЫ СИМУЛЯЦИИ)
# =====================================================================
TICKER_ASSET = "BTC-USD"     # Высоковолатильный целевой актив (Bitcoin)
TICKER_INDEX = "^IXIC"      # Макро-контекст: индекс NASDAQ Composite
START_DATE = "2023-01-01"   # Старт исторического периода
END_DATE = "2026-05-01"     # Финал исторического периода

INPUT_WIDTH = 8             # Ширина временного окна скользящего тренда
FEATURES_DIM = 3            # Цена BTC + Объем BTC + Индекс NASDAQ
HIDDEN_DIM = 64             # Внутренняя емкость кубика Лего

BATCH_SIZE = 32
EPOCHS = 20                 # Количество эпох
START_LR = 0.003

# =====================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =====================================================================
print(f"Скачиваю исторические данные {TICKER_ASSET} и {TICKER_INDEX}...")
asset_data = yf.download(TICKER_ASSET, start=START_DATE, end=END_DATE)
index_data = yf.download(TICKER_INDEX, start=START_DATE, end=END_DATE)

# Склеиваем по датам
df = pd.DataFrame(index=asset_data.index)
df['Target_Close'] = asset_data['Close']
df['Target_Volume'] = asset_data['Volume']
df['Index_Close'] = index_data['Close']
df = df.ffill().dropna()

raw_matrix = df.values
min_val = raw_matrix.min(axis=0)
max_val = raw_matrix.max(axis=0)
normalized_matrix = (raw_matrix - min_val) / (max_val - min_val)

X_list, Y_list = [], []
for i in range(len(normalized_matrix) - INPUT_WIDTH):
    X_list.append(normalized_matrix[i : i + INPUT_WIDTH, :])
    Y_list.append([normalized_matrix[i + INPUT_WIDTH, 0]])

X_all = torch.tensor(np.array(X_list), dtype=torch.float32)
Y_all = torch.tensor(np.array(Y_list), dtype=torch.float32)

# 80% данных на обучение, последние 20% — строгий бэктест реальной торговли
train_size = int(len(X_all) * 0.8)
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

# Сохраняем ненормализованные цены закрытия тестового периода для симулятора
test_dates = df.index[INPUT_WIDTH + train_size:]
real_test_prices = df['Target_Close'].values[INPUT_WIDTH + train_size:]

print(f"Загружено {len(df)} торговых сессий.")
print(f"Период бэктестинга: с {test_dates[0].strftime('%Y-%m-%d')} по {test_dates[-1].strftime('%Y-%m-%d')}")
print(f"Количество дней для симуляции торговли: {len(real_test_prices)}\n")


# =====================================================================
# 2. МНОГОСЛОЙНАЯ ВЕКТОРНАЯ Z-ЛЕГО-ТОПОЛОГИЯ
# =====================================================================
class MultivariateLegoBlock(nn.Module):
    def __init__(self, features_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim * 2)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        out = self.net(combined)
        out_a, out_b = torch.chunk(out, 2, dim=-1)
        return out_a, out_b

class TrueMultivariateLegoLayer(nn.Module):
    def __init__(self, input_width, features_dim, hidden_dim):
        super().__init__()
        self.input_width = input_width
        self.features_dim = features_dim
        self.num_blocks = input_width - 1
        self.blocks = nn.ModuleList([MultivariateLegoBlock(features_dim, hidden_dim) for _ in range(self.num_blocks)])

    def forward(self, x):
        x_transposed = x.transpose(0, 1)
        channels = [x_transposed[i] for i in range(self.input_width)]
        
        # Каноничный Z-ход (Справа налево во времени)
        out_a, out_b = self.blocks[0](channels[self.input_width - 2], channels[self.input_width - 1])
        channels[self.input_width - 1] = out_b
        carrier_signal = out_a
        
        for i in range(1, self.num_blocks):
            input_idx = (self.input_width - 2) - i
            out_a, out_b = self.blocks[i](channels[input_idx], carrier_signal)
            channels[input_idx + 1] = out_b
            carrier_signal = out_a
            
        # !!! ИСПРАВЛЕНО !!! Записываем финальный левый хвост транзита в нулевой канал списка
        channels[0] = carrier_signal
        
        # Теперь channels — это чистый кортеж/список Тензоров, stack отработает безупречно
        return torch.stack(channels, dim=1)

class MultivariateLegoNet(nn.Module):
    def __init__(self, input_width, features_dim, hidden_dim):
        super().__init__()
        self.lego_layer = TrueMultivariateLegoLayer(input_width, features_dim, hidden_dim)
        self.regressor = nn.Linear(input_width * features_dim, 1)

    def forward(self, x):
        h = self.lego_layer(x)
        h_flat = h.reshape(h.size(0), -1)
        return self.regressor(h_flat)


# =====================================================================
# 3. КРОСС-ЭНТРОПИЙНОЕ ОБУЧЕНИЕ (HUBER LOSS)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultivariateLegoNet(input_width=INPUT_WIDTH, features_dim=FEATURES_DIM, hidden_dim=HIDDEN_DIM).to(device)
criterion = nn.HuberLoss(delta=0.1)
optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)

print("Обучаю Z-LegoNet на высоковолатильном графике Биткоина...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, batch_count = 0, 0
    indices = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        if len(batch_idx) < BATCH_SIZE: continue
        
        x_batch = X_train[batch_idx].to(device)
        y_batch = Y_train[batch_idx].to(device)
        
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    scheduler.step()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f" Эпоха {epoch+1:02d}/{EPOCHS:02d} | Средний Huber Loss: {total_loss/batch_count:.6f}")


# =====================================================================
# 4. СИМУЛЯТОР РЕАЛЬНОЙ ТОРГОВЛИ (БЭКТЕСТИНГ СТРАТЕГИИ)
# =====================================================================
model.eval()
with torch.no_grad():
    pred_y = model(X_test.to(device))
    
    # Денормализуем предсказания в чистые доллары
    min_close = min_val[0]
    max_close = max_val[0]
    predicted_prices = (pred_y * (max_close - min_close) + min_close).cpu().numpy().flatten()

# Параметры симуляции
initial_capital = 10000.0  # Стартовый капитал: 10 000 долларов
capital = initial_capital
position = 0.0             # Количество Биткоинов на балансе робота
in_position = False

print("\n--- Запуск симулятора торговли торгового робота ---")

# Цикл симуляции день за днем
for t in range(len(real_test_prices) - 1):
    current_price = real_test_prices[t]
    predicted_next_price = predicted_prices[t]
    
    # СИГНАЛ НА ПОКУПКУ: Модель прогнозирует рост, а мы еще в кэше
    if predicted_next_price > current_price and not in_position:
        position = capital / current_price
        capital = 0.0
        in_position = True
        
    # СИГНАЛ НА ПРОДАЖУ: Модель прогнозирует падение, а мы удерживаем Биткоин
    elif predicted_next_price < current_price and in_position:
        capital = position * current_price
        position = 0.0
        in_position = False

# Закрываем позицию в самый последний день, чтобы посчитать чистый кэш
if in_position:
    capital = position * real_test_prices[-1]
    position = 0.0

# Расчет эффективности (ИСПРАВЛЕНО: берем строго нулевой индекс первого дня тестового окна)
robot_return = ((capital - initial_capital) / initial_capital) * 100
bh_return = ((real_test_prices[-1] - real_test_prices[0]) / real_test_prices[0]) * 100

print(f"Стартовый капитал робота:  ${initial_capital:,.2f}")
print(f"Итоговый капитал робота:   ${capital:,.2f}")
print("-" * 50)
print(f"Доходность Z-LegoNet бота: {robot_return:.2f}%")
print(f"Доходность 'Buy and Hold':  {bh_return:.2f}%")
print("-" * 50)

if robot_return > bh_return:
    print("Результат: Робот на базе Z-LegoNet успешно обыграл рынок!")
else:
    print("Результат: Пассивное удержание рынка оказалось выгоднее робота.")

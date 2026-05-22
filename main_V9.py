import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
import os

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (ПАРАМЕТРЫ ДЛЯ ЭКСПЕРИМЕНТОВ)
# =====================================================================
TICKER_ASSET = "AAPL"       # Целевая акция для прогноза (Apple)
TICKER_INDEX = "^GSPC"      # Рыночный индекс для контекста (S&P 500)
START_DATE = "2023-01-01"   # С какой даты брать историю
END_DATE = "2026-01-01"     # По какую дату

INPUT_WIDTH = 8             # Ширина временного окна (сколько дней смотрим назад)
FEATURES_DIM = 3            # Количество фич на один шаг времени (Цена + Объем + Индекс)
HIDDEN_DIM = 64             # Внутренняя емкость кубика Лего для переваривания фич

BATCH_SIZE = 32
EPOCHS = 15
START_LR = 0.003

# =====================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА МНОГОМЕРНЫХ ДАННЫХ ИЗ YFINANCE
# =====================================================================
print("Скачиваю данные из Yahoo Finance...")
asset_data = yf.download(TICKER_ASSET, start=START_DATE, end=END_DATE)
index_data = yf.download(TICKER_INDEX, start=START_DATE, end=END_DATE)

# Собираем многомерный датафрейм
df = pd.DataFrame(index=asset_data.index)
df['Target_Close'] = asset_data['Close']
df['Target_Volume'] = asset_data['Volume']
df['Index_Close'] = index_data['Close']
df = df.dropna()

# Нормализуем данные методом MinMax
raw_matrix = df.values
min_val = raw_matrix.min(axis=0)
max_val = raw_matrix.max(axis=0)
normalized_matrix = (raw_matrix - min_val) / (max_val - min_val)

# Нарезаем матрицу скользящим окном под нашу Лего-топологию
X_list, Y_list = [], []
for i in range(len(normalized_matrix) - INPUT_WIDTH):
    X_list.append(normalized_matrix[i : i + INPUT_WIDTH, :])
    Y_list.append([normalized_matrix[i + INPUT_WIDTH, 0]]) # Цель — цена (индекс 0)

X_all = torch.tensor(np.array(X_list), dtype=torch.float32) 
Y_all = torch.tensor(np.array(Y_list), dtype=torch.float32) 

# Делим на Train/Test
train_size = int(len(X_all) * 0.8)
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

print(f"Загружено {len(normalized_matrix)} торговых дней.")
print(f"Форма X_train: {X_train.shape} | Форма Y_train: {Y_train.shape}\n")


# =====================================================================
# 2. МНОГОМЕРНАЯ Z-ЛЕГО-ТОПОЛОГИЯ (MULTIVARIATE)
# =====================================================================
class MultivariateLegoBlock(nn.Module):
    """ Векторная деталь Лего. Перемалывает два многомерных канала. """
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
        
        # Цепочка из N-1 уникальных векторных кубиков Лего
        self.blocks = nn.ModuleList([MultivariateLegoBlock(features_dim, hidden_dim) for _ in range(self.num_blocks)])

    def forward(self, x):
        batch_size = x.size(0)
        
        # Разрезаем временное окно на N отдельных дней-векторов формы [Batch, FEATURES_DIM]
        x_transposed = x.transpose(0, 1) 
        channels = [x_transposed[i] for i in range(self.input_width)]
        
        # --- Z-ТОПОЛОГИЯ (Справа налево во времени) ---
        # ИСПРАВЛЕНО: Обращаемся к первому кубику строго по индексу [0]
        out_a, out_b = self.blocks[0](channels[self.input_width - 2], channels[self.input_width - 1])
        
        channels[self.input_width - 1] = out_b
        carrier_signal = out_a 
        
        # Сдвигаемся лесенкой назад по истории
        for i in range(1, self.num_blocks):
            input_idx = (self.input_width - 2) - i
            out_a, out_b = self.blocks[i](channels[input_idx], carrier_signal)
            
            channels[input_idx + 1] = out_b
            carrier_signal = out_a
            
        channels[0] = carrier_signal
        
        # Собираем всё обратно в один трехмерный тензор
        final_tensor = torch.stack(channels, dim=1) 
        return final_tensor


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
# 3. ЦИКЛ ОБУЧЕНИЯ НА РЕАЛЬНОМ РЫНКЕ (HUBER LOSS)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device} | Направление: BACKWARD (Z-Топология)")

model = MultivariateLegoNet(input_width=INPUT_WIDTH, features_dim=FEATURES_DIM, hidden_dim=HIDDEN_DIM).to(device)
criterion = nn.HuberLoss(delta=0.1) 
optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)

print(f"--- Старт обучения многомерной Z-LegoNet на акциях {TICKER_ASSET} ---")
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
    print(f"Епоха {epoch+1:02d}/{EPOCHS:02d} | Средний Huber Loss: {total_loss/batch_count:.6f}")


# =====================================================================
# 4. ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ
# =====================================================================
model.eval()
with torch.no_grad():
    # Берем последние 3 дня из тест-выборки
    test_x = X_test[-3:].to(device)
    test_y = Y_test[-3:].to(device)
    pred_y = model(test_x)
    
    # ИСПРАВЛЕНО: Извлекаем экстремумы строго для столбца цены закрытия (индекс 0)
    min_close = min_val[0]
    max_close = max_val[0]
    
    real_prices = test_y * (max_close - min_close) + min_close
    pred_prices = pred_y * (max_close - min_close) + min_close
    
    print(f"\n--- Результаты прогноза курса {TICKER_ASSET} (в долларах) ---")
    for i in range(3):
        print(f"Тест {i+1} | Реальная цена закрытия завтра: ${real_prices[i, 0].item():.2f}")
        print(f"       | Прогноз векторной Z-LegoNet: ${pred_prices[i, 0].item():.2f}")
        print("-" * 55)

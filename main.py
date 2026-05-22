import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
import os

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (ВНУТРИДНЕВНЫЕ НАСТРОЙКИ)
# =====================================================================
TICKER_ASSET = "BTC-USD"
TIMEFRAME = "1h"            # Часовые свечи
PERIOD = "2y"               # Максимально доступная глубина истории для 1h

INPUT_WIDTH = 12            # Окно анализа истории (12 часов)
FEATURES_DIM = 3            # Фичи: Ratio, Spread, Shadows
HIDDEN_DIM = 128            # Внутренний размер кубика Лего

BATCH_SIZE = 64             
EPOCHS = 10
START_LR = 0.002

# =====================================================================
# 1. СБОР И СТРУКТУРНАЯ РАЗМЕТКА ЧАСОВОГО ДАТАСЕТА
# =====================================================================
print(f"Скачиваю внутридневные данные ({TIMEFRAME}) для {TICKER_ASSET}...")
df = yf.download(TICKER_ASSET, period=PERIOD, interval=TIMEFRAME)
df = df.ffill().dropna()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Вычисляем внутридневные структурные фичи
df['Spread'] = (df['High'] - df['Low']) / df['Close']
df['Body'] = (df['Open'] - df['Close']).abs()
df['Shadows'] = ((df['High'] - df['Low']) - df['Body']) / df['Close']
df['Volume_Density'] = df['Volume'] / (df['High'] - df['Low'] + 1e-8)

# Разметка часовых аномалий (Volume Climax)
threshold = df['Volume_Density'].mean() + 2.0 * df['Volume_Density'].std()
df['Target_Climax'] = (df['Volume_Density'] > threshold).astype(int)

feature_cols = ['Volume_Density', 'Spread', 'Shadows']
raw_features = df[feature_cols].values
targets = df['Target_Climax'].values

# Z-score нормализация
mean_f = raw_features.mean(axis=0)
std_f = raw_features.std(axis=0)
normalized_features = (raw_features - mean_f) / (std_f + 1e-8)

# Нарезка скользящим окном
X_list, Y_list = [], []
for i in range(len(normalized_features) - INPUT_WIDTH):
    X_list.append(normalized_features[i : i + INPUT_WIDTH, :])
    Y_list.append([targets[i + INPUT_WIDTH]])

X_all = torch.tensor(np.array(X_list), dtype=torch.float32)
Y_all = torch.tensor(np.array(Y_list), dtype=torch.float32)

train_size = int(len(X_all) * 0.8)
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

pos_count = Y_train.sum().item()
total_count = Y_train.numel()
print(f"Загружено часовых свечей: {len(df)}. Зон жатвы: {int(df['Target_Climax'].sum())}")
print(f"Дисбаланс классов в обучении: {pos_count / total_count * 100:.1f}% аномалий.")
print(f"Форма X_train: {X_train.shape} | Форма Y_train: {Y_train.shape}\n")


# =====================================================================
# 2. МНОГОСЛОЙНАЯ ВЕКТОРНАЯ Z-ЛЕГО-ТОПОЛОГИЯ (ИСПРАВЛЕНО)
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
        
        # --- Z-ТОПОЛОГИЯ (Справа налево) ---
        # ИСПРАВЛЕНО: Первый кубик Лего теперь вызывается строго по индексу [0]
        out_a, out_b = self.blocks[0](channels[self.input_width - 2], channels[self.input_width - 1])
        channels[self.input_width - 1] = out_b
        carrier_signal = out_a
        
        for i in range(1, self.num_blocks):
            input_idx = (self.input_width - 2) - i
            out_a, out_b = self.blocks[i](channels[input_idx], carrier_signal)
            channels[input_idx + 1] = out_b
            carrier_signal = out_a
            
        # !!! ИСПРАВЛЕНО !!! Финальный транзит сохраняется на нулевое место, список не затирается
        channels[0] = carrier_signal
        return torch.stack(channels, dim=1)

class LegoClimaxDetector(nn.Module):
    def __init__(self, input_width, features_dim, hidden_dim):
        super().__init__()
        self.lego_layer = TrueMultivariateLegoLayer(input_width, features_dim, hidden_dim)
        self.classifier = nn.Linear(input_width * features_dim, 1)

    def forward(self, x):
        h = self.lego_layer(x)
        h_flat = h.reshape(h.size(0), -1)
        return self.classifier(h_flat)


# =====================================================================
# 3. ЦИКЛ ОБУЧЕНИЯ (BCE LOSS С ПОДДЕРЖКОЙ ВЕСОВ)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LegoClimaxDetector(input_width=INPUT_WIDTH, features_dim=FEATURES_DIM, hidden_dim=HIDDEN_DIM).to(device)

pos_weight_value = (total_count - pos_count) / pos_count
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)

print(f"Запуск обучения Z-LegoNet на часовых интервалах ({TIMEFRAME})...")
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
        logits = model(x_batch)
        
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    scheduler.step()
    print(f" Эпоха {epoch+1:02d}/{EPOCHS:02d} | Средний BCE Loss: {total_loss/batch_count:.6f}")


# =====================================================================
# 4. МЕТРИКИ ОЦЕНКИ НА ДИСКРЕТНОЙ ТЕСТОВОЙ ВЫБОРКЕ
# =====================================================================
model.eval()
with torch.no_grad():
    test_logits = model(X_test.to(device))
    probabilities = torch.sigmoid(test_logits).cpu().numpy().flatten()
    real_targets = Y_test.cpu().numpy().flatten()

predictions = (probabilities > 0.5).astype(int)

true_positives = np.sum((predictions == 1) & (real_targets == 1))
false_positives = np.sum((predictions == 1) & (real_targets == 0))
false_negatives = np.sum((predictions == 0) & (real_targets == 1))

precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (true_positives + false_negatives + 1e-8)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

print(f"\n--- Результаты анализа ЧАСОВОЙ ({TIMEFRAME}) ликвидности ---")
print(f"Всего тестовых часов проверено: {len(real_targets)}")
print(f"Реальное количество зон жатвы в тесте: {int(np.sum(real_targets))}")
print(f"Робот успешно обнаружил (True Positives): {true_positives}")
print(f"Ложные срабатывания (False Positives):    {false_positives}")
print("-" * 50)
print(f"Precision (Точность детекции манипуляций): {precision*100:.1f}%")
print(f"Recall (Процент пойманных зон жатвы):     {recall*100:.1f}%")
print(f"Метрика F1-Score:                         {f1_score:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (ИГРАЙТЕСЬ С ЭТИМИ ПАРАМЕТРАМИ ТУТ)
# =====================================================================
DIRECTION = "backward"  # "forward" (прямая лесенка) или "backward" (Z-лесенка справа налево)

# Главный параметр, определяющий геометрию кубиков Лего!
INPUT_WIDTH = 8         # Ширина входного вектора. Количество блоков автоматичеки станет (INPUT_WIDTH - 1)

BATCH_SIZE = 32
EPOCHS = 20
START_LR = 0.005


# =====================================================================
# 1. ГЕНЕРАЦИЯ ЧИСЛОВЫХ ДАННЫХ (Окно из N чисел -> 1 число в будущем)
# =====================================================================
def generate_sliding_window_data(num_samples=2000, window_size=8):
    X_list, Y_list = [], []
    for _ in range(num_samples):
        phase = random.uniform(0, 2 * math.pi)
        trend = random.uniform(-0.01, 0.01)
        
        full_series = []
        for t in range(window_size + 1):
            val = math.sin(0.4 * t + phase) + (trend * t) + random.normalvariate(0, 0.05)
            full_series.append(val)
            
        X_list.append(torch.tensor(full_series[:-1], dtype=torch.float32)) 
        Y_list.append(torch.tensor([full_series[-1]], dtype=torch.float32)) 
        
    return torch.stack(X_list), torch.stack(Y_list)

# Генерируем выборки
X_all, Y_all = generate_sliding_window_data(num_samples=1500, window_size=INPUT_WIDTH)
train_size = 1200
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

print(f"--- Статистика Истинной Лего-Топологии ---")
print(f"Ширина входа: {INPUT_WIDTH}. Количество кубиков Лего в слое: {INPUT_WIDTH - 1}")
print(f"Форма X_train: {X_train.shape} | Форма Y_train (скалярный прогноз): {Y_train.shape}\n")


# =====================================================================
# 2. ИСТИННАЯ ЛЕГО-ТОПОЛОГИЯ (ИСПРАВЛЕНА ИНДЕКСАЦИЯ MODULELIST)
# =====================================================================
class LegoBlock(nn.Module):
    """ Базовая деталь Лего 2x2. Берет 2 числа, возвращает 2 числа. """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1) 
        out = F.relu(self.linear(combined))
        out_a, out_b = torch.chunk(out, 2, dim=-1)
        return out_a, out_b


class TrueLegoLayer(nn.Module):
    def __init__(self, input_width, direction):
        super().__init__()
        self.input_width = input_width
        self.direction = direction
        self.num_blocks = input_width - 1
        
        # Каждый кубик Лего в лесенке имеет СВОИ УНИКАЛЬНЫЕ ВЕСА
        self.blocks = nn.ModuleList([LegoBlock() for _ in range(self.num_blocks)])

    def forward(self, x):
        batch_size = x.size(0)
        
        # Разрезаем входной батч на список тензоров формы [batch_size, 1]
        channels = list(torch.chunk(x, self.input_width, dim=-1))
        
        if self.direction == "backward":
            # =========================================================
            # Z-ТОПОЛОГИЯ: Справа налево со смещением на 1 шаг
            # =========================================================
            # Стартуем с самого конца вектора (ИСПРАВЛЕНО: берем кубик по индексу [0])
            out_a, out_b = self.blocks[0](channels[self.input_width - 2], channels[self.input_width - 1])
            
            channels[self.input_width - 1] = out_b
            carrier_signal = out_a 
            
            # Шагаем лесенкой влево по остальным кубикам
            for i in range(1, self.num_blocks):
                input_idx = (self.input_width - 2) - i 
                
                out_a, out_b = self.blocks[i](channels[input_idx], carrier_signal)
                
                channels[input_idx + 1] = out_b 
                carrier_signal = out_a          
                
            channels[0] = carrier_signal 
            
        else:
            # =========================================================
            # FORWARD ТОПОЛОГИЯ: Слева направо со смещением на 1 шаг
            # =========================================================
            # Стартуем с начала вектора (ИСПРАВЛЕНО: берем кубик по индексу [0])
            out_a, out_b = self.blocks[0](channels[0], channels[1])
            channels[0] = out_a
            carrier_signal = out_b 
            
            for i in range(1, self.num_blocks):
                input_idx = i + 1
                out_a, out_b = self.blocks[i](carrier_signal, channels[input_idx])
                channels[input_idx - 1] = out_a
                carrier_signal = out_b
                
            channels[self.input_width - 1] = carrier_signal
            
        # Склеиваем каналы обратно в вектор исходной ширины
        return torch.cat(channels, dim=-1)


class LegoTimeSeriesNet(nn.Module):
    def __init__(self, input_width, direction="backward"):
        super().__init__()
        self.lego_layer = TrueLegoLayer(input_width, direction)
        self.regressor = nn.Linear(input_width, 1)

    def forward(self, x):
        h = self.lego_layer(x)
        return self.regressor(h)


# =====================================================================
# 3. ЦИКЛ ОБУЧЕНИЯ (HUBER LOSS)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device} | Направление: {DIRECTION.upper()}")

model = LegoTimeSeriesNet(input_width=INPUT_WIDTH, direction=DIRECTION).to(device)
criterion = nn.HuberLoss(delta=0.5) 
optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)

print(f"--- Старт обучения истинной Лего-лесенки ---")
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
    print(f"Эпоха {epoch+1:02d}/{EPOCHS:02d} | Средний Huber Loss: {total_loss/batch_count:.6f}")


# =====================================================================
# 4. ПРОВЕРКА ТЕСТОВОГО ИНФЕРЕНСА
# =====================================================================
model.eval()
with torch.no_grad():
    test_x = X_test[0:3].to(device) 
    test_y = Y_test[0:3].to(device)
    pred_y = model(test_x)
    
    print("\n--- Результаты инференса (Прогноз на 1 шаг вперед) ---")
    for i in range(3):
        print(f"Пример {i+1} | Входное окно: {[round(n, 2) for n in test_x[i].tolist()]}")
        print(f"         | Реальное будущее число: {test_y[i, 0].item():.4f}")
        print(f"         | Прогноз Лего-лесенки:   {pred_y[i, 0].item():.4f}")
        print("-" * 65)

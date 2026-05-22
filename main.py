import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (ИГРАЙТЕСЬ С ЭТИМИ ПАРАМЕТРАМИ ТУТ)
# =====================================================================
# Архитектура сети
EMBED_DIM = 256       # Ширина сети (Размерность скрытых векторов)
NUM_LAYERS = 4        # Глубина сети (Количество слоев)

# Параметры контекста и обучения
SEQ_LEN = 128         # Объем контекстного окна
BATCH_SIZE = 32       # Размер батча
START_LR = 0.003      # Начальная скорость обучения (можно взять чуть выше, так как она будет падать)
ETA_MIN = 0.0001      # Минимальная скорость обучения в самом конце
EPOCHS = 7            # Увеличим до 7 эпох, чтобы дать расписанию раскрыться

# Параметры инференса (Z-Генерация идет в обратную сторону!)
GEN_TEMPERATURE = 0.8 # Температура сэмплирования
GEN_LENGTH = 200       # Длина генерируемого текста

INPUT_FILE = "input.txt"


# =====================================================================
# 1. ПОДГОТОВКА ДАТАСЕТА И СИМВОЛЬНОГО ТОКЕНИЗАТОРА
# =====================================================================
if not os.path.exists(INPUT_FILE):
    print(f"Файл {INPUT_FILE} не найден. Создаю демонстрационный файл...")
    demo_text = (
        "квадратный бежевый табурет стоял в углу. деревянное сочное кресло привлекало внимание. "
        "старая серая табуретка сломалась вчера. новое вкусное спелое яблоко лежало на столе. "
    ) * 400
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(demo_text)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

print(f"--- Статистика Z-датасета ---")
print(f"Размер считанного текста: {len(raw_text)} символов.")
print(f"Размер словаря: {vocab_size}")

data_indices = [char2idx[ch] for ch in raw_text]

def get_z_batches(data, batch_size, seq_len):
    num_chunks = len(data) // (seq_len + 1)
    x_list = []
    y_list = []
    
    for i in range(num_chunks):
        start_idx = i * (seq_len + 1)
        chunk = data[start_idx : start_idx + seq_len + 1]
        
        # Разворачиваем кусок текста для Z-топологии
        chunk = chunk[::-1]
        
        x_list.append(torch.tensor(chunk[:-1]))
        y_list.append(torch.tensor(chunk[1:]))
        
        if len(x_list) == batch_size:
            yield torch.stack(x_list), torch.stack(y_list)
            x_list, y_list = [], []


# =====================================================================
# 2. МНОГОСЛОЙНАЯ АРХИТЕКТУРА LEGO-GPT
# =====================================================================
class WideLegoBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, current_token_embed, carrier):
        combined = torch.cat([current_token_embed, carrier], dim=-1)
        out = F.relu(self.linear(combined))
        next_token_pred, next_carrier = torch.chunk(out, 2, dim=-1)
        return next_token_pred, next_carrier


class LegoLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.block = WideLegoBlock(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        carrier = torch.zeros(B, C, device=x.device)
        layer_outputs = []
        
        for t in range(T):
            current_embed = x[:, t, :]
            pred, carrier = self.block(current_embed, carrier)
            layer_outputs.append(pred)
            
        h = torch.stack(layer_outputs, dim=1)
        return self.ln(x + h)


class ZDeepLegoGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([LegoLayer(embed_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.classifier(h)
        return logits

    def generate_char(self, input_indices):
        x = torch.tensor([input_indices], device=self.embedding.weight.device)
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        last_logits = self.classifier(h[:, -1, :])
        return last_logits


# =====================================================================
# 3. ОБУЧЕНИЕ С ДИНАМИЧЕСКИМ ШАГОМ (LR SCHEDULER)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство вычислений: {device}\n")

model = ZDeepLegoGPT(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=START_LR)

# Добавляем Косинусное расписание. Оно будет уменьшать LR каждую эпоху.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

print(f"--- Запуск обучения Z-DeepLegoGPT + Cosine LR Scheduler ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Получаем текущий шаг обучения до шага оптимизатора (для красивого вывода)
    current_lr = optimizer.param_groups[0]['lr']
    
    for x_batch, y_batch in get_z_batches(data_indices, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = y_batch[..., :-1].contiguous()
        
        loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    # Делаем шаг расписания в конце каждой эпохи
    scheduler.step()
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Эпоха {epoch+1:02d}/{EPOCHS:02d} | Средний Loss: {avg_loss:.4f} | Текущий LR: {current_lr:.6f}")


# =====================================================================
# 4. АВТОРЕГРЕССИОННАЯ Z-ГЕНЕРАЦИЯ
# =====================================================================
def generate_z_text(model, seed_text, gen_length, temperature):
    model.eval()
    with torch.no_grad():
        reversed_seed = seed_text[::-1]
        print(f"\nОбратная затравка (будущее контекста): '{seed_text}'")
        
        generated_indices = []
        for ch in reversed_seed:
            if ch in char2idx:
                generated_indices.append(char2idx[ch])
            else:
                generated_indices.append(random.choice(list(char2idx.values())))
        
        for _ in range(gen_length):
            context_indices = generated_indices[-SEQ_LEN:]
            logits = model.generate_char(context_indices)
            
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            generated_indices.append(next_char_idx)
            
        raw_output_text = "".join([idx2char[idx] for idx in generated_indices])
        human_readable_text = raw_output_text[::-1]
        print(f"Результат Z-DeepLegoGPT:\n{human_readable_text}\n")

# Итоговая проверка результатов
generate_z_text(model, seed_text="скам.", gen_length=GEN_LENGTH, temperature=GEN_TEMPERATURE)
generate_z_text(model, seed_text="бизнес.", gen_length=GEN_LENGTH, temperature=GEN_TEMPERATURE)

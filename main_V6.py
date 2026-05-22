import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# =====================================================================
# 1. ПОДГОТОВКА РЕАЛЬНОГО ДАТАСЕТА И СИМВОЛЬНОГО ТОКЕНИЗАТОРА
# =====================================================================
INPUT_FILE = "input.txt"

if not os.path.exists(INPUT_FILE):
    # Если файла вдруг нет, создадим демонстрационный текстовый корпус
    print(f"Файл {INPUT_FILE} не найден. Создаю демонстрационный файл...")
    demo_text = (
        "квадратный бежевый табурет стоял в углу. деревянное сочное кресло привлекало внимание. "
        "старая серая табуретка сломалась вчера. новое вкусное спелое яблоко лежало на столе. "
    ) * 100  # Дублируем для объема
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(demo_text)

# Читаем весь текст
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Символьная токенизация (символы — это идеальные n-граммы для улавливания окончаний)
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

print(f"Размер считанного текста: {len(raw_text)} символов.")
print(f"Размер словаря (уникальных символов): {vocab_size}")

# Переводим весь текст в индексы
data_indices = [char2idx[ch] for ch in raw_text]

# Функция для нарезки текста на батчи фиксированной длины (Window Chunking)
def get_batches(data, batch_size=32, seq_len=64):
    # Нарезаем данные на куски длиной seq_len
    num_chunks = len(data) // (seq_len + 1)
    
    x_list = []
    y_list = []
    
    for i in range(num_chunks):
        start_idx = i * (seq_len + 1)
        chunk = data[start_idx : start_idx + seq_len + 1]
        
        # X — текущие символы, Y — сдвинутые на 1 вперед (цели GPT)
        x_list.append(torch.tensor(chunk[:-1]))
        y_list.append(torch.tensor(chunk[1:]))
        
        if len(x_list) == batch_size:
            yield torch.stack(x_list), torch.stack(y_list)
            x_list, y_list = [], []


# =====================================================================
# 2. МНОГОСЛОЙНАЯ АРХИТЕКТУРА DEEP-LEGO-GPT
# =====================================================================
class WideLegoBlock(nn.Module):
    """ Базовый каузальный кубик Лего """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, current_token_embed, carrier):
        combined = torch.cat([current_token_embed, carrier], dim=-1)
        out = F.relu(self.linear(combined))
        next_token_pred, next_carrier = torch.chunk(out, 2, dim=-1)
        return next_token_pred, next_carrier


class LegoLayer(nn.Module):
    """ 
    Один полноценный слой LegoNet. 
    Содержит свой собственный кубик Лего, который прогоняется по всей цепочке времени.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.block = WideLegoBlock(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        B, T, C = x.size()
        
        # Инициализируем чистую память для текущего слоя
        carrier = torch.zeros(B, C, device=x.device)
        layer_outputs = []
        
        # Проходим по цепочке шагов во времени
        for t in range(T):
            current_embed = x[:, t, :]
            pred, carrier = self.block(current_embed, carrier)
            layer_outputs.append(pred)
            
        # Собираем выходы шагов обратно во временную шкалу
        h = torch.stack(layer_outputs, dim=1) # [B, T, C]
        
        # Residual Connection (Сквозная связь) + Нормализация
        return self.ln(x + h)


class DeepLegoGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Стек из нескольких независимых слоев LegoNet (Многослойность!)
        self.layers = nn.ModuleList([LegoLayer(embed_dim) for _ in range(num_layers)])
        
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        h = self.embedding(x)
        
        # Сигнал последовательно проходит через все этажи нашей Лего-башни
        for layer in self.layers:
            h = layer(h)
            
        logits = self.classifier(h)
        return logits

    # Оптимизированный пошаговый инференс для генерации текста по одному символу
    def generate_char(self, input_indices):
        # input_indices: список индексов текущей истории текста
        seq_len = len(input_indices)
        x = torch.tensor([input_indices], device=self.embedding.weight.device)
        
        # Прямой проход по всей накопленной истории (так как слои многоэтажные)
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
            
        # Забираем логиты только для самого последнего, свежего символа на выходе
        last_logits = self.classifier(h[:, -1, :]) # [1, vocab_size]
        return last_logits


# =====================================================================
# 3. НАСТРОЙКА И ОБУЧЕНИЕ НА РЕАЛЬНОМ ТЕКСТЕ
# =====================================================================
# Параметры сети
BATCH_SIZE = 16
SEQ_LEN = 32      # Размер контекстного окна (сколько символов видит сеть за раз)
EMBED_DIM = 64    # Ширина векторов внутри кубиков Лего
NUM_LAYERS = 3    # Сколько слоев (кубиков) ставим друг на друга вглубь

# Проверяем доступность GPU для ускорения обучения реального текста
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство вычислений: {device}")

model = DeepLegoGPT(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

print("\nНачинаем глубокое обучение DeepLegoGPT на вашем input.txt...")
for epoch in range(5): # 5 эпох для демонстрации
    total_loss = 0
    batch_count = 0
    
    # Генератор батчей по вашему файлу
    for x_batch, y_batch in get_batches(data_indices, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(x_batch) # [B, T, vocab_size]
        
        # GPT-сдвиг внутри батча
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = y_batch[..., :-1].contiguous()
        
        loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
        loss.backward()
        
        # Обрезаем градиенты (Gradient Clipping), чтобы глубокие слои не «взрывались»
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Эпоха {epoch+1:02d} | Средний Loss на корпусе: {avg_loss:.4f}")


# =====================================================================
# 4. АВТОРЕГРЕССИОННАЯ ГЕНЕРАЦИЯ С ТЕМПЕРАТУРОЙ (СИМВОЛЬНАЯ)
# =====================================================================
def generate_live_text(model, seed_text, gen_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        print(f"\nЗатравка для глубокой генерации: '{seed_text}'")
        
        # Переводим начальный текст в индексы. Если символов нет в словаре, берем случайный
        generated_indices = []
        for ch in seed_text:
            if ch in char2idx:
                generated_indices.append(char2idx[ch])
            else:
                generated_indices.append(random.choice(list(char2idx.values())))
        
        # Посимвольно дописываем текст
        for _ in range(gen_length):
            # Передаем историю. Чтобы не перегружать память глубоких слоев, 
            # ограничим историю размером нашего окна контекста (SEQ_LEN)
            context_indices = generated_indices[-SEQ_LEN:]
            
            logits = model.generate_char(context_indices)
            
            # Сэмплирование с температурой
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            generated_indices.append(next_char_idx)
            
        # Декодируем символы обратно в строку
        result_text = "".join([idx2char[idx] for idx in generated_indices])
        print(f"Результат DeepLegoGPT:\n{result_text}")

# Запускаем генерацию. В качестве seed_text передайте несколько первых символов из вашего input.txt
# Например, если там про табуреты, напишем "квадратн"
generate_live_text(model, seed_text="квадратн", gen_length=80, temperature=0.7)
generate_live_text(model, seed_text="деревян", gen_length=80, temperature=0.7)

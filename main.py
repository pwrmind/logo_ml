import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import pickle

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ (МАСШТАБИРОВАНИЕ МОДЕЛИ)
# =====================================================================
VOCAB_SIZE = 1000     # Размер BPE словаря
EMBED_DIM = 256       # Ширина сети (Размерность скрытых векторов)
NUM_LAYERS = 4        # Глубина сети (Количество слоев)

SEQ_LEN = 64          # Контекстное окно в ТОКЕНАХ (64 токена ≈ 200 символов)
BATCH_SIZE = 32       # Размер батча
START_LR = 0.002      # Начальная скорость обучения
ETA_MIN = 0.0001      # Минимальная скорость обучения
EPOCHS = 5            # Количество эпох

GEN_TEMPERATURE = 0.8 # Температура сэмплирования
GEN_LENGTH = 60       # Длина генерируемого текста в ТОКЕНАХ

INPUT_FILE = "input.txt"
MODEL_PATH = "lego_bpe_z_gpt.pt"
TOKENIZER_PATH = "bpe_tokenizer.pkl"


# =====================================================================
# 1. СТАБИЛЬНЫЙ BPE ТОКЕНИЗАТОР (БЕЗ ЗАВИСАНИЙ И С КОРРЕКТНЫМ ДЕКОДОМ)
# =====================================================================
class SimpleBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        print("Обучаю BPE-токенизатор на тексте...")
        ids = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            ids = self._merge(ids, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # Применяем слияния строго в том порядке, в котором они учились
        for pair, idx in self.merges.items():
            if len(ids) < 2:
                break
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


# =====================================================================
# 2. ПОДГОТОВКА ДАННЫХ
# =====================================================================
if not os.path.exists(INPUT_FILE):
    print(f"Файл {INPUT_FILE} не найден. Создаю демо-корпус...")
    demo_text = (
        "критическое мышление в EdTech экосистеме развивает симулякры и масштабные тренды. "
        "квадратный бежевый табурет стоял в углу. деревянное сочное кресло привлекало внимание бизнеса. "
    ) * 1000
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(demo_text)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Очищаем старые дефектные кэши
if os.path.exists(TOKENIZER_PATH): os.remove(TOKENIZER_PATH)
if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)

tokenizer = SimpleBPETokenizer(vocab_size=VOCAB_SIZE)
tokenizer.train(raw_text)
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

data_indices = tokenizer.encode(raw_text)
print(f"Размер исходного текста: {len(raw_text)} символов.")
print(f"После BPE токенизации: {len(data_indices)} токенов (сжатие в {len(raw_text)/len(data_indices):.2f} раза!).")

def get_z_bpe_batches(data, batch_size, seq_len):
    num_chunks = len(data) // (seq_len + 1)
    x_list, y_list = [], []
    for i in range(num_chunks):
        start_idx = i * (seq_len + 1)
        chunk = data[start_idx : start_idx + seq_len + 1]
        
        # Каноничный Z-ход (разворачиваем токены задом наперед)
        chunk = chunk[::-1]
        
        x_list.append(torch.tensor(chunk[:-1]))
        y_list.append(torch.tensor(chunk[1:]))
        
        if len(x_list) == batch_size:
            yield torch.stack(x_list), torch.stack(y_list)
            x_list, y_list = [], []


# =====================================================================
# 3. АРХИТЕКТУРА LEGO-GPT (РЕКУРРЕНТНЫЙ КОНВЕЙЕР)
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
            pred, carrier = self.block(x[:, t, :], carrier)
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
        return self.classifier(h)

    def generate_token(self, input_indices):
        x = torch.tensor([input_indices], device=self.embedding.weight.device)
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h[:, -1, :])


# =====================================================================
# 4. ИСПРАВЛЕННОЕ ОБУЧЕНИЕ (ТОЧНЫЙ CAUSAL SHIFT)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}\n")

model = ZDeepLegoGPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

print(f"--- Запуск BPE-обучения Z-DeepLegoGPT ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss, batch_count = 0, 0
    current_lr = optimizer.param_groups[0]['lr']
    
    for x_batch, y_batch in get_z_bpe_batches(data_indices, BATCH_SIZE, SEQ_LEN):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        logits = model(x_batch) # [B, T, VOCAB_SIZE]
        
        # !!! ИСПРАВЛЕННЫЙ ЦЕЛЕВОЙ СДВИГ !!!
        # Поскольку y_batch уже изначально сдвинут в генераторе батчей на 1 шаг вперед,
        # мы сравниваем их напрямую «точка-в-точку» без срезов [:-1], выпрямляя в плоскость.
        loss = criterion(logits.view(-1, VOCAB_SIZE), y_batch.view(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    scheduler.step()
    print(f"Эпоха {epoch+1:02d}/{EPOCHS:02d} | Loss: {total_loss/batch_count:.4f} | LR: {current_lr:.6f}")
    
torch.save(model.state_dict(), MODEL_PATH)
print("Веса модели успешно сохранены на диск.\n")


# =====================================================================
# 5. ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ ТЕКСТА (БЕЗ СЛОМАННЫХ БАЙТОВ UTF-8)
# =====================================================================
def generate_z_bpe_text(model, seed_text, gen_tokens_len, temperature):
    model.eval()
    with torch.no_grad():
        print(f"\nЗатравка (Будущее для Z-модели): '{seed_text}'")
        
        # Кодируем затравку и разворачиваем индексы для Z-прохода
        seed_ids = tokenizer.encode(seed_text)
        reversed_seed = seed_ids[::-1]
        
        generated_indices = list(reversed_seed)
        
        for _ in range(gen_tokens_len):
            context_indices = generated_indices[-SEQ_LEN:]
            logits = model.generate_token(context_indices)
            
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            generated_indices.append(next_token_idx)
            
        # !!! ИСПРАВЛЕНИЕ ОБРЫВА ДЕКОДЕРА !!!
        # Перед тем как скармливать токены BPE-декодеру, мы ОБЯЗАНЫ 
        # вернуть их в нормальный хронологический порядок. Тогда байты UTF-8 склеятся верно.
        correct_order_indices = generated_indices[::-1]
        
        human_readable_text = tokenizer.decode(correct_order_indices)
        print(f"Результат Z-BPE-LegoGPT:\n{human_readable_text}\n")

# Тесты корректной генерации
generate_z_bpe_text(model, seed_text="бизнеса.", gen_tokens_len=GEN_LENGTH, temperature=GEN_TEMPERATURE)

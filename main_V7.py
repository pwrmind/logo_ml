import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import pickle

# =====================================================================
# 0. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ
# =====================================================================
VOCAB_SIZE = 1000
EMBED_DIM = 256
NUM_LAYERS = 4
SEQ_LEN = 64
BATCH_SIZE = 32
START_LR = 0.002
ETA_MIN = 0.0001
EPOCHS = 5
GEN_TEMPERATURE = 0.75
GEN_LENGTH = 50

INPUT_FILE = "input.txt"
MODEL_PATH = "lego_bpe_z_gpt.pt"
TOKENIZER_PATH = "bpe_tokenizer.pkl"


# =====================================================================
# 1. ИСПРАВЛЕННЫЙ BPE ТОКЕНИЗАТОР
# =====================================================================
class SimpleBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        print("Обучаю BPE-токенизатор на тексте...")
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        num_merges = self.vocab_size - 256
        current_ids = list(ids)

        for i in range(num_merges):
            stats = self._get_stats(current_ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            current_ids = self._merge(current_ids, top_pair, new_idx)
            self.merges[top_pair] = new_idx

        for idx in range(256):
            self.vocab[idx] = bytes([idx])
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            available_pairs = [p for p in stats.keys() if p in self.merges]
            if not available_pairs:
                break
            best_pair = min(available_pairs, key=lambda p: self.merges[p])
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


# =====================================================================
# 2. ЗАГРУЗКА ДАННЫХ И ТОКЕНИЗАТОРА (С УЧЁТОМ КЭШИРОВАНИЯ)
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

# Загружаем или обучаем токенизатор
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("Загружен сохранённый токенизатор.")
else:
    tokenizer = SimpleBPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(raw_text)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Токенизатор обучен и сохранён.")

data_indices = tokenizer.encode(raw_text)
print(f"Размер текста: {len(raw_text)} символов, {len(data_indices)} токенов "
      f"(сжатие в {len(raw_text) / len(data_indices):.2f} раза).")

# ---------------------------------------------------------------------
# Генератор батчей с перемешиванием
# ---------------------------------------------------------------------
def get_z_bpe_batches(data, batch_size, seq_len):
    """Создаёт обратные чанки, перемешивает и выдаёт батчами."""
    chunks = []
    # длина чанка seq_len+1, чтобы получить пару x,y
    total_len = len(data)
    # шагаем с перекрытием? Оригинал делил без перекрытия на блоки.
    # Сохраним то же: num_chunks = total_len // (seq_len + 1)
    num_chunks = total_len // (seq_len + 1)
    for i in range(num_chunks):
        start = i * (seq_len + 1)
        chunk = data[start : start + seq_len + 1]
        chunk_rev = chunk[::-1]          # обратный порядок
        x = torch.tensor(chunk_rev[:-1]) # все, кроме последнего
        y = torch.tensor(chunk_rev[1:])  # сдвиг на шаг назад
        chunks.append((x, y))
    random.shuffle(chunks)
    # Формируем батчи
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        if len(batch_chunks) < batch_size:
            continue  # отбрасываем неполный батч (можно и оставить, но упростим)
        x_batch = torch.stack([c[0] for c in batch_chunks])
        y_batch = torch.stack([c[1] for c in batch_chunks])
        yield x_batch, y_batch


# =====================================================================
# 3. АРХИТЕКТУРА LEGO-GPT
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
# 4. ОБУЧЕНИЕ ИЛИ ЗАГРУЗКА МОДЕЛИ
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}\n")

model = ZDeepLegoGPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=START_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

if os.path.exists(MODEL_PATH):
    print(f"Загружена готовая модель '{MODEL_PATH}'.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
else:
    print("Запуск обучения Z-DeepLegoGPT...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batch_count = 0
        current_lr = optimizer.param_groups[0]['lr']

        for x_batch, y_batch in get_z_bpe_batches(data_indices, BATCH_SIZE, SEQ_LEN):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)                     # (B, seq_len, vocab_size)
            # Прямое сравнение со всеми целевыми токенами
            loss = criterion(logits.view(-1, VOCAB_SIZE), y_batch.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / batch_count if batch_count else 0.0
        print(f"Эпоха {epoch+1:02d}/{EPOCHS:02d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Модель сохранена.\n")


# =====================================================================
# 5. ГЕНЕРАЦИЯ ТЕКСТА (ОБРАТНАЯ + ПРЯМАЯ ВЫДАЧА)
# =====================================================================
def generate_z_bpe_text(model, seed_text, gen_tokens_len, temperature):
    """
    Получает seed (фрагмент из будущего) и генерирует текст в прошлое,
    затем возвращает итоговую строку в нормальном (прямом) порядке.
    """
    model.eval()
    with torch.no_grad():
        seed_ids = tokenizer.encode(seed_text)
        if not seed_ids:
            return ""   # пустая затравка
        # Инвертированный порядок для обратной модели
        generated = seed_ids[::-1]

        for _ in range(gen_tokens_len):
            context = generated[-SEQ_LEN:]
            logits = model.generate_token(context)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

        # Возвращаем прямой порядок: разворачиваем весь список
        forward_ids = generated[::-1]
        return tokenizer.decode(forward_ids)

# Демонстрация
print("=" * 60)
print("Генерация текста (режим: будущее → прошлое)")
print("=" * 60)

seed1 = "бизнеса."
text1 = generate_z_bpe_text(model, seed1, GEN_LENGTH, GEN_TEMPERATURE)
print(f"\nЗатравка (будущее): '{seed1}'")
print(f"Сгенерированный контекст:\n{text1}")

seed2 = "симулякры."
text2 = generate_z_bpe_text(model, seed2, GEN_LENGTH, GEN_TEMPERATURE)
print(f"\nЗатравка (будущее): '{seed2}'")
print(f"Сгенерированный контекст:\n{text2}")
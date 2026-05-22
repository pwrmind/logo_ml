import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# =====================================================================
# 1. ТОКЕНИЗАЦИЯ НА УРОВНЕ N-ГРАММ (Корни + Окончания)
# =====================================================================
vocab = [
    "<PAD>", "<BOS>", "<EOS>", 
    "квадратн", "бежев", "деревян",  # Корни прилагательных
    "ый", "ая", "ое",                # Окончания (N-граммы)
    "табурет", "табуретк", "кресл"   # Существительные (задающие род)
]

token2idx = {t: i for i, t in enumerate(vocab)}
idx2token = {i: t for i, t in enumerate(vocab)}
vocab_size = len(vocab)

# Правила согласования n-грамм
# Род существительного строго определяет тип окончания для обоих прилагательных
def generate_ngram_data(num_samples=1000):
    dataset = []
    options = [
        ("табурет", "ый"),    # Мужской род
        ("табуретк", "ая"),   # Женский род
        ("кресл", "ое")       # Средний род
    ]
    roots = ["квадратн", "бежев", "деревян"]
    
    for _ in range(num_samples):
        noun, ending = random.choice(options)
        root1 = random.choice(roots)
        root2 = random.choice(roots)
        while root1 == root2: 
            root2 = random.choice(roots)
        
        # Цепочка n-грамм: <BOS> корень1+окончание корень2+окончание существительное <EOS>
        tokens = ["<BOS>", root1, ending, root2, ending, noun, "<EOS>"]
        indices = [token2idx[t] for t in tokens]
        dataset.append(torch.tensor(indices))
    return dataset

# Генерируем датасет
dataset = generate_ngram_data(1000)
train_data = dataset[:800]
test_data = dataset[800:]


# =====================================================================
# 2. АРХИТЕКТУРА LEGO-GPT (РЕКУРРЕНТНЫЙ КОНВЕЙЕР С ОБЩИМИ ВЕСАМИ)
# =====================================================================
class SharedLegoBlock(nn.Module):
    """ Единственный кубик Лего. Его веса неизменны во времени (Weight Sharing). """
    def __init__(self, embed_dim):
        super().__init__()
        # Принимает текущую n-грамму и вектор контекста из прошлого.
        # Возвращает скрытый вектор для предсказания будущего и обновленный контекст.
        self.linear = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, current_token_embed, carrier):
        combined = torch.cat([current_token_embed, carrier], dim=-1)
        out = F.relu(self.linear(combined))
        next_token_pred, next_carrier = torch.chunk(out, 2, dim=-1)
        return next_token_pred, next_carrier


class RealLegoGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.block = SharedLegoBlock(embed_dim)
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        embeds = self.embedding(x)
        
        # Начальное состояние контекста инициализируется нулями
        carrier = torch.zeros(batch_size, self.embed_dim, device=x.device)
        
        predictions = []
        # Честный последовательный проход. Будущее скрыто физикой цикла `for`.
        for t in range(seq_len):
            current_token = embeds[:, t, :]
            pred, carrier = self.block(current_token, carrier)
            predictions.append(pred)
            
        output_tensor = torch.stack(predictions, dim=1) # [batch_size, seq_len, embed_dim]
        logits = self.classifier(output_tensor)
        return logits


# =====================================================================
# 3. GPT-ОБУЧЕНИЕ СО СДВИГОМ (SHIFT ЛОГИТОВ И ЦЕЛЕЙ)
# =====================================================================
model = RealLegoGPT(vocab_size=vocab_size, embed_dim=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("Обучаем настоящий конвейерный Lego-GPT на n-граммах...")
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    random.shuffle(train_data)
    for sequence in train_data:
        optimizer.zero_grad()
        
        seq = sequence.unsqueeze(0)
        logits = model(seq)
        
        # Классический сдвиг GPT: предсказание токена t должно равняться токену t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = seq[..., 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = shift_logits.argmax(dim=-1)
        correct += (preds == shift_targets).sum().item()
        total += shift_targets.numel()
        
    print(f"Эпоха {epoch+1:02d} | Loss: {total_loss/len(train_data):.4f} | Точность n-грамм: {(correct/total)*100:.1f}%")


# =====================================================================
# 4. АВТОРЕГРЕССИОННЫЙ ИНФЕРЕНС С ТЕМПЕРАТУРОЙ И СЭМПЛИРОВАНИЕМ
# =====================================================================
def generate_lego_text_with_temperature(model, start_root, max_tokens=7, temperature=1.0):
    model.eval()
    with torch.no_grad():
        generated_indices = [token2idx["<BOS>"], token2idx[start_root]]
        
        # Инициализируем стартовую память
        carrier = torch.zeros(1, model.embed_dim)
        
        # Шаг 1: Прогреваем память стартовыми токенами (<BOS> и начальный корень)
        embeds = model.embedding(torch.tensor([generated_indices]))
        pred = None
        for t in range(embeds.size(1)):
            pred, carrier = model.block(embeds[:, t, :], carrier)
            
        # Получаем логиты для третьего токена (первого окончания)
        logits = model.classifier(pred) # [1, vocab_size]
        
        # Применяем температуру и сэмплирование [1]
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token_idx = torch.multinomial(probs[0], num_samples=1).item()
        generated_indices.append(next_token_idx)
        
        # Шаг 2: Генерируем текст дальше авторегрессионно со сложностью O(1)
        for _ in range(max_tokens - 3):
            last_token_tensor = torch.tensor([[generated_indices[-1]]])
            last_embed = model.embedding(last_token_tensor)[:, 0, :]
            
            # Один шаг через кубик Лего — вычисляем будущее, обновляем память
            pred, carrier = model.block(last_embed, carrier)
            
            # Применяем температуру и вероятностное сэмплирование [1]
            logits = model.classifier(pred) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Извлекаем случайный токен на основе распределения вероятностей
            next_token_idx = torch.multinomial(probs[0], num_samples=1).item()
            generated_indices.append(next_token_idx)
            
            if idx2token[next_token_idx] == "<EOS>":
                break
                
        readable = [idx2token[idx] for idx in generated_indices]
        print(f"Результат Lego-GPT ({temperature=}): {' + '.join(readable)}")


# =====================================================================
# 5. ЗАПУСК ГЕНЕРАЦИИ ДЛЯ ПРОВЕРКИ РАЗНООБРАЗИЯ
# =====================================================================
# Запустим генерацию с одной и той же затравкой несколько раз, 
# чтобы увидеть, как температура раскрывает креативность сети.
print("\nПроверяем генерацию с температурным сэмплированием:")
generate_lego_text_with_temperature(model, start_root="деревян", temperature=0.8)
generate_lego_text_with_temperature(model, start_root="деревян", temperature=0.8)
generate_lego_text_with_temperature(model, start_root="деревян", temperature=0.8)
generate_lego_text_with_temperature(model, start_root="деревян", temperature=0.8)

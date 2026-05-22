import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# =====================================================================
# 1. СЛОЖНЫЙ ДАТАСЕТ (ШУМ + НЕОДНОЗНАЧНОСТЬ)
# =====================================================================
# Словарь включает в себя маску, подлежащие, глаголы, предлоги, шум и объекты
vocab = ["<MASK>", "я", "он", "она", "кот", "сова", "купил", "съел", "нашел", "видит",
         "в", "на", "новое", "вкусное", "сочное", "спелое", "старое", "серое", "быстрое",
         "яблоко", "мышь", "зерно"]

word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Набор случайных прилагательных для создания шума в середине предложения
fillers = ["новое", "вкусное", "сочное", "спелое", "старое", "серое", "быстрое"]

# Правила логики (Кто -> Что делает -> С чем)
# Обратите внимание: для одного и того же конца предложения правильными могут быть 
# несколько подлежащих. Модели придется выучить эти вероятности.
rules = [
    (["я", "он", "она"], ["купил", "съел", "видит"], "яблоко"),
    (["кот", "сова", "она"], ["нашел", "съел", "видит"], "мышь"),
    (["кот", "я", "он"], ["нашел", "видит"], "зерно")
]

def generate_complex_sentence():
    # Выбираем случайное логическое правило
    subjects, verbs, object_word = random.choice(rules)
    
    subj = random.choice(subjects)
    verb = random.choice(verbs)
    
    # Случайный шум из прилагательных в середине предложения
    adj1 = random.choice(fillers)
    adj2 = random.choice(fillers)
    adj3 = random.choice(fillers)
    adj4 = random.choice(fillers)
    prep = random.choice(["в", "на"])
    
    # Строим строго фиксированную длину из 8 слов
    tokens = [subj, verb, prep, adj1, adj2, adj3, adj4, object_word]
    return tokens

def make_advanced_data(num_samples=500):
    dataset = []
    for _ in range(num_samples):
        tokens = generate_complex_sentence()
        target_word = tokens[0]  # Цель для предсказания — первое слово
        target_idx = word2idx[target_word]
        
        # Маскируем первое слово (заменяем на <MASK>, индекс 0)
        tokens[0] = "<MASK>"
        input_indices = [word2idx[t] for t in tokens]
        dataset.append((torch.tensor(input_indices), torch.tensor(target_idx)))
    return dataset

# Создаем независимые выборки для обучения и для проверки (валидации)
train_data = make_advanced_data(400)
test_data = make_advanced_data(100)


# =====================================================================
# 2. РАСШИРЕННАЯ АРХИТЕКТУРА Z-ЛЕГЕНДЫ (КУБИКИ ЛЕГО)
# =====================================================================
class WideLegoBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Принимает два вектора размера embed_dim, возвращает объединенный вектор
        self.linear = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, x1, x2):
        # x1, x2 имеют форму [batch_size, embed_dim]
        combined = torch.cat([x1, x2], dim=-1)  # Склеиваем в [batch_size, embed_dim * 2]
        out = F.relu(self.linear(combined))
        
        # Режем вектор ровно пополам на два независимых выхода
        out_a, out_b = torch.chunk(out, 2, dim=-1)
        return out_a, out_b


class LegoZBert(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Таблица эмбеддингов переводящая ID слов в скрытые векторы
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Для 8 слов нам нужно ровно 7 кубиков Лего
        self.num_blocks = seq_len - 1
        self.blocks = nn.ModuleList([WideLegoBlock(embed_dim) for _ in range(self.num_blocks)])
        
        # Финальный классификатор: берет вектор первого слова (маски)
        # и предсказывает логиты для каждого слова из словаря
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, 8] (индексы слов в предложении)
        
        # Переводим индексы в эмбеддинги. Форма: [batch_size, 8, embed_dim]
        embeds = self.embedding(x)
        
        # Разрезаем на список из 8 отдельных тензоров-слов формы [batch_size, embed_dim]
        words = [embeds[:, i, :] for i in range(self.seq_len)]
        
        reversed_outputs = []
        
        # --- ШАГ 1: Стартуем с конца (правый верхний угол буквы Z, индексы 6 и 7) ---
        out_a, out_b = self.blocks[0](words[self.seq_len - 2], words[self.seq_len - 1])
        reversed_outputs.append(out_b)
        carrier_signal = out_a  # Контекстный вектор, который начинает течь влево
        
        # --- ШАГ 2: Двигаемся по Z-лесенке влево ---
        for i in range(1, self.num_blocks):
            input_idx = (self.seq_len - 2) - i  # Индексы идут в обратном порядке: 5, 4, 3...
            
            # Передаем кубику текущее слово и накопленный справа контекст (carrier)
            out_a, out_b = self.blocks[i](words[input_idx], carrier_signal)
            
            reversed_outputs.append(out_b)
            carrier_signal = out_a  # Обновляем транзитный контекст
            
        reversed_outputs.append(carrier_signal)
        final_outputs = reversed_outputs[::-1]  # Разворачиваем список к исходному порядку от 0 до 7
        
        # Извлекаем контекстный вектор для первого слова (где была маска), 
        # который впитал в себя всю информацию справа налево
        masked_word_context = final_outputs[0]  # [batch_size, embed_dim]
        
        # Проектируем вектор на размерность словаря для классификации
        logits = self.classifier(masked_word_context)  # [batch_size, vocab_size]
        return logits


# =====================================================================
# 3. ЦИКЛ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# =====================================================================
# Инициализация модели и оптимизатора
model = LegoZBert(vocab_size=vocab_size, embed_dim=16) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Начинаем жесткий стресс-тест лесенки...")
for epoch in range(10):
    # Режим обучения
    model.train()
    total_loss = 0
    correct_train = 0
    
    random.shuffle(train_data)
    for inputs, target in train_data:
        optimizer.zero_grad()
        
        # Добавляем размерность батча: [8] -> [1, 8]
        inputs = inputs.unsqueeze(0)
        
        outputs = model(inputs)
        loss = criterion(outputs, target.unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if outputs.argmax(dim=-1) == target:
            correct_train += 1
            
    # Режим валидации (проверка на данных, которые модель не видела)
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for inputs, target in test_data:
            inputs = inputs.unsqueeze(0)
            outputs = model(inputs)
            if outputs.argmax(dim=-1) == target:
                correct_test += 1
                
    train_acc = (correct_train / len(train_data)) * 100
    test_acc = (correct_test / len(test_data)) * 100
    print(f"Эпоха {epoch+1:02d} | Loss: {total_loss/len(train_data):.4f} | Точность Обучения: {train_acc:.1f}% | Точность ТЕСТА: {test_acc:.1f}%")


# =====================================================================
# 4. ПРОВЕРКА ИНФЕРЕНСА НА КОНКРЕТНОМ ПРИМЕРЕ
# =====================================================================
print("\nПроверяем работу модели на случайном шумном примере:")
model.eval()
with torch.no_grad():
    # Тестовая фраза со случайным набором прилагательных в середине. В конце стоит 'зерно'.
    # По правилам (rules) для слова 'зерно' подходят ответы: 'кот', 'я', 'он'. Слове 'сова' или 'она' подходить не должны.
    test_phrase = ["<MASK>", "видит", "на", "старое", "быстрое", "серое", "вкусное", "зерно"]
    test_tensor = torch.tensor([[word2idx[w] for w in test_phrase]])
    
    logits = model(test_tensor)
    
    # Выведем топ-3 наиболее вероятных слов по мнению модели
    probabilities = F.softmax(logits, dim=-1)[0]
    top_v, top_i = torch.topk(probabilities, 3)
    
    print(f"Входное предложение: {test_phrase}")
    print("Топ-3 предсказания модели на месте <MASK>:")
    for idx, (val, index) in enumerate(zip(top_v, top_i)):
        print(f"  {idx+1}. '{idx2word[index.item()]}' с уверенностью {val.item()*100:.1f}%")

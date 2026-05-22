import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# =====================================================================
# 1. ДАТАСЕТ ДЛЯ ПРОВЕРКИ КОНТЕКСТА В ЦЕНТРЕ (Маскируем предлог)
# =====================================================================
vocab = ["<MASK>", "я", "он", "она", "кот", "сова", "купил", "съел", "нашел", "видит",
         "в", "на", "новое", "вкусное", "сочное", "спелое", "старое", "серое", "быстрое",
         "яблоко", "мышь", "зерно"]

word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

fillers = ["новое", "вкусное", "сочное", "спелое", "старое", "серое", "быстрое"]
rules = [
    (["я", "он", "она"], ["купил", "съел", "видит"], "яблоко"),
    (["кот", "сова", "она"], ["нашел", "съел", "видит"], "мышь"),
    (["кот", "я", "он"], ["нашел", "видит"], "зерно")
]

def generate_complex_sentence():
    subjects, verbs, object_word = random.choice(rules)
    subj = random.choice(subjects)
    verb = random.choice(verbs)
    
    adj1 = random.choice(fillers)
    adj2 = random.choice(fillers)
    adj3 = random.choice(fillers)
    adj4 = random.choice(fillers)
    
    # Жестко свяжем выбор предлога с глаголом: купил/нашел -> "в" магазине, съел/видит -> "на" столе
    # Это заставит модель смотреть на контекст СЛЕВА (на глагол), чтобы угадать предлог!
    prep = "в" if verb in ["купил", "нашел"] else "на"
    
    return [subj, verb, prep, adj1, adj2, adj3, adj4, object_word]

def make_bidirectional_data(num_samples=600):
    dataset = []
    for _ in range(num_samples):
        tokens = generate_complex_sentence()
        target_word = tokens[2]  # !!! ТЕПЕРЬ ЦЕЛЬ — ПРЕДЛОГ (ИНДЕКС 2, В ЦЕНТРЕ) !!!
        
        # Маскируем ровно третье слово
        tokens[2] = "<MASK>"
        input_indices = [word2idx[t] for t in tokens]
        dataset.append((torch.tensor(input_indices), torch.tensor(word2idx[target_word])))
    return dataset

train_data = make_bidirectional_data(500)
test_data = make_bidirectional_data(100)


# =====================================================================
# 2. ДВУНАПРАВЛЕННАЯ ЛЕГО-СЕТЬ
# =====================================================================
class WideLegoBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        out = F.relu(self.linear(combined))
        out_a, out_b = torch.chunk(out, 2, dim=-1)
        return out_a, out_b


class LegoBiNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Создаем два независимых набора кубиков Лего для двух направлений
        self.forward_blocks = nn.ModuleList([WideLegoBlock(embed_dim) for _ in range(seq_len - 1)])
        self.backward_blocks = nn.ModuleList([WideLegoBlock(embed_dim) for _ in range(seq_len - 1)])
        
        # Финальный классификатор принимает удвоенный вектор признаков (склеенный из fwd и bwd)
        self.classifier = nn.Linear(embed_dim * 2, vocab_size)

    def _run_forward_ladder(self, words):
        # Проход слева направо
        outputs = []
        out_a, out_b = self.forward_blocks[0](words[0], words[1])
        outputs.append(out_a)
        carrier = out_b
        
        for i in range(1, self.seq_len - 1):
            out_a, out_b = self.forward_blocks[i](carrier, words[i + 1])
            outputs.append(out_a)
            carrier = out_b
        outputs.append(carrier)
        return outputs

    def _run_backward_ladder(self, words):
        # Проход справа налево (буква Z)
        outputs = []
        out_a, out_b = self.backward_blocks[0](words[self.seq_len - 2], words[self.seq_len - 1])
        outputs.append(out_b)
        carrier = out_a
        
        for i in range(1, self.seq_len - 1):
            input_idx = (self.seq_len - 2) - i
            out_a, out_b = self.backward_blocks[i](words[input_idx], carrier)
            outputs.append(out_b)
            carrier = out_a
        outputs.append(carrier)
        return outputs[::-1] # возвращаем к нормальному порядку 0 -> 7

    def forward(self, x):
        embeds = self.embedding(x)
        words = [embeds[:, i, :] for i in range(self.seq_len)]
        
        # Запускаем оба потока параллельно
        fwd_outputs = self._run_forward_ladder(words)
        bwd_outputs = self._run_backward_ladder(words)
        
        # Нам нужно предсказать слово на позиции индекса 2 (где стоит маска)
        # Склеиваем вектор из прямого прохода и вектор из обратного прохода для этой позиции
        target_fwd = fwd_outputs[2]
        target_bwd = bwd_outputs[2]
        
        combined_context = torch.cat([target_fwd, target_bwd], dim=-1) # Размеренность: embed_dim * 2
        
        logits = self.classifier(combined_context)
        return logits


# =====================================================================
# 3. СУПЕР-ОБУЧЕНИЕ
# =====================================================================
model = LegoBiNet(vocab_size=vocab_size, embed_dim=16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Обучаем Двунаправленный Лего-Трансформер...")
for epoch in range(5):
    model.train()
    total_loss = 0
    correct_train = 0
    random.shuffle(train_data)
    
    for inputs, target in train_data:
        optimizer.zero_grad()
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        loss = criterion(outputs, target.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if outputs.argmax(dim=-1) == target:
            correct_train += 1
            
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
    print(f"Эпоха {epoch+1:02d} | Loss: {total_loss/len(train_data):.4f} | Точность обучения: {train_acc:.1f}% | Точность ТЕСТА: {test_acc:.1f}%")

# =====================================================================
# 4. ПРОВЕРКА ДВУСТОРОННЕГО МЫШЛЕНИЯ
# =====================================================================
print("\nПроверяем чтение с двух сторон:")
model.eval()
with torch.no_grad():
    # Глагол 'нашел' требует предлог 'в'
    test_phrase = ["кот", "нашел", "<MASK>", "новое", "вкусное", "сочное", "спелое", "мышь"]
    test_tensor = torch.tensor([[word2idx[w] for w in test_phrase]])
    logits = model(test_tensor)
    pred_idx = logits.argmax(dim=-1).item()
    print(f"  Вход: {test_phrase}")
    print(f"  Ответ модели на месте <MASK>: '{idx2word[pred_idx]}' (Ожидается: 'в')")
    
    # Глагол 'видит' требует предлог 'на'
    test_phrase2 = ["кот", "видит", "<MASK>", "новое", "вкусное", "сочное", "спелое", "мышь"]
    test_tensor2 = torch.tensor([[word2idx[w] for w in test_phrase2]])
    logits2 = model(test_tensor2)
    pred_idx2 = logits2.argmax(dim=-1).item()
    print(f"  Вход: {test_phrase2}")
    print(f"  Ответ модели на месте <MASK>: '{idx2word[pred_idx2]}' (Ожидается: 'на')")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# =====================================================================
# 1. ДАТАСЕТ С НЕОДНОЗНАЧНОСТЬЮ
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
    prep = random.choice(["в", "на"])
    
    return [subj, verb, prep, adj1, adj2, adj3, adj4, object_word]

def make_advanced_data(num_samples=500):
    dataset = []
    for _ in range(num_samples):
        tokens = generate_complex_sentence()
        target_word = tokens[0]  # Цель для предсказания — первое слово
        
        tokens[0] = "<MASK>"
        input_indices = [word2idx[t] for t in tokens]
        dataset.append((torch.tensor(input_indices), torch.tensor(word2idx[target_word])))
    return dataset

train_data = make_advanced_data(400)
test_data = make_advanced_data(100)


# =====================================================================
# 2. УНИВЕРСАЛЬНАЯ АРХИТЕКТУРА С ДИНАМИЧЕСКИМ НАПРАВЛЕНИЕМ
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


class LegoConfigurableNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, seq_len=8, direction="backward"):
        super().__init__()
        assert direction in ["forward", "backward"], "Направление должно быть 'forward' или 'backward'"
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.direction = direction
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.num_blocks = seq_len - 1
        self.blocks = nn.ModuleList([WideLegoBlock(embed_dim) for _ in range(self.num_blocks)])
        
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)
        words = [embeds[:, i, :] for i in range(self.seq_len)]
        
        outputs = []

        if self.direction == "backward":
            # =========================================================
            # ОБРАТНОЕ НАПРАВЛЕНИЕ (Буква Z): Справа налево (из будущего в прошлое)
            # =========================================================
            # Стартуем с правого верхнего угла (последние два слова)
            out_a, out_b = self.blocks[0](words[self.seq_len - 2], words[self.seq_len - 1])
            outputs.append(out_b)  # Сохраняем правый выход
            carrier_signal = out_a # Левый выход — транзитный контекст (течет влево)
            
            # Двигаемся по лесенке влево
            for i in range(1, self.num_blocks):
                input_idx = (self.seq_len - 2) - i
                out_a, out_b = self.blocks[i](words[input_idx], carrier_signal)
                outputs.append(out_b)
                carrier_signal = out_a
                
            outputs.append(carrier_signal)
            # Так как собирали справа налево (7 -> 0), разворачиваем список обратно
            final_outputs = outputs[::-1]
            
            # Для предсказания первого замаскированного слова берем самый левый контекст
            target_context = final_outputs[0]

        else:
            # =========================================================
            # ПРЯМОЕ НАПРАВЛЕНИЕ (Зеркальная Z): Слева направо (из прошлого в будущее)
            # =========================================================
            # Стартуем с левого верхнего угла (первые два слова)
            out_a, out_b = self.blocks[0](words[0], words[1])
            outputs.append(out_a)  # Сохраняем левый выход
            carrier_signal = out_b # Правый выход — транзитный контекст (течет вправо)
            
            # Двигаемся по лесенке вправо
            for i in range(1, self.num_blocks):
                input_idx = i + 1
                out_a, out_b = self.blocks[i](carrier_signal, words[input_idx])
                outputs.append(out_a)
                carrier_signal = out_b
                
            outputs.append(carrier_signal)
            final_outputs = outputs # Здесь порядок уже правильный (0 -> 7)
            
            # Для предсказания первого слова в прямой сети контекста «из будущего» нет.
            # Но мы можем посмотреть, что накопилось в самом конце последовательности (канал 7)
            target_context = final_outputs[0] # Берем первый доступный элемент для классификатора

        logits = self.classifier(target_context)
        return logits


# =====================================================================
# 3. ТЕСТИРОВАНИЕ И СРАВНЕНИЕ ДВУХ НАПРАВЛЕНИЙ
# =====================================================================
def train_and_evaluate(direction_name):
    print(f"\n{'=' * 60}\nЗАПУСК ОБУЧЕНИЯ ДЛЯ НАПРАВЛЕНИЯ: {direction_name.upper()}\n{'=' * 60}")
    
    model = LegoConfigurableNet(vocab_size=vocab_size, embed_dim=16, direction=direction_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(5): # Сделаем 5 эпох для быстрой демонстрации разницы
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

    # Инференс на тесте
    model.eval()
    with torch.no_grad():
        test_phrase = ["<MASK>", "видит", "на", "старое", "быстрое", "серое", "вкусное", "зерно"]
        test_tensor = torch.tensor([[word2idx[w] for w in test_phrase]])
        logits = model(test_tensor)
        probabilities = F.softmax(logits, dim=-1)
        top_v, top_i = torch.topk(probabilities, 2)
        
        print(f"\nПроверка после обучения ('{direction_name}'):")
        print(f"  Вход: {test_phrase}")
        print(f"  Топ-1: '{idx2word[top_i[0][0].item()]}' ({top_v[0][0].item()*100:.1f}%)")
        print(f"  Топ-2: '{idx2word[top_i[0][1].item()]}' ({top_v[0][1].item()*100:.1f}%)")

# Запускаем оба эксперимента по очереди
train_and_evaluate(direction_name="backward")
train_and_evaluate(direction_name="forward")

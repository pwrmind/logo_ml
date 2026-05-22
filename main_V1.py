import torch
import torch.nn as nn
import torch.nn.functional as F

class LegoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Принимает 2 сигнала, возвращает 2 сигнала
        self.linear = nn.Linear(2, 2)

    def forward(self, x1, x2):
        combined_input = torch.cat([x1, x2], dim=-1)
        combined_output = F.relu(self.linear(combined_input))
        out_a, out_b = torch.chunk(combined_output, 2, dim=-1)
        return out_a, out_b


class LegoZLadderNet(nn.Module):
    def __init__(self, input_width=8):
        super().__init__()
        self.input_width = input_width
        self.num_blocks = input_width - 1
        self.blocks = nn.ModuleList([LegoBlock() for _ in range(self.num_blocks)])

    def forward(self, x):
        """
        x: Входной тензор формы [batch_size, 8]
        """
        # Разрезаем вход на 8 каналов
        inputs = torch.chunk(x, self.input_width, dim=-1)
        
        # Сюда мы будем складывать результаты. 
        # Чтобы не путаться с индексами, мы будем собирать их справа налево 
        # и в конце просто развернем список.
        reversed_outputs = []
        
        # --- ШАГ 1: Стартуем с правого верхнего угла буквы Z (каналы 6 и 7) ---
        out_a, out_b = self.blocks[0](inputs[self.input_width - 2], inputs[self.input_width - 1])
        
        # Допустим, out_b — это правый выход (уходит в финальный результат для канала 7)
        # а out_a — это левый выход (транзит, который двигается влево)
        reversed_outputs.append(out_b)
        carrier_signal = out_a
        
        # --- ШАГ 2: Двигаемся по лесенке влево ---
        # Проходим по оставшимся кубикам и входам в обратном порядке: от 5 до 0
        for i in range(1, self.num_blocks):
            # Берем индекс входного канала, двигаясь справа налево
            input_idx = (self.input_width - 2) - i  # На первой итерации это будет 5, затем 4...
            
            # Передаем кубику текущий транзит и следующий вход слева
            out_a, out_b = self.blocks[i](inputs[input_idx], carrier_signal)
            
            # Сохраняем правый выход и обновляем транзит, идущий влево
            reversed_outputs.append(out_b)
            carrier_signal = out_a
            
        # У самого последнего (левого нижнего) кубика забираем его последний левый выход
        reversed_outputs.append(carrier_signal)
        
        # Так как мы собирали выходы от 7-го к 0-му, разворачиваем список обратно
        final_outputs = reversed_outputs[::-1]
        
        # Склеиваем в итоговый тензор шириной 8
        return torch.cat(final_outputs, dim=-1)

# Тестируем новую Z-архитектуру
model = LegoZLadderNet(input_width=8)
test_input = torch.randn(3, 8)

test_output = model(test_input)

print("Входная форма:", test_input.shape)   # [3, 8]
print("Выходная форма:", test_output.shape) # [3, 8]

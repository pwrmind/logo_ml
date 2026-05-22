import torch
import torch.nn as nn
import torch.nn.functional as F

class LegoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Принимает 2 сигнала, возвращает 2 сигнала
        self.linear = nn.Linear(2, 2)

    def forward(self, x1, x2):
        # x1 и x2 — это скаляры (или батчи скаляров размера [batch_size, 1])
        # Склеиваем их вместе для подачи в слой
        combined_input = torch.cat([x1, x2], dim=-1)
        
        # Пропускаем через нейроны кубика с активацией
        combined_output = F.relu(self.linear(combined_input))
        
        # Разделяем обратно на два раздельных выхода
        out_a, out_b = torch.chunk(combined_output, 2, dim=-1)
        return out_a, out_b


class LegoLadderNet(nn.Module):
    def __init__(self, input_width=8):
        super().__init__()
        self.input_width = input_width
        
        # Для входного слоя шириной 8 нам нужно ровно 7 кубиков Лего
        self.num_blocks = input_width - 1
        self.blocks = nn.ModuleList([LegoBlock() for _ in range(self.num_blocks)])

    def forward(self, x):
        """
        x: Входной тензор формы [batch_size, 8]
        """
        batch_size = x.size(0)
        
        # Разрезаем входной тензор на 8 отдельных каналов формы [batch_size, 1]
        inputs = torch.chunk(x, self.input_width, dim=-1)
        
        final_outputs = []
        
        # --- ШАГ 1: Первый кубик Лего (особый, берет первые два входа) ---
        out_a, out_b = self.blocks[0](inputs[0], inputs[1])
        final_outputs.append(out_a) # Первый выход идет в финальный результат
        
        # Переменная для хранения «транзитного» сигнала, идущего к следующему кубику
        carrier_signal = out_b 
        
        # --- ШАГ 2: Остальные кубики лесенкой ---
        for i in range(1, self.num_blocks):
            # Каждому следующему кубику передаем транзитный сигнал и свежий вход из общего слоя
            out_a, out_b = self.blocks[i](carrier_signal, inputs[i + 1])
            
            final_outputs.append(out_a) # Записываем первый выход кубика
            carrier_signal = out_b      # Обновляем транзитный сигнал для следующего шага
            
        # У самого последнего кубика забираем и второй выход тоже, как вы просили
        final_outputs.append(carrier_signal)
        
        # Склеиваем все 8 выходов обратно в один монолитный тензор
        return torch.cat(final_outputs, dim=-1)

# Создаем тестовую модель и батч из 3 примеров с шириной входа 8
model = LegoLadderNet(input_width=8)
test_input = torch.randn(3, 8)

# Прямой проход
test_output = model(test_input)

print("Входная форма:", test_input.shape)   # [3, 8]
print("Выходная форма:", test_output.shape) # [3, 8]

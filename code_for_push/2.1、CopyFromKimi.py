"""
name：2024 & 2、CopyFromKimi
time：2024/6/16 13:41
author：yxy
content：
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设定随机种子以保证结果可复现
torch.manual_seed(0)

# 假设 temp_array 和 wind_array 是已经加载的NumPy数组
# 这里需要根据实际数据结构进行调整
temp_path = "./global/temp.npy"
wind_path = "./global/wind.npy"
temp_array = np.load(temp_path)
wind_array = np.load(wind_path)

# 数据预处理：重塑数据以符合LSTM的输入要求
# 假设数据形状为 (time_steps, num_sites, 1)
num_sites = temp_array.shape[1]
time_steps = temp_array.shape[0]

# 将数据转换为PyTorch张量
temp_tensor = torch.tensor(temp_array.reshape(-1, 1), dtype=torch.float32)
wind_tensor = torch.tensor(wind_array.reshape(-1, 1), dtype=torch.float32)

# 创建数据集和数据加载器
temp_dataset = TensorDataset(temp_tensor, temp_tensor)  # 使用同一数据集作为特征和目标
wind_dataset = TensorDataset(wind_tensor, wind_tensor)

batch_size = 64
temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True)
wind_loader = DataLoader(wind_dataset, batch_size=batch_size, shuffle=True)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        output = self.linear(hn[-1])  # 取最后一个隐藏状态
        return output


# 实例化模型
input_size = 1  # 特征数量
hidden_layer_size = 50  # 隐藏层大小
output_size = 1  # 输出数量
model = LSTMModel(input_size, hidden_layer_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    for data, target in temp_loader:  # 假设使用温度数据作为示例
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    test_output = model(temp_tensor.unsqueeze(1))  # 增加维度以匹配LSTM输入

# 保存模型的预测结果
predictions = test_output.numpy().reshape(time_steps, num_sites)
np.save('temp_predictions.npy', predictions)

# 注意：这个示例代码假设了数据已经是二维的，并且每个站点只有一个特征。
# 实际上，你可能需要根据你的数据结构和任务要求进行适当的调整。

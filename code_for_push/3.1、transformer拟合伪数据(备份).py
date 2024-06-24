"""
name：2024 & 3、transformer拟合伪数据
time：2024/6/18 21:50
author：yxy
content：防止代码3.1、transformer拟合伪数据.py被该乱 | 备份
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
torch.manual_seed(0)
np.random.seed(0)


# 生成伪时序数据
def generate_sine_wave_data(seq_length, num_samples):
    data = np.zeros((num_samples, seq_length))
    for i in range(num_samples):
        t = np.arange(seq_length)
        data[i] = np.sin(t) + np.random.normal(scale=0.2, size=t.shape)  # 添加噪声
    return data


# 定义时序数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.positional_encoding = self._generate_positional_encoding(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]     #
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])  # 取序列的最后一个时间点作为输出
        return out.view(-1,1)

    def _generate_positional_encoding(self, dim, max_len=4):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# 配置参数
seq_length = 10  # 序列长度
input_dim = 10  # 输入特征维度
num_heads = 2  # 注意力头数
num_encoder_layers = 2  # 编码器层数
num_classes = 1  # 输出类别数
num_samples = 100  # 样本数量

# 生成数据
data = generate_sine_wave_data(seq_length, num_samples)
data = torch.tensor(data, dtype=torch.float32)      # 将数据转换为torch.tensor
target = data[:, -1]  # 用序列的最后一个时间点作为目标

# 创建数据集和数据加载器
dataset = TimeSeriesDataset(data, target)
# ,batch_size=4 一次训练4个批次(即，4个向量),shuffle每次都是随机抽取(直到抽完)。
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)        # 创建对应的dataset，并以参数的形式传递个DataLoader

# 初始化模型
model = TransformerModel(input_dim, num_heads, num_encoder_layers, num_classes)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
loss_keep = []       # 保存损失函数
for epoch in range(num_epochs):
    model.train()
    loss_ = 0
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # loss是一个张量，需要保存loss里面的值。# 张量里面的值。
        loss_ = loss.item()

    # 每一轮保存一个loss
    # 直接.append就行，不要重新赋值
    loss_keep.append(loss_)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# 持久化--损失函数
import json
with open('loss.json', 'w', encoding='utf-8') as f:
    loss_obj = {"loss":loss_keep}
    json.dump(loss_obj, f, ensure_ascii=False, indent=4)

# 测试模型
model.eval()

all_predictions = []
all_targets = []
with torch.no_grad():
    for batch in dataloader:
        inputs, targets = batch
        predictions = model(inputs)
        # 这里可以添加代码来评估模型的性能，例如计算RMSE等
        all_predictions.extend(predictions.cpu().numpy())  # 收集预测值
        all_targets.extend(targets.cpu().numpy())  # 收集目标值

# 将列表转换为 numpy 数组以便于绘图
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# 可视化预测结果和实际值
plt.figure(figsize=(10, 5))
plt.plot(all_targets, label='Actual Values')
plt.plot(all_predictions, label='Predictions', linestyle='--')
plt.title('Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
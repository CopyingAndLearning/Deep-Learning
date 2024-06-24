"""
name：2024 & 4.2、拟合情况2
time：2024/6/22 19:46
author：yxy
content：
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# -------------------- 数据加载 ------------------------ #
## 假设已经加载了数据
# temp_data = np.load('global/temp.npy')  # (17544, 3850, 1)
# wind_data = np.load('global/wind.npy')  # (17544, 3850, 1)
# global_data = np.load('global/global_data.npy')  # (5848, 4, 9, 3850)

## 防止重复加载数据
temp_data = torch.rand((17544,3850,1))
wind_data = torch.rand((17544,3850,1))
global_data = torch.rand((5848,4,9,3850))

# 预处理数据，例如标准化、重塑等
# ...

# ------------------ 构建数据集 ------------------------- #
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels_temp, labels_wind):
        self.features = features
        self.labels_temp = labels_temp
        self.labels_wind = labels_wind

    def __len__(self):
        return len(self.labels_temp)

    def __getitem__(self, idx):
        return self.features[idx], self.labels_temp[idx], self.labels_wind[idx]

# --------------------- 构建模型 ------------------------- #
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        # 定义模型各部分
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_encoder_layers)
        self.decoder = nn.Linear(input_dim, num_classes)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (seq_length, num_features)
        # 在神经网络中，经过对应的变换后shap保持稳定
        x = self.transformer_encoder(x)
        out = self.decoder(x)
        out = self.fc(out)
        return out


# 假设进行了必要的预处理，使得协变量和因变量具有相同的时间步长
# 例如，将协变量从 (5848, 4, 9, 3850) 重塑为 (5848, 3850 * 4 * 9)

reshape_var = global_data.reshape(-1, 3850)
# 将协变量和因变量组合为一个特征矩阵
features = np.concatenate((reshape_var, temp_data.squeeze(-1), wind_data.squeeze(-1)), axis=-1)


dataset = TimeSeriesDataset(features, temp_data, wind_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(input_dim=10, num_heads=2, num_encoder_layers=2, num_classes=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
# 训练循环
for epoch in range(num_epochs):
    model.train()
    for features, labels_temp, labels_wind in dataloader:
        features = features.to(device)
        labels_temp = labels_temp.to(device)
        labels_wind = labels_wind.to(device)

        # 前向传播
        outputs = model(features)
        loss_temp = criterion(outputs[:, :1], labels_temp)
        loss_wind = criterion(outputs[:, 1:], labels_wind)
        loss = loss_temp + loss_wind

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"第{epoch}轮,loss={loss.item()}")
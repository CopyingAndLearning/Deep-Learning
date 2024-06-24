"""
name：2024 & 3、test
time：2024/6/18 21:50
author：yxy
content：
"""
import math
import numpy as np

# -------------------- 参数定义 ------------------------ #
import torch
torch.manual_seed(0)
np.random.seed(0)
# ----------------------- 定义参数&训练 ------------------------------#
## 定义参数
### 序列长度是单个序列中元素（或时间步）的数量。(比如：一天24h，每一个小时的数据才有意义)   # 实际意义
### 有几个向量。                                                               # 代码意义
seq_length = 10  # 序列长度
### 输入特征维度是影响最终结果的因变量。             # 实际意义
#### 比如，1h中有风速、风向决定了温度。这两个就是特征维度。
### 每个向量的维度，或说列表中的元素的个数           # 代码意义
input_dim = 10  # 输入特征维度
num_heads = 2  # 注意力头数
num_encoder_layers = 2  # 编码器层数
num_classes = 1  # 输出类别数
num_samples = 100  # 样本数量

# --------------------  伪数据的生成  --------------------- #
## 封装
def generate_sine_wave_data(seq_length, num_samples):
    data = np.zeros((num_samples, seq_length))
    for i in range(num_samples):
        t = np.arange(seq_length)
        data[i] = np.sin(t) + np.random.normal(scale=0.2, size=t.shape)  # 添加噪声
    return data

## 测试
# ### 现实中就是要将对应真实存在的非数值 | 数值类型的数据进行对应的编码后，才能放入模型当中使用。
# num_samples = 100   # 样本数量
# seq_length = 10     # 序列长度
# data = np.zeros((num_samples, seq_length))      # 创建对应的矩阵（宽 * 高）：向量个数 * 序列长度 = 100 * 10
# for i in range(num_samples):
#     t = np.arange(seq_length)       # [0,1,...,9]
#     # 对张量(数组)求sin操作，并加上对应的噪声
#     # 将数据放到对应的零矩阵当中
#     data[i] = np.sin(t) + np.random.normal(scale=0.2, size=t.shape)  # 添加噪声
#
# # 对得到的数据调用，绘制时间序列图
# # print(data)
# # print(data.shape)
# # print(np.sin([math.pi/6,math.pi/3,math.pi/2,math.pi]))        # [5.00000000e-01 8.66025404e-01 1.00000000e+00 1.22464680e-16]
#
# ## 数组切片
# target = data[:, -1]  # 用序列的最后一个时间点作为目标
# # print(target)




# -------------------- 模型的构建 ------------------------------ #
## 创建时序数据集
from torch.utils.data import Dataset, DataLoader                # 创建对应数据集必须继承的父类。
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):                                          # 通过len(对象)可以返回类中的长度。可以自定义处理方法。
        return len(self.target)

    def __getitem__(self, idx):                                 # 类中的魔术方法，是该类可以像数组一样被访问。
        return self.data[idx], self.target[idx]

## 生成数据
data = generate_sine_wave_data(seq_length, num_samples)
data = torch.tensor(data, dtype=torch.float32)      # 将数据转换为torch.tensor
target = data[:, -1]  # 用序列的最后一个时间点作为目标
dataset = TimeSeriesDataset(data, target)
# print(target.shape)       # torch.Size([100])
# print(data.shape)         # torch.Size([100, 10])

## 创建对应的dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# print(list(dataloader)[0])          # 可以看到数组0的位置有4个元素 # 所以这就是batch_size的原因   # 只有某些函数能够接受lambda表达式
# print(list(dataloader)[0][0])

## 构建类
from torch import nn
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes

        # 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,batch_first=True)     # d_model规定向量输入的维度，num_heads是多头注意力的头数
        # tfm的编码器
        ## 编码器由多个编码器层堆叠而成 # num_encoder_layers决定了编码器层数
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 位置编码器
        ## self._generate_positional_encoding(input_dim)，这就是self.__func__ 可以无视位置，调用类中的任意方法。（为什么：可能是因为类方法底层都是静态方法吧）
        self.positional_encoding = self._generate_positional_encoding(input_dim)
        # 全连接层，做预测用的
        ## input_dim 输入向量的维度([0,1]维度为2), num_classes为分类的类别，这你做的预测所以分类类别为1.
        ### 增加模型复杂度，拟合数据的能力更强   ### 因为他会反馈 tfm提取特征是否是想要的特征，进而提升模型的精度
        self.fc = nn.Linear(input_dim, num_classes)

    # forwa函数在底层会自动被类中的魔术方法：__call__()自动调用
    ## nn.Module  1534 - 1541
    ### If we don't have any hooks, we want to skip the rest of the logic in
        # # this function, and just call forward.
        # if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
        #         or _global_backward_pre_hooks or _global_backward_hooks
        #         or _global_forward_hooks or _global_forward_pre_hooks):
        #     return forward_call(*args, **kwargs)
    def forward(self, x):
        # print(x)
        # print(x.size(0))
        x = x + self.positional_encoding[:x.size(0), :]
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])  # 取序列的最后一个时间点作为输出
        return out.view(-1,1)       # 解决维度不匹配的问题

    # 位置编码器
    ## 知道怎么计算就行，至于原理和修改，以后的事情
    ### 通过max_len限制每次训练的最大批量。 # max_len要进行对应的修改
    ### 即一次性训练几个向量，对几个向量进行位置编码。
    #### 为什么位置编码器产生的数据只需要简单相加。
    def _generate_positional_encoding(self, dim, max_len=4):     # max_len，即一次给多少个向量进行位置编码，进而得到特征向量。通常是和模型的batch相匹配。
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# ----------------------- 训练 ------------------------------#
## 模型调用
### input_dim是输入的维度，num_heads是注意力机制的头数
model = TransformerModel(input_dim, num_heads, num_encoder_layers, num_classes)

### 直接调用模型参数
model.load_state_dict(torch.load('model_parameters_100.pth'))

# ## 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# ## 训练模型
# ### 训练次数
# num_epochs = 100
#
# ### 更新全连接层的参数。
# #### 模型训练的部分，勿打开。
# for epoch in range(num_epochs):
#     # 开启训练模式
#     model.train()
#     # 获取批次数据
#     for batch in dataloader:
#         # 输入和输出 # 自监督
#         inputs, targets = batch
#         # 梯度清零
#         ## 梯度不清零，会造成梯度的累加，影响模型的精度。# 不是影响参数，切记。是影响改变参数的变量。
#         optimizer.zero_grad()
#         # 获得输出，并通过对应的损失函数计算损失
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         # 根据损失，反向传播更新参数
#         loss.backward()         # 计算需要更新的参数的值，例如：y=kx+b中的k下降了多少。
#         optimizer.step()        # 把上述得到的值在升级网络中进行更新
#     # 打印训练的情况
#     # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
#
# ## 训练完之后保存参数
# torch.save(model.state_dict(), 'model_parameters.pth')


# ---------------------- 预测部分 ----------------------------------------- #
## 开启评估模式
model.eval()

## 用原数据集进行拟合，并绘制图像
import matplotlib.pyplot as plt         # 导入绘图的包
all_predictions = []                    # 简单的数据结构，用于获取对应的数据
all_targets = []

loss_history = []                       # 用于计算损失函数
# print(len(dataloader))                  # 打印dataloader的长度     # 25 * 4 = 100
with torch.no_grad():                 # 禁用梯度计算，减少CPU的耗费。# 当然这是在训练完之后查看 # 有些情况下需要再训练时查看，这样的话才比较好
    for batch in dataloader:
        inputs, targets = batch
        # print(targets)
        for input in inputs:
            ## 预测的时候要对每个inputs里面的值进行拟合
            predictions = model(input)
            # 这里可以添加代码来评估模型的性能，例如计算RMSE等
            all_predictions.extend(predictions.cpu().numpy())  # 收集预测值
        all_targets.extend(targets.cpu().numpy())  # 收集目标值

# print(len(all_targets))       # 100
# print(len(all_predictions))   # 25
# print(all_targets)
# print(all_predictions)

# 将列表转换为 numpy 数组以便于绘图
## 为什么会出现维度不匹配的问题
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# 可视化预测结果和实际值
# plt.figure(figsize=(10, 5))
# # plt.plot(all_targets[:25], label='Actual Values')
# plt.plot(all_targets, label='Actual Values')
# plt.plot(all_predictions, label='Predictions', linestyle='--')
# plt.title('Predictions vs Actual Values')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# ------------------ 绘制损失函数的图像 -------------------- #
# 详情件3.1、test3


# ---------------- 定量指标的评测 ------------------------ #
## 转换为torch方便计算
all_predictions = torch.tensor(all_predictions)
all_targets = torch.tensor(all_targets)

## MSE
### 真实值 - 预测值  # 类似于均方误差。
mse_loss = torch.nn.MSELoss()
mse = round(mse_loss(all_predictions.unsqueeze(1), all_targets.unsqueeze(1).unsqueeze(1)).item(),5)
print(f"MSE: {mse}")

## R^2
# ss_res = torch.sum((all_predictions - all_targets) ** 2)
# ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
# r2 = 1 - (ss_res / ss_tot)
# print(f"R² Score: {r2}")
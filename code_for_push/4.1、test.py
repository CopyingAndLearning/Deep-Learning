"""
name：2024 & 4.1 test
time：2024/6/22 17:54
author：yxy
content：思考数据的操作维度 | 真实数据的维度调整
"""

import numpy as np
import math
import torch

def generate_sine_wave_data(seq_length, num_samples):
    data = np.zeros((num_samples, seq_length))
    for i in range(num_samples):
        t = np.arange(seq_length)
        data[i] = np.sin(t) + np.random.normal(scale=0.2, size=t.shape)  # 添加噪声
    return data

# 位置编码器参数的数据最后要加到对应的维度上
def _generate_positional_encoding(self, dim, max_len=4):     # max_len，即一次给多少个向量进行位置编码，进而得到特征向量。通常是和模型的batch相匹配。
    pe = torch.zeros(max_len, dim)      # 4 * dim  #  4 * 4
    # print(pe.shape)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # print(position.shape)               # 4*1
    # 最计算用的，并将其赋值到对应的矩阵当中
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    print(div_term.shape)               # 2
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # print(pe.shape)                   # 并没用到所谓的矩阵乘法
    return pe.unsqueeze(0)

data = generate_sine_wave_data(10,100)
print(data.shape)
print((_generate_positional_encoding(data,4)).shape)
"""
name：2024 & 4.1、生成多对多的数据并拟合
time：2024/6/22 17:19
author：yxy
content：
"""
import torch
def generate_sine_wave_data(seq_length, num_samples, num_features):
    data = torch.zeros(num_samples, seq_length, num_features)
    for i in range(num_samples):
        t = torch.arange(seq_length)
        for j in range(num_features):
            data[i] = torch.sin(t) + torch.rand(t.shape) * 0.2  # 添加噪声
    return data

data = generate_sine_wave_data(1,100,4)
print(data)
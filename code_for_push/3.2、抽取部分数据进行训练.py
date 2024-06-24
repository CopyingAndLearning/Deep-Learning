"""
name：2024 & 3、抽取部分数据进行训练
time：2024/6/18 21:46
author：yxy
content：
"""

# import numpy as np
#
# # 假设 data 是包含所有数据的 NumPy 数组
# # 假设 targets 是包含对应目标值的数组
# np.random.seed(42)  # 设置随机种子以确保结果可复现
# indices = np.arange(data.shape[0])  # 获取数据索引
# np.random.shuffle(indices)  # 随机打乱索引
#
# split_size = int(0.1 * len(data))  # 假设我们想要10%的数据作为训练集
# train_indices = indices[:split_size]
# train_data = data[train_indices]
# train_targets = targets[train_indices]
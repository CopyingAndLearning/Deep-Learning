"""
name：2024 & 3.1 test2
time：2024/6/18 22:50
author：yxy
content：输入维度和多头的数量关系
"""

# # 假设 d_model 是模型的维度
# d_model = 512  # 例如，每个输入向量的维度是 512
#
# # 假设 num_heads 是多头注意力机制中头的数量
# num_heads = 8  # 8 能够整除 512，所以这里没有问题
#
# # 确保 d_model % num_heads == 0
# assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

# 2024-6-19
## 可迭代对象
# a = [1, 2, 3]
# b = [4, 5, 6]
# a.extend(b)
# # 现在 a 是 [1, 2, 3, 4, 5, 6]

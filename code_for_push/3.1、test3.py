"""
name：2024 & 3.1、test3
time：2024/6/22 11:39
author：yxy
content：保存变量参数
"""

# import pickle
#
# # 假设有一个列表
# data = [1, 2, 3, 4, 5]
#
# # 使用pickle保存数据
# # 用特殊的数据结构进行保存，读取也需要特殊的数据结构。
# with open('data.pkl', 'wb') as f:
#     pickle.dump(data, f)

import json

# 假设有一个字典
# data = {'name': 'Kimi', 'age': 1, 'skills': ['聊天', '搜索', '文件处理']}

# # 将字典保存为JSON文件
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

# 打开文件并读取JSON数据
# 数据的读取和其他文件读取是一样的
with open('loss.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 打印读取的数据
print(data)

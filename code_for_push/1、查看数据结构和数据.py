"""
name：2024 & 1、查看数据结构和数据
time：2024/6/16 12:29
author：yxy
content：数据预处理 | 查看数据结构和数据 | 绘制时序图
"""

import numpy as np      # 读取数据
import pandas as pd     # 处理数据
from sklearn.preprocessing import MinMaxScaler    # 数据归一化处理


# 指定.npy文件的路径
temp_path = "./global/temp.npy"
wind_path = "./global/wind.npy"

# 使用numpy的load函数来加载.npy文件
temp_array = np.load(temp_path)
wind_array = np.load(wind_path)


# 将NumPy数组转换为DataFrame(代码的作用)
column_names = ["temp","wind"]                 # 根据实际情况调整列名(备注)

# -------- 封装(x) --------- #
# #判断np数据的维度
# def judge_npdim_ToDf(array):
#     if array.ndim == 2:
#         df = pd.DataFrame(array, columns=column_names)
#     else:
#         # 如果数组是一维的，需要先将其重塑为二维
#         df = pd.DataFrame(array.reshape(-1, 1), columns=column_names)
#     return df
# df1 = judge_npdim_ToDf(temp_array)
# df2 = judge_npdim_ToDf(wind_array)
# ------------------------ #


# -------- 读取数据，并展示 --------- #
#分别创建两个不同的df结构数据
df1 = pd.DataFrame(temp_array.reshape(-1,1),columns=[column_names[0]])      # columns接受的参数为一个集合类型的。
df2 = pd.DataFrame(wind_array.reshape(-1,1),columns=[column_names[1]])
# 合并两个df
df = pd.concat([df1, df2], axis=1)     # 指定合并的维度
# 显示DataFrame的前5行
# print(df.head())    # 传递参数为多少，则查看多少列

# --------- 去重、去空、归一化 ----------- #

##### 去空 #####
## 无空值 ##
# 获取含有缺失值的行索引       # 列
# df的索引可以接受那些索引值为布尔类型的情况
# rows_with_missing_values = df[df.isnull().any(axis=1)].index        # .any表示对于任意的数据而言，可以指定列 # axis指定列，默认是行
# # 获取含有缺失值的列索引     # 行
# columns_with_missing_values = df.columns[df.isnull().any()]
#
# # 打印结果
# print(rows_with_missing_values)
# print(columns_with_missing_values)          # 发现没有空的行值或列值。
# print(df.isnull().any())                    # 打印相关信息

# 其他查看行和列空值
# 显示DataFrame的信息，包括每列的非空值数量
# df.info()

##### 去重 #####
## 去重前后差值太大，故不去重！
# 查看去重前后值的形状是否相等,判断是否存在重复值
# print(df.shape)                         # 输出：(67544400, 2)
# df.drop_duplicates(inplace=True)        # 原地删除重复值 # 原地删除后,原数组发生了变化
# print(df.shape)                         # 输出：(129506, 2)

##### 归一化 #####
## 归一化，可以加快模型的收敛速度
# 如果，效果不好，回头来修改一下就行。

# 假设df是你的DataFrame
# 选择需要标准化的列
# df_standardized = df.copy()                # 建议不拷贝，数据量太大了。浪费时间

# 对每一列分别进行标准化
# for column in df.columns:                 # 获取colums中的每一个column，对每一个column进行归一化
#     if df[column].dtype.kind in 'biufc':  # 检查数据类型是否为数值型
#         scaler = MinMaxScaler(feature_range=(0,1))      # 将值映射到0-1空间
#         df[column] = scaler.fit_transform(df[[column]])

### 联合归一化
# 选择所有数值列
# numeric_cols = df.select_dtypes(include=['number']).columns
# # 初始化 MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# # 对所有数值列进行归一化
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# # 查看归一化后的数据
# print(df.head())


# ------------------------ 查看global_data.npy数据，并进行预处理 -------------------------- #
global_data = np.load('./global/global_data.npy')
## 打印数据格式
print(global_data.shape)            # (5848, 4, 9, 3850)

# 将对应的global_data 转换为 pd.df格式之后读取前5行。
## gddf = global_data_dataframe
# gddf = pd.DataFrame(global_data)
# print(gddf.head())

# 关键字del是手动删除对应的变量
"""
name：2024 & 2、数据预处理
time：2024/6/18 21:14
author：yxy
content：
"""

import matplotlib.pyplot as plt
import numpy as np      # 读取数据
import pandas as pd     # 处理数据


# 指定.npy文件的路径
temp_path = "./global/temp.npy"
wind_path = "./global/wind.npy"

# 使用numpy的load函数来加载.npy文件
temp_array = np.load(temp_path)
wind_array = np.load(wind_path)
# print(temp_array.shape)       # (17544, 3850, 1)，即为17544小时,3850站点,1数据。
# print(temp_array[:,:1,:].shape)     # 对数据进行切片，并选取某一个站点的数据作为时序图绘制指标

# 假设time和temperature是从文件或数据源中获取的时间和温度数据
time = np.array([i for i in range(len(temp_array))])  # 时间数据，这里只是示例
# print(np.array(time).shape)
temperature = temp_array[:,:1,:]  # 对应的温度数据

# print(np.array(temperature).shape)          # 将数据转换numpy类型，并打印出对应的形状，得到(1, 17544, 1, 1)
# 降维度为1的，全部去掉
temperature = np.squeeze(temperature)          #  (17544,)
# print(temperature.shape)

# 绘制线图
# plt.plot(time, temperature, marker='o')  # marker='o' 表示在数据点处添加圆圈标记
plt.plot(time, temperature)

# 添加标题和轴标签
plt.title('Temperature at Station 17 Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature')

# 显示网格（非必要）
plt.grid(True)

# 显示图表
plt.show()
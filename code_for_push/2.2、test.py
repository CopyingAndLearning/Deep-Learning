"""
name：2024 & 2、test
time：2024/6/18 21:26
author：yxy
content：
"""

import pandas as pd
import matplotlib.pyplot as plt

# 假设df是包含时间序列数据的DataFrame，列名为'Time'和'Temperature'
df = pd.DataFrame({
    'Time': pd.to_datetime(['2024-06-18 00:00:00', '2024-06-19 00:00:00', '2024-06-20 00:00:00', '2024-06-21 00:00:00']),
    'Temperature': [50, 150, 200, 250]
})

# 设置时间列为索引
df.set_index('Time', inplace=True)

# 使用pandas内置的plot方法绘图
df['Temperature'].plot()

# 添加标题和轴标签
plt.title('Temperature at Station 17 Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
"""
name：2024 & 3.1 损失函数图像的绘制
time：2024/6/22 12:03
author：yxy
content：
"""
import json

import matplotlib.pyplot as plt
with open('./loss_100.json','r',encoding="utf-8") as file:
    data = json.load(file)
    loss_history = data["loss"]

# 是哪个地方决定了，图像的绘制是折线图，还是时序图。
# 时序图，只是数据过多的时候显示的情况。理论上来说，数据足够大的情况都可以绘制为时序图。
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
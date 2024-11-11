## torch
### torch.
#### torch.mean
【时间】11/11
``` python
import torch
# 创建一个三维张量
tensor = torch.randn(2, 3, 4)  # 假设形状为 (2, 3, 4)
# 计算整个张量的平均值
mean_value = tensor.mean(-2)
print(mean_value.shape)  # 输出一个标量值 # (2,4)
```
这样理解：
- [1,2,3],[4,5,6]]，如果按照行求平均值，那么行的平均值为(1+2+3)/3=2, (4,5,6)/3=5，所以[2,3]按行维度求求完（也就是axis=1,最内层的维度），维度变为[2]；
- 同理，按行求(axis=0)，维度变为[3]

【总结】按哪个维度求，也就是按哪个括号求，求完之后的维度消失；括号从内到外分别表示不同的维度；

## paper&网络
#### 网络的组成
【时间】11/11
[!CAUTION]
- 一般认为神经网络是由BackBone\Neck\Head组成，但我认为神经网络可由Limb\Backbone\Neck、head组成

其中：
Feet的目的是进行数据特征提取仟的一些预处理操作，通常设计到数据预处理、数据分析、数据准备和数据结构；

BackBone的目的是为了提取数据，比如VIT使用的Transformer来抽取特征，抽取出一些基本的数据特征；

Neck的目的是进一步融合信息，例如U-Net网络中的多特征融合思路和各种魔改的注意力机制等，使模型对于数据特征的整合更好；

Head的目的是确定下游任务，例如线性回归就接全连接层、图像分割就接卷积层等，通过设计不同的任务和特殊设计（根据数据集设计的）的损失函数来完成对应的下游任务；

- So, Take an open-source project frist thing we should to do is to recognize which of the four sections above does each section belong to.

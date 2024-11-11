## torch
#### torch.mean
- 11/5
``` python
import torch
# 创建一个三维张量
tensor = torch.randn(2, 3, 4)  # 假设形状为 (2, 3, 4)
# 计算整个张量的平均值
mean_value = tensor.mean(-2)
print(mean_value.shape)  # 输出一个标量值 # (2,4)
```
这样理解：
①[[1,2,3],[4,5,6]]，如果按照行求平均值，那么行的平均值为(1+2+3)/3=2, (4,5,6)/3=5，所以[2,3]按行维度求求完（也就是axis=1,最内层的维度），维度变为[2]；
②同理，按行求(axis=0)，维度变为[3]
总结：按哪个维度求，也就是按哪个括号求，求完之后的维度消失；括号从内到外分别表示不同的维度；

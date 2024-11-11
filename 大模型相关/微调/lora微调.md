【目的】想做lora微调已经很久了，但是一直没有时间和动力做，今天听到本科大四学生的谈到Lora微调，又重新是我回忆起了该做lora微调了。

【思考】

- 既要接触新的内容，也要接触这些新内容的基础内容来源；给定一个方向后，自己要花时间去学习，去思考；少说一些没用的话，多做一些自己想做，却没有做导致自己焦虑的事情；

- 确定到底是要学习AI应用，还是AI原理的使用；
- 知识的获取也是一步一步来的;
- 自己所缺乏的是一个把我学过的知识串起来的一个过程；

【明白】到底是想学习lora的原理，还是学会lora怎么微调的步骤

* 既要看网络博客教程，也要看论文；
  看教程的时间 >> 看论文的时间
  直到看论文特别顺手时；

* |        | 原生代码 | 封装的库 |
  | :----: | :------: | :------: |
  | 方便性 |    ×     |    √     |
  | 理解性 |    √     |    ×     |
  |  上手  |    快    |    慢    |

  



#### lora原生代码

【时间】2024年11月11日

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 本质上是低秩分解
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5):
        """
        	in_features: 输入维度
        	out_features: 输出维度
        	merge: 是否使用预训练权重，即全量还上增量微调
        	lora_aplha: 控制原始权重的比值
        """
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha

        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            # 全0初始化
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.lora_aplpha / self.rank
            self.linear.weight.requires_grad = False

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        if self.rank > 0 and self.merge:
            output = F.linear(x, self.linear.weight + self.lora_b @ self.lora_a * self.scale, self.linear.bias)
            output = self.dropout(output)
            return output
        else:
            return self.dropout(self.linear(x))
```

##### 对小白解释

- [ ] Q&A

Q1：怎样进行代码的验证
A1：不需要验证，这段代码来自于论文当中，论文已经证明了。只需要知道这段代码是如何具体执行的就行；

- [ ] 代码的解释

nn.Identity()：神经网络里面的空操作，即输入经过这一层后输出的内容和输出一样

nn.init.kaiming_uniform_：对指定参数进行初始化，也就是将初始参数分布转换为均匀分布；其中，a指的是ReLU函数的负斜率



##### 对大佬解释

* 微调的思路和对其他模型的微调思路一样，使模型能够适应自己的数据；
* 参数值是进行累加的即：
  $h = W_0 \cdot x +  \delta W \cdot x$​
  其中，$\delta W$是可正可负的；所以，对应的参数的更新可以作用到原来的参数矩阵上面；
* $\delta W$可以被拆分称为A和B两个低秩的矩阵，AB维度是远远小于W矩阵的，对AB的操作的计算量也是远远小于直接对W矩阵的操作的
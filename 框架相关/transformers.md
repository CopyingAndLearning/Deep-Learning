【注】按时间来，自己每次学习进行分类的话，会增加额外开销

【目的】会用transformers的权重，来微调预训练模型；学会transformers；

【版本】transformers：4.46.2

## 11

### 11/18

#### DetrConfig.

##### from_pretrained.

`DetrConfig.from_pretrained` 是一个用于加载预训练模型配置的方法，它属于 Hugging Face Transformers 库中的 `DetrConfig` 类。这个方法可以从 Hugging Face 模型库中加载一个预训练模型的配置，或者从本地路径加载配置文件。使用 `from_pretrained` 方法可以方便地获取模型的配置参数，如模型结构、超参数等，这些参数对于初始化模型是必要的。

``` python
from transformers import DetrConfig, DetrModel

# 加载预训练模型的配置
config = DetrConfig.from_pretrained('facebook/detr-resnet-50')

# 使用加载的配置初始化模型
model = DetrModel(config)
```

* 实在不知道config里面的内容的化，那么使用print打印看它的输出；

`总结：`预训练配置文件加载；



#### Transformers库微调视觉模型的一般方法

##### 目标检测

`自己初学时的疑惑：`

- [ ] Q：如何调整对应的头部来适应不同的下游任何；
  A：先找到对应的全连接层，通过修改全连接层的输入和输出来确定自己对应的任务；
- [ ] Q：目标检测的输出如果直接接入一个全连接层
  是修改分类的网络输出，还是修改回归的网络？
  A：当然是修改分类的网络，回归网络的唯一任务就是预测回归框；
- [ ] Q：判断是修改头部，还是在头部前一层进行额外的操作？
  A：建议是修改头部，在头部的前一层也是可以做其他操作的，方法同理；先定位对应的网络层，在将网络层修改为自己想要的输出格式

`理解：`目标检测网络的微调，只需要修改分类网络即可；要修改特定的头部，特定的头部可以通过定位的方法解决；

`步骤：`①使用dir找到目标检测网络的最后一层，通常是class_label结尾的（或者是查看网络结构）；②修改为自己想要的输出层即可；③准备对应类型的数据集（比如：CoCo类型或XML类型）；

``` python
print(model)    # 打印模型的网络结构
print(list(model.children())[-2])    # 获取模型最后的一个元素

```



### 11/19

- [ ] 建议直接使用hugging_face的load_dataset加载

##### XML转为COCO数据集

`疑惑：`

- [ ] 怎么转？转了之后如何让Transformers库进行使用？
- [ ] coco数据集有多个类别，分为检测、分割等；

`步骤：`

`之后：`创建对应的dataloader



##### 数据集后缀名

`.parquet:`是数据集的后缀



##### 创建parquet数据集

`数据格式：`目标检测的coco数据集；parquet数据集可以存储对应的二进制数据；

``` json
{
    'image_id': 366, 
    'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=500x290 at 0x201F696C6A0>,
    'width': 500, 
    'height': 500, 
    'objects': 
    {
    	'id': [1932, 1933, 1934],
		'area': [27063, 34200, 32431],
		'bbox': [[29.0, 11.0, 97.0, 279.0], [201.0, 1.0, 120.0, 285.0], [382.0, 0.0, 113.0, 287.0]],
		'category': [0, 0, 0]
	}
}
```

`xml2parquet:`

步骤：①创建对应parquet数据集；②one-hot编码；③区域(area)的连续性；

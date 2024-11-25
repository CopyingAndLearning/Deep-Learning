### 11

#### 11/19

##### pip安装库

`需求:`安装的库在目前的pip没有，在安装时可以加上-U参数，进行下载最新的库；

``` bash
pip install -U xx
```

pip --upgrade --user albumentations == pip -U albumentations

`理解：`安装一个库，如果网上有这个库，但是pip没有这个，就可以用这个方法下载对应的库；



#### 11/25

##### 异步

``` python
import asyncio

# 模拟异步下载数据的函数
async def download_data():
    print("开始下载数据...")
    await asyncio.sleep(3)  # 模拟下载耗时操作
    print("数据下载完成")
    return "下载的数据内容"

# 模拟另一个异步任务，比如处理数据
async def process_data():
    print("开始处理数据...")
    await asyncio.sleep(2)  # 模拟数据处理耗时操作
    print("数据处理完成")
    return "处理后的数据结果"

# 主函数，同时执行下载和处理数据的任务
async def main():
    # 创建下载数据的任务
    download_task = asyncio.create_task(download_data())
    # 创建处理数据的任务
    process_task = asyncio.create_task(process_data())

    # 等待下载任务完成
    download_result = await download_task
    print("下载结果:", download_result)

    # 等待处理任务完成
    process_result = await process_task
    print("处理结果:", process_result)

# 运行主函数
asyncio.run(main())
```

【理解】

* 类似于这样一个过程：下载一个文件，并对其进行处理。
  ①正常情况，下载的文件和处理文件的过程是串行的，也就是下载完了对应的文件才能进行处理，这样时间复杂度为o(2n)
  ②但是下载的同时，可以进行处理，这就极大的减少了时间复杂度。
* 使用方法，将普通函数加上对应的关键字aysnc，并用asyncio.create即可；
  使用async不使用await，不等待后续的其他await声明的任务，直接打印。
  使用async，同时使用await，那么所有的await的内容几乎是在同一时刻输入；
* 可以理解为JS里面AJAX的同步操作；


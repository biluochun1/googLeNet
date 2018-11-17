# GoogleNet 物品识别模型

这是一个基于tflearn深度学习个框架和GoogLeNet网络的物品识别模型。同时还有一个小型的flask应用显示训练结果。


[a1]: http://tflearn.org/
[a2]: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf


## 如何使用

1. 整理数据集到 `images/<DATASET_NAME>/<LABEL_NAME>` 如下所示. 你也可以使用 `python3 dump_17flowers.py` 下载样例数据集 [17 Category Flower Dataset][b1] 
   
   ```
    images  
      |
      |---17flowers  
             |
             |--- jpg
                   |
                   |--- 0 (folder containing all label=0 samples)
                   |    |
                   |    |----image_0001.jpg  (label=0 sample)
                   |    |----image_0002.jpg  (label=0 sample)
                   |    |----image_0003.jpg  (label=0 sample)
                   |    ...
                   |
                   |--- 1 (folder containing all label=1 samples)
                   |--- 2 (folder containing all label=2 samples)
                   |--- 3 (folder containing all label=3 samples)
                   ...
                   |--- 15  
                   |--- 16  
             
   ```

2. 使用如下命令行开始训练

```
python3 train.py --model_name 17flowers --label_size 17
```

最新的模型将会保存到 `models/<DATASET_NAME>`. 如果训练被中断，程序将找到最新的模型继续训练。
   
   
## 一个小型flask应用

你可以使用如下命令行启动flask应用。另外你可以通过修改app.py里的host参数来部署到服务器等。

```python3 app.py --model_name 17flowers --label_size 17```

之后打开你的浏览器, 前往 [http://localhost:8883](). 填写一个图片的url并提交，之后返回结果。

## 一些细节

在这个程序中，修改了`data_util.py`，它将调整原始图像的大小，并将它们缓存到pickle文件中，每个文件仅包含500个图像。在训练过程中，它将一次加载500幅图像进行小批处理，然后丢弃这些图像并加载另一个小批处理。所以，不管你的原始样本体积有多大，都要确保你的机器能够工作。

## 对不同数据集的参数调整

对于你的数据集, 在`train.py` 和 `app.py`确保修改这些参数

1. `model_name`: 数据集名称

2. `label_size`: 标签大小

## 模块化对象

我们将GoogLeNet封装为了一个类， `lib/googlenet.py`. 

除了网络的定义，它还有其他以下功能： 
1. `fit`: 做训练 
2. `predict`: 做预测 
3. `get_data`: 获取某个数据集缓存


## 训练过程解释

举个例子

```
scope_name, label_size = '17flowers', 17
```
17 flowers 这个数据集得到 17个标签

```
gnet = GoogLeNet(img_size=227, label_size=label_size, gpu_memory_fraction=0.4, scope_name=scope_name)
```
初始化GoogleNet

```
down_sampling = {str(n): 10000 for n in range(17)}
```
解决数据集的`偏差问题` 可以通过  `down_sampling` 来限制每个标签之获取一定的样本. 同时可以设定参数，0~1范围内，来取部分样本。

之后我们调用 `get_data` 来收集图像，同时不可浪费内存

```
pkl_files = gnet.get_data(dirname=scope_name,down_sampling=down_sampling)
```

最后，一个一个训练它们！

```
for f in pkl_files:
    X, Y = pickle.load(gzip.open(f, 'rb'))
    gnet.fit(X, Y, n_epoch=10)    
```



# MetaChain

## 代码目录
### [data](data)     ---存放文章中所涉及的五个数据集
###### （由于当前论文还在审稿状态，第5个数据集我们仅公布了部分）
###### [BPD1.csv](data%2FBPD1.csv)
###### [BPD2.csv](data%2FBPD2.csv)
###### [BPD3.csv](data%2FBPD3.csv)
###### .......
### [ensemble](ensemble)   ---集成学习代码及实验结果

[light_ensemble.py](ensemble%2Flight_ensemble.py) 是第1-4个数据集的代码

[light_ensemble_5.py](ensemble%2Flight_ensemble_5.py) 是第5个数据集的代码

### [Meta](Meta)   ---元学习代码及实验结果


[MetaReg.py](Meta%2FMetaReg.py)

## 运行方式

###### （将git文件下载后不要改变数据集文件夹和代码文件夹的相对路径）

#### 集成学习

````
# dataset用于指定使用那个数据集 
# task用于指定运行第几个任务

python light_ensemble.py --batch_size=10  --learning_rate=0.001 --dataset=1 --task=1
````

#### 元学习

````
# targetdatasettargetdataset用于指定使用那个数据集 
# task用于指定运行第几个任务
# test_batch_size 用于指定在目标任务中使用几个样本进行训练

python MetaReg.py --targetdataset=5 --task=1  --test_batch_size=10
````
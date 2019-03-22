# 基于BERT 的中文数据集下的命名实体识别(NER)



在上课老师课程作业发布的中文数据集下使用BERT来训练命名实体识别NER任务。

之前也用了CRF进行识别，这次使用BERT来进行训练，也算是对BERT源码进行一个阅读和理解吧。

虽然之前网上也有很多使用BERT的例子和教程，但是我觉得都不是很完整，有些缺乏注释对新手不太友好，有些则是问题不同修改的代码也不同，自己也在路上遇到了不少的坑。所以这个还是对新手使用BERT很友好的。

### 数据集

tmp 文件夹下

![1553264280882](images\1553264280882.png)

如上图，对数据集进行了分割，其中source是训练集中文，target是训练集的label。

test1 测试集，test_tgt 测试集label。     dev 验证集   dev-lable 验证集label。

### 代码

其实BERT需要根据具体的问题来修改相对应的代码，NER算是序列标注一类的问题，可以算分类问题吧。

然后修改的主要是run_classifier.py部分即可，我把修改后的存放到了run_classifier.py里。

之后会对其需要修改的部分进行解释。



### 实验结果

待更新





有时间了更新剩下的readme 。 包括对源码的注释等====





> 参考 ：https://github.com/google-research/bert   官方
>
>             https://github.com/kyzhouhzau/BERT-NER  基于英文的NER，但是代码注释不太多。
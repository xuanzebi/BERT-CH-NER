## 基于BERT 的中文数据集下的命名实体识别(NER)

基于上课老师课程作业发布的中文数据集下使用BERT来训练命名实体识别NER任务。

之前也用了Bi+LSTM+CRF进行识别，效果也不错，这次使用BERT来进行训练，也算是对BERT源码进行一个阅读和理解吧。

虽然之前网上也有很多使用BERT的例子和教程，但是我觉得都不是很完整，有些缺乏注释对新手不太友好，有些则是问题不同修改的代码也不同，自己也在路上遇到了不少的坑。所以记录一下。

### 数据集

tmp 文件夹下

![1553264280882](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553264280882.png)

如上图，对数据集进行了分割，其中source是训练集中文，target是训练集的label。

test1 测试集，test_tgt 测试集label。     dev 验证集   dev-lable 验证集label。

#### 类别

![1553304765330](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553304765330.png)

其中共设置了10个类别，PAD是当句子长度未达到max_seq_length时，补充0的类别。

CLS是每个句首前加一个标志[CLS]的类别，SEP是句尾同理。（因为BERT处理句子是会在句首句尾加上这两个符号。）



### 代码

其实BERT需要根据具体的问题来修改相对应的代码，NER算是序列标注一类的问题，可以算分类问题吧。

然后修改的主要是run_classifier.py部分即可，我把修改下游任务后的代码放到了run_NER.py里。

之后会对其需要修改的部分进行解释。

待更新==



### 训练

首先下载BERT基于中文预训练的模型，

```shell
export BERT_BASE_DIR=/opt/xxx/chinese_L-12_H-768_A-12
export NER_DIR=/opt/xxx/tmp
python run_NER.py \
          --task_name=NER \
          --do_train=true \
          --do_eval=true \
          --do_predict=true \
          --data_dir=$NER_DIR/ \
          --vocab_file=$BERT_BASE_DIR/vocab.txt \
          --bert_config_file=$BERT_BASE_DIR/bert_config.json \
          --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
          --max_seq_length=256 \     # 根据实际句子长度可调
          --train_batch_size=32 \    # 可调
          --learning_rate=2e-5 \
          --num_train_epochs=3.0 \
          --output_dir=$BERT_BASE_DIR/output/

```



### 实验结果

![1553304598242](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553304598242.png)

可以基于验证集看到的准确率召回率都在95%以上。

下面可以看看预测测试集的几个例子。

![1553305073652](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553305073652.png)

下图为使用BERT预测的类别。可以与真实的类别对比看到预测还是很准确的。

![1553305053823](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553305053823.png)

真实类别如下图。

![1553305543516](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553305543516.png)



### 总结

其实在读了BERT的论文后，结合代码进行下游任务的微调能够理解的更深刻。

其实改造下游任务主要是把自己数据改造成它们需要的格式，然后将输出类别根据需要改一下，然后模型的metric函数改一下就整体差不多了。

如下图根据具体的下游任务修改label即可。如下图的第四个就是在NER上进行修改，

![1553306691480](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553306691480.png)

### BERT论文

这留个空给自己总结一下BERT论文里的知识和自己的理解。





> 参考 ：
>
> https://github.com/google-research/bert   官方
>
> https://github.com/kyzhouhzau/BERT-NER  基于英文的NER，但是代码注释不太多。







有时间了更新剩下的readme 、 包括对源码的注释和如何根据NER下游任务来修改代码等====
## 基于BERT 的中文数据集下的命名实体识别(NER)

### 一 搜狐比赛   

<https://www.biendata.com/competition/sohu2019/>

在搜狐这个文本比赛中写了一个baseline，使用了bert以及bert+lstm+crf来进行实体识别。

其后只使用BERT的结果如下，具体评测方案请看比赛说明，这里的话只做了实体部分，情感全部为POS进行的测试得分。

![1557228899471](https://github.com/xuanzebi/BERT-NER/blob/master/images/1557228899471.png)

使用bert+lstm+crf 结果如下

![1557228995787](https://github.com/xuanzebi/BERT-NER/blob/master/images/1557228995787.png)

##### 训练验证测试

```shell
export BERT_BASE_DIR=/opt/hanyaopeng/souhu/data/chinese_L-12_H-768_A-12
export NER_DIR=/opt/hanyaopeng/souhu/data/data_v2
python run_souhuv2.py \
                    --task_name=NER \
                    --do_train=true
                    --do_eval=true \
                    --do_predict=true \
                    --data_dir=$NER_DIR/ \
                    --output_dir=$BERT_BASE_DIR/outputv2/ \
                    --train_batch_size=32 \
                    --vocab_file=$BERT_BASE_DIR/vocab.txt \
                    --max_seq_length=256 \
                    --learning_rate=2e-5 \
                    --num_train_epochs=10.0 \
                    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
                    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \

```

#### 代码

在souhu文件下

- souhu_util.py  文件是取得预测的label后，转换为实体的数据 处理代码。
- lstm_crf_layer.py 是lstm+crf层的代码
- run_souhu.py 只用bert的代码
- run_souhuv2.py  bert+lstm+crf

### 二 

基于上课老师课程作业发布的中文数据集下使用BERT来训练命名实体识别NER任务。

之前也用了Bi+LSTM+CRF进行识别，效果也不错，这次使用BERT来进行训练，也算是对BERT源码进行一个阅读和理解吧。

虽然之前网上也有很多使用BERT的例子和教程，但是我觉得都不是很完整，有些缺乏注释对新手不太友好，有些则是问题不同修改的代码也不同，自己也在路上遇到了不少的坑。所以记录一下。

### 数据集

tmp 文件夹下

![1553264280882](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553264280882.png)

如上图，对数据集进行了分割，其中source是训练集中文，target是训练集的label。

test1 测试集，test_tgt 测试集label。     dev 验证集   dev-lable 验证集label。

#### 注意

因为在处理中文时，会有一些奇怪的符号，比如\u3000等，需要你提前处理，否则label_id和inputs_id对应不上，因为bert自带的tokenization会处理掉这些符号。所以可以使用bert自带的BasicTokenizer来先将数据文本与处理一下。

```python
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
text = tokenizer.tokenize(text)
text = ''.join([l for l in text])
```

#### 数据格式

```python
 需要将数据处理成如下格式，一个句子对应一个label.句子和label的每个字都用空格分开。
 如: line = [我 爱 国 科 大 哈 哈]   str
     label = [O O B I E O O]       str的type 用空格分开
    
具体请看代码中的NerProcessor 和 NerBaiduProcessor
```

#### 类别

![1553304765330](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553304765330.png)

其中共设置了10个类别，PAD是当句子长度未达到max_seq_length时，补充0的类别。

CLS是每个句首前加一个标志[CLS]的类别，SEP是句尾同理。（因为BERT处理句子是会在句首句尾加上这两个符号。）



### 代码

其实BERT需要根据具体的问题来修改相对应的代码，NER算是序列标注一类的问题，可以算分类问题吧。

然后修改的主要是run_classifier.py部分即可，我把修改下游任务后的代码放到了run_NER.py里。

代码中除了数据部分的预处理之外，还需要自己修改一下评估函数、损失函数。



### 训练

首先下载BERT基于中文预训练的模型（BERT官方github页面可下载），存放到BERT_BASE_DIR文件夹下，之后将数据放到NER_DIR文件夹下。即可开始训练。

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

其实改造下游任务主要是把自己数据改造成它们需要的格式，然后将输出类别根据需要改一下，然后修改一下评估函数和损失函数。

如下图根据具体的下游任务修改label即可。如下图的第四个就是在NER上进行修改，

![1553306691480](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553306691480.png)

之后会写一篇Attention is all you need 和 bert论文的详解，会结合代码来解释一下细节，比如Add & Norm是如何实现的，为什么要Add & Norm。 

欢迎大家关注我的博客。



> 参考 ：
>
> https://github.com/google-research/bert   官方
>
> https://github.com/kyzhouhzau/BERT-NER  基于英文的NER，但是代码注释不太多。



## 基于BERT 的中文数据集下的命名实体识别(NER)

基于tensorflow官方代码修改。

### 环境

Tensorflow: 1.13

Python: 3.6 

tensorflow2.0 会报错。

##  搜狐比赛   

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

#### 注意

因为在处理中文时，会有一些奇怪的符号，比如\u3000等，需要你提前处理，否则label_id和inputs_id对应不上，因为bert自带的tokenization会处理掉这些符号。所以可以使用bert自带的BasicTokenizer来先将数据文本预处理一下从而与label对应上。

```python
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
text = tokenizer.tokenize(text)
text = ''.join([l for l in text])
```



## 二

基于上课老师课程作业发布的中文数据集下使用BERT来训练命名实体识别NER任务。

之前也用了Bi+LSTM+CRF进行识别，效果也不错，这次使用BERT来进行训练，也算是对BERT源码进行一个阅读和理解吧。

虽然之前网上也有很多使用BERT的例子和教程，但是我觉得都不是很完整，有些缺乏注释对新手不太友好，有些则是问题不同修改的代码也不同，自己也在路上遇到了不少的坑。所以记录一下。

### 数据集

tmp 文件夹下

![1553264280882](https://github.com/xuanzebi/BERT-NER/blob/master/images/1553264280882.png)

如上图，对数据集进行了分割，其中source是训练集中文，target是训练集的label。

test1 测试集，test_tgt 测试集label。     dev 验证集   dev-lable 验证集label。



#### 数据格式

```python
 需要将数据处理成如下格式，一个句子对应一个label.句子和label的每个字都用空格分开。
 如: line = [我 爱 国 科 大 哈 哈]   str
     label = [O O B I E O O]       str的type 用空格分开
    
具体请看代码中的NerProcessor 和 NerBaiduProcessor
```

#### 注意

BERT分词器在对字符分词会遇到一些问题。

比如 输入叩 问 澳 门 =- =- =- 贺 澳 门 回 归 进 入 倒 计 时 ，label :O O B-LOC I-LOC O O O O B-LOC I-LOC O O O O O O O  

会把输入的=- 处理成两个字符，所以会导致label对应不上，需要手动处理一下。比如如下每次取第一个字符的label。 其实这个问题在处理英文会遇到，WordPiece会将一个词分成若干token,所以需要手动处理（这只是一个简单处理方式）。

```
    la = example.label.split(' ')

    tokens_a = []
    labellist = []

    for i,t in enumerate(example.text_a.split(' ')):
        tt = tokenizer.tokenize(t)
        if len(tt) == 1 :
            tokens_a.append(tt[0])
            labellist.append(la[i])
        elif len(tt) > 1:
            tokens_a.append(tt[0])
            labellist.append(la[i])

    assert len(tokens_a) == len(labellist)
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

首先下载BERT基于中文预训练的模型（BERT官方github页面可下载），存放到BERT_BASE_DIR文件夹下，之后将数据放到NER_DIR文件夹下。即可开始训练。sh run.sh

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

之后会写一篇Attention is all you need 和 bert论文的详解，会结合代码来解释一下细节，比如Add & Norm是如何实现的，为什么要Add & Norm。 ==  感觉不用写了 bert已经火遍大街了   不重复造轮子了。建议大家直接莽源代码和论文。

最后BERT还有很多奇淫技巧需要大家来探索。。比如可以取中间层向量来拼接，再比如冻结中间层等等。



后来自己又用pytorch版本的BERT做了几个比赛和做实验发论文，个人觉得pytorch版本的bert更简单好用，更方便的冻结BERT中间层，还可以在训练过程中梯度累积，直接继承BERTmodel就可以写自己的模型了。

（自己用pytorch又做了NER的BERT实验，想开源但是懒得整理....哪天闲了再开源吧  ps 网上已经一大堆开源了233）

pytorch真香..改起来比tensorflow简单多了.. 

个人建议 如果自己做比赛或者发论文做实验用pytorch版本.. pytorch已经在学术界称霸了..但是工业界tensorflow还是应用很广。   

> 参考 ：
>
> https://github.com/google-research/bert   
>
> https://github.com/kyzhouhzau/BERT-NER 
>
> https://github.com/huggingface/transformers   pytorch版本



##### 今天又出来一个叼模型，20项任务全面碾压BERT，CMU全新XLNet预训练模型屠榜（已开源）

留坑，哈哈  读读论文看看代码去。

> <https://mp.weixin.qq.com/s/29y2bg4KE-HNwsimD3aauw>
>
> <https://github.com/zihangdai/xlnet>



好吧 前几天又看见了谷歌开源的T5模型，从XLNet、RoBERTa、ALBERT、SpanBERT发展到现在T5....根本顶不住.. 现在NLP比赛基本也都被预训练霸榜了..不用预训练根本拿不到好成绩...




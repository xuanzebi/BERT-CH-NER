import codecs
import json
from tqdm import tqdm
import re
import tokenization

# tokenizer = FullTokenizer(vocab_file='/opt/hanyaopeng/souhu/data/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
input_file = '/opt/hanyaopeng/souhu/data/data_v2/coreEntityEmotion_test_stage1.txt'

with open(input_file, encoding='utf-8') as f:
    test_data = []
    for l in tqdm(f):
        data = json.loads(l.strip())
        news_id = data['newsId']
        title = data['title']
        title = tokenizer.tokenize(title)
        title = ''.join([l for l in title])
        content = data['content']
        sentences = []
        ans = '' + title
        for seq in re.split(r'[\n。]', content):
            seq = tokenizer.tokenize(seq)
            seq = ''.join([l for l in seq])
            if len(seq) > 0:
                if len(seq) + len(ans) <= 254:
                    if len(ans) == 0:
                        ans = ans + seq
                    else:
                        ans = ans + '。' + seq
                elif len(seq) + len(ans) > 254 and len(seq) + len(ans) < 350 and len(ans) < 150:
                    if len(ans) == 0:
                        ans = ans + seq + '。'
                    else :
                        ans = ans + '。' + seq + '。'
                    sentences.append(ans)
                    ans = ''
                else:
                    ans = ans + '。'
                    sentences.append(ans)
                    ans = ''
        if len(ans) != 0:
            sentences.append(ans)
        for seq in sentences:
            label = ['O'] * len(seq)
            l = ' '.join([la for la in label])
            w = ' '.join([word for word in seq])
            test_data.append((news_id, w, l))



import codecs
with codecs.open("/opt/hanyaopeng/souhu/data/data_v2/test_samplev2.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False)

fr = open('/opt/hanyaopeng/souhu/data/chinese_L-12_H-768_A-12/outputv2/test_predictionv2.txt', 'r', encoding='utf-8')
result = fr.readlines()

test_sample = json.load(open('/opt/hanyaopeng/souhu/data/data_v2/test_samplev2.json', encoding='utf-8'))
entity = {}
for i in range(len(test_sample)):
    a = result[i].split(' ')[1:-1]  # label
    t = test_sample[i][1].split(' ')  # seq
    newsid = test_sample[i][0]
    if newsid not in entity:
        entity[newsid] = []
    if len(t) > 254: # max_seq_length
        t = t[:254]
    ent = {}
    assert len(a) == len(t)
    j = 0
    while j < len(a):
        if a[j] == 'S':
            entity[newsid].append(t[j])
            j += 1
        elif a[j] == 'B':
            flag = j
            k = j + 1
            while k < len(a):
                if a[k] == 'E':
                    ti = ''.join([la for la in t[flag:k+1]])
                    entity[newsid].append(ti)
                    j = k + 1
                    break
                elif a[k] == 'O':
                    j = k+1
                    break
                else:
                    k += 1
            j = k
        else:
            j += 1

    if i % 100000 == 0:
        print(i)

res = {}
for i in entity.keys():
    res[i] = []
    items = {}
    for j in entity[i]:
        if j in items:
            items[j] += 1
        else:
            items[j] = 1
    ans = sorted(items.items(),key=lambda x:x[1],reverse=True)
    if len(ans) >= 3: # 取 存放的实体数量
        res[i] = [i[0] for i in ans[:3]]
    else :
        res[i] = [i[0] for i in ans]

res_file = open("/opt/hanyaopeng/souhu/data/sub/subbmission4.txt", 'w', encoding='utf-8')

for i in res:
    ent = res[i]
    for en in range(len(ent)):
        ent[en] = ent[en].replace(",",'')
        ent[en] = ent[en].replace("，",'')
    emos = []
    for j in range(len(ent)):
        emos.append('POS')
    row = i +'\t'+','.join(ent)+'\t'+','.join(emos) + '\n'
    res_file.write(row)
    res_file.flush()
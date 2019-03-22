import random

#
# fr = open('source.txt', encoding='utf-8')
# fw_l = open('target.txt', encoding='utf-8')
#
# max = 0
# for i in fr.readlines():
#     i = i.strip().split()
#     if len(i) > max:
#         max = len(i)
# print(max)
# # import json
# lines = []
# label = []
# for i in fr.readlines():
#     lines.append(i.strip())
# for i in fw_l.readlines():
#     label.append(i.strip())
#
# fr2 = open('test1.txt', encoding='utf-8')
# for i in fr2.readlines():
#     lines.append(i.strip())
# fw2 = open('test_tgt.txt', encoding='utf-8')
# for i in fw2.readlines():
#     label.append(i.strip())
#
# res = []
#
# for i in range(len(lines)):
#     res.append([lines[i], label[i]])
#
# random.shuffle(res)
#
# test = res[0:600]
# fww = open('test.txt','w',encoding='utf-8')
# fww.write(str(test))
#
a = [['s', 's', 'a'], ['e', 't', 'q']]
# print(a[:2])
import os

# pred_out = os.path.join('C:/nlp/bert/tmp', "test_pred.txt")
# with open(pred_out, 'w+') as writer:
fw = open(os.path.join('C:/nlp/bert/tmp', "test_prediction.txt"), 'w', encoding='utf-8')
for i in a:
    out = " ".join(id for id in i if id != 'a') + '\n'
    fw.write(out)

import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import re

######################################################
# 为GCN链接预测准备数据，移除训练测试用的边集
######################################################

np.random.seed(0)
def write_pos(G1):
    for n1 in G1.nodes():
        for n2 in G1.nodes():
            if n1 == n2:
                continue
            elif (n1,n2) in G1.edges:
                fout.write(n1 +' '+ n2 +'\t'+ '1' +'\n')
            elif (n2,n1) in G1.edges:
                continue
            else:
                f.write(n1 + ' ' + n2 + '\t' + '0' + '\n')



G1=nx.read_edgelist('data/edges_final.txt')
with open('data/train_pos_full.txt','w') as fout:
    with open('data/train_neg_full.txt', 'w') as f:
        write_pos(G1)

f = open('data/train_pos_full.txt')
pos = f.readlines()
print(len(pos))
f = open('data/train_neg_full.txt')
neg = f.readlines()
pos_len = len(pos)
all_sample = int(pos_len*0.5) # 删去的边数
print(all_sample)
pos_idx = random.sample(range(pos_len), all_sample)
train_idx = random.sample(pos_idx, int(all_sample*0.8)) # 作为训练集的正样本索引
test_pos = set(pos_idx)-set(train_idx)                  # 作为测试集的正样本索引
neg_len = len(neg)
neg_idex = random.sample(range(neg_len), all_sample)
neg_idx = random.sample(neg_idex, int(all_sample*0.8))  # 作为训练集的负样本索引
test_neg = set(neg_idex)-set(neg_idx)                   # 作为测试集的负样本索引
pos_train = [pos[i] for i in pos_idx]
pos_test = [pos[i] for i in test_pos]
neg_train = [neg[i] for i in neg_idx]
neg_test = [neg[i] for i in test_neg]
f = open('data/edges_final.txt')
edges = f.readlines()
print(len(edges))
dict = {}
with open('data/test_pos_fff.txt','w') as fout:
    for w in pos_test:
        dict[w.strip('\n').split('\t')[0]] = 1
        fout.write(w)
with open('data/test_neg_fff.txt','w') as fout:
    for w in neg_test:
        fout.write(w)

with open('data/train_fff.txt','w') as fout:
    for w in pos_train:
        dict[w.strip('\n').split('\t')[0]] = 1
        fout.write(w)
    for w in neg_train:
        fout.write(w)
print(len(dict))
i=0
for w in edges:
        if w.strip('\n') in dict:
            edges.pop(i)
            i=i-1
        i=i+1
f = open('data/edges_fff.txt','w')
for w in edges:
    f.write(w)


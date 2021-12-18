import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import re
import csv

######################################################
# 使用Pearson、完整特征、社区划分、RA等经典算法做逻辑回归
######################################################

# pearson 相关系数
def pear():
    f = open('feature_107.txt',encoding='utf-8')
    feature = f.readlines()
    rows = len(feature)
    datamat = np.zeros((rows, 1406))
    row = 0
    feat = []
    for line in feature:
        line = line.strip().split(' ')
        feat.append(line)
        datamat[row, :] = line[1:]
        row += 1
    per = {}
    for i in range(row):
        for j in range(i,row):
            per[(feat[i][0],feat[j][0])] = pearsonr(datamat[i,:],datamat[j,:])[0]
            per[(feat[j][0],feat[i][0])] = per[(feat[i][0],feat[j][0])]

    # print(per[(236, 347)])
    return per
    # print(per[(2,1)])


# 基于邻居
def wei():
    G1=nx.read_edgelist('edges_107.txt')
    G2 = nx.Graph()
    # print(max(nx.edge_betweenness_centrality(G1).items(),key=lambda item:item[1])[0])
    w = {}
    # print(np.log(G1.degree('1')))
    for u,v in G1.edges():
        G2.add_edge(u,v,weight=1/(np.log(G1.degree(u))+np.log(G1.degree(v))+1))
    # for node in G1.nodes():
    #     for n in G1.neighbors(node):
    #         w[(int(node),int(n))] = 1/(np.log(G1.degree(node))+np.log(G1.degree(n))+1)
    nx.write_weighted_edgelist(G2,'edges_weight_full.txt')

def write_pear_pos(G1,pear):
    for n1 in G1.nodes():
        for n2 in G1.nodes():
            if n1 == n2:
                continue
            elif (n1,n2) in G1.edges:
                fout.write(n1+' '+ n2 +'\t'+ '1' +'\t'+ str(pear[(n1,n2)])+'\t'+str(1/(np.log(G1.degree(n1))+np.log(G1.degree(n2))+1))+'\n')
            elif (n2,n1) in G1.edges:
                fout.write(n1 +' '+ n2 +'\t'+ '1' +'\t'+ str(pear[(n2,n1)])+'\t'+str(1/(np.log(G1.degree(n1))+np.log(G1.degree(n2))+1))+'\n')

def write_pear_neg(G1,pear):
    for n1 in G1.nodes():
        for n2 in G1.nodes():
            if n1 == n2:
                continue
            elif (n1,n2) not in G1.edges and (n2,n1) not in G1.edges:
                f.write(n2 + ' ' + n1 + '\t' + '0' + '\t' + str(pear[(n2, n1)])+'\t'+str(1/(np.log(G1.degree(n1))+np.log(G1.degree(n2))+1)) + '\n')

p = pear()
G1=nx.read_edgelist('edges_107.txt')
with open('full_train_pos.txt','w') as fout:
    write_pear_pos(G1,p)
with open('full_train_neg.txt','w') as f:
    write_pear_neg(G1,p)

# 分正负训练集测试集
f = open('full_train_pos.txt')
pos = f.readlines()
f = open('full_train_neg.txt')
neg = f.readlines()
pos_len = len(pos)
train_sample = int(pos_len*0.8)
pos_idx = random.sample(range(pos_len), train_sample)
test_pos = set(np.arange(pos_len))-set(pos_idx)
neg_len = len(neg)
neg_idex = random.sample(range(neg_len), pos_len)
neg_idx = random.sample(neg_idex, train_sample)
test_neg = set(neg_idex)-set(neg_idx)
pos_train = [pos[i] for i in pos_idx]
pos_test = [pos[i] for i in test_pos]
neg_train = [neg[i] for i in neg_idx]
neg_test = [neg[i] for i in test_neg]

with open('train_pos.txt','w') as fout:
    for w in pos_train:
        fout.write(w)
with open('test_pos.txt','w') as fout:
    for w in pos_test:
        fout.write(w)
with open('train_neg.txt','w') as fout:
    for w in neg_train:
        fout.write(w)
with open('test_neg.txt','w') as fout:
    for w in neg_test:
        fout.write(w)
with open('test.txt','w') as fout:
    for w in neg_test:
        fout.write(w)
    for w in pos_test:
        fout.write(w)
with open('train.txt','w') as fout:
    for w in pos_train:
        fout.write(w)
    for w in neg_train:
        fout.write(w)

# 加入社区分类结果
f = open('k_means_107.txt')
a = f.readlines()
nodeclass = {}
for w in a:
    w = w.strip('\n').split(' ')
    nodeclass[w[0]] = w[1]

# 加入特征
f = open('feature_107.txt')
a = f.readlines()
feature = {}
for w in a:
    w = w.strip('\n').split(' ')
    feature[w[0]] = [int(w[i]) for i in range(1,len(w))]

# 加入RA
f = open('train_RA.txt')
a = f.readlines()
train_x = []
train_y = []
# 构建训练数据
for w in a:
    w = w.strip()
    w = re.split('\t| ',w)
    k = np.abs(np.array(feature[w[0]])-np.array(feature[w[1]]))**2
    if nodeclass[w[0]]==nodeclass[w[1]]:
        train_x.append(k.tolist()+[float(w[3]),float(w[4]),1])
    else:
        train_x.append(k.tolist()+[float(w[3]),float(w[4]),0])
    # train_x.append(float(w[4]))
    train_y.append(int(w[2]))

# 构建测试数据
f = open('test_RA.txt')
a = f.readlines()
test_x = []
test_y = []
for w in a:
    w = w.strip()
    w = re.split('\t| ', w)
    k = np.abs(np.array(feature[w[0]])-np.array(feature[w[1]]))**2
    # test_x.append(k.tolist()+[float(w[4])])
    if nodeclass[w[0]] == nodeclass[w[1]]:
        test_x.append(k.tolist()+[float(w[3]),float(w[4]),1])
    else:
        test_x.append(k.tolist()+[float(w[3]),float(w[4]),0])
    # test_x.append([float(w[2]),float(w[3])])
    # test_x.append(float(w[4]))
    test_y.append(int(w[2]))

test_x = np.array(test_x).reshape(-1,1409)
print(test_x.shape)
test_y = np.array(test_y).reshape(-1,1)
train_x = np.array(train_x).reshape(-1,1409)
train_y = np.array(train_y).reshape(-1,1)
# 逻辑回归
modelLR=LogisticRegression()
modelLR.fit(train_x,train_y)
print(modelLR.score(test_x,test_y))
pre = modelLR.predict(test_x)
fpr,tpr,threshold = roc_curve(test_y, pre) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
print(roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

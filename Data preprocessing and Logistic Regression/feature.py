import numpy as np
import networkx as nx
import re
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


###########################
# 构建特征矩阵与特征名称矩阵
# 构建不同类别特征矩阵
###########################

def featname(num):
    # 获得特征名称
    f = open('data/facebook/%s.featnames'%num)
    featname = f.readlines()
    name = []
    for i in range(len(featname)):
        featname[i] = featname[i].strip("\n").split(' ')
        name.append(featname[i][1]+' '+featname[i][2]+' '+featname[i][3])
    return name

def feat(num,dic):
    # 获得节点完整特征
    for k in range(len(num)):
        f = open('data/facebook/%s.feat'%num[k])
        name = featname(num[k])
        feat = f.readlines()
        for i in range(len(feat)):
            feat[i] = feat[i].strip("\n").split(' ')
            for j in range(1,len(feat[i])):
                if feat[i][j]=='1':
                    dic.setdefault(name[j-1],[]).append(int(feat[i][0]))
    return dic

def egofeat(num,dic):
    # 获得中心点特征
    for k in range(len(num)):
        f = open('data/facebook/%s.egofeat'%num[k])
        name = featname(num[k])
        feat = f.readlines()
        for i in range(len(feat)):
            feat[i] = feat[i].strip("\n").split(' ')
            for j in range(len(feat[i])):
                if feat[i][j]=='1':
                    dic.setdefault(name[j],[]).append(num[k])
    return dic


dic = feat([0,107,348,414,686,698,1684,1912,3437,3980],{})
dic = egofeat([0,107,348,414,686,698,1684,1912,3437,3980],dic)
dic = sorted(dic.items(),key=lambda x:x[0])
name = {}
for w in dic:
    name[w[0]]=str(w[1])
name = sorted(name.items(),key=lambda x:x[0])
with open('data/total feature.txt', "w") as fout:
    for w in name:
        fout.write(w[0]+'\t'+w[1]+'\n')

# 通过excel分列排序，可以获得featnames和feature_all.txt
f = open('data/feature_all.txt')
feat = f.readlines()
for i in range(len(feat)):
    feat[i] = feat[i].strip().split('\t')
    feat[i][1] = feat[i][1].replace("[", '').replace(']', '').replace(",", ' ').replace('\n', '')
    feat[i][1] = feat[i][1].strip(' ').split()

a = [0 for _ in range(len(feat))]
feature = [list(a) for _ in range(4039)]
for i in range(len(feat)): # feature数
    for w in feat[i][1]:
        feature[int(w)][i] = 1

with open('data/feature.txt', "w") as fout:
    i = 0
    for w in feature:
        fout.write(str(i)+'\t'+str(w)+'\n')
        i = i+1

f = open('data/feature.txt')
feat = f.readlines()
for i in range(len(feat)):
    feat[i] = feat[i].strip().split('\t')
    feat[i][1] = feat[i][1].replace("[", '').replace(']', '').replace(",", '').replace('\n', '')

# 获得完整特征，按节点编号排序
with open('data/feature_final.txt', "w") as fout:
    for w in feat:
        fout.write(w[0]+' '+w[1]+'\n')


# 把feature按类别分开
f = open('data/feature_final.txt')
feature = f.readlines()
for i in range(len(feature)):
    feature[i] = feature[i].strip().split(' ')
    feature[i] = [int(w) for w in feature[i]]
    feature[i] = feature[i][1:]
print(feature[2])

f = open('data/featurenames.txt')
names = f.readlines()
dic = {}
for i in range(len(names)):
    names[i] = re.split('[;\s]',names[i])
    if names[i][1] not in dic:
        dic[names[i][1]] = []
    dic[names[i][1]].append(int(names[i][0]))
maxi = {}
mini = {}
for w in dic:
    maxi[w] = max(dic[w])
    mini[w] = min(dic[w])
for w in dic:
    path = 'feature/feature_'+w+'.txt'
    with open(path, "w") as fout:
        for i in range(len(feature)):
            # if mini[w]==maxi[w]:
            #     t = [i] + [feature[i][mini[w]]]
            # else:
            t = [i]+feature[i][mini[w]:maxi[w]+1]
            t = str(t).replace("[", '').replace(']', '').replace(",", '').replace('\n', '')
            fout.write(t+'\n')





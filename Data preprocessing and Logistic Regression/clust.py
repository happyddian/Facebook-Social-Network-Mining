import random
import re
import numpy as np

#############################################
# 根据聚类的结果，获得某个类别出现次数较多的特征
#############################################

f = open('k_means_107.txt')
a = f.readlines()
nodeclass = {}
for w in a:
    w = w.strip('\n').split(' ')
    if w[1] in nodeclass:
        nodeclass[w[1]].append(w[0])
    else:
        nodeclass[w[1]] = []
        nodeclass[w[1]].append(w[0])

f = open('feature_107.txt')
a = f.readlines()
nodefeat = {}
for w in a:
    w = w.strip('\n').split(' ')
    w[1:] = [int(w[i]) for i in range(1,len(w))]
    nodefeat[w[0]] = np.array(w[1:])

f = open('featurenames.txt')
a = f.readlines()
featname = {}
for w in a:
    w = w.strip('\n').split(' ')
    featname[w[0]] = w[1]+' '+w[2]+' '+w[3]

classfeat = {}
for w in nodeclass:
    classfeat[w] = np.zeros((1,len(nodefeat['107'])))
    for n in nodeclass[w]:
        classfeat[w] = classfeat[w]+nodefeat[n]

classname = {}
for w in classfeat:
    classname[w] = []
    for i in range(1,10):
        k = featname[str(np.argsort(-classfeat[w])[0][i])]
        classname[w].append(k)
    print(classname[w])
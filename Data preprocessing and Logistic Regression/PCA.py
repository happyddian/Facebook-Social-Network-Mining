import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

###########################
# 使用PCA对特征降维
###########################

f = open('data/feature_all.txt')
feature = f.readlines()
rows = len(feature)
datamat = np.zeros((rows, 1406))
row = 0
for line in feature:
    line = line.strip().split(' ')
    datamat[row, :] = line[1:]                #feature矩阵
    row += 1
print(pearsonr(datamat[4025,:],datamat[0,:])) #Pearson相关系数
pca = PCA(100)   #降到100维
pca.fit(datamat)                  #训练
newX=pca.fit_transform(datamat)
plt.figure()
plt.plot(pca.explained_variance_, 'k', linewidth=2)
plt.xlabel('n_components', fontsize=16)
plt.ylabel('explained_variance_', fontsize=16)
plt.show()

i=0
line = []
for f in newX:
    a = str(f).replace("[", '').replace(']', '').replace('\n', '').replace('  ', ' ').split()  # feature矩阵
    a = [i] + a
    a[1:] = [float(i) for i in a[1:]]
    a = str(a)
    a = a.replace("[", '').replace(']', '').replace(",", '').replace('\n', '')
    line.append(a)
    i = i + 1

with open('data/feature_PCA.txt', "w") as fout:
    for w in line:
        fout.write(w+'\n')

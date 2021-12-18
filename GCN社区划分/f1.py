import sklearn
from sklearn.metrics import f1_score, accuracy_score

# 计算F1-score

f = open('data/pre.txt')
pre = f.readlines()
f = open('data/lab.txt')
label = f.readlines()
f = open('data/index.txt')
idx = f.readlines()

p = []
true = []
for i in range(len(idx)):
  idx[i] = idx[i].strip('\n')
  idx[i] = int(idx[i])
for i in range(len(pre)):
  if i in idx:
    pre[i] = pre[i].strip('\n')
    pre[i] = int(pre[i])
    label[i] = label[i].strip('\n')
    label[i] = int(label[i])
    p.append(pre[i])
    true.append(label[i])
for ave in ['micro', 'macro', 'weighted']:
  print(f1_score(true,p,average = ave))
print(accuracy_score(true,p))
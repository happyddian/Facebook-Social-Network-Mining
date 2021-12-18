data：各种数据集，包括最终得到的结果数据以及中间数据
feature：通过运行feature.py得到的不同类别的特征数据集
feature.py：构建特征矩阵，获得对应的特征名称同时得到的不同类别的特征数据集
L_regression.py：使用Pearson、完整特征、社区划分、RA等经典算法做逻辑回归
PCA.py：PCA特征降维
prepare_data.py：为GCN链接预测准备数据
clust.py：根据聚类的结果，获得某个类别出现次数较多的特征，但由于数据经过脱敏，无法为客户进行画像
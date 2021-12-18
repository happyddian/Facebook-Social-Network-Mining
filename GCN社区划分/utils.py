import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random

def get_splits(y,):
    # 分训练集测试集验证集
    idx_list = np.arange(len(y))
    train_percent = 0.8
    train_size = int(train_percent*len(y))
    index = random.sample(range(len(y)), train_size)
    idx_train = [idx_list[i] for i in index]
    idx_val_test = list(set(idx_list) - set(idx_train))
    idx_val = idx_val_test[0:100]
    idx_test = idx_val_test[100:]
    f=open('data/index.txt','w')
    for w in idx_test:
      f.write(str(w)+'\n')
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])
    return y_train, y_val, y_test,train_mask, val_mask, test_mask

def load_data_v2():
    # 读入数据
    #f = open('feature1.txt')
    f = open('data/feature_107_new.txt')
    feat = f.readlines()
    name = []
    for i in range(len(feat)):
      feat[i] = feat[i].strip("\n").split(' ')
      feat[i] = feat[i][1:]
    features = sp.csr_matrix(feat, dtype=np.float32)
    
    #f = open('label.txt')
    f = open('data/label_107.txt')
    label = f.readlines()
    name = []
    idx = []
    for i in range(len(label)):
      label[i] = label[i].strip("\n").split('\t')
      name.append(int(label[i][1]))
      idx.append(int(label[i][0]))
    onehot_labels = encode_onehot(name)
    
    # build graph
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    #f = open('edges_final.txt')
    f = open('data/edges_107_new.txt')
    edge = f.readlines()
    edges = np.zeros((len(edge), 2))
    for i in range(len(edge)):
      edge[i] = edge[i].strip("\n").split(' ')
      edges[i, :] = edge[i][:]  
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = convert_symmetric(adj, )
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    
def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def encode_onehot(labels):
    # onehot编码
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
    

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import pdb
import os
import random

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
    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
    
def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X 
    
def get_splits(y, ):
    # 划分训练集测试集验证集
    idx_list = np.arange(len(y))
    train_percent = 0.8
    np.random.seed(0)
    train_size = int(train_percent*len(y))
    index = random.sample(range(len(y)), train_size)

    idx_train = [idx_list[i] for i in index]
    idx_val_test = list(set(idx_list) - set(idx_train))
    idx_val = idx_val_test[0:60]
    idx_test = idx_val_test[60:]
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_v1():
    # 导入数据
    f = open('data/feature1.txt')
    feat = f.readlines()
    name = []
    for i in range(len(feat)):
      feat[i] = feat[i].strip("\n").split(' ')
      feat[i] = feat[i][1:]
    features = sp.csr_matrix(feat, dtype=np.float32)

    f = open('data/label.txt')
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
    
    f = open('data/edges_f.txt')
    edge = f.readlines()
    edges = np.zeros((len(edge), 2))
    for i in range(len(edge)):
      edge[i] = edge[i].strip("\n").split(' ')
      edges[i, :] = edge[i][:]  
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
                        
    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = convert_symmetric(adj, )
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )



def transferLabel2Onehot(labels, N):
    # onehot编码
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y


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
    features = features.astype(np.float32)
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


def construct_feed_dict(features, support, length, link_labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = {}
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['length']: length})
    feed_dict.update({placeholders['link_labels']: link_labels})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})

    return feed_dict

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import os
import sys
import time

from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Data, DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Embedding
from torch_geometric.nn import GATConv, GCNConv, GINConv, TransformerConv
from torch_geometric.nn import global_max_pool as gmp
from prefetch_generator import BackgroundGenerator 

from memory_profiler import profile


# In[ ]:


def evaluate(label, predict, fold):
    roc = roc_auc_score(label, predict)
    pr = average_precision_score(label, predict)
    return {'Fold':fold, 'ROC AUC':roc, 'PR AUC':pr}


# In[ ]:


def global_select_concat(feature, batch, x):
    feature = feature[x==-1]
    batch_size = batch[-1].item() + 1
    return feature.view(batch_size, -1)


# In[ ]:





# In[ ]:


class GCN_Model(torch.nn.Module):
    def __init__(self, word_sizes, embed_dim,  n_output=2):
        super(GCN_Model, self).__init__()

        self.embed_dim = embed_dim
        #layers
        self.embedding = Embedding(word_sizes, self.embed_dim)
        self.gcn1 = GATConv(embed_dim, 50)
        self.gcn2 = GATConv(50, 50)
        self.gcn3 = GATConv(50, 100)
        self.fc1 = Linear(200, 100)
        self.output = nn.Linear(100, n_output)
        
    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        feature = [torch.zeros(self.embed_dim).to(x.get_device()) if point==-1 else self.embedding(point) for point in x]
        feature = self.gcn1(torch.stack(feature), edge_index)
        feature = F.relu(feature)
        feature = self.gcn2(feature, edge_index)
        feature = F.relu(feature)
        feature = self.gcn3(feature, edge_index)
        feature = F.relu(feature)
        feature = global_select_concat(feature, batch, x)
        feature = self.fc1(feature)
        feature = F.dropout(feature, 0.2)
        out = self.output(feature)
        return out


# In[ ]:


class MyOwnDataset(Dataset):
    def __init__(self, root, indexs, graph_map, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.indexs = indexs
        self.graph_map = graph_map
            
    def len(self):
        return len(self.indexs)

    def get(self, idx):
        drug, pathway, label = self.graph_map[self.indexs[idx]]
        with open('{}/graphs/{}+{}+{}.pkl'.format(self.root, drug, pathway, label), 'rb') as file:
            data = pickle.load(file)
        return data


# In[ ]:


print('Start Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

substructure_type = 'molecular graph'
embed_dim = 11
print('Type: ', substructure_type, 'Dim: ', embed_dim)

kf = KFold(n_splits=10, shuffle=True)
if not os.path.exists('model_tmp'):
    os.mkdir('model_tmp')
if not os.path.exists('results'):
    os.mkdir('results')

with open('{}/graphs/graph_map.pkl'.format(substructure_type), 'rb') as file:
    graph_map = pickle.load(file)
with open('{}/entities2id.pkl'.format(substructure_type), 'rb') as file:
    entities2id = pickle.load(file) 
with open('{}/train_test_val_indexs.pkl'.format(substructure_type), 'rb') as file:
    train_test_val_indexs = pickle.load(file)
    
    
results = []
for fold, (train_indexs, test_indexs, val_indexs) in enumerate(train_test_val_indexs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GCN_Model(len(entities2id), embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_model_file = 'model_tmp/{}+{}+{}+GATConv.pt'.format(substructure_type, embed_dim, fold)
    early_stopping = EarlyStopping(file=best_model_file, patience=5)
    
    train_dataset = MyOwnDataset(substructure_type, train_indexs, graph_map)
    test_dataset = MyOwnDataset(substructure_type, test_indexs, graph_map)
    val_dataset = MyOwnDataset(substructure_type, val_indexs, graph_map)
    train_loader = BackgroundGenerator(DataLoader(train_dataset, batch_size=1000, shuffle=True))
    test_loader = BackgroundGenerator(DataLoader(test_dataset, batch_size=1000))
    val_loader = BackgroundGenerator(DataLoader(val_dataset, batch_size=1000))
    if not os.path.exists(best_model_file):
        model.train()
        for epoch in range(1000):
            model.train()

            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                logits = model(data)
                loss = F.cross_entropy(logits, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                break

            if epoch%10 == 0:
                model.eval()
                test_loss = 0
                for data in test_loader:
                    data = data.to(device)
                    logits = model(data)
                    test_loss = roc_auc_score(data.y.cpu().tolist(), logits.cpu()[:,1].tolist())
                    break

                print('{} Epoch {}: train loss {}, test ROC {}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch, train_loss, test_loss))
                early_stopping(-test_loss, model)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    break

    model = torch.load(best_model_file, map_location=device)
    labels = []
    predicts = []
    for data in val_loader:
        data = data.to(device)
        logits = model(data)
        labels.extend(data.y.cpu().tolist())
        predicts.extend(F.softmax(logits, dim=1).cpu()[:,1].tolist())
    result = evaluate(labels, predicts, fold)
    print(result)
    results.append(result)
results = pd.DataFrame(results)
results['Type'] = substructure_type
results['Dim'] = embed_dim
results.to_csv('results/{}+{}+GATConv.csv'.format(substructure_type, embed_dim))


# In[ ]:


import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# In[ ]:


def evaluate(label, predict, fold):
    roc = roc_auc_score(label, predict)
    pr = average_precision_score(label, predict)
    return {'Fold':fold, 'ROC AUC':roc, 'PR AUC':pr}


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, file='finish_model.pkl'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.file = file

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, self.file)                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


import os
from time import time
import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset
import json

### Code adapted from GraphEBM
def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])
def get_max_atoms(data_name):
    max_atoms=9
    if  data_name == 'ames_25_train1_neg':
        max_atoms=55
    elif data_name == 'ames_25_train1_pos':
         max_atoms=54
    elif data_name == 'ames_33_train1_neg':
         max_atoms=55
    elif data_name == 'ames_33_train1_pos':
         max_atoms=54
    elif data_name == 'ames_50_train1_neg':
         max_atoms=55
    elif data_name == 'ames_50_train1_pos':
         max_atoms=54
    elif data_name == 'ames_40_train1_neg':
         max_atoms=55
    elif data_name == 'ames_40_train1_pos':
         max_atoms=54
    elif data_name == 'bbb_martins_25_train1_neg':
         max_atoms=123
    elif data_name == 'bbb_martins_25_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_33_train1_neg':
         max_atoms=123
    elif data_name == 'bbb_martins_33_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_50_train1_neg':
         max_atoms=123
    elif data_name == 'bbb_martins_50_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_40_train1_neg':
         max_atoms=132
    elif data_name == 'bbb_martins_40_train1_pos':
         max_atoms=76
    elif data_name == 'cyp1a2_veith_25_train1_neg':
         max_atoms=123
    elif data_name == 'cyp1a2_veith_25_train1_pos':
         max_atoms=106
    elif data_name == 'cyp1a2_veith_33_train1_neg':
         max_atoms=123
    elif data_name == 'cyp1a2_veith_33_train1_pos':
         max_atoms=85
    elif data_name == 'cyp1a2_veith_50_train1_neg':
         max_atoms=123
    elif data_name == 'cyp1a2_veith_50_train1_pos':
         max_atoms=106
    elif data_name == 'cyp1a2_veith_40_train1_neg':
         max_atoms=123
    elif data_name == 'cyp1a2_veith_40_train1_pos':
         max_atoms=106
    elif data_name == 'cyp2c19_veith_25_train1_neg':
         max_atoms=85
    elif data_name == 'cyp2c19_veith_25_train1_pos':
         max_atoms=67
    elif data_name == 'cyp2c19_veith_33_train1_neg':
         max_atoms=101
    elif data_name == 'cyp2c19_veith_33_train1_pos':
         max_atoms=67
    elif data_name == 'cyp2c19_veith_50_train1_neg':
         max_atoms=101
    elif data_name == 'cyp2c19_veith_50_train1_pos':
         max_atoms=106
    elif data_name == 'cyp2c19_veith_40_train1_neg':
         max_atoms=114
    elif data_name == 'cyp2c19_veith_40_train1_pos':
         max_atoms=106
    elif data_name == 'herg_karim_25_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_25_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_33_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_33_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_50_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_50_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_40_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_40_train1_pos':
         max_atoms=50
    elif data_name == 'lipophilicity_astrazeneca_25_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_25_train1_pos':
         max_atoms=72
    elif data_name == 'lipophilicity_astrazeneca_33_train1_neg':
         max_atoms=65
    elif data_name == 'lipophilicity_astrazeneca_33_train1_pos':
         max_atoms=72
    elif data_name == 'lipophilicity_astrazeneca_50_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_50_train1_pos':
         max_atoms=58
    elif data_name == 'lipophilicity_astrazeneca_40_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_40_train1_pos':
         max_atoms=61
    return max_atoms

def get_atomic_num_list(data_name):

    if data_name == 'ames_25_train1_neg':
        atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_33_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_33_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_50_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_50_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_40_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_40_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_25_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_25_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53,0]
    elif data_name == 'bbb_martins_33_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17,0]
    elif data_name == 'bbb_martins_33_train1_pos':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 11, 15, 16, 17,0]
    elif data_name == 'bbb_martins_50_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_50_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53,0]
    elif data_name == 'bbb_martins_40_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 35,0]   
    elif data_name == 'cyp1a2_veith_25_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 78, 19, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_25_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 28, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_33_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 25, 26, 27, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_33_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_40_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 14, 24, 25, 26, 27, 30, 33, 34, 35, 51, 53,0]
    elif data_name == 'cyp1a2_veith_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 14, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_50_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 78, 20, 24, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_50_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 80, 28, 29, 34, 35, 53,0]
    elif data_name == 'cyp2c19_veith_25_train1_neg':
         atomic_num_list=[1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 78, 20, 26, 29, 33, 35, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_25_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_33_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 19, 20, 80, 25, 26, 29, 30, 33, 34, 35, 44, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_33_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 28, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_40_train1_neg':
         atomic_num_list=[1, 3, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 80, 20, 25, 26, 29, 30, 33, 35, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 26, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_50_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 19, 20, 25, 26, 29, 30, 33, 34, 35, 44, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_50_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 29, 35, 53,0]
    elif data_name == 'herg_karim_25_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 79, 16, 17, 15, 34, 35, 53,0]
    elif data_name == 'herg_karim_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'herg_karim_33_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'herg_karim_33_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'herg_karim_40_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'herg_karim_40_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'herg_karim_50_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 79, 34, 35, 53,0]
    elif data_name == 'herg_karim_50_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 14, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_25_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_33_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_33_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_40_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_40_train1_pos':
         atomic_num_list=[1, 34, 35, 5, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_50_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_50_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,0]
          
    return atomic_num_list

def get_transform_fn(dataset):
    if dataset == 'QM9':
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F
            x_ = np.zeros((9, 5))
            indices = np.where(x >= 6, x - 6, 4)
            x_[np.arange(9), indices] = 1
            x = torch.tensor(x_).to(torch.float)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                    axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float)
            return x, adj


    elif dataset == 'ZINC250k' or  dataset in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            zinc250k_atomic_num_list = get_atomic_num_list(dataset)
            x_ = np.zeros((get_max_atoms(dataset), len(zinc250k_atomic_num_list)), dtype=np.float32)
            for i in range(get_max_atoms(dataset)):
                ind = zinc250k_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float)
            return x, adj

    return transform

def dataloader(config, get_graph_list=False):
    start_time = time()
    
    mols = load_mol(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    with open(os.path.join(config.data.dir, f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)
        
    if config.data.data  in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
    
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx if i<len(mols)]
    train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data))
    test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data))

    if get_graph_list:
        train_mols_nx = [nx.DiGraph(np.array(adj)) for x, adj in train_dataset]
        test_mols_nx = [nx.DiGraph(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx

    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader

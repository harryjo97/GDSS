from time import time
import pickle
import json
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils.mol_utils import mols_to_nx, smiles_to_mols


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='ZINC250k', choices=['ZINC250k', 'QM9'])
args = parser.parse_args()

dataset = args.dataset
start_time = time()

with open(f'data/valid_idx_{dataset.lower()}.json') as f:
    test_idx = json.load(f)

if dataset == 'QM9':
    test_idx = test_idx['valid_idxs']
    test_idx = [int(i) for i in test_idx]
    col = 'SMILES1'
elif dataset == 'ZINC250k':
    col = 'smiles'
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={dataset}")

smiles = pd.read_csv(f'data/{dataset.lower()}.csv')[col]
test_smiles = [smiles.iloc[i] for i in test_idx]
nx_graphs = mols_to_nx(smiles_to_mols(test_smiles))
print(f'Converted the test molecules into {len(nx_graphs)} graphs')

with open(f'data/{dataset.lower()}_test_nx.pkl', 'wb') as f:
    pickle.dump(nx_graphs, f)

print(f'Total {time() - start_time:.2f} sec elapsed')

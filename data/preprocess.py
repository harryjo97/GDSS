### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import argparse
import time
from utils.data_frame_parser import DataFrameParser
from utils.numpytupledataset import NumpyTupleDataset
from utils.smile_to_graph import GGNNPreprocessor


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='ZINC250k', choices=[ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos'])
args = parser.parse_args()

start_time = time.time()
data_name = args.dataset
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


if data_name == 'ZINC250k':
    max_atoms = 38
    path = 'data/zinc250k.csv'
    smiles_col = 'smiles'
    label_idx = 1
elif data_name == 'QM9':
    max_atoms = 9
    path = 'data/qm9.csv'
    smiles_col = 'SMILES1'
    label_idx = 2
elif data_name in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
    max_atoms =get_max_atoms(data_name)
    path = 'data/{}.csv'.format(data_name)
    smiles_col = 'smiles'
    label_idx = 1
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

print(f'Preprocessing {data_name} data')
df = pd.read_csv(path, index_col=0)
# Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
# 'smiles' column contains '\n', need to remove it.
# Here we do not remove \n, because it represents atom N with single bond
labels = df.keys().tolist()[label_idx:]
parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
result = parser.parse(df, return_smiles=True)

dataset = result['dataset']
smiles = result['smiles']

NumpyTupleDataset.save(f'data/{data_name.lower()}_kekulized.npz', dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

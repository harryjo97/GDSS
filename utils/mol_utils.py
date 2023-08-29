import numpy as np
import pandas as pd
import json
import networkx as nx

import re
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


ATOM_VALENCY = {5:3,6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1, 11:1, 1:1,28:3,80:1,\
 78:4,14:4,34:-2,29:2,20:2,33:3,25:2, 50:4,3:1, 51:5, 24:6, 30:2, 26:2, 19:1, 44:1,79:1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
AN_TO_SYMBOL = {5:'B',6: 'C', 7: 'N', 8: 'O', 9: 'F',11:'Na', 15: 'P', 16: 'S', 17: 'Cl',\
 28:'Ni', 35: 'Br', 53: 'I', 1:'H', 80:'Hg', 78:'Pt', 14:'Si', 34:'Se', 29:'Cu', 20:'Ca', 33:'As',\
 25:'Mn',50:'Sn', 3:'Li', 51:'Sb', 24:'Cr', 30:'Zn', 26:'Fe', 19:'K', 44:'Ru',79:'Au'}


def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def load_smiles(dataset='QM9'):
    if dataset == 'QM9':
        col = 'SMILES1'
    elif dataset in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')
    
    df = pd.read_csv(f'data/{dataset.lower()}.csv')

    with open(f'data/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)
    
    if dataset in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
    
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])
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
         atomic_num_list[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 35,0]
    elif data_name == 'cyp1a2_veith_25_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 78, 19, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_25_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 80, 28, 29, 34, 35, 53,0]
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
         atomic_num_list[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 19, 20, 80, 25, 26, 29, 30, 33, 34, 35, 44, 50, 51, 53,0]
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


def gen_mol(x, adj, dataset, largest_connected_comp=True):    
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()

    if dataset == 'QM9':
        atomic_num_list = [6, 7, 8, 9, 0]
    else:
        atomic_num_list = get_atomic_num_list(dataset)
    # mols_wo_correction = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    # mols_wo_correction = [mol for mol in mols_wo_correction if mol is not None]
    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct: num_no_correct += 1
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        mols.append(vcmol)
    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def construct_mol(x, adj, atomic_num_list): # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]              # 9,
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)            # 9, 9
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1                                # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    #    atomic_num=atom.GetAtomicNum(),
                    #    formal_charge=atom.GetFormalCharge(),
                    #    chiral_tag=atom.GetChiralTag(),
                    #    hybridization=atom.GetHybridization(),
                    #    num_explicit_hs=atom.GetNumExplicitHs(),
                    #    is_aromatic=atom.GetIsAromatic())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
                    #    bond_type=bond.GetBondType())
        
        nx_graphs.append(G)
    return nx_graphs

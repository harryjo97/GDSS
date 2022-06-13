import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


# -------- Create initial node features --------
def init_features(init, adjs=None, nfeat=10):

    if init=='zeros':
        feature = torch.zeros((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='ones':
        feature = torch.ones((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='deg':
        feature = adjs.sum(dim=-1).to(torch.long)
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            print(feature.max())
            raise NotImplementedError(f'max_feat_num mismatch')
    else:
        raise NotImplementedError(f'{init} not implemented')

    flags = node_flags(adjs)

    return mask_x(feature, flags)


# -------- Sample initial flags tensor from the training graph set --------
def init_flags(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])

    return flags


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


# -------- Quantize generated graphs --------
def quantize(adjs, thr=0.5):
    adjs_ = torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))
    return adjs_


# -------- Quantize generated molecules --------
# adjs: 32 x 9 x 9
def quantize_mol(adjs):                         
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def adjs_to_graphs(adjs, is_cuda=False):
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


# -------- Check if the adjacency matrices are symmetric --------
def check_sym(adjs, print_val=False):
    sym_error = (adjs-adjs.transpose(-1,-2)).abs().sum([0,1,2])
    if not sym_error < 1e-2:
        raise ValueError(f'Not symmetric: {sym_error:.4e}')
    if print_val:
        print(f'{sym_error:.4e}')


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


# -------- Create padded adjacency matrices --------
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def graphs_to_tensor(graph_list, max_node_num):
    adjs_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        adj = nx.to_numpy_matrix(g, nodelist=node_list)
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        adjs_list.append(padded_adj)

    del graph_list

    adjs_np = np.asarray(adjs_list)
    del adjs_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    del adjs_np

    return adjs_tensor 


def graphs_to_adj(graph, max_node_num):
    max_node_num = max_node_num

    assert isinstance(graph, nx.Graph)
    node_list = []
    for v, feature in graph.nodes.data('feature'):
        node_list.append(v)

    adj = nx.to_numpy_matrix(graph, nodelist=node_list)
    padded_adj = pad_adjs(adj, node_number=max_node_num)

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    del padded_adj

    return adj


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair

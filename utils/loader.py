import torch
import random
import numpy as np

from models.ScoreNetwork_A import ScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from sde import VPSDE, VESDE, subVPSDE

from losses import get_sde_loss_fn
from solver import get_pc_sampler, S4_solver
from evaluation.mmd import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    elif torch.backends.mps.is_available(): 
         device = torch.device('mps' )
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    
    elif torch.backends.mps.is_available(): 
        model=model.to(torch.device("mps"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
        from utils.data_loader_mol import dataloader
        return dataloader(config, get_graph_list)
    else:
        from utils.data_loader import dataloader
        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device):
    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num  = config_train.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if config_module.predictor == 'S4':
        get_sampler = S4_solver
    else:
        get_sampler = get_pc_sampler

    if config_train.data.data in ['QM9', 'ZINC250k']:
        shape_x = (10000, max_node_num, config_train.data.max_feat_num)
        shape_adj = (10000, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_25_train1_pos']:
          shape_x = (676, max_node_num, config_train.data.max_feat_num)
          shape_adj = (676, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_25_train1_neg']:
          shape_x = (597, max_node_num, config_train.data.max_feat_num)
          shape_adj = (597, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_33_train1_pos']:
          shape_x = (901, max_node_num, config_train.data.max_feat_num)
          shape_adj = (901, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_33_train1_neg']:
          shape_x = (797, max_node_num, config_train.data.max_feat_num)
          shape_adj = (797, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_40_train1_pos']:
          shape_x = (1087, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1087, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_40_train1_neg']:
          shape_x = (949, max_node_num, config_train.data.max_feat_num)
          shape_adj = (949, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_50_train1_pos']:
          shape_x = (1377, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1377, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_50_train1_neg']:
          shape_x = (1170, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1170, max_node_num, max_node_num)



    elif config_train.data.data in ['bbb_martins_25_train1_pos']:
          shape_x = (285, max_node_num, config_train.data.max_feat_num)
          shape_adj = (285, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_25_train1_neg']:
          shape_x = (70, max_node_num, config_train.data.max_feat_num)
          shape_adj = (70, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_33_train1_pos']:
          shape_x = (361, max_node_num, config_train.data.max_feat_num)
          shape_adj = (361, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_33_train1_neg']:
          shape_x = (112, max_node_num, config_train.data.max_feat_num)
          shape_adj = (112, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_40_train1_pos']:
          shape_x = (435, max_node_num, config_train.data.max_feat_num)
          shape_adj = (435, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_40_train1_neg']:
          shape_x = (133, max_node_num, config_train.data.max_feat_num)
          shape_adj = (133, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_50_train1_pos']:
          shape_x = (551, max_node_num, config_train.data.max_feat_num)
          shape_adj = (551, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_50_train1_neg']:
          shape_x = (159, max_node_num, config_train.data.max_feat_num)
          shape_adj = (159, max_node_num, max_node_num)


    elif config_train.data.data in ['cyp1a2_veith_25_train1_pos']:
          shape_x = (1048, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1048, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_25_train1_neg']:
          shape_x = (1153, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1153, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp1a2_veith_33_train1_pos']:
          shape_x = (1358, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1358, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_33_train1_neg']:
          shape_x = (1577, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1577, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp1a2_veith_40_train1_pos']:
          shape_x = (1604, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1604, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_40_train1_neg']:
          shape_x = (1916, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1916, max_node_num, max_node_num)
       
    elif config_train.data.data in ['cyp1a2_veith_50_train1_pos']:
          shape_x = (2064, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2064, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_50_train1_neg']:
          shape_x = (2338, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2338, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_25_train1_pos']:
          shape_x = (986, max_node_num, config_train.data.max_feat_num)
          shape_adj = (986, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_25_train1_neg']:
          shape_x = (1230, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1230, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_33_train1_pos']:
          shape_x = (1332, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1332, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_33_train1_neg']:
          shape_x = (1623, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1623, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_40_train1_pos']:
          shape_x = (1651, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1651, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_40_train1_neg']:
          shape_x = (1893, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1893, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_50_train1_pos']:
          shape_x = (1978, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1978, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_50_train1_neg']:
          shape_x = (2455, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2455, max_node_num, max_node_num)


    
    elif config_train.data.data in ['herg_karim_25_train1_pos']:
          shape_x = (3562, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3562, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_25_train1_neg']:
          shape_x = (3497, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3497, max_node_num, max_node_num)

    elif config_train.data.data in ['herg_karim_33_train1_pos']:
          shape_x = (1152, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1152, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_33_train1_neg']:
          shape_x = (1201, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1201, max_node_num, max_node_num)
        
    elif config_train.data.data in ['herg_karim_40_train1_pos']:
          shape_x = (1890, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1890, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_40_train1_neg']:
          shape_x = (1874, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1874, max_node_num, max_node_num)

    elif config_train.data.data in ['herg_karim_50_train1_pos']:
          shape_x = (2350, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2350, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_50_train1_neg']:
          shape_x = (2356, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2356, max_node_num, max_node_num)

    elif config_train.data.data in ['lipophilicity_astrazeneca_25_train1_pos']:
          shape_x = (629, max_node_num, config_train.data.max_feat_num)
          shape_adj = (629, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_25_train1_neg']:
          shape_x = (106, max_node_num, config_train.data.max_feat_num)
          shape_adj = (106, max_node_num, max_node_num)
 
    elif config_train.data.data in ['lipophilicity_astrazeneca_33_train1_pos']:
          shape_x = (824, max_node_num, config_train.data.max_feat_num)
          shape_adj = (824, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_33_train1_neg']:
          shape_x = (156, max_node_num, config_train.data.max_feat_num)
          shape_adj = (156, max_node_num, max_node_num)

    elif config_train.data.data in ['lipophilicity_astrazeneca_40_train1_pos']:
          shape_x = (980, max_node_num, config_train.data.max_feat_num)
          shape_adj = (980, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_40_train1_neg']:
          shape_x = (196, max_node_num, config_train.data.max_feat_num)
          shape_adj = (196, max_node_num, max_node_num)
 
    elif config_train.data.data in ['lipophilicity_astrazeneca_50_train1_pos']:
          shape_x = (1208, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1208, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_50_train1_neg']:
          shape_x = (262, max_node_num, config_train.data.max_feat_num)
          shape_adj = (262, max_node_num, max_node_num)

    else:
        shape_x = (config_train.data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        
    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj, 
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                snr=config_module.snr, scale_eps=config_module.scale_eps, 
                                n_steps=config_module.n_steps, 
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn


def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth, 
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final, 
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv':config_m.conv}
    else:
        params_x = {'model_type':config_m.x, 'max_feat_num':max_feat_num, 'depth':config_m.depth, 'nhid':config_m.nhid}
    params_adj = {'model_type':config_m.adj, 'max_feat_num':max_feat_num, 'max_node_num':config.data.max_node_num, 
                    'nhid':config_m.nhid, 'num_layers':config_m.num_layers, 'num_linears':config_m.num_linears, 
                    'c_init':config_m.c_init, 'c_hid':config_m.c_hid, 'c_final':config_m.c_final, 
                    'adim':config_m.adim, 'num_heads':config_m.num_heads, 'conv':config_m.conv}
    return params_x, params_adj


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
    path = f'./checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    ckpt_dict= {'config': ckpt['model_config'], 'params_x': ckpt['params_x'], 'x_state_dict': ckpt['x_state_dict'],
                'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt['adj_state_dict']}
    if config.sample.use_ema:
        ckpt_dict['ema_x'] = ckpt['ema_x']
        ckpt_dict['ema_adj'] = ckpt['ema_adj']
    if return_ckpt:
        ckpt_dict['ckpt'] = ckpt
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    elif torch.backends.mps.is_available(): 
         model.to(torch.device("mps"))
    return model


def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels

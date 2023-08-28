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
          shape_x = (2083, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2083, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_25_train1_neg']:
          shape_x = (1738, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1738, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_33_train1_pos']:
          shape_x = (1858, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1858, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_33_train1_neg']:
          shape_x = (1538, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1538, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_40_train1_pos']:
          shape_x = (1672, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1672, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_40_train1_neg']:
          shape_x = (1386, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1386, max_node_num, max_node_num)

    elif config_train.data.data in ['ames_50_train1_pos']:
          shape_x = (1382, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1382, max_node_num, max_node_num)
    elif config_train.data.data in ['ames_50_train1_neg']:
          shape_x = (1165, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1165, max_node_num, max_node_num)



    elif config_train.data.data in ['bbb_martins_25_train1_pos']:
          shape_x = (811, max_node_num, config_train.data.max_feat_num)
          shape_adj = (811, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_25_train1_neg']:
          shape_x = (255, max_node_num, config_train.data.max_feat_num)
          shape_adj = (255, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_33_train1_pos']:
          shape_x = (735, max_node_num, config_train.data.max_feat_num)
          shape_adj = (735, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_33_train1_neg']:
          shape_x = (213, max_node_num, config_train.data.max_feat_num)
          shape_adj = (213, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_40_train1_pos']:
          shape_x = (661, max_node_num, config_train.data.max_feat_num)
          shape_adj = (661, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_40_train1_neg']:
          shape_x = (192, max_node_num, config_train.data.max_feat_num)
          shape_adj = (192, max_node_num, max_node_num)

    elif config_train.data.data in ['bbb_martins_50_train1_pos']:
          shape_x = (545, max_node_num, config_train.data.max_feat_num)
          shape_adj = (545, max_node_num, max_node_num)
    elif config_train.data.data in ['bbb_martins_50_train1_neg']:
          shape_x = (166, max_node_num, config_train.data.max_feat_num)
          shape_adj = (166, max_node_num, max_node_num)


    elif config_train.data.data in ['cyp1a2_veith_25_train1_pos']:
          shape_x = (3012, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3012, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_25_train1_neg']:
          shape_x = (3592, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3592, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp1a2_veith_33_train1_pos']:
          shape_x = (2702, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2702, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_33_train1_neg']:
          shape_x = (3168, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3168, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp1a2_veith_40_train1_pos']:
          shape_x = (2456, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2456, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_40_train1_neg']:
          shape_x = (2829, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2829, max_node_num, max_node_num)
       
    elif config_train.data.data in ['cyp1a2_veith_50_train1_pos']:
          shape_x = (1996, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1996, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp1a2_veith_50_train1_neg']:
          shape_x = (2407, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2407, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_25_train1_pos']:
          shape_x = (3077, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3077, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_25_train1_neg']:
          shape_x = (3573, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3573, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_33_train1_pos']:
          shape_x = (2731, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2731, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_33_train1_neg']:
          shape_x = (3180, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3180, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_40_train1_pos']:
          shape_x = (2412, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2412, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_40_train1_neg']:
          shape_x = (2910, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2910, max_node_num, max_node_num)

    elif config_train.data.data in ['cyp2c19_veith_50_train1_pos']:
          shape_x = (2085, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2085, max_node_num, max_node_num)
    elif config_train.data.data in ['cyp2c19_veith_50_train1_neg']:
          shape_x = (2348, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2348, max_node_num, max_node_num)


    
    elif config_train.data.data in ['herg_karim_25_train1_pos']:
          shape_x = (3562, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3562, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_25_train1_neg']:
          shape_x = (3497, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3497, max_node_num, max_node_num)

    elif config_train.data.data in ['herg_karim_33_train1_pos']:
          shape_x = (3115, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3115, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_33_train1_neg']:
          shape_x = (3160, max_node_num, config_train.data.max_feat_num)
          shape_adj = (3160, max_node_num, max_node_num)
        
    elif config_train.data.data in ['herg_karim_40_train1_pos']:
          shape_x = (2824, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2824, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_40_train1_neg']:
          shape_x = (2824, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2824, max_node_num, max_node_num)

    elif config_train.data.data in ['herg_karim_50_train1_pos']:
          shape_x = (2364, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2364, max_node_num, max_node_num)
    elif config_train.data.data in ['herg_karim_50_train1_neg']:
          shape_x = (2342, max_node_num, config_train.data.max_feat_num)
          shape_adj = (2342, max_node_num, max_node_num)

    elif config_train.data.data in ['lipophilicity_astrazeneca_25_train1_pos']:
          shape_x = (1817, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1817, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_25_train1_neg']:
          shape_x = (388, max_node_num, config_train.data.max_feat_num)
          shape_adj = (388, max_node_num, max_node_num)
 
    elif config_train.data.data in ['lipophilicity_astrazeneca_33_train1_pos']:
          shape_x = (1622, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1622, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_33_train1_neg']:
          shape_x = (338, max_node_num, config_train.data.max_feat_num)
          shape_adj = (338, max_node_num, max_node_num)

    elif config_train.data.data in ['lipophilicity_astrazeneca_40_train1_pos']:
          shape_x = (1466, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1466, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_40_train1_neg']:
          shape_x = (298, max_node_num, config_train.data.max_feat_num)
          shape_adj = (298, max_node_num, max_node_num)
 
    elif config_train.data.data in ['lipophilicity_astrazeneca_50_train1_pos']:
          shape_x = (1238, max_node_num, config_train.data.max_feat_num)
          shape_adj = (1238, max_node_num, max_node_num)
    elif config_train.data.data in ['lipophilicity_astrazeneca_50_train1_neg']:
          shape_x = (232, max_node_num, config_train.data.max_feat_num)
          shape_adj = (232, max_node_num, max_node_num)

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

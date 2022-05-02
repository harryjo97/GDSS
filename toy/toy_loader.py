import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os

from toy_models import ScoreNetwork
from toy_data import data_distribution
from toy_losses import get_sde_loss_fn
from toy_solver import get_pc_sampler
from toy_sde import VPSDE, VESDE


def load_data_distribution(mus, covs, batch_size, device):
    x = data_distribution(mus, covs, batch_size)
    return torch.from_numpy( x ).to(device=device, dtype=torch.float)


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


def load_device(gpu):
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'
    return device


def load_model_optimizer(config, params, device):
    model = ScoreNetwork(**params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, 
                                    weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    return model, optimizer, scheduler


def load_sampling_fn(configt, config, device):
    sde1 = load_sde(configt.sde1)
    sde2 = load_sde(configt.sde2)
    cs = config.sample
    sampling_fn = get_pc_sampler(sde1=sde1, sde2=sde2, 
                                    shape=(configt.train.batch_size, 2),
                                    predictor=cs.predictor, corrector=cs.corrector,
                                    snr=cs.snr, scale_eps=cs.scale_eps, n_steps=cs.n_steps, sampling_steps=1,
                                    probability_flow=False, continuous=True, denoise=True, 
                                    eps=cs.eps, device=device)
    return sampling_fn


def load_model_from_ckpt(params, state_dict, device):
    model = ScoreNetwork(**params)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def load_params(configm):
    params = {'num_layers': configm.num_layers, 'input_dim': configm.input_dim, 
                'hidden_dim': configm.hidden_dim, 'output_dim': configm.output_dim}
    return params


def load_sde(configs):
    if configs.type=='VP':
        sde = VPSDE(configs.beta_min, configs.beta_max, configs.num_scales)
    elif configs.type=='VE':
        sde = VESDE(configs.beta_min, configs.beta_max, configs.num_scales)
    else:
        raise NotImplementedError(f'SDE {configs.type} not implemented.')
    return sde


def load_loss_fn(config):
    sde1 = load_sde(config.sde1)
    sde2 = load_sde(config.sde2)
    loss_fn = get_sde_loss_fn(sde1, sde2, train=True, reduce_mean=config.train.reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_data_settings(configd):

    coms = configd.coms
    rho = configd.rho

    mus = [[coms[0],coms[1]],[-coms[0],-coms[1]]]
    covs = np.array([[[1,rho],[rho,1]],
                    [[1,rho],[rho,1]],]) * configd.norm

    return mus, covs


def plot(gen_list, ckpt='test', tick_size=16):

    x = np.array(gen_list)
    plt.figure(figsize=(6, 5))
    cmap = plt.cm.get_cmap('viridis', 5)
    plt.scatter(x[:, 0], x[:, 1], c='r', s=0.5)

    R = 1.0
    plt.xlim(-R,R)
    plt.ylim(-R,R)
    plt.axis('off')

    fig_dir = './toy/plot/'
    plt.tight_layout()
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir, ckpt),
                bbox_inches='tight',
                dpi=200,
                transparent=False)
    plt.close()


def save_gen_list(gen_list, exp_name):
    if not(os.path.isdir('./toy/samples/')):
        os.makedirs(os.path.join('./toy/samples/'))
    with open('./toy/samples/{}.pkl'.format( exp_name), 'wb') as f:
            pickle.dump(obj=gen_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './toy/samples/{}.pkl'.format(exp_name)
    return save_dir




import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import argparse
import yaml
from easydict import EasyDict as edict

from toy_data import data_distribution
from toy_loader import load_seed, load_device, load_params, load_model_optimizer, \
                    load_loss_fn, load_data_settings, load_data_distribution,  \
                    load_model_from_ckpt, load_sampling_fn, plot, save_gen_list
from toy_logger import set_log, Logger, train_log

class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        load_seed(self.config.seed)
        self.device = load_device(self.config.gpu)

        self.params1 = load_params(self.config.model.x1)
        self.params2 = load_params(self.config.model.x2)
        self.model1, self.optimizer1, self.scheduler1 = load_model_optimizer(self.config, self.params1, self.device)
        self.model2, self.optimizer2, self.scheduler2 = load_model_optimizer(self.config, self.params2, self.device)

        self.loss_fn = load_loss_fn(self.config)

        self.mus, self.covs = load_data_settings(config.data)

    def train(self, ckpt):

        self.ckpt = ckpt
        set_log(self.ckpt)
        logger = Logger(str(os.path.join('./toy/logs', f'{self.ckpt}.log')), mode='a')
        logger.log(f'{ckpt}')
        train_log(logger, self.config)

        for epoch in trange(0, (self.config.train.epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train1 = []
            self.train2 = []
            self.test1 = []
            self.test2 = []
            t_start = time.time()

            self.model1.train()
            self.model2.train()
            # -------- Train --------
            x = load_data_distribution(self.mus, self.covs, self.config.train.batch_size, self.device)
            
            loss1, loss2 = self.loss_fn(self.model1, self.model2, x)
            loss1.backward()
            loss2.backward()

            torch.nn.utils.clip_grad_norm_(self.model1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 1.0)

            self.optimizer1.step()
            self.optimizer2.step()

            self.train1.append(loss1.item())
            self.train2.append(loss2.item())

            self.scheduler1.step()
            self.scheduler2.step()

            self.model1.eval()
            self.model2.eval()

            x = load_data_distribution(self.mus, self.covs, self.config.train.val_batch_size, self.device)
            with torch.no_grad():
                loss1, loss2 = self.loss_fn(self.model1, self.model2, x)
                self.test1.append(loss1.item())
                self.test2.append(loss2.item())

            mean_train1 = np.mean(self.train1)
            mean_train2 = np.mean(self.train2)
            mean_test1 = np.mean(self.test1)
            mean_test2 = np.mean(self.test2)

            logger.log(f'{epoch:02d}|{time.time()-t_start:.2f}s| '
                        f'test1: {mean_test1:.3e}| train1: {mean_train1:.3e}| '
                        f'test2: {mean_test2:.3e}| train2: {mean_train2:.3e}| ', verbose=False)

        torch.save({ 
            'config': self.config,
            'params1': self.params1,
            'params2': self.params2,
            'state_dict1': self.model1.state_dict(), 
            'state_dict2': self.model2.state_dict(),
            }, f'./toy/checkpoints/{self.ckpt}.pth')


class Sampler(object):

    def __init__(self, config):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device(self.config.gpu)

    def sample(self, ckpt):
        self.ckpt = ckpt
        path = f'./toy/checkpoints/{self.ckpt}.pth'
        self.ckpt_dict = torch.load(path, map_location=self.device)
        self.configt = self.ckpt_dict['config']

        self.model1 = load_model_from_ckpt(self.ckpt_dict['params1'], self.ckpt_dict['state_dict1'], self.device)
        self.model2 = load_model_from_ckpt(self.ckpt_dict['params2'], self.ckpt_dict['state_dict2'], self.device)

        self.sampling_fn = load_sampling_fn(self.configt, self.config, self.device)

        # -------- Generate samples --------
        load_seed(self.config.seed)
        gen_list = []
        for r in range(self.config.sample.num_sampling_rounds):
            x = self.sampling_fn(self.model1, self.model2)
            gen_list.extend(x.cpu().numpy())

        fig_name = f'{self.ckpt}'
        plot(gen_list, fig_name+'.png')
        save_gen_list(gen_list, fig_name)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Toy experiment')
    parser.add_argument('--config', type=str, default='toy/toy_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader))
    config.gpu = args.gpu
    config.seed = args.seed

    trainer = Trainer(config)
    sampler = Sampler(config)

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    trainer.train(ts)
    sampler.sample(ts)

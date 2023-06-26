import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config) 
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in [ 'ames_25_train1_neg','ames_25_train1_pos','ames_33_train1_neg','ames_33_train1_pos','ames_40_train1_neg','ames_40_train1_pos','ames_50_train1_neg','ames_50_train1_pos','bbb_martins_25_train1_neg','bbb_martins_25_train1_pos','bbb_martins_33_train1_neg','bbb_martins_33_train1_pos','bbb_martins_50_train1_neg','bbb_martins_50_train1_pos','bbb_martins_40_train1_neg','bbb_martins_40_train1_pos','cyp1a2_veith_25_train1_neg','cyp1a2_veith_25_train1_pos','cyp1a2_veith_33_train1_neg','cyp1a2_veith_33_train1_pos','cyp1a2_veith_50_train1_neg','cyp1a2_veith_50_train1_pos','cyp1a2_veith_40_train1_neg','cyp1a2_veith_40_train1_pos','cyp2c19_veith_25_train1_neg','cyp2c19_veith_25_train1_pos','cyp2c19_veith_33_train1_neg','cyp2c19_veith_33_train1_pos','cyp2c19_veith_50_train1_neg','cyp2c19_veith_50_train1_pos','cyp2c19_veith_40_train1_neg','cyp2c19_veith_40_train1_pos','herg_karim_25_train1_neg','herg_karim_25_train1_pos','herg_karim_33_train1_neg','herg_karim_33_train1_pos','herg_karim_50_train1_neg','herg_karim_50_train1_pos','herg_karim_40_train1_neg','herg_karim_40_train1_pos','lipophilicity_astrazeneca_25_train1_neg','lipophilicity_astrazeneca_25_train1_pos','lipophilicity_astrazeneca_33_train1_neg','lipophilicity_astrazeneca_33_train1_pos','lipophilicity_astrazeneca_50_train1_neg','lipophilicity_astrazeneca_50_train1_pos','lipophilicity_astrazeneca_40_train1_neg','lipophilicity_astrazeneca_40_train1_pos']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
        
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])

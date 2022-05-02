import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')

        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

        if verbose:
            print(str)


def set_log(exp_name='toy'):

    if not(os.path.isdir(f'./toy/logs/')):
        os.makedirs(os.path.join(f'./toy/logs/'))
    ckpt_dir = os.path.join(f'./toy/logs/')
    
    if not(os.path.isdir(f'./toy/checkpoints/')):
        os.makedirs(os.path.join(f'./toy/checkpoints/'))
    ckpt_dir = os.path.join(f'./toy/checkpoints/')



def train_log(logger, config):
    configm = config.model
    logger.log('-'*100)
    logger.log(f'({config.sde1.type}: {config.sde1.beta_min:.2f}, {config.sde1.beta_max:.1f})')
    logger.log(f'LAYERS1={configm.x1.num_layers}  HID1={configm.x1.hidden_dim}  '
                f'LAYERS1={configm.x2.num_layers}  HID1={configm.x2.hidden_dim}')
    logger.log(f'EPOCHS={config.train.epochs}  LR={config.train.lr}  BS={config.train.batch_size}')
    logger.log('-'*100)
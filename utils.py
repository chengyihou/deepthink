import os
import sys
import errno
import os.path as osp
import numpy as np
import torch



def mkdir_if_missing(directory):

    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Logger(object):

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



def save_networks(networks, result_dir, name='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    # 在 log/models/HRRP 下创建 checkpoints
    weights = networks.state_dict()

    filename = '{}/checkpoints/{}.pth'.format(result_dir, name) # loss 默认没存，只存了模型名字
    torch.save(weights, filename)

    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_criterion.pth'.format(result_dir, name)
        torch.save(weights, filename)



def load_networks(networks, result_dir, name='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}.pth'.format(result_dir, name)
    networks.load_state_dict(torch.load(filename,weights_only=True))

    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_criterion.pth'.format(result_dir, name)
        criterion.load_state_dict(torch.load(filename,weights_only=True))

    return networks, criterion

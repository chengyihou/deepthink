import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import random

from eval import test_1
from models import ConvNet
from HRRP_OSR import HRRP_OSR # 绿色是类，黄色是函数
from utils import Logger, save_networks, load_networks
from plot import plot_tsne_by_class
from train import train
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='HRRP', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log') # 后面和model_path结合
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=32) # net main hrrp_osr
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model") # 0.005 con1 0.01

# scheduler
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.15,help="weight for center loss") # 0.1 # 0.3 最终
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='ConvNet')

# misc
parser.add_argument('--nz', type=int, default=50) # 用于输入到G的通道数
parser.add_argument('--ns', type=int, default=1) # 输入到G的噪声的维度
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--eval', action='store_true', help="Eval", default= False)
# parser.add_argument('--cs', action='store_true', help="Confusing Sample", default= False)



def main_worker(options):
    torch.manual_seed(options['seed']) # 设置随机种子，确保结果可复现
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()

    if options['use_cpu']: use_gpu = False
    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed']) # 设置所有GPU的随机种子，确保结果可复现
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    Data = HRRP_OSR(known = options['known'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_classes'] = Data.num_classes # 已知类的个数

    # Model
    print("Creating model: {}".format(options['model']))
    net = ConvNet(num_classes=options['num_classes'])  # 后面改
    feat_dim = 64

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )
    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    # use_gpu
    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    # file_name 
    model_path = os.path.join(options['outf'], 'models', options['dataset']) # log/models/HRRP
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_name = ('{}_{}_{}'.format(options['model'], options['loss'],options['known']))

    # eval 
    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion = criterion)
        results = test_1(net, criterion, testloader, outloader, epoch=0, **options)
        plot_tsne_by_class(results['test_x'], results['out_x'], known, results['labels_testloader'])
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['acc'], results['AUROC'], 0))
        return results

    # optimizer
    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

    # train and test epoch
    start_time = time.time()
    train_losses = []
    test_losses = []

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        train_loss = train(net, criterion, optimizer, trainloader, epoch, **options)
        train_losses.append(train_loss)

        if epoch >= 20:
            results = test_1(net, criterion, testloader, outloader, epoch = epoch, **options)

        save_networks(net, model_path, file_name, criterion = criterion)

        if options['stepsize'] > 0: scheduler.step()
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results


# main
if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()

    n = 5
    numbers = list(range(10))
    known = random.sample(numbers, n)
    unknown = list(set(list(range(0, 10))) - set(known))

    options.update(
        {
            'known': known,
            'unknown': unknown,
        }
    )

    # dir_path 用于存放实验结果
    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results', dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = options['dataset'] + '.csv'

    # complement
    res = main_worker(options)

    # save results
    res['unknown'] = unknown
    res['known'] = known
    results[str(1)] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name))
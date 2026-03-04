import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import numpy as np
from collections import defaultdict


def train(net, criterion, optimizer, trainloader, epoch=None, **options):

    net.train()

    losses = AverageMeter() # # losses是个类，包括avg，count，sum，val等
    torch.cuda.empty_cache()
    loss_all = 0

    for batch_idx, (data, labels) in enumerate(trainloader):
        # batch_idx 第几个batch
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad() 
            x, y = net(data, True, labels)
            logits, loss = criterion(x, y, labels)
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            # len(trainloader) 返回的是 trainloader 中包含的批次数（即迭代次数）
        loss_all += losses.avg
    # 循环终止

    return loss_all / len(trainloader) 









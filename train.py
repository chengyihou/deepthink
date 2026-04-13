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
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t train_Loss_val: {:.6f} train_loss_avg: {:.6f}" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            # len(trainloader) 返回的是 trainloader 中包含的批次数（即迭代次数）
        loss_all += losses.avg
    # 循环终止

    return loss_all / len(trainloader) 



def train_center_Dist(net, criterion, optimizer, trainloader, epoch=None, **options):

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
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t train_Loss_val: {:.6f} train_loss_avg: {:.6f}" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            # len(trainloader) 返回的是 trainloader 中包含的批次数（即迭代次数）
        loss_all += losses.avg
    # 循环终止

    centers = compute_epoch_centers(
            net=net,
            trainloader=trainloader,
            num_classes=options["num_classes"],
            feat_dim=options["feat_dim"],
            use_gpu=options["use_gpu"],
        )

    current_epoch = epoch + 1 # 保存 centers
    if current_epoch % 1 == 0:
        np.savetxt("centers.csv", centers.detach().cpu().numpy(), delimiter=",", fmt="%.8f")
        print(f"centers saved to centers.csv at epoch {current_epoch}")

    return loss_all / len(trainloader), centers



def compute_epoch_centers(net, trainloader, num_classes, feat_dim, use_gpu):
    # C 已知类别数，D 特征维度  
    net.eval()

    device = next(net.parameters()).device
    class_sums = torch.zeros(num_classes, feat_dim, device=device) # (C, D)
    class_counts = torch.zeros(num_classes, device=device) # (C,) 记录每个类的样本数量

    with torch.no_grad():
        for data, labels in trainloader:
            if use_gpu:
                data = data.cuda()
                labels = labels.cuda()

            feats, _ = net(data, True)   # feats: (B, D)
            # feats = F.normalize(feats, p=2, dim=1)  # 采用 L2 归一化，用归一化后的特征计算类中心
            class_sums.index_add_(0, labels, feats)
            class_counts.index_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=torch.float32)
            )

    class_counts = class_counts.clamp_min(1.0).unsqueeze(1)   # (C, 1)
    centers = class_sums / class_counts
    centers = F.normalize(centers, p=2, dim=1)

    return centers



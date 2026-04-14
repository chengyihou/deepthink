import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import numpy as np
from collections import defaultdict
from scipy.stats import weibull_min


# ============================================================
# 1. 基础的概率训练
# ============================================================


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



def fit_openmax_stats(
    net,
    trainloader,
    num_classes,
    use_gpu,
    save_path="openmax_stats.npz",
    tailsize=20,
    min_fit_samples=5, # min_fit_samples=5：某类至少要有 5 个正确样本，才允许做 Weibull 拟合
):
    """
    训练结束后统计 OpenMax 所需参数：
    1. 每类 MAV（正确分类样本的 logit 均值）
    2. 每类到 MAV 的距离分布尾部
    3. 每类 Weibull 拟合参数
    """
    net.eval()

    logits_by_class = [[] for _ in range(num_classes)] # 创建一个长度为 num_classes 的列表，每个元素又是一个空列表
                                                       # logits_by_class[c] 用来存第 c 类所有“正确分类样本”的 logits
    with torch.no_grad():
        for data, labels in trainloader:
            if use_gpu:
                data = data.cuda()
                labels = labels.cuda()

            feats, logits = net(data, True)
            pred = torch.argmax(logits, dim=1)

            correct_mask = pred.eq(labels)     # bull 掩码
            if correct_mask.sum().item() == 0: # 如果一整个批都没有对的就跳过
                continue

            correct_logits = logits[correct_mask]
            correct_labels = labels[correct_mask]

            for c in range(num_classes):
                cls_mask = correct_labels.eq(c)
                if cls_mask.sum().item() > 0:
                    logits_by_class[c].append(correct_logits[cls_mask].detach().cpu())

    mavs = []
    weibull_shape = np.zeros(num_classes, dtype=np.float32)     # 每类对应一个参数
    weibull_loc = np.zeros(num_classes, dtype=np.float32)
    weibull_scale = np.zeros(num_classes, dtype=np.float32)
    valid_classes = np.zeros(num_classes, dtype=np.int32)       # 有效已知类，防止有的类一个分对的也没有

    for c in range(num_classes):
        if len(logits_by_class[c]) == 0: # 不用看
            mavs.append(torch.zeros(num_classes, dtype=torch.float32))
            print(f"class {c}: no correctly classified samples, skip Weibull fitting")
            continue

        cls_logits = torch.cat(logits_by_class[c], dim=0)   # (Nc, C)
        mav = cls_logits.mean(dim=0)                        # (C,)
        mavs.append(mav)

        dists = torch.norm(cls_logits - mav.unsqueeze(0), p=2, dim=1).numpy() # 欧氏距离

        if len(dists) < min_fit_samples:
            print(f"class {c}: only {len(dists)} correct samples, skip Weibull fitting")
            continue

        tail = np.sort(dists)[-min(tailsize, len(dists)):]
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
            weibull_shape[c] = np.float32(shape)
            weibull_loc[c] = np.float32(loc)
            weibull_scale[c] = np.float32(scale)
            valid_classes[c] = 1
            print(
                f"class {c}: fitted Weibull with {len(tail)} tail samples "
                f"(shape={shape:.6f}, scale={scale:.6f})"
            )
        except Exception as e:
            print(f"class {c}: Weibull fit failed: {e}")

    mavs = torch.stack(mavs, dim=0).numpy().astype(np.float32)   # (C, C)

    np.savez( # 把多个数组一次性保存到一个文件里的函数
        save_path,
        mavs=mavs,
        weibull_shape=weibull_shape,
        weibull_loc=weibull_loc,
        weibull_scale=weibull_scale,
        valid_classes=valid_classes,
        tailsize=np.int32(tailsize),
        min_fit_samples=np.int32(min_fit_samples),
    )

    print(f"OpenMax stats saved to {save_path}")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import weibull_min

from sklearn.metrics import confusion_matrix, roc_auc_score
from utils import AverageMeter
from sklearn.metrics import f1_score



try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None



def load_openmax_stats(stats_path):
    data = np.load(stats_path, allow_pickle=False)
    stats = {
        "mavs": data["mavs"],
        "weibull_shape": data["weibull_shape"],
        "weibull_loc": data["weibull_loc"],
        "weibull_scale": data["weibull_scale"],
        "valid_classes": data["valid_classes"],
    }
    return stats



def openmax_recalibrate(raw_logits, stats, alpha=3):
    """
    raw_logits: (B, C) 的原始分类输出 y，不是 softmax 概率
    返回:
        openmax_logits: (B, C+1)
        openmax_probs:  (B, C+1)
    """
    device = raw_logits.device
    dtype = raw_logits.dtype
    batch_size, num_classes = raw_logits.shape
    topk = min(alpha, num_classes)  # 取的 top_k 个优先级的分类

    mavs = torch.tensor(stats["mavs"], device=device, dtype=dtype)   # (C, C)
    weibull_shape = stats["weibull_shape"]
    weibull_loc = stats["weibull_loc"]
    weibull_scale = stats["weibull_scale"]
    valid_classes = stats["valid_classes"]

    revised_logits = raw_logits.clone()
    unknown_scores = torch.zeros(batch_size, device=device, dtype=dtype) # 一个 batch

    _, topk_indices = torch.topk(raw_logits, k=topk, dim=1) # 取一个张量里最大的前 k 个值，以及它们的下标。

    for i in range(batch_size):
        for rank in range(topk):
            c = int(topk_indices[i, rank].item())

            if valid_classes[c] == 0:
                w_score = 0.0
            else:
                dist = torch.norm(raw_logits[i] - mavs[c], p=2).item() # 第 i 个样本的 logit 到类 c 的欧氏距离
                w_score = weibull_min.cdf(
                    dist,
                    weibull_shape[c],
                    loc=weibull_loc[c],
                    scale=weibull_scale[c],
                )

            rank_weight = float(topk - rank) / float(topk) # 衰减权重，和 topk 的排名有关
            old_score = raw_logits[i, c]
            new_score = old_score * (1.0 - rank_weight * w_score)

            revised_logits[i, c] = new_score
            unknown_scores[i] += (old_score - new_score)

    openmax_logits = torch.cat([revised_logits, unknown_scores.unsqueeze(1)], dim=1) # 在行上堆叠，加到 k+1 维
    openmax_probs = F.softmax(openmax_logits, dim=1)

    return openmax_logits, openmax_probs



def test_openmax(net, criterion, testloader, outloader, stats, alpha=3, epoch=None, **options):
    net.eval()

    correct_test, num_test = 0, 0
    torch.cuda.empty_cache()

    test_loss = AverageMeter()
    loss_all = 0

    labels_testloader = []
    labels_outloader = []
    pred_label_test = []
    pred_label_out = []

    test_x = []
    out_x = []

    num_classes = options["num_classes"]

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()

            x, raw_logits = net(data, True)
            _, loss = criterion(x, raw_logits, labels)

            openmax_logits, openmax_probs = openmax_recalibrate(raw_logits, stats, alpha=alpha)

            pred_idx = torch.argmax(openmax_probs, dim=1)
            pred_open = pred_idx.clone()
            pred_open[pred_open == num_classes] = -1

            num_test += labels.size(0)
            correct_test += (pred_open == labels).sum().item()

            labels_testloader.append(labels.cpu().numpy())
            pred_label_test.append(pred_open.cpu().numpy())
            test_x.append(x.cpu())

            test_loss.update(loss.item(), labels.size(0))
            if (batch_idx + 1) % options["print_freq"] == 0:
                print(
                    "Batch {}/{}\t test_Loss_val: {:.6f} test_loss_avg: {:.6f}".format(
                        batch_idx + 1, len(testloader), test_loss.val, test_loss.avg
                    )
                )
            loss_all += test_loss.avg

        for data, labels in outloader:
            oodlabel = torch.zeros_like(labels) - 1

            if options["use_gpu"]:
                data = data.cuda()

            x, raw_logits = net(data, True)
            openmax_logits, openmax_probs = openmax_recalibrate(raw_logits, stats, alpha=alpha)

            pred_idx = torch.argmax(openmax_probs, dim=1)
            pred_open = pred_idx.clone()
            pred_open[pred_open == num_classes] = -1

            labels_outloader.append(oodlabel.cpu().numpy())
            pred_label_out.append(pred_open.cpu().numpy())
            out_x.append(x.cpu())

    test_x = torch.cat(test_x, dim=0)
    out_x = torch.cat(out_x, dim=0)

    acc = float(correct_test) * 100.0 / float(num_test)

    labels_testloader = np.concatenate(labels_testloader, axis=0)  # 真实测试集标签
    labels_outloader = np.concatenate(labels_outloader, axis=0)    # 全 -1
    pred_label_test = np.concatenate(pred_label_test, axis=0)      # 
    pred_label_out = np.concatenate(pred_label_out, axis=0)

    y_true_f1 = np.concatenate([labels_testloader, labels_outloader], axis=0)
    y_pred_f1 = np.concatenate([pred_label_test, pred_label_out], axis=0)

    all_labels = list(range(num_classes)) + [-1]
    f1_macro = f1_score(y_true_f1, y_pred_f1, average="macro", labels=all_labels)
    matrix = confusion_matrix(y_true_f1, y_pred_f1, labels=all_labels)

    print("Acc: {:.5f}, F1 Macro: {:.6f}".format(acc, f1_macro))
    print("Confusion matrix:")
    print(matrix)

    results = dict()
    results["acc"] = acc
    results["Test Loss"] = loss_all / len(testloader)
    results["f1_macro"] = f1_macro
    results["confusion_matrix"] = matrix
    results["test_x"] = test_x                          # plot
    results["out_x"] = out_x                            # plot
    results["labels_testloader"] = labels_testloader    # plot

    return results
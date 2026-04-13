import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, roc_auc_score
from utils import AverageMeter
from sklearn.metrics import f1_score

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _save_confusion_matrix(matrix, label_names, save_path):
    if plt is None:
        print("matplotlib is not installed, skip saving confusion matrix figure.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def test_1(net, criterion, testloader, outloader, epoch=None, **options):

    net.eval()

    correct_test, total_test = 0, 0
    torch.cuda.empty_cache()
    logits_testloader, logits_outloader = [], []
    labels_testloader, labels_outloader = [], []
    test_loss = AverageMeter()
    loss_all = 0
    thr = 0
    test_x = []
    out_x = []

    with torch.no_grad():
        
        for batch_idx, (data, labels) in enumerate(testloader):
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            predictions = logits.data.max(1)[1]
            total_test += labels.size(0)
            correct_test += (predictions == labels.data).sum().item() # prediction 是张量， 布尔向量 -> 求和成一个张量标量 -> .item()变普通数字
            logits_testloader.append(logits.data.cpu().numpy())
            labels_testloader.append(labels.data.cpu().numpy())
            test_x.append(x.data.cpu())
            test_loss.update(loss.item(), labels.size(0))

            if (batch_idx+1) % options['print_freq'] == 0:
                    print("Batch {}/{}\t test_Loss_val:{:.6f} test_loss_avg:{:.6f}" \
                        .format(batch_idx+1, len(testloader), test_loss.val, test_loss.avg))
                    # len(testloader) 返回的是 testloader 中包含的批次数（即迭代次数）
            loss_all += test_loss.avg

        for data, labels in outloader:
            oodlabel = torch.zeros_like(labels) - 1 # outloader的标签全是-1，表示未知类样本
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, _ = criterion(x, y)
                           
            logits_outloader.append(logits.data.cpu().numpy())
            labels_outloader.append(oodlabel.data.cpu().numpy())
            out_x.append(x.cpu())

            

    test_x = torch.cat(test_x, dim=0)
    out_x = torch.cat(out_x, dim=0)

    acc = float(correct_test) * 100.0 / float(total_test)
    print("Acc: {:.5f}".format(acc))

    logits_testloader = np.concatenate(logits_testloader, 0)
    logits_outloader = np.concatenate(logits_outloader, 0)
    np.savetxt("logits_outloader.csv", logits_outloader, delimiter=",", fmt="%.8f")
    print("logits_outloader has been saved to logits_outloader.csv")
    labels_testloader = np.concatenate(labels_testloader, 0)
    labels_outloader = np.concatenate(labels_outloader, 0)

    logits_test = np.max(logits_testloader, axis=1)
    logits_out = np.max(logits_outloader, axis=1) # ？ ？ ？
    pred_label_test = np.argmax(logits_testloader, axis=1)
    pred_label_out = np.argmax(logits_outloader, axis=1)
    pred_label = np.concatenate([pred_label_test, pred_label_out], 0)
    true_label = np.concatenate([labels_testloader, labels_outloader], 0)
    pred_prob_all = np.concatenate([logits_test, logits_out], 0)

    judge = (pred_prob_all >= thr).astype(np.float32)
    pred_label_judge = pred_label * judge + (-1) * (1 - judge)
    matrix = confusion_matrix(true_label, pred_label_judge)

    precision = np.divide(
        np.diag(matrix),
        np.sum(matrix, axis=0),
        out=np.zeros_like(np.diag(matrix), dtype=float),
        where=(np.sum(matrix, axis=0) != 0),
    )
    recall = np.divide(
        np.diag(matrix),
        np.sum(matrix, axis=1),
        out=np.zeros_like(np.diag(matrix), dtype=float),
        where=(np.sum(matrix, axis=1) != 0),
    )

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall != 0),
    )
    f1 = np.nan_to_num(f1)

    y_true = np.concatenate(
        [np.ones_like(logits_test, dtype=np.int32), np.zeros_like(logits_out, dtype=np.int32)], axis=0
    )
    y_score = np.concatenate([logits_test, logits_out], axis=0).astype(np.float64)
    auroc = roc_auc_score(y_true, y_score)
    print("AUROC: {:.6f}".format(auroc))

    results = dict()
    results["acc"] = acc
    results["AUROC"] = auroc
    results["Test Loss"] = loss_all / len(testloader)
    results["test_x"] = test_x
    results["out_x"] = out_x
    results["labels_testloader"] = labels_testloader
    results["precision_macro"] = float(np.mean(precision))
    results["recall_macro"] = float(np.mean(recall))
    results["f1_macro"] = float(np.mean(f1))
    return results



def test_center_Dist(net, criterion, testloader, outloader, centers, epoch = None, **options):
    net.eval()

    # 参数定义
    correct_test, num_test = 0, 0
    torch.cuda.empty_cache()

    test_loss = AverageMeter()
    loss_all = 0

    labels_testloader = []  # 定义了四个集合
    labels_outloader = []
    pred_label_test = []
    pred_label_out = []

    known_min_dist = []
    unknown_min_dist = []
    test_x = []
    out_x = []

    if options["use_gpu"]:
        centers = centers.cuda()
        centers = F.normalize(centers, p=2, dim=1)
        
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)

            x = F.normalize(x, p=2, dim=1)             # (B, D)
            sim = torch.matmul(x, centers.t())         # (B, C)
            dist = 1.0 - sim                           # (B, C)
            min_dist, pred_label = dist.min(dim=1)     # 测试（预测）的最小距离和对应类中心索引
            num_test += labels.size(0)
            correct_test += (pred_label == labels.data).sum().item()
          
            known_min_dist.append(min_dist.data.cpu().numpy())     # 已知类样本特征与各已知类类中心的最小余弦距离，用于计算AUROC
            pred_label_test.append(pred_label.cpu().numpy())       # 已知类的预测标签，T-sne需要
            labels_testloader.append(labels.data.cpu().numpy())    # 已知类的真实标签，T-sne需要
            test_x.append(x.data.cpu())                            # 已知类的特征，T-sne需要

            test_loss.update(loss.item(), labels.size(0))
            if (batch_idx+1) % options['print_freq'] == 0:
                    print("Batch {}/{}\t test_Loss_val: {:.6f} test_loss_avg: {:.6f}" \
                        .format(batch_idx+1, len(testloader), test_loss.val, test_loss.avg))
                    # len(testloader) 返回的是 testloader 中包含的批次数（即迭代次数）
            loss_all += test_loss.avg

        for data, labels in outloader:
            oodlabel = torch.zeros_like(labels) - 1
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, _ = criterion(x, y)

            x = F.normalize(x, p=2, dim=1)             # (B, D)
            sim = torch.matmul(x, centers.t())         # (B, C)
            dist = 1.0 - sim                           # (B, C)
            min_dist, pred_label  = dist.min(dim=1)    # 测试（预测）的最小距离和对应类中心索引

            unknown_min_dist.append(min_dist.data.cpu().numpy())    # 未知类样本特征与各已知类类中心的最小余弦距离
            pred_label_out.append(pred_label.data.cpu().numpy())    # 未知类的预测标签
            labels_outloader.append(oodlabel.data.cpu().numpy())    # 未知类的真实标签（全是-1）
            out_x.append(x.cpu())       
                                        # 未知类的特征
    test_x = torch.cat(test_x, dim=0)
    out_x = torch.cat(out_x, dim=0)



    # ACC
    acc = float(correct_test) * 100.0 / float(num_test)



    # AUROC
    known_min_dist = np.concatenate(known_min_dist, axis=0)
    unknown_min_dist = np.concatenate(unknown_min_dist, axis=0)

    y_true_auc = np.concatenate([
        np.zeros_like(known_min_dist, dtype=np.int32),
        np.ones_like(unknown_min_dist, dtype=np.int32)
    ], axis=0)
    y_score_auc = np.concatenate([known_min_dist, unknown_min_dist], axis=0).astype(np.float64)
    auroc = roc_auc_score(y_true_auc, y_score_auc)



    # macro f1 score
    thr = 0.999
    labels_testloader = np.concatenate(labels_testloader, axis=0)
    labels_outloader = np.concatenate(labels_outloader, axis=0)
    pred_label_test = np.concatenate(pred_label_test, axis=0)
    pred_label_out = np.concatenate(pred_label_out, axis=0)

    pred_label_test_open = pred_label_test.copy()
    pred_label_out_open = pred_label_out.copy()
    pred_label_test_open[known_min_dist > thr] = -1
    pred_label_out_open[unknown_min_dist > thr] = -1

    y_true_f1 = np.concatenate([labels_testloader, labels_outloader], axis=0)
    y_pred_f1 = np.concatenate([pred_label_test_open, pred_label_out_open], axis=0)

    all_labels = list(range(options["num_classes"])) + [-1]
    f1_macro = f1_score(y_true_f1, y_pred_f1, average='macro', labels=all_labels)
    matrix = confusion_matrix(y_true_f1, y_pred_f1, labels=all_labels)

    print("Acc: {:.5f}, AUROC: {:.6f}, F1 Macro: {:.6f}".format(acc, auroc, f1_macro))
    print("Confusion matrix:")
    print(matrix)

    # 结果输出
    results = dict()
    results["acc"] = acc
    results["AUROC"] = auroc
    results["Test Loss"] = loss_all / len(testloader)
    results["test_x"] = test_x
    results["out_x"] = out_x
    results["labels_testloader"] = labels_testloader
    results["f1_macro"] = f1_macro
    results["confusion_matrix"] = matrix
    return results






def test_center_Dist_thr_from_train(net, criterion, testloader, outloader, centers, thr = 0.0, epoch = None, **options):
    net.eval()

    # 参数定义
    correct_test, num_test = 0, 0
    torch.cuda.empty_cache()

    test_loss = AverageMeter()
    loss_all = 0

    labels_testloader = []  # 定义了四个集合
    labels_outloader = []
    pred_label_test = []
    pred_label_out = []

    known_min_dist = []
    unknown_min_dist = []
    test_x = []
    out_x = []

    if options["use_gpu"]:
        centers = centers.cuda()
        
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)

            x = F.normalize(x, p=2, dim=1)             # (B, D)
            sim = torch.matmul(x, centers.t())         # (B, C)
            dist = 1.0 - sim                           # (B, C)
            min_dist, pred_label = dist.min(dim=1)     # 测试（预测）的最小距离和对应类中心索引
            num_test += labels.size(0)
            correct_test += (pred_label == labels.data).sum().item()
          
            known_min_dist.append(min_dist.data.cpu().numpy())     # 已知类样本特征与各已知类类中心的最小余弦距离，用于计算AUROC
            pred_label_test.append(pred_label.cpu().numpy())       # 已知类的预测标签，T-sne需要
            labels_testloader.append(labels.data.cpu().numpy())    # 已知类的真实标签，T-sne需要
            test_x.append(x.data.cpu())                            # 已知类的特征，T-sne需要

            test_loss.update(loss.item(), labels.size(0))
            if (batch_idx+1) % options['print_freq'] == 0:
                    print("Batch {}/{}\t test_Loss_val: {:.6f} test_loss_avg: {:.6f}" \
                        .format(batch_idx+1, len(testloader), test_loss.val, test_loss.avg))
                    # len(testloader) 返回的是 testloader 中包含的批次数（即迭代次数）
            loss_all += test_loss.avg

        for data, labels in outloader:
            oodlabel = torch.zeros_like(labels) - 1
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, _ = criterion(x, y)

            x = F.normalize(x, p=2, dim=1)             # (B, D)
            sim = torch.matmul(x, centers.t())         # (B, C)
            dist = 1.0 - sim                           # (B, C)
            min_dist, pred_label  = dist.min(dim=1)    # 测试（预测）的最小距离和对应类中心索引

            unknown_min_dist.append(min_dist.data.cpu().numpy())    # 未知类样本特征与各已知类类中心的最小余弦距离
            pred_label_out.append(pred_label.data.cpu().numpy())    # 未知类的预测标签
            labels_outloader.append(oodlabel.data.cpu().numpy())    # 未知类的真实标签（全是-1）
            out_x.append(x.cpu())       
                                        # 未知类的特征
    test_x = torch.cat(test_x, dim=0)
    out_x = torch.cat(out_x, dim=0)



    # ACC
    acc = float(correct_test) * 100.0 / float(num_test)



    # AUROC
    known_min_dist = np.concatenate(known_min_dist, axis=0)
    unknown_min_dist = np.concatenate(unknown_min_dist, axis=0)

    y_true_auc = np.concatenate([
        np.zeros_like(known_min_dist, dtype=np.int32),
        np.ones_like(unknown_min_dist, dtype=np.int32)
    ], axis=0)
    y_score_auc = np.concatenate([known_min_dist, unknown_min_dist], axis=0).astype(np.float64)
    auroc = roc_auc_score(y_true_auc, y_score_auc)



    # macro f1 score
    thr = 0.999
    labels_testloader = np.concatenate(labels_testloader, axis=0)
    labels_outloader = np.concatenate(labels_outloader, axis=0)
    pred_label_test = np.concatenate(pred_label_test, axis=0)
    pred_label_out = np.concatenate(pred_label_out, axis=0)

    pred_label_test_open = pred_label_test.copy()
    pred_label_out_open = pred_label_out.copy()
    pred_label_test_open[known_min_dist > thr] = -1
    pred_label_out_open[unknown_min_dist > thr] = -1

    y_true_f1 = np.concatenate([labels_testloader, labels_outloader], axis=0)
    y_pred_f1 = np.concatenate([pred_label_test_open, pred_label_out_open], axis=0)

    all_labels = list(range(options["num_classes"])) + [-1]
    f1_macro = f1_score(y_true_f1, y_pred_f1, average='macro', labels=all_labels)
    matrix = confusion_matrix(y_true_f1, y_pred_f1, labels=all_labels)

    print("Acc: {:.5f}, AUROC: {:.6f}, F1 Macro: {:.6f}".format(acc, auroc, f1_macro))
    print("Confusion matrix:")
    print(matrix)

    # 结果输出
    results = dict()
    results["acc"] = acc
    results["AUROC"] = auroc
    results["Test Loss"] = loss_all / len(testloader)
    results["test_x"] = test_x
    results["out_x"] = out_x
    results["labels_testloader"] = labels_testloader
    results["f1_macro"] = f1_macro
    results["confusion_matrix"] = matrix
    return results

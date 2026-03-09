import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils import AverageMeter


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
            oodlabel = torch.zeros_like(labels) - 1
            if options["use_gpu"]:
                data, labels = data.cuda(), labels.cuda()
            x, y = net(data, True)
            logits, _ = criterion(x, y, labels)
            logits_outloader.append(logits.data.cpu().numpy())
            labels_outloader.append(oodlabel.data.cpu().numpy())
            out_x.append(x.cpu())

            

    test_x = torch.cat(test_x, dim=0)
    out_x = torch.cat(out_x, dim=0)

    acc = float(correct_test) * 100.0 / float(total_test)
    print("Acc: {:.5f}".format(acc))

    logits_testloader = np.concatenate(logits_testloader, 0)
    logits_outloader = np.concatenate(logits_outloader, 0)
    labels_testloader = np.concatenate(labels_testloader, 0)
    labels_outloader = np.concatenate(labels_outloader, 0)

    logits_test = np.max(logits_testloader, axis=1)
    logits_out = np.max(logits_outloader, axis=1)
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

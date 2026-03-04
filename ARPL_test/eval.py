import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.autograd import Variable
from utils import AverageMeter

def test_1(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()

    correct_test, total_test, n = 0, 0, 0
    torch.cuda.empty_cache()
    logits_testloader, logits_outloader, _labels = [], [], []
    test_loss = AverageMeter()
    loss_all = 0
    thr = 0
    # loss_r_all = 0
    # test_loss_r = AverageMeter()
    # pred_testloader, pred_outloader, labels_testloader, labels_outloader = [], [], [], []
    # open_labels = torch.zeros(50000)
    # probs = torch.zeros(50000)
    test_x=[]
    out_x=[]

    with torch.no_grad():

        # 对测试集和未知类数据集进行迭代，计算模型的输出、损失、置信度，并记录预测结果和标签。
        for data, labels in testloader: 
            bsz = labels.size(0) # batch size

            if options['use_gpu']: # 如果使用GPU，将数据和标签转移到GPU上进行计算。
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False): # 测试没必要追踪梯度
                x, y = net(data, True)
                logits, loss = criterion(x, y, labels)
                # 用于计算ACC
                predictions = logits.data.max(1)[1]
                total_test += labels.size(0) # 累加测试集的总样本数
                correct_test += (predictions == labels.data).sum() # 计算正确预测的数量，并累加到 correct_test 变量中。
                # 保存预测结果和标签                    
                logits_testloader.append(logits.data.cpu().numpy())
                labels_testloader.append(labels.data.cpu().numpy())
                test_x.append(x.data.cpu())  # 将 x 转移到 CPU 并保存

            test_loss.update(loss.item(), labels.size(0))
            loss_all += test_loss.avg

        # 对未知类数据集进行迭代，计算模型的输出、损失、置信度，并记录预测结果和标签。
        for batch_idx, (data, labels) in enumerate(outloader):
            bsz = labels.size(0) # batch size
            oodlabel = torch.zeros_like(labels) - 1  # 未知类的标签全为 -1
            
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y, labels) #  logits 是模型输出的概率分布，_ 是损失值（在测试阶段通常不关心损失）, logits 是二维的
                # 用于计算matrix
                logits_outloader.append(logits.data.cpu().numpy()) 
                labels_outloader.append(oodlabel.data.cpu().numpy())  # 全是 -1
                out_x.append(x.cpu())

    test_x = torch.cat(test_x, dim=0)  # 按第 0 维（batch 维度）拼接
    out_x = torch.cat(out_x,dim=0)


    # Accuracy 闭集正确率
    acc = float(correct_test) * 100. / float(total_test)
    print('Acc: {:.5f}'.format(acc)) # 闭集


    # 整合
    logits_testloader = np.concatenate(logits_testloader, 0) # logits 按行进行堆积，每一行代表每个样本到中心点的距离
    logits_outloader = np.concatenate(logits_outloader, 0) # (batch_size,num_classes)
    labels_testloader = np.concatenate(labels_testloader, 0)
    labels_outloader = np.concatenate(labels_outloader, 0)


    # F1
    logits_test, logits_out = np.max(logits_testloader, axis=1), np.max(logits_outloader, axis=1) # 每行的最大值，也就是每个样本的最大置信度
    pred_label_test, pred_label_out = np.argmax(logits_testloader, axis=1), np.argmax(logits_outloader, axis=1)  # 测试集和未知类数据集的预测标签，取每行最大值的索引作为预测的类别标签
    pred_label = np.concatenate([pred_label_test, pred_label_out], 0) # 预测的标签，前半部分是闭集预测，后半部分是未知类预测
    true_label = np.concatenate([labels_testloader, labels_outloader], 0) # 真实的标签，前半部分是闭集标签，后半部分是未知类标签（全是 -1）
    pred_prob_all = np.concatenate([logits_test, logits_out], 0) # 预测的置信度，前半部分是闭集预测的最大置信度，后半部分是未知类预测的最大置信度

    judge = (pred_prob_all >= thr).astype(np.float32) # 根据阈值判断是否为已知类，pred_prob_all 是每个样本的最大置信度，thr 是预设的阈值，如果置信度大于等于阈值，则认为是已知类（1），否则认为是未知类（0）
    pred_label_judge = pred_label * judge + (-1) * (1 - judge) # 最终的预测标签，如果被判断为已知类，则使用 pred_label 中的预测标签，否则标签为 -1（未知类）

    matrix = confusion_matrix(true_label, pred_label_judge) # 混淆矩阵，标签顺序包括所有已知类和未知类

    precision = np.divide(
            np.diag(matrix),
            np.sum(matrix, axis=0),
            out=np.zeros_like(np.diag(matrix), dtype=float),
            where=(np.sum(matrix, axis=0) != 0)
        )

    recall = np.divide(
            np.diag(matrix),
            np.sum(matrix, axis=1),
            out=np.zeros_like(np.diag(matrix), dtype=float),
            where=(np.sum(matrix, axis=1) != 0)
        )
    
    # nan solve
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    # recall_divide
    racall_unknown = recall[0]
    recall_known = recall[1:]
    recall_known = np.mean(recall_known)

    f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall != 0)
        )

    f1 = np.nan_to_num(f1)
    # F1-macro
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)


    # AUROC
    # 已知(测试集)=1，未知(outloader)=0
    y_true = np.concatenate(
        [np.ones_like(logits_test, dtype = np.int32),np.zeros_like(logits_out, dtype = np.int32)], axis=0)
    y_score = np.concatenate(
        [logits_test, logits_out], axis=0).astype(np.float64)
    
    auroc = roc_auc_score(y_true, y_score)
    print("AUROC: {:.6f}".format(auroc))

    results = dict()
    results['acc'] = acc
    results['AUROC'] = auroc
    results['Test Loss'] = loss_all / len(testloader)
    results['test_x'] = test_x
    results['out_x'] = out_x
    results['labels_testloader'] = labels_testloader

    return results


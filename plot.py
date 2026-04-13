import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler  # 新增导入标准化模块
from sklearn.decomposition import PCA




def plot_tsne_by_class(test_ft, out_ft, known, test_labels, class_names=None):

    # chinese font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows可用
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # convert tensors to numpy arrays if they are on GPU
    if test_ft.is_cuda:
        # test_ft = F.normalize(test_ft, p=2, dim=1)
        test_ft = F.normalize(test_ft, p=2, dim=1)
        test_ft = test_ft.detach().cpu().numpy()
    if out_ft.is_cuda:
        # out_ft = F.normalize(out_ft, p=2, dim=1)
        out_ft = F.normalize(out_ft, p=2, dim=1)
        out_ft = out_ft.detach().cpu().numpy()

    # combine features for t-SNE
    combined_ft = np.concatenate([test_ft, out_ft], axis=0)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=150) 
    # tsne = TSNE(n_components=2, random_state=42, perplexity=75) # 应该画距离直方图
    tsne_result = tsne.fit_transform(combined_ft)

    # cut the t-SNE result back into known and unknown parts
    test_2d = tsne_result[:len(test_ft)] # 已知类别的t-SNE结果
    out_2d = tsne_result[len(test_ft):len(tsne_result)] # 未知类别的t-SNE结果

    # plotting settings
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体为SimHei，Windows系统可用
    plt.rcParams['axes.unicode_minus'] = False 
    colors = plt.cm.get_cmap('tab10', len(known)) # 为每个已知类别分配不同的颜色

    plt.figure(figsize=(10, 8))

    # plot known classes with different colors
    for i in range(len(known)):
        mask = (test_labels == i)
        plt.scatter(
            test_2d[mask, 0], test_2d[mask, 1],
            linestyle='None',  # 关键：去掉横线
            color=colors(i), marker='o', alpha=0.6,
            s=50, edgecolors='w', linewidths=0.5
        )

    # plot unknown classes in dim gray
    plt.scatter(
        out_2d[:, 0], out_2d[:, 1],
        linestyle='None',  # 关键：去掉横线
        color='dimgray', marker='o', alpha=0.6,
        s=50, edgecolors='w', linewidths=0.5,
        label='Unknown'
    )

    # create legend handles for known classes
    legend_handles = []
    for i in range(len(known)):
        legend_handles.append(
            plt.Line2D([], [], color=colors(i), marker='o',
                       markersize=10, alpha=0.6, linestyle='',
                       label=f'{class_names[i]} 样本' if class_names else f'{i+1}'
                       )
        )


    # add legend handle for unknown class
    plt.xlabel('第一维投影', fontsize=22, fontname='SimSun')
    plt.ylabel('第二维投影', fontsize=22, fontname='SimSun')

    # font size settings for ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # add legend for known and unknown classes
    plt.legend(
        handles=legend_handles,
        # title='Classes',
        # bbox_to_anchor=(1.05, 1),
        loc='lower right',
        fontsize=18
    )

    # adjust layout and save figure
    plt.tight_layout()
    save_path = "tsne_plot.png"  # 固定文件名
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()





def plot_features_2d_by_class(test_ft, out_ft, known, test_labels, class_names=None):
    # chinese font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # convert tensors to numpy arrays if they are on GPU
    if hasattr(test_ft, "is_cuda") and test_ft.is_cuda:
        test_ft = test_ft.detach().cpu().numpy()
    elif hasattr(test_ft, "detach"):
        test_ft = test_ft.detach().cpu().numpy()

    if hasattr(out_ft, "is_cuda") and out_ft.is_cuda:
        out_ft = out_ft.detach().cpu().numpy()
    elif hasattr(out_ft, "detach"):
        out_ft = out_ft.detach().cpu().numpy()

    if hasattr(test_labels, "is_cuda") and test_labels.is_cuda:
        test_labels = test_labels.detach().cpu().numpy()
    elif hasattr(test_labels, "detach"):
        test_labels = test_labels.detach().cpu().numpy()

    # ensure features are 2D
    if test_ft.shape[1] != 2 or out_ft.shape[1] != 2:
        raise ValueError(f"Expected 2D features, but got test_ft {test_ft.shape}, out_ft {out_ft.shape}")

    colors = plt.cm.get_cmap('tab10', len(known))

    plt.figure(figsize=(10, 8))

    # plot known classes with different colors
    for i in range(len(known)):
        mask = (test_labels == i)
        plt.scatter(
            test_ft[mask, 0], test_ft[mask, 1],
            linestyle='None',
            color=colors(i), marker='o', alpha=0.6,
            s=50, edgecolors='w', linewidths=0.5
        )

    # plot unknown class
    plt.scatter(
        out_ft[:, 0], out_ft[:, 1],
        linestyle='None',
        color='dimgray', marker='o', alpha=0.6,
        s=50, edgecolors='w', linewidths=0.5,
        label='Unknown'
    )

    # legend handles
    legend_handles = []
    for i in range(len(known)):
        legend_handles.append(
            plt.Line2D(
                [], [], color=colors(i), marker='o',
                markersize=10, alpha=0.6, linestyle='',
                label=f'{class_names[i]} 样本' if class_names else f'{i+1}'
            )
        )

    legend_handles.append(
        plt.Line2D(
            [], [], color='dimgray', marker='o',
            markersize=10, alpha=0.6, linestyle='',
            label='Unknown'
        )
    )

    plt.xlabel('第一维特征', fontsize=22, fontname='SimSun')
    plt.ylabel('第二维特征', fontsize=22, fontname='SimSun')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=18
    )

    plt.tight_layout()
    save_path = "feature_2d_plot.png"
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()

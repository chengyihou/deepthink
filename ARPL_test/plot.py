import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler  # 新增导入标准化模块
from sklearn.decomposition import PCA

def plot_tsne_by_class(test_ft, out_ft, known, test_labels, class_names=None):

    # chinese font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows可用
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # convert tensors to numpy arrays if they are on GPU
    if test_ft.is_cuda:
        test_ft = test_ft.cpu().detach().numpy()
    if out_ft.is_cuda:
        out_ft = out_ft.cpu().detach().numpy()

    # combine features for t-SNE
    combined_ft = np.concatenate([test_ft, out_ft], axis=0)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=75)
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
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=900)
    plt.show()
import argparse
import datetime
import importlib
import os
import random
import time
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np

from pathlib import Path
from torch.optim import lr_scheduler
from eval import test_1, test_center_Dist
from HRRP_OSR import HRRP_OSR
from models import ConvNet
from models_NS_RFF import NS_CLF_Softmax  
from models_NS_RFF import NS_CLF_L2Softmax
from models_NS_RFF import NS_CLF_ResNet_L2Softmax
from plot import plot_tsne_by_class
from plot import plot_features_2d_by_class
from train import train, train_center_Dist, compute_epoch_centers
from utils import load_networks, save_networks

parser = argparse.ArgumentParser("Training")


# Dataset
parser.add_argument("--dataset", type=str, default="HRRP")
parser.add_argument("--dataroot", type=str, default="./data")
parser.add_argument("--outf", type=str, default="./log")
parser.add_argument("--out-num", type=int, default=50)

# Optimization
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for model")

# Scheduler
parser.add_argument("--max-epoch", type=int, default=50)
parser.add_argument("--stepsize", type=int, default=30)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--num-centers", type=int, default=1)

# Model
parser.add_argument("--weight-pl", type=float, default=0.15)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--model", type=str, default="NSRFF", choices=["ConvNet", "NSRFF"])
parser.add_argument("--seq-len", type=int, default=4096)
parser.add_argument("--d1", type=int, default=8) # 通道
parser.add_argument("--d2", type=int, default=24)
parser.add_argument("--z-dim", type=int, default=256)
parser.add_argument("--arc-s", type=float, default=10.0)
parser.add_argument("--arc-m", type=float, default=0.0)

# Misc
parser.add_argument("--nz", type=int, default=50)
parser.add_argument("--ns", type=int, default=1)
parser.add_argument("--eval-freq", type=int, default=1)
parser.add_argument("--print-freq", type=int, default=10)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use-cpu", action="store_true")
parser.add_argument("--save-dir", type=str, default="../log")
parser.add_argument("--loss", type=str, default="Softmax")
# parser.add_argument("--eval", action="store_true", default = False)
parser.add_argument("--eval", action="store_true", default = False)


# Function to build model
def build_model(options):
    if options["model"] == "NSRFF":
        net = NS_CLF_ResNet_L2Softmax(
            # out_channels=options["num_classes"],
            out_channels = 3,
            d1=options["d1"], # ??? 
            d2=options["d2"],
            z_dim=options["z_dim"],
            arc_s=options["arc_s"],
            arc_m=options["arc_m"],
        )
        feat_dim = options["z_dim"] # 特征维度

        # net = NS_CLF_Softmax(
        #     in_channels=3,
        #     out_channels=3,
        #     d1=options["d1"],
        #     d2=options["d2"],
        #     z_dim=options["z_dim"],
        # )
        # feat_dim = options["z_dim"] # 特征维度

    else:
        net = ConvNet(num_classes=options["num_classes"])
        feat_dim = 64
    return net, feat_dim



# Main function
def main_worker(options):
    torch.manual_seed(options["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = options["gpu"]
    use_gpu = torch.cuda.is_available() and not options["use_cpu"]

    if use_gpu:
        print(f"Currently using GPU: {options['gpu']}")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options["seed"])
    else:
        print("Currently using CPU")



    # Dataset
    print(f"{options['dataset']} Preparation")
    data = HRRP_OSR(
        known=options["known"],
        model=options["model"],
        seq_len=options["seq_len"],
        use_gpu=use_gpu,
        batch_size=options["batch_size"],
    )
    trainloader, testloader, outloader = data.train_loader, data.test_loader, data.out_loader
    options["num_classes"] = data.num_classes
    print(f"Creating model: {options['model']}")
    net, feat_dim = build_model(options)



    # Loss function
    options.update({"feat_dim": feat_dim, "use_gpu": use_gpu})
    loss_module = importlib.import_module("loss." + options["loss"])
    criterion = getattr(loss_module, options["loss"])(**options)



    # Move to GPU if available
    if use_gpu:
        # net = nn.DataParallel(net).cuda() # 多卡训练
        net = net.cuda()
        criterion = criterion.cuda()



    # Create directory for saving models
    model_path = os.path.join(options["outf"], "models", options["dataset"]) # 单纯创建一个路径名
    os.makedirs(model_path, exist_ok=True) # 如果目录不存在就创建，已存在就跳过，不报错
    file_name = (
        f"{options['model']}_{options['loss']}_{options['known']}"
        f"_d1{options['d1']}_d2{options['d2']}_z{options['z_dim']}"
        f"_s{options['arc_s']}_m{options['arc_m']}"
    )



    # Evaluation
    if options["eval"]:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        centers = torch.tensor(np.loadtxt("centers.csv", delimiter=","),
        dtype=torch.float32
    )
        results = test_center_Dist(net, criterion, testloader, outloader, centers=centers, epoch=0, **options)
        # plot_features_2d_by_class(
        #     results["test_x"],
        #     results["out_x"],
        #     options["known"],
        #     results["labels_testloader"],
        # ) # 直接的2维特征可视化，
        plot_tsne_by_class(                                                                          
            results["test_x"],
            results["out_x"],
            options["known"],
            results["labels_testloader"],
        )

        print(
            "Acc (%): {:.3f}\t AUROC (%): {:.3f}\t".format(results["acc"], results["AUROC"])
        )
        return results



    # Optimizer and Scheduler
    params_list = [{"params": net.parameters()}, {"params": criterion.parameters()}]

    optimizer = torch.optim.SGD(
    params_list, 
    lr=options["lr"], 
    momentum=0.9, 
    weight_decay=5e-4
    ) # sgd

    # optimizer = torch.optim.Adam( 
    # params_list,
    # lr=options["lr"],
    # betas=(0.9, 0.999),
    # weight_decay=5e-4
    # ) # adam

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 120], gamma=0.5)



    # # Training and Evaluation -------------prob
    # start_time = time.time()
    # results = None
    # for epoch in range(options["max_epoch"]):
    #     print(f"==> Epoch {epoch + 1}/{options['max_epoch']}")
    #     train(net, criterion, optimizer, trainloader, epoch, **options)

    #     if epoch >= 3:
    #         results = test_1(net, criterion, testloader, outloader, epoch=epoch, **options)
    #     save_networks(net, model_path, file_name, criterion=criterion)
        
    #     if options["stepsize"] > 0:
    #         scheduler.step()

    # if results is None:
    #     results = test_1(net, criterion, testloader, outloader, epoch=options["max_epoch"], **options)

    # elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    # print(f"Finished. Total elapsed time (h:m:s): {elapsed}")
    # return results



    # Training and Evaluation -------------distance
    start_time = time.time()
    auroc_history = []
    results = None
    for epoch in range(options["max_epoch"]):
        print(f"==> Epoch {epoch + 1}/{options['max_epoch']}")
        _, centers = train_center_Dist(net, criterion, optimizer, trainloader, epoch, **options)
        auroc = np.nan   # 1
        if epoch >= 3:
            results = test_center_Dist(net, criterion, testloader, outloader, centers = centers, epoch=epoch, **options)
            auroc = results["AUROC"]
        auroc_history.append({
        "epoch": epoch + 1,
        "AUROC": auroc
        })
        save_networks(net, model_path, file_name, criterion=criterion)
        
        if options["stepsize"] > 0:
            scheduler.step()

    if results is None:
        results = test_1(net, criterion, testloader, outloader, epoch=options["max_epoch"], **options)

    np.savetxt("centers.csv", centers.detach().cpu().numpy(), delimiter=",", fmt="%.8f")
    auroc_csv_path = os.path.join(dir_path, f"{options['dataset']}_auroc_per_epoch.csv")
    pd.DataFrame(auroc_history).to_csv(auroc_csv_path, index=False, encoding="utf-8-sig")
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(f"Finished. Total elapsed time (h:m:s): {elapsed}")
    return results



if __name__ == "__main__":
    args = parser.parse_args()
    options = vars(args)

    # n = 3
    # numbers = list(range(4))
    # known = random.sample(numbers, n)
    # unknown = list(set(list(range(0, 4))) - set(known))
    
    known = [0,1,3]
    unknown = [2]

    # unknown = list(set(list(range(0, 4))) - set(known))
    options.update({"known": known,
                    "unknown": unknown
                    }
                )

    dir_name = f"{options['model']}_{options['loss']}"
    dir_path = os.path.join(options["outf"], "results", dir_name)
    os.makedirs(dir_path, exist_ok=True)

    res = main_worker(options)
    res["unknown"] = unknown
    res["known"] = known
    df = pd.DataFrame({"1": res})
    df.to_csv(os.path.join(dir_path, options["dataset"] + ".csv"))

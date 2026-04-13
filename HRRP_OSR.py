import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path



class HRRP_Dataset(Dataset):
    def __init__(self, data, targets, model = "NSRFF", seq_len = 4096, transform = None):
        self.targets = targets
        self.transform = transform
        self.model = model
        self.seq_len = seq_len # length
        tensor = torch.from_numpy(data).float()   # data shape: (N, 8192)

        if model == "NSRFF":
            real = tensor[:, :seq_len]                  # (N, 4096)
            imag = tensor[:, seq_len:]                  # (N, 4096)
            tensor = torch.stack([real, imag], dim=-1)  # (N, 4096, 2)
            tensor = tensor.unsqueeze(1)                # (N, 1, 4096, 2)
        else:
            tensor = tensor.unsqueeze(1)

        self.data = tensor # (N, 1, 4096, 2) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, target



class HRRPFilter(HRRP_Dataset):
    def __filter__(self, known):
        targets = self.targets
        mask, new_targets = [], []

        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i) # mask 不就是 [0,len-1] 的 list 吗 
                new_targets.append(known.index(targets[i])) # 标签重排

        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)



class HRRP_OSR(object):
    def __init__(self, known, model="NSRFF", seq_len=4096, use_gpu=True, num_workers=0, batch_size=32):
  
        data_dir = Path(__file__).resolve().parent.parent
        # data_dir = Path("./data")

        train_y_path = data_dir / "train_y.csv"
        train_x_path = data_dir / "train_x.csv"
        test_y_path = data_dir / "test_y.csv"
        test_x_path = data_dir / "test_x.csv"

        self.train_y = pd.read_csv(train_y_path, header=None).values.reshape(-1)
        self.train_x = pd.read_csv(train_x_path, header=None).values.astype(np.float32)
        self.test_y = pd.read_csv(test_y_path, header=None).values.reshape(-1)
        self.test_x = pd.read_csv(test_x_path, header=None).values.astype(np.float32)

        # self.num_classes = len(known) # 规定的已知类别数量
        # self.known = known
        # self.unknown = list(set(list(range(0, 4))) - set(known))


        self.num_classes = len(known)
        self.known = known
        self.unknown = [x for x in range(4) if x not in known]

        pin_memory = True if use_gpu else False

        kwargs = dict(model = model, seq_len = seq_len)

        trainset = HRRPFilter(self.train_x, self.train_y, model=model, seq_len=seq_len)
        trainset.__filter__(self.known)

        testset = HRRPFilter(self.test_x, self.test_y, model=model, seq_len=seq_len)
        testset.__filter__(self.known)

        outset = HRRPFilter(self.test_x, self.test_y, model=model, seq_len=seq_len)
        outset.__filter__(self.unknown)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,  # 单线程
        )

        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            drop_last = True,         # drop 如果要丢，就需要注意数据预处理的问题
            shuffle = True,           # 测试没有必要 shuffle
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        self.out_loader = DataLoader(
            outset,
            batch_size=batch_size,
            drop_last = True,
            shuffle = True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        print("Train:", len(trainset), "Test:", len(testset), "Out:", len(outset))

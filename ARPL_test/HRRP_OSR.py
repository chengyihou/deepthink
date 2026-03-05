import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



class HRRP_Dataset(Dataset):
    def __init__(self, data, targets, model = "ConvNet", seq_len = 1280, transform = None):
        self.targets = targets
        self.transform = transform
        self.model = model
        self.seq_len = seq_len # length
        tensor = torch.from_numpy(data).float()

        if model == "NSRFF":
            expected_cols = 2 * seq_len # 2560
            if tensor.size(1) != expected_cols:
                raise ValueError(
                    f"NSRFF input expects {expected_cols} columns (2*seq_len), got {tensor.size(1)}."
                )
            tensor = tensor.view(-1, seq_len, 2).unsqueeze(1) # (N, 1, T, 2)
        else:
            tensor = tensor.unsqueeze(1)

        self.data = tensor

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
                mask.append(i)
                new_targets.append(known.index(targets[i]))

        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)



class HRRP_OSR(object):
    def __init__(self, known, model="ConvNet", seq_len=1280, use_gpu=True, num_workers=0, batch_size=32):
        self.train_y = pd.read_csv("train_y.csv", header=None).values.reshape(-1)
        self.train_x = pd.read_csv("train_x.csv", header=None).values.astype(np.float32)
        self.test_y = pd.read_csv("test_y.csv", header=None).values.reshape(-1)
        self.test_x = pd.read_csv("test_x.csv", header=None).values.astype(np.float32)

        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        pin_memory = True if use_gpu else False
        kwargs = dict(model=model, seq_len=seq_len)

        trainset = HRRPFilter(self.train_x, self.train_y, **kwargs)
        trainset.__filter__(known)

        testset = HRRPFilter(self.test_x, self.test_y, **kwargs)
        testset.__filter__(known)

        outset = HRRPFilter(self.test_x, self.test_y, **kwargs)
        outset.__filter__(self.unknown)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        self.out_loader = DataLoader(
            outset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        print("Train:", len(trainset), "Test:", len(testset), "Out:", len(outset))

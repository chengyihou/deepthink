import numpy as np
import torch
import pandas as pd
from torch.onnx.symbolic_opset8 import ones_like
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split



class HRRP_Dataset(Dataset):
    def __init__ (self, data, targets, transform=None): #666
        self.data = data # 引用变量给类变量 
        self.targets = targets
        self.transform = transform

        self.data = torch.from_numpy(data) # 转成tensor
        self.data = self.data.unsqueeze(1) # 在第1维增加一个维度，变成 (N, 1, 256)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index] 

        if self.transform:
            sample = self.transform(sample)

        return sample, target
    


class HRRP_filter(HRRP_Dataset): 
    def __filter__ (self,known):
        
        targets = self.targets
        mask, new_targets = [], [] 

        for i in range(len((targets))):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))

        self.targets = np.array(new_targets) # 确定位置和对应位置上有什么
        mask = torch.tensor(mask).long() # 第一个tensor
        self.data = torch.index_select(self.data, 0, mask)  



class HRRP_OSR(object):
    def __init__(self, known, use_gpu=True, num_workers = 8, batch_size = 32):
        self.df_train_y = pd.read_csv('train_y.csv',header=None)
        self.train_y = self.df_train_y.values # 把 pandas.DataFrame 转成“纯 NumPy 数组”，
        self.train_y = self.train_y.reshape(-1) # 变成一维数组

        self.df_train_x = pd.read_csv('train_x.csv',header=None)
        self.train_x = self.df_train_x.values.astype(np.float32) # 转成浮点数

        self.df_test_y = pd.read_csv('test_y.csv',header=None)
        self.test_y = self.df_test_y.values
        self.test_y = self.test_y.reshape(-1)

        self.df_test_x = pd.read_csv('test_x.csv',header=None)
        self.test_x = self.df_test_x.values.astype(np.float32)
        
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        pin_memory = True if use_gpu else False

        # trainset
        trainset = HRRP_filter(self.train_x, self.train_y)
        trainset.__filter__(known)

        # testset
        testset = HRRP_filter(self.test_x, self.test_y)
        testset.__filter__(known)

        # outset
        outset = HRRP_filter(self.test_x, self.test_y)
        outset.__filter__(self.unknown)

        # 创建DataLoader
        self.train_loader = DataLoader(trainset, batch_size = batch_size, drop_last=True, pin_memory=pin_memory)
        self.test_loader = DataLoader(testset, batch_size = batch_size, drop_last=True, shuffle=True, pin_memory=pin_memory)
        self.out_loader = DataLoader(outset, batch_size = batch_size, drop_last=True, shuffle=True, pin_memory=pin_memory)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
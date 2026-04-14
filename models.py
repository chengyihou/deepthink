import torch
import torch.nn as nn
from torch.ao.nn.quantized import Dropout
from torch.nn import functional as F, LeakyReLU
# from models.ABN import MultiBatchNorm
import torch.nn.init as init



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # 先进行BN和ReLU，再进行卷积（预激活结构）
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual  # 无需再次激活
        return nn.ReLU(inplace=True)(out)



def weights_init(m):
    classname = m.__class__.__name__
    # 仅处理卷积层、批归一化层和全连接层
    if classname in ['Conv1d', 'BatchNorm1d', 'Linear']:
        if hasattr(m, 'weight'):
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.05)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)



class ConvNet(torch.nn.Module): # 64 用0.01    2 用低一点 0.005或更低
    def __init__(self, num_classes):
        # 假设输入32，1，101
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 9, 1, padding=4,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(32, 32, 9, 1, padding=4,bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(0.2)
        self.mx2 = nn.MaxPool1d(2,2)


        self.conv3 = nn.Conv1d(32, 64, 9, 1, padding=4,bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.LeakyReLU(0.2) # (32,64,25)
        self.conv4 = nn.Conv1d(64, 64, 9, 1, padding=4, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.LeakyReLU(0.2)  # (32,64,25)
        self.mx4 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.conv5 = nn.Conv1d(64, 128, 9, 1, padding=4, bias=False)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.LeakyReLU(0.2)  # (32,64,25)
        self.conv6 = nn.Conv1d(128, 128, 9, 1, padding=4, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.LeakyReLU(0.2)  # (32,64,25)
        self.mx6 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.dr = nn.Dropout(0.3)
        self.mx_all = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64) # 512 到 128
        self.fc2 = nn.Linear(64,num_classes)

        self.encoder = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,

            self.conv2,
            self.bn2,
            self.relu2,
            self.mx2,

            self.conv3,
            self.bn3,
            self.relu3,

            self.conv4,
            self.bn4,
            self.relu4,
            self.mx4,

            self.conv5,
            self.bn5,
            self.relu5,

            self.conv6,
            self.bn6,
            self.relu6,
            self.mx6,

            self.mx_all,
        )


        self.apply(weights_init)

    def forward(self, x, rf = False, labels = None):
        x = self.encoder(x)          # (B, 128, 1)
        x = x.view(x.size(0), -1)    # 或 torch.flatten(x, 1)
        x = self.dr(x)
        feat = self.fc1(x)
        y = self.fc2(feat)

        return (feat, y) if rf else y
    



import torch
import torch.nn as nn
from torch.ao.nn.quantized import Dropout
from torch.nn import functional as F, LeakyReLU
# from models.ABN import MultiBatchNorm
import torch.nn.init as init




# 输入数据的归一化
class NormalizedModel(nn.Module):
    def __init__(self) -> None:
        super(NormalizedModel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor: # 正态标准化
        # mean = input.mean(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        # std = input.std(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        mean = input.mean()
        std = input.std()
        normalized_input = (input - mean)/std
        return normalized_input



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
    def __init__(self, num_classes, feat_dim, d = 4, in_channel = 2):
        # 假设输入32，1，101
        super().__init__()

        self.main_module = nn.Sequential(
            NormalizedModel(),
            nn.Conv2d(in_channels=in_channel, out_channels=d,      kernel_size=(3, 3), stride=1, padding=(1, 1)), # 64 64
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=d,           out_channels =d * 2, kernel_size=(3, 3), stride=2, padding=(1, 1)),# 32 32
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=d * 2,       out_channels =d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)), # 16 16
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=d * 4,       out_channels =d * 8, kernel_size=(3, 3), stride=2, padding=(1, 1)), # 8 8
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=d * 8,       out_channels=d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)), # 4 4
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(in_channels=d * 16, out_channels=d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)), # 2 2
            # nn.BatchNorm2d(d * 32),
            # nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16 * d, feat_dim)  # 按照 4096 来算的话，out_dim 就是特征维度
        )
        self.fc_classifier = nn.Linear(feat_dim, num_classes)
        self.apply(weights_init)

    def forward(self, x, rf=False, labels=None):
        
        feat = self.features(x)
        y = self.fc_classifier(feat)

        return (feat, y) if rf else y
    
    def features(self, x):
        n, _, t, _ = x.shape
        if t != 4096:
            raise ValueError(f"NS-RFF expects T=4096, got T={t}.")
        x_img = x.view(n, 1, t, 2).permute(0, 3, 1, 2).flatten().view(n, -1, 64, 64)
        return self.main_module(x_img).view(n, -1)
    
    
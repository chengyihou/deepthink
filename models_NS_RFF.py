import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import freq_compensation, phase_compensation



# ============================================================
# 1. 基础模块：残差块与参数初始化，包括网络和输入数据
# ============================================================



# 定义残差块 ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential( # 这里定义了残差块内连续的 2 个卷积层
            nn.Conv2d(inchannel,  outchannel, kernel_size=3, stride=1,      padding=1, bias=False),       # 先加通道
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False), # 再降维
            nn.BatchNorm2d(outchannel),
         )
        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel: # shortcut，这里为了跟 2 个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x) # 将 2 个卷积层的输出跟处理过的x相加, 实现 ResNet 的基本结构
        out = F.relu(out)
        return out



# 网络参数的初始化
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
    
    # def forward(self, input: torch.Tensor) -> torch.Tensor: # L2 归一化
    #     # input: [B, C, H, W] 或你的特征形状
    #     norm = torch.sqrt(torch.sum(input ** 2, dim=(1, 2, 3), keepdim=True) + 1e-12)
    #     normalized_input = input / norm
    #     return normalized_input

    # def forward(self, input: torch.Tensor) -> torch.Tensor: # L2 归一化
    #     # input: [B, C, H, W] 或你的特征形状
          # norm = torch.norm(input, p=2, dim=(1,2,3), keepdim=True)
          # norm = torch.clamp(norm, min=1e-12)
          # output = input / norm



# ============================================================
# 2. 网络结构
#    ArcMarginProduct 实现 ArcFace 的角度间隔分类
#    网络层的设计
# ============================================================



# ArcMarginProduct 实现了 ArcFace 的核心逻辑
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=10.0, m=0.0, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) # Xavier初始化权重

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, label=None):
        if label is None:
            return F.linear(features, self.weight)

        # 传标签的情况
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output



# NS-RFF 模型采用的CNN网络
class BaseCLF2(nn.Module):
    def __init__(self, in_channels=2, out_dim=1, d=4):
        super().__init__()
        self.main_module = nn.Sequential(
            NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=d,      kernel_size=(3, 3), stride=1, padding=(1, 1)), # 64 64
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
            # nn.Conv2d(in_channels=d * 4, out_channels=out_dim, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(in_channels=d * 32, out_channels=out_dim, kernel_size=(2, 2), stride=1, padding=(0, 0)),
            nn.Linear(16 * d, out_dim)  # 按照 4096 来算的话，out_dim 就是特征维度
        )

    def forward(self, x):
        return self.features(x)

    def features(self, x):
        n, _, t, _ = x.shape
        if t != 4096:
            raise ValueError(f"NS-RFF expects T=4096, got T={t}.")
        x_img = x.view(n, 1, t, 2).permute(0, 3, 1, 2).flatten().view(n, -1, 64, 64)
        return self.main_module(x_img).view(n, -1)



# 标准 resnet 18 
class ResNet18(nn.Module):
    def __init__(self, ResBlock, in_channel, out_dim = 3, d=64): # 多了个 resnet 的输入通道数
        super(ResNet18, self).__init__()
        self.inchannel = d
        # 先定义
        self.layer1 = self.make_layer(ResBlock, d * 2, 2, stride=1)   # 不降维？
        self.layer2 = self.make_layer(ResBlock, d * 4, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, d * 8, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, d * 16, 2, stride=2)        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))             # 添加自适应池化层
        self.fc = nn.Linear(d * 16, out_dim)
        # 再串联
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.adaptive_pool,
            nn.Flatten(),
            self.fc
        )
                     
    # 重复残差块, 把同一种残差块连续堆叠 num_blocks 次, 组成一个 stage
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)      # 列表, 只有第一个 block 降维, 后面 num_blocks-1 个 block 不降维
        layers = []                                      # 一个layer有两个残差快，只有第一个残差快降维，第二个不降维，每个残差快都有残差结构
 
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        return out
    
    def features(self, x):
        n, _, t, _ = x.shape
        if t != 4096:
            raise ValueError(f"NS-RFF expects T=4096, got T={t}.")
        x_img = x.view(n, 1, t, 2).permute(0, 3, 1, 2).flatten().view(n, -1, 64, 64)
        return self.main_module(x_img).view(n, -1)



# ============================================================
# 3. 各网络模块的组合
#    包括频偏估计、频偏补偿、相位估计与相位补偿
#    模块组合
# ============================================================



# 同步模块，实现补偿
class Synchronization(nn.Module):
    def __init__(self, d=4, process_sampling_rate = 122880000): # 122.88MHz
        super().__init__()
        self.freq_estimation = BaseCLF2(2, out_dim=1, d=d)
        self.phase_estimation = BaseCLF2(2, out_dim=1, d=d)
        self.process_sampling_rate = process_sampling_rate

    def forward(self, x):
        n, _, t, _ = x.shape
        freq_offset = self.freq_estimation(x.view(n, 1, t, 2)).view(-1)
        aligned = freq_compensation(x.view(n, t, -1), freq_offset, PROCESS_SAMPLING_RATE=self.process_sampling_rate)
        phase_offset = self.phase_estimation(aligned.view(n, 1, t, 2)).view(-1)
        aligned = phase_compensation(aligned.view(n, t, -1), phase_offset)
        return aligned.view(n, 1, t, 2), freq_offset, phase_offset



# NS-RFF-HP 模型，包含了同步模块、特征提取模块和 ArcMarginProduct 模块
class NS_CLF_L2Softmax(nn.Module):
    def __init__(self, out_channels=4, d1=8, d2=24, z_dim=256, arc_s=10.0, arc_m=0.0): # arc_s
        super().__init__()
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=arc_s, m=arc_m)
        self.z_dim = z_dim

    def forward(self, x, rf=False, labels=None): # 原始的NS-RFF-HP
        feat = self.features(x)
        logits = self.output(feat, labels)
        return (feat, logits) if rf else logits
    
    # def forward(self, x, rf=False, labels=None): # 在原始的NS-RFF-HP基础上加上测试时的L2归一化
    #       feat = self.features(x)
    #       if self.training:
    #           logits = self.output(feat, labels)
    #       else:
    #           feat_norm = F.normalize(feat, p=2, dim=1)
    #           weight_norm = F.normalize(self.output.weight, p=2, dim=0)
    #           logits = F.linear(feat_norm, weight_norm)
    #       return (feat, logits) if rf else logits
    
    # def forward(self, x, rf=False, labels=None): # 去掉HP，训练测试时均用普通归一化
    #     feat = self.features(x)
    #     feat_norm = F.normalize(feat, p=2, dim=1)
    #     weight_norm = F.normalize(self.output.weight, p=2, dim=0)
    #     logits = F.linear(feat_norm, weight_norm)
    #     return (feat, logits) if rf else logits
    

    def features(self, x):
        n, _, t, _ = x.shape
        aligned, _, _ = self.synchronization(x)
        out = self.main_module.features(aligned).view(n, -1)
        return out



# NS-RFF 模型，包含了同步模块、特征提取模块和普通的线性分类器
class NS_CLF_Softmax(nn.Module): # 实际是最简单的 softmax + NS-RFF 模块 ，没有ArcMarginProduct
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=256, d=d2)
        self.output = nn.Linear(256, out_channels)

    def forward(self, x, rf = False, labels=None):
        feat = self.features(x)
        feat = F.normalize(feat, p=2, dim=1)
        logits = self.output(feat)
        return (feat, logits) if rf else logits

    def features(self, x):
        N, _, T, _ = x.shape
        aligned, _, _ = self.synchronization(x)
        out = self.main_module.features(aligned).view(N, -1)
        return out
    


# NS-RFF-HP 模型，包含了同步模块、特征提取模块和 ArcMarginProduct 模块
class NS_CLF_ResNet_L2Softmax(nn.Module):
    def __init__(self, out_channels=4, d1=8, d2=24, z_dim=256, arc_s=10.0, arc_m=0.0): # arc_s
        super().__init__()
        self.synchronization = Synchronization(d=d1)
        self.main_module = ResNet18(ResBlock, in_channel=2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=arc_s, m=arc_m)
        self.z_dim = z_dim

    def forward(self, x, rf=False, labels=None): # 原始的NS-RFF-HP
        feat = self.features(x)
        logits = self.output(feat, labels)
        return (feat, logits) if rf else logits

    def features(self, x):
        n, _, t, _ = x.shape
        aligned, _, _ = self.synchronization(x)
        out = self.main_module.features(aligned).view(n, -1)
        return out


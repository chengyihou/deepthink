import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import freq_compensation, phase_compensation



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
            nn.Linear(16 * d, out_dim)
        )

    def forward(self, x):
        return self.features(x)

    def features(self, x):
        n, _, t, _ = x.shape
        if t != 4096:
            raise ValueError(f"NS-RFF expects T=4096, got T={t}.")
        x_img = x.view(n, 1, t, 2).permute(0, 3, 1, 2).flatten().view(n, -1, 64, 64)
        return self.main_module(x_img).view(n, -1)



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

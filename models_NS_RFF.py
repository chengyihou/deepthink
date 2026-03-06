import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import freq_compensation, phase_compensation



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=10.0, m=0.0, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

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
            nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d, out_channels=d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d * 2, out_channels=d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d * 4, out_channels=d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d * 8, out_channels=d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d * 16, out_channels=d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(d * 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=d * 32, out_channels=out_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
        )

    def forward(self, x):
        return self.features(x)

    def features(self, x):
        n, _, t, _ = x.shape
        if t != 1280:
            raise ValueError(f"NS-RFF expects T=1280, got T={t}.")
        x_img = x.view(n, 1, t, 2).permute(0, 3, 1, 2).flatten().view(n, -1, 16, 80)
        return self.main_module(x_img).view(n, -1)



class Synchronization(nn.Module):
    def __init__(self, d=4, process_sampling_rate=16000):
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
    def __init__(self, out_channels=10, d1=8, d2=24, z_dim=512, arc_s=10.0, arc_m=0.0):
        super().__init__()
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=arc_s, m=arc_m)
        self.z_dim = z_dim

    def forward(self, x, rf=False, labels=None):
        feat = self.features(x)
        logits = self.output(feat, labels)
        return (feat, logits) if rf else logits

    def features(self, x):
        n, _, t, _ = x.shape
        aligned, _, _ = self.synchronization(x)
        return self.main_module.features(aligned).view(n, -1)

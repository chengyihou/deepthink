import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Dist(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes  # 类别数 3
        self.centers = (
                nn.Parameter(0.1 * torch.randn(num_classes, self.feat_dim))) # 初始化类中心
        
    def forward(self, features, center=None, metric='cosine'):
        if metric == 'cosine':
            features = F.normalize(features, p=2, dim=1)   # (B, D)
            center = F.normalize(center, p=2, dim=1)       # (C, D)
            sim = torch.matmul(features, center.t())       # (B, C)
            dist = 1.0 - sim                               # (B, C)
            return dist

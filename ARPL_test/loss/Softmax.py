import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, **options): 
        super(Softmax, self).__init__()
        self.temp = options['temp'] # temp是缓和参数

    def forward(self, x, y, labels=None):
        logits = F.softmax(y, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(y / self.temp, labels)
        return logits, loss
    
    # super(Softmax, self).__init__() 是为了正确初始化 nn.Module，让 PyTorch 能“看见”你的子模块、参数，并建立计算图。
    # module是所有loss的基class
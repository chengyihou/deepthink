import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist



class center_cosine_dist_Loss(nn.CrossEntropyLoss):

    def __init__(self, **options):
        super(center_cosine_dist_Loss, self).__init__()
        self.dist = Dist(options['dist'])
        self.temp = options['temp'] # temp是缓和参数
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        self.points = self.Dist.centers
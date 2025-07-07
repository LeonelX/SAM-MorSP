import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import register_loss
from builder import build_model
    
@register_loss("SoftCLDice")    
class SoftCLDice(nn.Module):
    def __init__(self,
                 half_size=2,
                 iter=50,
                 sigmoid=False,
                 epsilon=1e-6,
                 loss_weight=1.0,
                 **kwargs):
        """[function to compute dice loss]
        
        parms:
            device: 'cpu' or 'cuda'
            half_size: the half size of pool kernel
            iter: number of iterations of skeletonize
            epsilon: use for smoothing cl dice

        Args:
            y_true ([float32]): [ground truth image] (1, H, W)
            y_pred ([float32]): [predicted image] (1, H, W)

        Returns:
            [float32]: [loss value]
        """
        super(SoftCLDice, self).__init__()
        self.sigmoid = sigmoid
        self.epsilon = epsilon
        self.soft_skeletonize = build_model(dict(
            type='Skeleton', half_size=half_size, iter=iter))
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        soft_sk = self.soft_skeletonize(torch.cat((y_pred, y_true), dim=0)) # 提取骨架
        skel_pred, skel_true = soft_sk[:y_pred.shape[0]], soft_sk[-y_true.shape[0]:]
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.epsilon)/(torch.sum(skel_pred)+self.epsilon)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.epsilon)/(torch.sum(skel_true)+self.epsilon)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return self.loss_weight * cl_dice
    

@register_loss("SkeletonEnergy")
class SkeletonEnergy(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=50,
                 epsilon=1e-6,
                 loss_weight=1.0,
                 alpha=0.05):
        """[function to compute dice loss]
        
        parms:
            device: 'cpu' or 'cuda'
            half_size: the half size of pool kernel
            iter: number of iterations of skeletonize
            epsilon: use for smoothing cl dice

        Args:
            y_true ([float32]): [ground truth image] (1, H, W)
            y_pred ([float32]): [predicted image] (1, H, W)

        Returns:
            [float32]: [loss value]
        """
        super(SkeletonEnergy, self).__init__()
        self.epsilon = epsilon
        self.soft_skeletonize = build_model(dict(
            type='SmoothSkeleton', half_size=half_size, iter=iter, alpha=alpha))
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.float(), y_true.float()
        skel_pred, skel_true = self.soft_skeletonize(y_pred), self.soft_skeletonize(y_true)
        return self.loss_weight * torch.norm(skel_pred-skel_true)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import register_model

# 绘制骨架
@register_model("Skeleton")
class Skeleton(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=20):
        super(Skeleton, self).__init__()
        self.trans = nn.MaxPool2d(kernel_size=2*half_size+1,
                                  stride=1,
                                  padding=half_size) # 最大池化
        self.iter = iter

    def erode(self, x): # 最小池化腐蚀
        return -self.trans(-x)

    def dilate(self, x): # 最大池化膨胀
        return self.trans(x)

    def open(self, x): # 开运算
        return self.dilate(self.erode(x))

    def close(self, x): # 闭运算
        return self.erode(self.dilate(x))

    def forward(self, x):
        skel = x - self.open(x)
        for _ in range(self.iter):
            x = self.erode(x)
            delta = x - self.open(x)
            skel = skel + delta
            
        return skel


# 绘制光滑骨架
@register_model("SmoothSkeleton")
class SmoothSkeleton(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=50,
                 alpha=0.05):
        super(SmoothSkeleton, self).__init__()
        
        self.ker = torch.ones((2*half_size+1, 2*half_size+1)).reshape((1, 1, 2*half_size+1, 2*half_size+1))
        self.h = half_size
        self.iter = iter
        self.alpha = alpha
        self.log_size = math.log((2*half_size+1) ** 2)
    
    def erode(self, x): # 光滑腐蚀
        return -self.dilate(-x)

    def dilate(self, x): # 光滑膨胀
        x = F.conv2d(F.pad(torch.exp(x/self.alpha),
                             pad=(self.h, self.h, self.h, self.h),
                             mode='replicate'), self.ker)
        return self.alpha * (torch.log(x) - self.log_size)

    def open(self, x): # 开运算
        return self.dilate(self.erode(x))

    def close(self, x): # 闭运算
        return self.erode(self.dilate(x))

    def forward(self, x):
        self.ker = self.ker.to(x.device)
        skel = torch.clamp(x - self.open(x), min=0, max=1)
        for _ in range(self.iter):
            x = self.erode(x)
            skel = skel + torch.clamp(x - self.open(x), min=0, max=1)
            
        return skel


# 对偶核函数版本光滑骨架
class SmoothSkeletonDual(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=50,
                 alpha=0.05):
        super(SmoothSkeletonDual, self).__init__()
        self.h = half_size
        self.iter = iter
        self.alpha = alpha
        self.ks = 2*half_size+1
        self.log_N = math.log(self.ks ** 2)
    

    def erode(self, x): # 光滑腐蚀
        return -self.dilate(-x)

    def dilate(self, x): # 光滑膨胀
        B, C, H, W = x.shape
        # 1) 划分窗口
        x_pad = F.pad(x, pad=(self.h, self.h, self.h, self.h), mode="reflect")
        unfold = nn.Unfold(kernel_size=self.ks, padding=0)
        patches = unfold(x_pad)  # (B, C*kernel_size*kernel_size, H*W)

        # 2) 滑动窗口计算softmax权重和主项
        softmax_weights = F.softmax(patches / self.alpha, dim=1)
        weighted_sum = (patches * softmax_weights).sum(dim=1).view(B, C, H, W)

        # 3) 计算 softmax 权重的熵
        entropy = - (softmax_weights * torch.log(softmax_weights + 1e-8)).sum(dim=1).view(B, C, H, W)
        
        return weighted_sum + self.alpha * (entropy - self.log_N)

    def open(self, x): # 开运算
        return self.dilate(self.erode(x))

    def close(self, x): # 闭运算
        return self.erode(self.dilate(x))

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        
        skel = torch.clamp(x - self.open(x), min=0, max=1)
        for j in range(self.iter):
            x = self.erode(x)
            skel = skel + torch.clamp(x - self.open(x), min=0, max=1)
            
        return skel


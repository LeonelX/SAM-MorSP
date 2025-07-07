import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import register_model
from builder import build_model

@register_model("MorSP")
class MorSP(nn.Module):
    def __init__(
            self,
            iterations=20,
            entropy_epsilon=1.0,
            lam= 1.0,
            eta = 2,
            ito = 1e-3,
            delta = 1e-1,
            ker_halfsize=2,
            skeleton=dict(type='SmoothSkeleton', half_size=2, iter=50),
            **kwargs
    ):
        """
        :param iterations: iterations number
        :param entropy_spsilon: entropy of softmax for smooth
        :param lam: weight of p in softmax
        :param eta: weight of penalty
        :param ito: learning rate of w
        :param delta: weight of partial energy
        :param ker_halfsize: the half size of STD kernel
        """
        super(MorSP, self).__init__()
        self.iterations = iterations
        # Fixed paramaters of Gaussian function
        self.sigma = torch.full((1, 1, 1), 5.0, dtype=torch.float, requires_grad=False)
        self.ker_halfsize = ker_halfsize
        # Fixed paramater of ADMM
        self.entropy_epsilon = entropy_epsilon
        self.lam = lam
        self.eta = eta
        self.ito = ito
        self.delta = delta
        # skeleton function
        self.get_skelton = build_model(skeleton)

    def forward(self, o, v):
        # compute STD paramaters
        p, q = self.Skel_STD(o.detach(), v.detach())
        u = (o - self.lam*p + self.eta*q) # without sigmoid
        return u

    def Skel_STD(self, o, v):
        u = torch.sigmoid(o / self.entropy_epsilon)

        # std kernel
        ker = self.STD_Kernel(self.sigma, self.ker_halfsize)
        ker = ker.to(o.device)
        
        # Initailize
        u = torch.sigmoid(o/self.entropy_epsilon).to(o.device) # (B, 1, 1024, 1024)
        q = v - u
        q = torch.clamp(q, min=-1, max=1)
        w = (u.clone()+v.clone())/2
        w.requires_grad_(True)
        
        # Iterations
        for i in range(self.iterations):
            with torch.no_grad():
                # 1. softmax
                p = F.conv2d(1.0 - 2.0 * u, ker, padding=self.ker_halfsize)
                u = torch.sigmoid((o - self.lam*p + self.eta*q)/self.entropy_epsilon)
                # 2. update dual var q
                q = q + (w-u) 
                q = torch.clamp(q, min=-1, max=1)
            
            # 3. update w with pytorch auto grad
            if i%5 == 0:
                if w.grad is not None:
                    w.grad.zero_()
                loss = self.energy(w, v)
                loss.backward()
                w.data = w.data - self.ito * (self.eta*q + self.delta*w.grad.data)
                w.grad.zero_()
        
        del w, u 
        return p, q

    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * sigma ** 2))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker
    
    def energy(self, w, v):
        return torch.norm(self.get_skelton(w)-self.get_skelton(v))
        # return torch.sum(abs(self.get_skelton(w)-self.get_skelton(v)))
    

# if __name__ == "__main__":
#     o = torch.rand((4, 1, 32, 32)).to('cuda')
#     v = torch.rand((4, 1, 32, 32)).to('cuda')
#     skel_module = MorSP()
#     mask = skel_module(o,v)
#     print(mask.shape)
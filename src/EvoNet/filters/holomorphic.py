import torch
import torch.nn as nn
from torch.autograd import grad


class Firewall(nn.Module):
    def __init__(self, model, iterate=100, lr=0.01):
        super().__init__()
        self.model = model
        self.iterate = iterate
        self.lr = lr

    def forward(self, x):
        def objective(_dx):
            return torch.max(torch.abs(self.model.conjugate(x + _dx)))
        
        dx = torch.zeros_like(x, requires_grad=True)
        ddx = torch.ones_like(dx)
        counter = 0
        while torch.max(torch.abs(ddx)) < 1e-3 and counter < self.iterate:
            counter += 1
            obj = objective(dx)
            ddx = grad(obj, dx)[0]
            dx = dx - self.lr * ddx
        
        return x + dx


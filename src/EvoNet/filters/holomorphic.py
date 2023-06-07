import torch
import torch.nn as nn
from torch.autograd import grad


class Firewall(nn.Module):
    def __init__(self, model, iterate=10, lr=0.01):
        super().__init__()
        self.model = model
        self.iterate = iterate
        self.lr = lr

    def forward(self, x):
        def objective(_dx):
            return torch.linalg.norm(self.model.conjugate(x + _dx)) + torch.linalg.norm(_dx)
        
        dx = torch.zeros_like(x, requires_grad=True)
        for _ in range(self.iterate):
            obj = objective(dx)
            ddx = grad(obj, dx)[0]
            dx = dx - self.lr * ddx
        
        return x + dx


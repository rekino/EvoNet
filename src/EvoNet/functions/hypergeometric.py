from typing import Any, Tuple
import torch
from torch.autograd import Function
import mpmath
import numpy as np
from scipy.special import hyp2f1


class Hyp0F1(Function):
    @staticmethod
    def forward(b, z):
        b = torch.tensor(b)
        result = float(mpmath.hyp0f1(b.item(), z.detach().item()))
        return torch.tensor(result, requires_grad=z.requires_grad)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        b, z = inputs
        ctx.save_for_backward(torch.tensor(b), z)

    @staticmethod
    def backward(ctx, grad_output):
        b, z = ctx.saved_tensors
        grad_input = Hyp0F1.apply(b+1, z) / b
        return None, grad_input * grad_output

    @staticmethod
    def vmap(info, in_dims, b, z):
        B, Z = torch.meshgrid(torch.tensor(b), z)
        out = torch.zeros_like(B)
        for i, j in np.ndindex(B.shape):
            out[i, j] = Hyp0F1.apply(B[i, j], Z[i, j])
        out = out.squeeze()
        return out, 0


class Hyp2F1(Function):
    @staticmethod
    def forward(a, b, c, x):
        x = x.detach()
        result = torch.asarray(hyp2f1(a, b, c, x), dtype=torch.float)
        return torch.tensor(result, requires_grad=x.requires_grad)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        a, b, c, x = inputs
        a = torch.tensor(a)
        b = torch.tensor(b)
        c = torch.tensor(c)
        ctx.save_for_backward(a, b, c, x)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, c, x = ctx.saved_tensors
        grad_input = a * b * Hyp2F1.apply(a+1, b+1, c+1, x) / c
        return None, None, None, grad_input * grad_output

    @staticmethod
    def vmap(info, in_dims, a, b, c, x):
        out = torch.zeros(*a.shape, *x.shape)
        for idx in np.ndindex(a.shape):
            out[idx] = Hyp2F1.apply(a[idx], b[idx], c[idx], x)
        out = out.squeeze()
        return out, 0

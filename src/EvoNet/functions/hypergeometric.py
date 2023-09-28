from typing import Any, Tuple
import torch
from torch.autograd import Function
import mpmath
import numpy as np


class Hyp0F1(Function):
    @staticmethod
    def forward(b, z):
        result = float(mpmath.hyp0f1(b.item(), z.detach().item()))
        return torch.tensor(result, requires_grad=True)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        b, z = inputs
        ctx.save_for_backward(b, z)

    @staticmethod
    def backward(ctx, grad_output):
        b, z = ctx.saved_tensors
        grad_input = Hyp0F1.apply(b+1, z) / b
        return None, grad_input * grad_output

    @staticmethod
    def vmap(info, in_dims, b, z):
        B, Z = torch.meshgrid(b, z)
        out = torch.zeros_like(B)
        for i, j in np.ndindex(B.shape):
            out[i, j] = Hyp0F1.apply(B[i, j], Z[i, j])
        return out, 0

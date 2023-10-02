from typing import Any, Tuple
import torch
from torch.autograd import Function
import mpmath
import numpy as np
from scipy.special import hyp2f1

hyp0f1 = np.vectorize(mpmath.hyp0f1)


class Hyp0F1(Function):
    @staticmethod
    def forward(b, z):
        z = z.detach() if z.requires_grad else z
        result = hyp0f1(b, z).astype('float32')
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


class Hyp2F1(Function):
    @staticmethod
    def forward(a, b, c, x):
        x = x.detach() if x.requires_grad else x
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

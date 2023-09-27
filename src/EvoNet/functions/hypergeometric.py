import torch
from torch.autograd import Function
import mpmath

class Hyp0F1(Function):
    @staticmethod
    def forward(ctx, b, z):
        # Forward pass using mpmath.hyp0f1
        result = float(mpmath.hyp0f1(b.item(), z.detach().item()))
        # Save input for backward pass
        ctx.save_for_backward(b, z)
        return torch.tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass using mpmath.hyp0f1 derivative
        b, z = ctx.saved_tensors
        grad_input = Hyp0F1.apply(b+1, z) / b
        return None, grad_input * grad_output

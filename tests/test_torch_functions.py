import unittest
import torch
from torch.autograd import grad

from src.EvoNet.functions import Hyp0F1


class TestTorchFunctions(unittest.TestCase):

    def test_hyp0f1_val(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = Hyp0F1.apply(b, z)
        true_res = torch.cosh(2 * torch.sqrt(z))

        assert torch.allclose(res, true_res)

    def test_hyp0f1_vmap(self):
        hyp0f1 = torch.vmap(Hyp0F1.apply, in_dims=(None, 0))

        z = torch.linspace(0, 1, 2, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = hyp0f1(b, z)
        true_res = torch.cosh(2 * torch.sqrt(z))

        assert torch.allclose(res, true_res)

        hyp0f1 = torch.vmap(Hyp0F1.apply, in_dims=(0, 0))

        z = torch.linspace(0, 1, 2, requires_grad=True)
        b = torch.linspace(0.5, 1.5, 2, requires_grad=False)
        res = hyp0f1(b, z)
        true_res_0 = torch.cosh(2 * torch.sqrt(z))
        temp = torch.sinh(2 * torch.sqrt(z)) / (2*torch.sqrt(z))
        true_res_1 = torch.where(z > 0, temp, 1)

        assert torch.allclose(res[0, :], true_res_0)
        assert torch.allclose(res[1, :], true_res_1)

    def test_hyp0f1_der(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = Hyp0F1.apply(b, z)

        dz = grad(res, z)[0]
        true_dz = torch.sinh(2 * torch.sqrt(z)) / torch.sqrt(z)

        assert torch.allclose(dz, true_dz)

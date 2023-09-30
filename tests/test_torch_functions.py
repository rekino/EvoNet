import unittest
import torch
from torch.autograd import grad

from src.EvoNet.functions import Hyp0F1, Hyp2F1


class TestHyp0F1Function(unittest.TestCase):

    def test_value(self):
        z = torch.tensor(1.0, requires_grad=False)
        b = 1/2
        res = Hyp0F1.apply(b, z)
        true_res = torch.cosh(2 * torch.sqrt(z))

        assert torch.allclose(res, true_res)

        b = torch.tensor(1/2, requires_grad=False)
        res = Hyp0F1.apply(b, z)

        assert torch.allclose(res, true_res)

    def test_vmap(self):
        hyp0f1 = torch.vmap(Hyp0F1.apply, in_dims=(None, 0))

        z = torch.linspace(0, 1, 2, requires_grad=False)
        b = 1/2
        res = hyp0f1(b, z)
        true_res = torch.cosh(2 * torch.sqrt(z))

        assert torch.allclose(res, true_res)

        hyp0f1 = torch.vmap(Hyp0F1.apply, in_dims=(0, 0))

        z = torch.linspace(0, 1, 2, requires_grad=False)
        b = torch.linspace(0.5, 1.5, 2, requires_grad=False)
        res = hyp0f1(b, z)
        true_res_0 = torch.cosh(2 * torch.sqrt(z))
        temp = torch.sinh(2 * torch.sqrt(z)) / (2*torch.sqrt(z))
        true_res_1 = torch.where(z > 0, temp, 1)

        assert torch.allclose(res[0, :], true_res_0)
        assert torch.allclose(res[1, :], true_res_1)

    def test_derivative(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = 1/2
        res = Hyp0F1.apply(b, z)

        dz = grad(res, z)[0]
        true_dz = torch.sinh(2 * torch.sqrt(z)) / torch.sqrt(z)

        assert torch.allclose(dz, true_dz)


class TestHyp2F1Function(unittest.TestCase):

    def test_value(self):
        x = torch.tensor(0.0, requires_grad=False)
        a, b, c = -1/2, 3/2, 3/2
        res = Hyp2F1.apply(a, b, c, x)
        true_res = torch.sqrt(1 - x)

        assert torch.allclose(res, true_res)

    def test_derivative(self):
        x = torch.tensor(0.0, requires_grad=True)
        a, b, c = -1/2, 3/2, 3/2
        res = Hyp2F1.apply(a, b, c, x)

        dx = grad(res, x)[0]
        true_dx = a * b / torch.sqrt(1 - x) / c

        assert torch.allclose(dx, true_dx)

    def test_vmap(self):
        a, b, c = -1/2, 3/2, 3/2
        x = torch.linspace(0, 0.9, 2)
        res = Hyp2F1.apply(a, b, c, x)
        true_res = torch.sqrt(1 - x)

        assert torch.allclose(res, true_res)

        a = a * torch.ones(2)
        b = b * torch.ones(2)
        c = torch.linspace(0.5, 1.5, 2)
        hyp2f1 = torch.vmap(Hyp2F1.apply, (None, None, 0, None))
        res = hyp2f1(a, b, c, x)
        true_res_0 = (1 - 2*x) / torch.sqrt(1 - x)
        true_res_1 = torch.sqrt(1 - x)

        assert torch.allclose(res[0, :], true_res_0)
        assert torch.allclose(res[1, :], true_res_1)

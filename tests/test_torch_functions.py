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
        hyp0f1 = Hyp0F1.apply

        z = torch.linspace(0, 1, 2, requires_grad=False)
        b = 1/2
        res = hyp0f1(b, z)
        true_res = torch.cosh(2 * torch.sqrt(z))

        assert torch.allclose(res, true_res)

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

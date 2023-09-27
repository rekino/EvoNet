import unittest
import torch
from torch.autograd import grad

from src.EvoNet.functions import Hyp0F1


class TestFunctions(unittest.TestCase):

    def test_hyp0f1_val(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = Hyp0F1.apply(b, z).detach().item()
        true_res = torch.cosh(2 * torch.sqrt(z)).detach().item()

        self.assertAlmostEqual(res, true_res)

    def test_hyp0f1_der(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = Hyp0F1.apply(b, z)

        dz = grad(res, z)[0].detach().item()
        true_dz = torch.sinh(2 * torch.sqrt(z)) / torch.sqrt(z)

        self.assertAlmostEqual(dz, true_dz.detach().item())

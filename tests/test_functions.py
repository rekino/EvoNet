import unittest
import torch
import numpy as np


from src.EvoNet.functions import Hyp0F1

class TestFunctions(unittest.TestCase):

    def test_hyp0f1(self):
        z = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.5, requires_grad=False)
        res = Hyp0F1.apply(b, z)

        self.assertAlmostEqual(res, np.cosh(2 * np.sqrt(z.detach())))

        pass
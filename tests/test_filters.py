import unittest
import torch
import numpy as np

from src.EvoNet.filters.holomorphic import Firewall
from src.EvoNet.layers.convolution import HolomorphicConv2d


class TestFilters(unittest.TestCase):
    def test_firewall(self):
        model = HolomorphicConv2d((3, 3), 2, torch.ones(1, 1, 1, 1), 1)
        firewall = Firewall(model, iterate=20, lr=0.001)

        x = torch.rand(4, 1, 3, 3)
        clean_x = firewall(x)

        out = model.conjugate(x).detach().numpy()
        clean_out = model.conjugate(clean_x).detach().numpy()

        clean_x = clean_x.detach().numpy()

        self.assertTrue(np.all(clean_x.shape == x.detach().numpy().shape))
        self.assertFalse(np.any(np.isinf(clean_x)))
        self.assertFalse(np.any(np.isnan(clean_x)))

        self.assertTrue(np.linalg.norm(clean_out) <= np.linalg.norm(out))

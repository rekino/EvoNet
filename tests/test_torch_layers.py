import unittest
import torch

from src.EvoNet.layers import SphericalLinear


class TestSphericalLinearLayer(unittest.TestCase):

    def test_constructor(self):
        layer = SphericalLinear(2, 3)
        self.assertIsInstance(layer, SphericalLinear)

    def test_forward(self):
        layer = SphericalLinear(2, 3)
        x = torch.eye(2)
        out = layer(x)

        self.assertEquals(out.shape, (2, 4))
        self.assertGreater(1, out[:, 1:].abs().max())

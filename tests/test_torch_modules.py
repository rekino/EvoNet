import unittest
import torch

from src.EvoNet.layers import SphericalLinear, SphericalHarmonic
from src.EvoNet.layers import BoundaryCondition


class TestSphericalLinearLayer(unittest.TestCase):

    def test_constructor(self):
        layer = SphericalLinear(2, 3)
        self.assertIsInstance(layer, SphericalLinear)

    def test_forward(self):
        layer = SphericalLinear(2, 3)
        x = torch.eye(2)
        out = layer(x)

        self.assertEquals(out.shape, (2, 4))
        self.assertGreaterEqual(1, out[:, 1:].abs().max())


class TestSphericalHarmonicActivation(unittest.TestCase):

    def test_constructor(self):
        module = SphericalHarmonic(2, 3, 4)

        self.assertIsInstance(module, SphericalHarmonic)

    def test_forward(self):
        module = SphericalHarmonic(2, 3, 4)
        x = torch.ones(2, 3)
        x[1, :] = torch.zeros(1, 3)
        out = module(x)

        self.assertEquals(out.shape, (12, 2, 2))
        assert torch.allclose(out[1, 1, :], 0*out[1, 1, :])
        assert torch.allclose(out[0, -1, :], out[0, -1, :].max())

        module = SphericalHarmonic(150, 3, 4, bc=BoundaryCondition.Neumann)
        out = module(x)

        self.assertEquals(out.shape, (12, 2, 2))
        assert torch.allclose(out[1, 1, :], 0*out[1, 1, :])

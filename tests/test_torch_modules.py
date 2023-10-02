import unittest
import torch
import numpy as np

import matplotlib.pyplot as plt

from src.EvoNet.layers import SphericalLinear, SphericalHarmonic


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
        module = SphericalHarmonic(64, 3, 4, bc='neumann')
        x = torch.ones(2, 3)
        x[1, :] = torch.zeros(1, 3)
        out = module(x)

        self.assertEquals(out.shape, (12, 2, 2))
        assert torch.allclose(out[1, 1, :], 0*out[1, 1, :])

        r = torch.linspace(0, 1, 20)
        t = torch.linspace(-1, 1, 20)

        R = r[:, None] * t[None, :]**0
        T = t[None, :] * r[:, None]**0

        x = torch.hstack([r[:, None], T])
        out = module(x)

        fig, ax = plt.subplots(3, 4)

        for k, l in np.ndindex((3, 4)):
            ax[k, l].contourf(R, T, out[k*4+l])

        plt.show()

import unittest
import torch
import numpy as np

from src.EvoNet.layers.convolution import ComplexLinear, HarmonicConv2d

class TestLayers(unittest.TestCase):

    def test_complex_linear(self):
        layer = ComplexLinear(3, 2)
        out = layer(torch.randn(4, 3) + 1j).detach().numpy()

        self.assertEqual(len(out.shape), 2)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 2)

        self.assertFalse(np.any(np.isinf(out)))
        self.assertFalse(np.any(np.isnan(out)))

    def test_harmonic_conv2d(self):
        layer = HarmonicConv2d((3, 3), 2, torch.ones(1, 1, 1, 1), 1)
        out = layer(2*torch.rand(4, 1, 3, 3)-1).detach().numpy()

        self.assertEqual(len(out.shape), 2)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 2)

        self.assertFalse(np.any(np.isinf(out)))
        self.assertFalse(np.any(np.isnan(out)))

import unittest

from src.EvoNet.layers import SphericalLinear


class TestSphericalLinearLayer(unittest.TestCase):

    def test_constructor(self):
        layer = SphericalLinear(2, 3)
        self.assertIsInstance(layer, SphericalLinear)

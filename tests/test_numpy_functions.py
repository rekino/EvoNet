import unittest
import numpy as np

from src.EvoNet.functions import spherical_harmonic


class TestNumpyFunctions(unittest.TestCase):

    def test_spherical_harmonic_val(self):
        Y = spherical_harmonic(1, 784)
        res = Y(1)

        assert np.isclose(res, 1)

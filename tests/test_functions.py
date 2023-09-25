import unittest
import torch
import numpy as np


from src.EvoNet.functions import Hyp0F1

class TestFunctions(unittest.TestCase):

    def test_hyp0f1(self):
        z = 1.0
        res = Hyp0F1.apply(0.5, z)

        self.assertAlmostEqual(res, np.cosh(2 * np.sqrt(z)))
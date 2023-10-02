import numpy as np
from mpmath import besseljzero, hyp0f1
import torch
import torch.nn as nn
from itertools import product
from scipy.optimize import brentq

from ..functions import Hyp0F1, Hyp2F1


class SphericalHarmonic(nn.Module):
    def __init__(self, xdim, rad_deg, ang_deg, bc='dirichlet') -> None:
        super().__init__()
        self.xdim = xdim
        self.rad_deg = rad_deg
        self.ang_deg = ang_deg
        self.bc = bc

        if bc == 'neumann':
            self._eigenvalues = self._neumann()
        elif bc == 'dirichlet':
            self._eigenvalues = self._dirichlet()
        else:
            raise NotImplementedError()

    def _dirichlet(self, k=None):
        k = self.rad_deg if k is None else k
        n, L = self.xdim, self.ang_deg
        eigenvalues = torch.zeros(k, L)

        for i in range(L):
            order = n/2 + i - 1
            eigenvalues[:, i] = torch.tensor([
                float(besseljzero(order, m+1)) for m in range(k)
                ])

        return eigenvalues

    def _neumann(self):
        n, l, k = self.xdim, self.ang_deg, self.rad_deg
        eigenvalues = torch.zeros(k, l)

        hyp0f1_vec = np.vectorize(hyp0f1)
        brentq_vec = np.vectorize(brentq)

        def Rp(lam):
            return l*hyp0f1_vec(n/2+l, -lam**2 / 4) - lam**2 * hyp0f1_vec(n/2+l+1, -lam**2 / 4) / (2*l + n)

        for i in range(l):
            order = n/2 + i - 1

            # m = 1
            # jpz = [besseljzero(order, 1)]
            # while len(jpz) <= k:
            #     zero = besseljzero(order, m+1)
            #     if Rp(jpz[-1]) * Rp(zero) < 0:
            #         jpz.append(float(zero))
            #     m += 1
            
            # jpz = np.asarray(jpz)
            # jpz = brentq_vec(Rp, jpz[:-1], jpz[1:])
            jpz = [float(besseljzero(order, m+1, derivative=1)) for m in range(k)]
            eigenvalues[:, i] = torch.tensor(jpz)

        return eigenvalues

    def forward(self, x):
        n, L, K = self.xdim, self.ang_deg, self.rad_deg
        lam = self._eigenvalues
        r = x[:, 0]
        t = x[:, 1:]

        result = []
        for k, l in product(range(K), range(L)):
            radial = -(r * lam[k, l])**2 / 4
            radial = r**l * Hyp0F1.apply(n/2 + l, radial)

            d = (n-3) / 2
            angular = Hyp2F1.apply(-(d+l), 1+l+d, 1+d, (1 - torch.abs(t))/2)
            angular /= Hyp2F1.apply(-(d+l), 1+l+d, 1+d, torch.zeros(1))
            angular *= (1 + torch.abs(t) / 2)**(-d)
            angular = torch.where(t < 0, (-1)**l * angular, angular)

            activation = radial[:, None] * angular

            result.append(activation)

        return torch.stack(result)

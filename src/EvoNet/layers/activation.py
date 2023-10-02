from mpmath import besseljzero
import torch
import torch.nn as nn
from itertools import product
from enum import Enum
import numpy as np
from mpmath import hyp0f1, absmax
from scipy.optimize import brentq

from ..functions import Hyp0F1, Hyp2F1


class BoundaryCondition(Enum):
    Dirichlet = 'dirichlet'
    Neumann = 'neumann'


class SphericalHarmonic(nn.Module):
    def __init__(self,
                 xdim, rad_deg, ang_deg,
                 bc=BoundaryCondition.Dirichlet
                 ) -> None:

        super().__init__()
        self.xdim = xdim
        self.rad_deg = rad_deg
        self.ang_deg = ang_deg
        self.bc = bc

        if bc == BoundaryCondition.Neumann:
            self._eigenvalues = self._neumann()
        elif bc == BoundaryCondition.Dirichlet:
            self._eigenvalues = self._dirichlet()
        else:
            raise NotImplementedError()

    def _dirichlet(self, K=None):
        K = self.rad_deg if K is None else K
        n, L = self.xdim, self.ang_deg
        eigenvalues = torch.zeros(K, L)

        bjz = np.vectorize(besseljzero)

        for i in range(L):
            order = n/2 + i - 1
            jpz = bjz(order, np.arange(K)+1)
            eigenvalues[:, i] = torch.tensor(jpz.astype('float32'))

        return eigenvalues

    def _neumann(self):
        n, L, K = self.xdim, self.ang_deg, self.rad_deg
        eigenvalues = torch.zeros(K, L)

        def Rp(lam, i):
            res = i * hyp0f1(n/2 + i, -lam**2/4)
            res -= lam**2 * hyp0f1(n/2 + i + 1, -lam**2/4) / (2*i + n)

            return res

        bjz = np.vectorize(besseljzero)
        h0f1 = np.vectorize(hyp0f1)
        bq = np.vectorize(brentq)

        for i in range(L):
            order = n/2 + i - 1

            lam_0 = besseljzero(order, 1, derivative=1)
            delta = 1
            for j in range(1000):
                if absmax(delta) < 1e-8:
                    break
                lam_1 = -lam_0**2 / 4
                lam_1 = h0f1(order + 1, lam_1) / h0f1(order + 2, lam_1)
                lam_1 = np.sqrt(i*(2*i+n) * lam_1)

                delta = lam_1 - lam_0
                lam_0 = float(lam_1)

            eigenvalues[0, i] = torch.tensor(lam_0)

            jz = bjz(order, np.arange(K)+1)
            jz = bq(Rp, jz[:-1], jz[1:], args=(i, ))
            eigenvalues[1:, i] = torch.tensor(jz)

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

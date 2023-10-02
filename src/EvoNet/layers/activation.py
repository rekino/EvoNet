import numpy as np
from mpmath import besseljzero as bjz
import torch
import torch.nn as nn
from itertools import product

from EvoNet.functions import Hyp0F1, Hyp2F1


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
            eigenvalues[:, i] = torch.tensor(
                [float(bjz(order, m+1)) for m in range(k)]
                )

        return eigenvalues

    def _neumann(self):
        n, l, k = self.xdim, self.ang_deg, self.rad_deg
        eigenvalues = torch.zeros(k, l)

        for i in range(l):
            order = n/2 + i - 1
            jpz = [float(bjz(order, m+1, derivative=1)) for m in range(k)]
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    module = SphericalHarmonic(2, 3, 4)

    r = torch.linspace(0, 1, 50)
    t = torch.linspace(-1, 1, 50)

    R = r[:, None] * t[None, :]**0
    T = t[None, :] * r[:, None]**0

    x = torch.hstack([r[:, None], T])
    out = module(x).numpy()

    print(np.abs(out).max())

    fig, ax = plt.subplots(3, 4)

    for k, l in np.ndindex((3, 4)):
        ax[k, l].contourf(R.numpy(), T.numpy(), out[k*4+l])

    plt.show()

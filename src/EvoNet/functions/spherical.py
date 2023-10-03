from scipy.integrate import odeint
import numpy as np
from scipy.interpolate import CubicSpline


def angular_ode(_y, _x, dim, eigen):
    if np.isclose(_x, 0, 1e-10):
        return [
            _y[1],
            -((dim-2) + eigen*_y[0])
        ]
    return [
        _y[1],
        -(_y[1]*(dim-2)/np.tan(_x) + eigen*_y[0])
    ]


def angular_int(x, dim, index):
    v = index * (index + dim - 2)
    y0 = [1.0, 0.0]
    return odeint(angular_ode, y0, x, args=(dim, v))


def spherical_harmonic(index, dim, knots=1000):
    x = np.linspace(0, 1, knots)
    t = np.arccos(np.flip(x))
    y = angular_int(t, dim, index)
    spl = CubicSpline(x, np.flip(y[:, 0]))

    def T(_x, der=0):
        tmp = spl(np.abs(_x), nu=der)
        return np.where(_x < 0, (-1)**index * tmp, tmp)

    return T

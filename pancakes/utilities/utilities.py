import numpy as np
from scipy.special import eval_legendre
from scipy.integrate import simps
from scipy.interpolate import (
    InterpolatedUnivariateSpline, RectBivariateSpline
)


def read_2darray(fname):
    data = np.genfromtxt(fname)
    dim1 = np.unique(data[:,0])
    dim2 = np.unique(data[:,1])

    vary_dim2 = False
    if data[0,0] == data[1,0]:
        vary_dim2 = True

    result = np.zeros([len(dim1), len(dim2)])
    counter = 0
    if vary_dim2:
        for i in range(len(dim1)):
            for j in range(len(dim2)):
                result[i, j] = data[counter, 2]
                counter += 1
    else:
        for i in range(len(dim2)):
            for j in range(len(dim1)):
                result[j, i] = data[counter, 2]
                counter += 1
    return dim1, dim2, result

def multipole(
    ell,
    s,
    mu,
    xi_smu
):
    """
    Decomposes the redshift-space cross-correlation function
    into multipoles.
    Args:
        ell: list of multipoles to calculate
        s: radial pairwise separation
        mu: cosine of the angle between pair separation
            and the line of sight.
        xi_smu: redshift-space correlation function.

    Returns: List containing all requested multipoles.
    """
    multipoles = [ ]
    for pole in ell:
        multipole = _multipole(
            pole, s, mu, xi_smu
            )
        multipoles.append(multipole)
    return multipoles

def _multipole(
    ell,
    s,
    mu,
    xi_smu
):
    """
    Calculate multipoles of the redshif-space
    correlation function.
    """

    multipole = np.zeros(xi_smu.shape[0])
    xi_func = RectBivariateSpline(
        s, mu, xi_smu,
        kx=1, ky=1, s=0
    )

    if mu.min() < 0:
        factor = 2
        mumin = -1
    else:
        factor = 1
        mumin=0

    x_axis = np.linspace(mumin, 1, 200)
    lmu = eval_legendre(ell, x_axis)

    for i in range(len(s)):
        y_axis = xi_func(s[i], x_axis) * (2 * ell + 1) / factor * lmu
        multipole[i] = simps(y_axis, x_axis)
    return multipole

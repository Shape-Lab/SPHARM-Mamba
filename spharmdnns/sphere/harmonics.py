"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from joblib import Parallel, delayed


def legendre(n, x, tol, tstart):
    """
    The Schmidt semi-normalized associated polynomials.

    Parameters
    __________
    n : int
        Degree of polynomial.
    x : 1D array, shape = [n_vertex]
        List of cosine values.
    tol : float
        Threshold for underflow.
    tstart: float
        Small value used to initialize recurrence at poles.

    Returns
    _______
    Y : 2D array, shape = [n + 1, n_vertex]
        Schmidt semi-normalized associated Legendre functions.

    Notes
    _____
    MATLAB: https://www.mathworks.com/help/matlab/ref/legendre.html
    """

    Y = np.zeros((n + 1, len(x)))

    if n == 0:
        Y[0] = 1
        return Y

    if n == 1:
        Y[0] = x
        Y[1] = np.sqrt(1.0 - x * x)
        return Y

    factor = np.sqrt(1.0 - x * x)
    rootn = np.sqrt(np.arange(2 * n + 1))
    pole = factor == 0
    factor[pole] = 1
    twocot = -2 * x / factor
    sn = (-factor) ** n

    Y[n] = np.sqrt(np.prod(1.0 - 1.0 / (2 * np.arange(1, n + 1)))) * sn
    Y[n - 1] = Y[n] * twocot * n / rootn[2 * n]
    for m in range(n - 2, -1, -1):
        Y[m] = (Y[m + 1] * twocot * (m + 1) - Y[m + 2] * rootn[n + m + 2] * rootn[n - m - 1]) / (rootn[n + m + 1] * rootn[n - m])

    idx = np.where(np.absolute(sn) < tol)[0]
    if idx.size > 0:
        v = 9.2 - np.log(tol) / (n * factor[idx])
        w = 1 / np.log(v)
        m1 = 1 + n * factor[idx] * v * w * (1.0058 + w * (3.819 - w * 12.173))
        m1 = np.minimum(n, np.floor(m1)).astype(int)

        Y[:, idx] = 0
        for mm1 in np.unique(m1):
            col = idx[m1 == mm1]
            neg = x[col] < 0

            Y[mm1 - 1, col[neg]] = np.sign((n + 1) % 2 - 0.5) * tstart
            Y[mm1 - 1, col[~neg]] = np.sign(mm1 % 2 - 0.5) * tstart
            for m in range(mm1 - 2, -1, -1):
                Y[m, col] = (Y[m + 1, col] * twocot[col] * (m + 1) - Y[m + 2, col] * rootn[n + m + 2] * rootn[n - m - 1]) / (
                    rootn[n + m + 1] * rootn[n - m]
                )
            sumsq = tol + (Y[: mm1 - 2, col] * Y[: mm1 - 2, col]).sum(0)
            Y[:mm1, col] /= np.sqrt(2 * sumsq - Y[0, col] * Y[0, col])

    Y[1:, pole] = 0
    Y[0, pole] = x[pole] ** n
    Y[1:] *= rootn[2]
    Y[1::2] *= -1

    return Y


def spharm_real(x, l, lbase=0, threads=1):
    """
    A set of real spherical harmonic bases using the Schmidt semi-normalized associated polynomials.
    The spherical harmonics will be generated from lbase to l.

    Parameters
    __________
    x : 2D array, shape = [n_vertex, 3]
        Array of 3D coordinates of the unit sphere.
    l : int
        Degree of spherical harmonics.
    lbase : int
        Base degree of spherical harmonics.
    threads : int
        Non-negative number of threads for parallel computing powered by joblib.

    Returns
    _______
    Y : 2D array, shape = [(l - lbase + 1) ** 2, n_vertex]
        Real spherical harmonic bases.
    """

    def basis(Y, theta, lfrom, lto, lbase, c, s, tol, tstart):
        cos = np.cos(-theta)
        for l in range(lfrom, lto, 1):
            center = (l + 1) * (l + 1) - l - 1 - lbase * lbase
            Y[center : center + l + 1] = legendre(l, cos, tol, tstart) * np.sqrt((2 * l + 1) / (4 * np.pi))
        if lfrom == 0:
            lfrom = 1
        for l in range(lfrom, lto, 1):
            center = (l + 1) * (l + 1) - l - 1 - lbase * lbase
            Y[center - 1 : center - l - 1 : -1] = Y[center + 1 : center + l + 1] * s[:l]
            Y[center + 1 : center + l + 1] *= c[:l]

    lbase = max(lbase, 0)

    phi = np.arctan2(x[:, 1], x[:, 0])
    theta = np.pi / 2 - np.arctan2(x[:, 2], np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))

    size = (l + 1) * (l + 1) - lbase * lbase
    Y = np.zeros((size, len(x)))

    m = np.arange(1, l + 1)
    deg = m[:, None] * phi[None, :]
    c, s = np.cos(deg), np.sin(deg)

    tol = np.sqrt(np.finfo(x.dtype).tiny)
    tstart = np.finfo(x.dtype).eps

    Parallel(n_jobs=threads, require="sharedmem")(delayed(basis)(Y, theta, n, n + 1, lbase, c, s, tol, tstart) for n in range(lbase, l + 1, 1))

    return Y

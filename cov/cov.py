from typing import Union
import numpy as np
import scipy as sp
import unittest
from numpy.typing import ArrayLike
from numpy.random import default_rng, Generator


def haar_measure(n, rng: Generator = 0):
    """
    A random orthogonal matrix distributed uniformly wrt Haar measure

    Adapted from http://arxiv.org/abs/math-ph/0609050 
    """
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)  # QR decomposition; Q orthogonal, R upper-triangular
    D = np.diagonal(R)  # take diagonals of upper-triangular R
    Ph = D / np.abs(D)  # "sign" of diagonals
    Q_ = Q @ np.diag(Ph)  # TODO: double-check

    return Q_


def covariance_matrix(spectrum: ArrayLike, rng: Union[int, Generator] = 0):
    """
    Randomly generate a covariate matrix with the given eigenspectrum


    Parameters
    ----------
    spectrum : ArrayLike, shape (n, )
        Eigenspectrum of the desired covariate matrix
    rng : Union[int, Generator], optional
        random number generator or random seed, by default 0

    Returns
    -------
    covariate_matrix : ArrayLike
        Random covariate matrix with the given eigenspectrum
    
    """
    spectrum = np.array(spectrum)
    assert len(spectrum.shape) == 1, "spectrum must be a vector"
    if isinstance(rng, int):
        rng = default_rng(seed=rng)
    n = spectrum.shape[0]

    # intuitively, we want to (uniformly) randomly rotate/reflect the diagonal matrix
    U = haar_measure(n, rng)
    cov = U.T @ np.diag(spectrum) @ U  # TODO: double-check

    return cov


def cov_to_corr(cov_matrix: ArrayLike):
    """
    Convert covariance matrix to correlation matrix


    Parameters
    ----------
    cov_matrix : ArrayLike
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert cov_matrix.shape[0] == cov_matrix.shape[1]
    variance = np.diag(cov_matrix)
    inverse_stdevs = np.diag(1 / np.sqrt(variance))
    corr = inverse_stdevs @ cov_matrix @ inverse_stdevs
    return corr

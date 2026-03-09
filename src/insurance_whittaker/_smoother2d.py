"""
Internal helpers for 2-D Whittaker-Henderson smoothing.

Uses direct Cholesky solve on the Kronecker-structured system for small-to-medium
tables (up to ~100x100 cells).

Reference: Biessy (2026), ASTIN Bulletin; WH R package compact.cpp.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from ._utils import penalty_matrix


def _eig_penalty(n: int, order: int) -> tuple[NDArray, NDArray]:
    """Eigendecompose the penalty matrix P = D'D for one dimension.

    Parameters
    ----------
    n:
        Dimension size.
    order:
        Difference order.

    Returns
    -------
    (eigenvalues, eigenvectors) — eigenvalues sorted ascending.
    """
    P = penalty_matrix(n, order)
    vals, vecs = eigh(P)
    return vals, vecs


def _build_2d_system(
    nx: int,
    nz: int,
    order_x: int,
    order_z: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Pre-compute eigendecompositions of the two 1-D penalty matrices.

    Returns
    -------
    (vals_x, vecs_x, vals_z, vecs_z)
    """
    vals_x, vecs_x = _eig_penalty(nx, order_x)
    vals_z, vecs_z = _eig_penalty(nz, order_z)
    return vals_x, vecs_x, vals_z, vecs_z


def _solve_2d_system(
    ab_Px: NDArray[np.float64],
    ab_Pz: NDArray[np.float64],
    nx: int,
    nz: int,
    W_vec: NDArray[np.float64],
    y_vec: NDArray[np.float64],
    lam_x: float,
    lam_z: float,
    order_x: int = 2,
    order_z: int = 2,
) -> tuple[NDArray[np.float64], float]:
    """Solve the 2-D WH system and return (theta_vec, log_det).

    Parameters
    ----------
    ab_Px, ab_Pz:
        Banded penalty matrices (kept for API symmetry; not used internally).
    nx, nz:
        Table dimensions.
    W_vec:
        Vectorised weights, length nx*nz, row-major (x varies slowest).
    y_vec:
        Vectorised response, length nx*nz.
    lam_x, lam_z:
        Smoothing parameters.
    order_x, order_z:
        Difference orders.

    Returns
    -------
    (theta_vec, log_det):
        Smoothed values and log-determinant of the system matrix.
    """
    Px_full = penalty_matrix(nx, order_x)
    Pz_full = penalty_matrix(nz, order_z)
    Iz = np.eye(nz)
    Ix = np.eye(nx)

    P_kron = lam_x * np.kron(Iz, Px_full) + lam_z * np.kron(Pz_full, Ix)
    A = np.diag(W_vec) + P_kron
    Wy = W_vec * y_vec

    from scipy.linalg import cho_factor, cho_solve
    c, low = cho_factor(A)
    theta_vec = cho_solve((c, low), Wy)

    sign, log_det = np.linalg.slogdet(A)
    if sign <= 0:
        log_det = 0.0

    return theta_vec, float(log_det)


def solve_2d_full(
    nx: int,
    nz: int,
    W_vec: NDArray[np.float64],
    y_vec: NDArray[np.float64],
    lam_x: float,
    lam_z: float,
    order_x: int = 2,
    order_z: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Solve the 2-D system and return theta, diag(V), and log-det.

    Parameters
    ----------
    nx, nz:
        Table dimensions.
    W_vec:
        Vectorised weights, length nx*nz.
    y_vec:
        Vectorised response, length nx*nz.
    lam_x, lam_z:
        Smoothing parameters.
    order_x, order_z:
        Difference orders.

    Returns
    -------
    (theta_vec, diag_v, log_det)
    """
    Px_full = penalty_matrix(nx, order_x)
    Pz_full = penalty_matrix(nz, order_z)
    Iz = np.eye(nz)
    Ix = np.eye(nx)

    P_kron = lam_x * np.kron(Iz, Px_full) + lam_z * np.kron(Pz_full, Ix)
    A = np.diag(W_vec) + P_kron
    Wy = W_vec * y_vec

    from scipy.linalg import cho_factor, cho_solve

    try:
        c, low = cho_factor(A)
        theta_vec = cho_solve((c, low), Wy)
        # Diagonal of A^{-1} via column-wise solve
        n_total = nx * nz
        diag_v = np.zeros(n_total)
        for i in range(n_total):
            e_i = np.zeros(n_total)
            e_i[i] = 1.0
            diag_v[i] = cho_solve((c, low), e_i)[i]
        sign, log_det = np.linalg.slogdet(A)
    except np.linalg.LinAlgError:
        # Fallback: use general solver
        from scipy.linalg import solve
        theta_vec = solve(A, Wy)
        diag_v = np.diag(np.linalg.inv(A))
        sign, log_det = np.linalg.slogdet(A)

    return theta_vec, diag_v, float(log_det)

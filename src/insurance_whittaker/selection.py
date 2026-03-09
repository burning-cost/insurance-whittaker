"""
Lambda selection criteria for Whittaker-Henderson smoothing.

Implements REML (recommended), GCV, AIC, and BIC.  All criteria work on
the full symmetric system matrix A = W + lambda * P.

Reference: Biessy (2026), 'Whittaker-Henderson Smoothing Revisited',
ASTIN Bulletin.  arXiv:2306.06932.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar, minimize

from ._utils import penalty_banded, penalty_matrix


# ---------------------------------------------------------------------------
# Core: build and solve the full system
# ---------------------------------------------------------------------------

def _build_full_system(
    P_full: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Return A = W + lambda * P as a full matrix.

    Parameters
    ----------
    P_full:
        Full penalty matrix, shape (n, n).
    W_diag:
        Diagonal weight vector, length n.
    lam:
        Smoothing parameter.

    Returns
    -------
    A, shape (n, n).
    """
    return np.diag(W_diag) + lam * P_full


def _solve_system(
    P_full: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    Wy: NDArray[np.float64],
    lam: float,
) -> tuple[NDArray, object, float]:
    """Solve A theta = Wy and return (theta, chol_factor, log_det).

    Parameters
    ----------
    P_full:
        Full penalty matrix, shape (n, n).
    W_diag:
        Diagonal weight vector.
    Wy:
        W @ y vector.
    lam:
        Smoothing parameter.

    Returns
    -------
    theta_hat:
        Smoothed values.
    cf:
        Cholesky factor tuple (from scipy.linalg.cho_factor).
    log_det:
        log|A|, computed from the Cholesky diagonal.
    """
    A = _build_full_system(P_full, W_diag, lam)
    cf = cho_factor(A)
    theta = cho_solve(cf, Wy)
    # log|A| = 2 * sum(log(diag(L))) where A = L L^T
    log_det = 2.0 * np.sum(np.log(np.abs(np.diag(cf[0]))))
    return theta, cf, log_det


def _edf_from_hat(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    lam: float,
) -> float:
    """Compute effective degrees of freedom edf = trace(H).

    H = (W + lam*P)^{-1} W.  edf = trace(H) = sum_i W_i * V_ii
    where V = A^{-1}.

    Parameters
    ----------
    ab_P:
        Penalty in upper banded storage (used only to get n and order).
    W_diag:
        Diagonal weight vector.
    lam:
        Smoothing parameter.

    Returns
    -------
    Effective degrees of freedom as a float.
    """
    n = len(W_diag)
    order = ab_P.shape[0] - 1
    P_full = penalty_matrix(n, order)
    A = _build_full_system(P_full, W_diag, lam)
    cf = cho_factor(A)
    diag_v = _diag_of_inverse(cf, n)
    return float(np.sum(W_diag * diag_v))


def _diag_of_inverse(cf: tuple, n: int) -> NDArray[np.float64]:
    """Compute diagonal of A^{-1} from Cholesky factor via column solves."""
    diag_v = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        v = cho_solve(cf, e_i)
        diag_v[i] = v[i]
    return diag_v


# ---------------------------------------------------------------------------
# REML criterion
# ---------------------------------------------------------------------------

def _log_det_P_nonzero(n: int, order: int) -> float:
    """Compute log-determinant of the non-zero eigenvalues of P = D'D."""
    P_full = penalty_matrix(n, order)
    eigs = np.linalg.eigvalsh(P_full)
    eigs_nz = eigs[eigs > 1e-10]
    return float(np.sum(np.log(eigs_nz)))


def reml_criterion(
    log_lam: float,
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
    log_det_P_nz: float,
) -> float:
    """Evaluate the REML criterion at log(lambda).

    REML = (dev + pen) / 2 + K / 2
    where K = log|A| - (n - order) * log(lam) - log_det_P_nz.

    Returns the REML value (minimise this).
    """
    lam = np.exp(log_lam)
    n = len(y)
    P_full = penalty_matrix(n, order)
    Wy = W_diag * y
    try:
        theta, cf, log_det_A = _solve_system(P_full, W_diag, Wy, lam)
    except np.linalg.LinAlgError:
        return 1e30

    resid = y - theta
    dev = float(np.sum(W_diag * resid ** 2))

    # Penalty via differences
    Dq_theta = theta.copy()
    for _ in range(order):
        Dq_theta = np.diff(Dq_theta)
    pen = lam * float(np.sum(Dq_theta ** 2))

    K = log_det_A - (n - order) * np.log(lam) - log_det_P_nz
    return (dev + pen) / 2.0 + K / 2.0


# ---------------------------------------------------------------------------
# GCV criterion
# ---------------------------------------------------------------------------

def gcv_criterion(
    log_lam: float,
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
    _log_det_P_nz: float = 0.0,
) -> float:
    """Evaluate the GCV criterion at log(lambda).

    GCV = n * dev / (n - edf)^2
    """
    lam = np.exp(log_lam)
    n = len(y)
    P_full = penalty_matrix(n, order)
    Wy = W_diag * y
    try:
        theta, cf, _ = _solve_system(P_full, W_diag, Wy, lam)
    except np.linalg.LinAlgError:
        return 1e30
    diag_v = _diag_of_inverse(cf, n)
    edf = float(np.sum(W_diag * diag_v))
    resid = y - theta
    dev = float(np.sum(W_diag * resid ** 2))
    denom = (n - edf) ** 2
    if denom < 1e-10:
        return 1e30
    return n * dev / denom


# ---------------------------------------------------------------------------
# AIC criterion
# ---------------------------------------------------------------------------

def aic_criterion(
    log_lam: float,
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
    _log_det_P_nz: float = 0.0,
) -> float:
    """Evaluate the AIC criterion at log(lambda).

    AIC = dev + 2 * edf
    """
    lam = np.exp(log_lam)
    n = len(y)
    P_full = penalty_matrix(n, order)
    Wy = W_diag * y
    try:
        theta, cf, _ = _solve_system(P_full, W_diag, Wy, lam)
    except np.linalg.LinAlgError:
        return 1e30
    diag_v = _diag_of_inverse(cf, n)
    edf = float(np.sum(W_diag * diag_v))
    resid = y - theta
    dev = float(np.sum(W_diag * resid ** 2))
    return dev + 2.0 * edf


# ---------------------------------------------------------------------------
# BIC criterion
# ---------------------------------------------------------------------------

def bic_criterion(
    log_lam: float,
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
    _log_det_P_nz: float = 0.0,
) -> float:
    """Evaluate the BIC criterion at log(lambda).

    BIC = dev + log(n) * edf
    """
    lam = np.exp(log_lam)
    n = len(y)
    P_full = penalty_matrix(n, order)
    Wy = W_diag * y
    try:
        theta, cf, _ = _solve_system(P_full, W_diag, Wy, lam)
    except np.linalg.LinAlgError:
        return 1e30
    diag_v = _diag_of_inverse(cf, n)
    edf = float(np.sum(W_diag * diag_v))
    resid = y - theta
    dev = float(np.sum(W_diag * resid ** 2))
    return dev + np.log(n) * edf


# ---------------------------------------------------------------------------
# Unified selector
# ---------------------------------------------------------------------------

CRITERIA = {
    "reml": reml_criterion,
    "gcv": gcv_criterion,
    "aic": aic_criterion,
    "bic": bic_criterion,
}


def select_lambda(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
    method: str = "reml",
) -> float:
    """Select the optimal lambda via 1-D minimisation over log(lambda).

    Uses Brent's method (golden section) via ``scipy.optimize.minimize_scalar``
    over the interval log(lambda) in [-6*log(10), 12*log(10)].

    Parameters
    ----------
    ab_P:
        Penalty in upper banded storage, shape (order+1, n).
    W_diag:
        Diagonal weight vector of length n.
    y:
        Response vector of length n.
    order:
        Difference order q.
    method:
        One of 'reml' (default), 'gcv', 'aic', 'bic'.

    Returns
    -------
    Optimal lambda (positive float).

    Raises
    ------
    ValueError
        If method is not recognised.
    """
    if method not in CRITERIA:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(CRITERIA.keys())}"
        )
    criterion_fn = CRITERIA[method]

    n = len(y)
    log_det_P_nz = 0.0
    if method == "reml":
        log_det_P_nz = _log_det_P_nz_cached(n, order)

    result = minimize_scalar(
        criterion_fn,
        bounds=(-6 * np.log(10), 12 * np.log(10)),
        method="bounded",
        args=(ab_P, W_diag, y, order, log_det_P_nz),
    )
    return float(np.exp(result.x))


def select_lambda_2d(
    ab_Px: NDArray[np.float64],
    ab_Pz: NDArray[np.float64],
    nx: int,
    nz: int,
    W_vec: NDArray[np.float64],
    y_vec: NDArray[np.float64],
    order_x: int,
    order_z: int,
    method: str = "reml",
) -> tuple[float, float]:
    """Select lambda_x and lambda_z jointly for 2D smoothing via Nelder-Mead.

    Parameters
    ----------
    ab_Px:
        Penalty for x-direction, upper banded.
    ab_Pz:
        Penalty for z-direction, upper banded.
    nx, nz:
        Dimensions of the table.
    W_vec:
        Vectorised weight array (row-major), length nx*nz.
    y_vec:
        Vectorised response (row-major), length nx*nz.
    order_x, order_z:
        Difference orders.
    method:
        Currently only 'reml' is supported for 2D.

    Returns
    -------
    (lambda_x, lambda_z) tuple.
    """
    from ._smoother2d import _solve_2d_system

    def neg_reml_2d(log_lams: NDArray) -> float:
        lx, lz = np.exp(log_lams[0]), np.exp(log_lams[1])
        try:
            theta, log_det = _solve_2d_system(
                ab_Px, ab_Pz, nx, nz, W_vec, y_vec, lx, lz, order_x, order_z
            )
        except Exception:
            return 1e30
        resid = y_vec - theta
        dev = float(np.sum(W_vec * resid ** 2))
        Dqx_theta = theta.reshape(nx, nz).copy()
        for _ in range(order_x):
            Dqx_theta = np.diff(Dqx_theta, axis=0)
        pen_x = lx * float(np.sum(Dqx_theta ** 2))
        Dqz_theta = theta.reshape(nx, nz).copy()
        for _ in range(order_z):
            Dqz_theta = np.diff(Dqz_theta, axis=1)
        pen_z = lz * float(np.sum(Dqz_theta ** 2))
        return (dev + pen_x + pen_z) / 2.0 + log_det / 2.0

    log_lam_bounds = (-6 * np.log(10), 10 * np.log(10))
    x0 = np.array([3.0, 3.0])  # start at lambda ~ 20
    res = minimize(
        neg_reml_2d, x0,
        method="L-BFGS-B",
        bounds=[log_lam_bounds, log_lam_bounds],
        options={"ftol": 1e-6, "maxiter": 200},
    )
    lx = float(np.clip(np.exp(res.x[0]), 1e-6, 1e12))
    lz = float(np.clip(np.exp(res.x[1]), 1e-6, 1e12))
    return lx, lz


# ---------------------------------------------------------------------------
# Compatibility shim for smoother.py (which imports _solve_banded_system)
# ---------------------------------------------------------------------------

def _solve_banded_system(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    Wy: NDArray[np.float64],
    lam: float,
) -> tuple[NDArray, object, float]:
    """Compatibility wrapper: solve using full Cholesky.

    Parameters
    ----------
    ab_P:
        Penalty in upper banded storage, shape (order+1, n).
    W_diag:
        Diagonal weight vector.
    Wy:
        W @ y vector.
    lam:
        Smoothing parameter.

    Returns
    -------
    (theta_hat, chol_factor, log_det)
    """
    n = len(W_diag)
    order = ab_P.shape[0] - 1
    P_full = penalty_matrix(n, order)
    return _solve_system(P_full, W_diag, Wy, lam)


# ---------------------------------------------------------------------------
# Cache for log_det_P_nz (expensive eigendecomposition)
# ---------------------------------------------------------------------------

_LOG_DET_CACHE: dict[tuple, float] = {}


def _log_det_P_nz_cached(n: int, order: int) -> float:
    """Cached version of _log_det_P_nonzero."""
    key = (n, order)
    if key not in _LOG_DET_CACHE:
        _LOG_DET_CACHE[key] = _log_det_P_nonzero(n, order)
    return _LOG_DET_CACHE[key]

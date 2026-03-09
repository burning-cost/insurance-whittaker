"""
Lambda selection criteria for Whittaker-Henderson smoothing.

Implements REML (recommended), GCV, AIC, and BIC.  All criteria operate
on the banded system (W + lambda * P) without forming the full matrix.

Reference: Biessy (2026), 'Whittaker-Henderson Smoothing Revisited',
ASTIN Bulletin.  arXiv:2306.06932.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.optimize import minimize_scalar, minimize

from ._utils import add_lambda_to_banded, penalty_banded


# ---------------------------------------------------------------------------
# Core: solve the banded system and compute derived quantities
# ---------------------------------------------------------------------------

def _solve_banded_system(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    Wy: NDArray[np.float64],
    lam: float,
) -> tuple[NDArray, NDArray, float]:
    """Solve (W + lam*P) theta = W*y and return (theta, chol_factor, log_det).

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
    theta_hat:
        Smoothed values.
    cb:
        Cholesky factor (upper banded form).
    log_det:
        log|W + lam*P|, computed from the Cholesky diagonal.
    """
    ab = add_lambda_to_banded(ab_P, W_diag, lam)
    cb = cholesky_banded(ab, lower=False)
    theta = cho_solve_banded((cb, False), Wy)
    log_det = 2.0 * np.sum(np.log(cb[0, :]))
    return theta, cb, log_det


def _edf_from_hat(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    lam: float,
) -> float:
    """Compute effective degrees of freedom edf = trace(H).

    H = (W + lam*P)^{-1} W.  By linearity:
        trace(H) = sum_i W_i * V_ii
    where V = (W + lam*P)^{-1}.

    Parameters
    ----------
    ab_P:
        Penalty in upper banded storage.
    W_diag:
        Diagonal weight vector.
    lam:
        Smoothing parameter.

    Returns
    -------
    Effective degrees of freedom as a float.
    """
    from ._utils import diag_of_inverse_banded
    order = ab_P.shape[0] - 1
    ab = add_lambda_to_banded(ab_P, W_diag, lam)
    diag_v = diag_of_inverse_banded(ab, order)
    return float(np.sum(W_diag * diag_v))


# ---------------------------------------------------------------------------
# REML criterion
# ---------------------------------------------------------------------------

def _log_det_P_nonzero(ab_P: NDArray[np.float64], n: int, order: int) -> float:
    """Compute log-determinant of the non-zero eigenvalues of P = D'D.

    P has exactly ``order`` zero eigenvalues (the null space of D^q).
    We compute the product of non-zero eigenvalues via the determinant of
    the reduced (n - order) x (n - order) matrix D D^T (same non-zero
    eigenvalues as D^T D).

    Parameters
    ----------
    ab_P:
        Upper banded penalty matrix P.
    n:
        Dimension of P.
    order:
        Difference order q.

    Returns
    -------
    Sum of logs of non-zero eigenvalues.
    """
    # Full penalty matrix for eigenvalue decomposition — only done once
    # during optimisation setup; n is typically <= 200.
    P_full = np.zeros((n, n))
    for k in range(ab_P.shape[0]):
        diag_vals = ab_P[k, : n - k]
        P_full += np.diag(diag_vals, k)
        if k > 0:
            P_full += np.diag(diag_vals, -k)
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
    """Evaluate the negative REML criterion at log(lambda).

    Following Biessy (2026), the REML criterion is::

        REML = (dev + pen) / 2 + K / 2

    where::

        K = log|W + lam*P| - (n - order) * log(lam) - log_det_P_nz

    We minimise the negative REML (i.e., maximise REML).

    The deviance for Gaussian data is dev = sum w_i (y_i - theta_i)^2
    and pen = lam * theta^T P theta.

    Parameters
    ----------
    log_lam:
        Natural log of lambda.
    ab_P:
        Penalty in upper banded storage.
    W_diag:
        Diagonal weight vector.
    y:
        Response vector.
    order:
        Difference order q.
    log_det_P_nz:
        Pre-computed log-determinant of non-zero eigenvalues of P.

    Returns
    -------
    Negative REML value (scalar, minimise this).
    """
    lam = np.exp(log_lam)
    n = len(y)
    Wy = W_diag * y
    theta, cb, log_det_A = _solve_banded_system(ab_P, W_diag, Wy, lam)

    # Deviance: sum w_i (y_i - theta_i)^2
    resid = y - theta
    dev = float(np.sum(W_diag * resid ** 2))

    # Penalty: lam * theta' P theta — compute via the full P reconstruction
    # More efficiently via: pen = lam * ||D theta||^2
    # P stored as banded; reconstruct quadratic form without full matrix
    # pen = lam * theta @ P_full @ theta — use difference operator
    # D^q theta norm: apply diff order times
    Dq_theta = theta.copy()
    for _ in range(order):
        Dq_theta = np.diff(Dq_theta)
    pen = lam * float(np.sum(Dq_theta ** 2))

    # K = log|A| - (n-order)*log(lam) - log_det_P_nz
    K = log_det_A - (n - order) * np.log(lam) - log_det_P_nz

    reml = (dev + pen) / 2.0 + K / 2.0
    return reml  # minimise (REML is already a negative log-marginal-likelihood)


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

    Returns the GCV value (minimise this).
    """
    lam = np.exp(log_lam)
    Wy = W_diag * y
    theta, _, _ = _solve_banded_system(ab_P, W_diag, Wy, lam)
    edf = _edf_from_hat(ab_P, W_diag, lam)
    n = len(y)
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

    Returns the AIC value (minimise this).
    """
    lam = np.exp(log_lam)
    Wy = W_diag * y
    theta, _, _ = _solve_banded_system(ab_P, W_diag, Wy, lam)
    edf = _edf_from_hat(ab_P, W_diag, lam)
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

    Returns the BIC value (minimise this).
    """
    lam = np.exp(log_lam)
    n = len(y)
    Wy = W_diag * y
    theta, _, _ = _solve_banded_system(ab_P, W_diag, Wy, lam)
    edf = _edf_from_hat(ab_P, W_diag, lam)
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
    over the interval log(lambda) in [-6*log(10), 12*log(10)], corresponding
    to lambda in [1e-6, 1e12].

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
        log_det_P_nz = _log_det_P_nz_cached(ab_P, n, order)

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
    from ._smoother2d import _build_2d_system, _solve_2d_system

    def neg_reml_2d(log_lams: NDArray) -> float:
        lx, lz = np.exp(log_lams[0]), np.exp(log_lams[1])
        try:
            theta, log_det = _solve_2d_system(
                ab_Px, ab_Pz, nx, nz, W_vec, y_vec, lx, lz
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
        # Simplified REML proxy: deviance + penalties + 0.5 * log_det
        return (dev + pen_x + pen_z) / 2.0 + log_det / 2.0

    x0 = np.array([3.0, 3.0])  # start at lambda ~ 20
    res = minimize(neg_reml_2d, x0, method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 0.01, "maxiter": 500})
    lx, lz = np.exp(res.x[0]), np.exp(res.x[1])
    return float(lx), float(lz)


# ---------------------------------------------------------------------------
# Cache for log_det_P_nz (expensive eigendecomposition)
# ---------------------------------------------------------------------------

_LOG_DET_CACHE: dict[tuple, float] = {}


def _log_det_P_nz_cached(
    ab_P: NDArray[np.float64],
    n: int,
    order: int,
) -> float:
    """Cached version of _log_det_P_nonzero."""
    key = (n, order)
    if key not in _LOG_DET_CACHE:
        _LOG_DET_CACHE[key] = _log_det_P_nonzero(ab_P, n, order)
    return _LOG_DET_CACHE[key]

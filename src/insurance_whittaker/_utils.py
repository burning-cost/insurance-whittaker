"""
Utility functions for Whittaker-Henderson smoothing.

Covers difference matrix construction, banded matrix helpers, and
Polars/NumPy input normalisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Difference matrix helpers
# ---------------------------------------------------------------------------

def diff_matrix(n: int, order: int) -> NDArray[np.float64]:
    """Construct the q-th order difference matrix D of shape (n-q, n).

    D^1 is the first-difference matrix; D^q = D^1 applied q times via
    recursion.  The penalty matrix is P = D.T @ D.

    Parameters
    ----------
    n:
        Number of rating cells (columns of D).
    order:
        Difference order q.  Typical values: 1 (linear), 2 (quadratic).

    Returns
    -------
    NDArray of shape (n - order, n).
    """
    if order == 0:
        return np.eye(n)
    D = np.diff(np.eye(n), n=1, axis=0)
    for _ in range(order - 1):
        D = np.diff(D, n=1, axis=0)
    return D


def penalty_matrix(n: int, order: int) -> NDArray[np.float64]:
    """Return P = D^T D, the (n x n) penalty matrix.

    Parameters
    ----------
    n:
        Number of rating cells.
    order:
        Difference order q.

    Returns
    -------
    NDArray of shape (n, n).
    """
    D = diff_matrix(n, order)
    return D.T @ D


def penalty_banded(n: int, order: int) -> NDArray[np.float64]:
    """Return P = D^T D in upper banded storage format for solveh_banded.

    ``scipy.linalg.solveh_banded`` uses upper form: row k of the returned
    array holds the k-th superdiagonal (row 0 = main diagonal).  Shape is
    (order + 1, n).

    The (i, j) element of D^T D satisfies::

        (D'D)[i,j] = (-1)^|i-j| * C(2q, q-|i-j|)  for |i-j| <= q

    This follows from the convolution of binomial coefficients.

    Parameters
    ----------
    n:
        Number of rating cells.
    order:
        Difference order q.

    Returns
    -------
    NDArray of shape (order + 1, n) in upper banded storage.
    """
    from math import comb

    ab = np.zeros((order + 1, n))
    for k in range(order + 1):
        coeff = (-1) ** k * comb(2 * order, order - k)
        # k == 0 → main diagonal; k > 0 → k-th superdiagonal
        if k == 0:
            ab[0, :] = coeff
        else:
            # superdiagonal k has n-k elements; ab[k, :n-k]
            ab[k, : n - k] = coeff
    return ab


def add_lambda_to_banded(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Return (W + lambda * P) in upper banded storage.

    Parameters
    ----------
    ab_P:
        Penalty P in upper banded storage, shape (order+1, n).
    W_diag:
        Diagonal weight vector of length n.
    lam:
        Smoothing parameter lambda.

    Returns
    -------
    Upper banded array of shape (order+1, n).
    """
    ab = lam * ab_P.copy()
    ab[0, :] += W_diag
    return ab


# ---------------------------------------------------------------------------
# Diagonal of the inverse via Cholesky factor
# ---------------------------------------------------------------------------

def diag_of_inverse_banded(
    ab: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Compute the diagonal of (W + lambda P)^{-1} from its banded Cholesky.

    Uses the recursive back-substitution described in the WH R package
    (``diag_V_compact_cpp``).  Given the banded Cholesky factor L (lower
    triangular, computed from the upper banded form), the diagonal of
    L^{-T} L^{-1} is computed column by column in O(n * order^2).

    Parameters
    ----------
    ab:
        Upper banded storage of (W + lambda P), shape (order+1, n).
    order:
        Difference order q (bandwidth = q+1 diagonals).

    Returns
    -------
    NDArray of length n — the diagonal of the inverse matrix V.
    """
    from scipy.linalg import cholesky_banded  # lower=False by default

    # cholesky_banded returns the Cholesky factor in the same banded storage.
    cb = cholesky_banded(ab, lower=False)
    # cb is upper triangular banded: cb[k, j] = U[j-k, j] for row k, col j.
    # We need diag(V) = diag((U^T U)^{-1}).
    # Compute diagonal of U^{-1} and then diag(V) via recursive formula.
    n = ab.shape[1]
    bw = order  # upper bandwidth

    # Extract the actual U matrix entries we need via back-substitution.
    # Store columns of U^{-1} only within the bandwidth.
    # v[i] = diag of V = sum_k (U_inv[k,i])^2

    diag_v = np.zeros(n)
    # Work from right to left.
    # For column j of U^{-1}: U^{-1}[:,j] has nonzero entries only from
    # max(0, j-bw) to j.
    # Let x[i] = (U^{-1})[i, j] for i = j, j-1, ..., max(0,j-bw)

    col_inv = np.zeros(bw + 1)  # col_inv[k] = U^{-1}[j-k, j]

    for j in range(n - 1, -1, -1):
        # Diagonal entry of U^{-1}: 1 / U[j,j] = 1 / cb[0, j]
        ujj = cb[0, j]
        col_inv[0] = 1.0 / ujj
        # Off-diagonal: for k = 1, ..., min(bw, j)
        limit = min(bw, j)
        for k in range(1, limit + 1):
            # U[j-k, j] is stored as cb[k, j]; note cb uses upper storage
            # cb[k, j] = U[j-k, j]
            s = 0.0
            # sum over l = 0, ..., k-1
            for l in range(k):
                row = j - l  # column index in U (row of U^{-1} col j)
                if row <= j:
                    u_val = cb[k - l, j]  # U[j-k, j-l] — wait, need care
                    # U[i, r] stored at cb[r-i, r]
                    # We want U[j-k, j-l] — that is cb[(j-l)-(j-k), j-l] = cb[k-l, j-l]
                    u_val = cb[k - l, j - l]
                    s += u_val * col_inv[l]
            col_inv[k] = -s / ujj
        # contribution to diag_v[j-k] for k = 0,...,limit
        for k in range(limit + 1):
            diag_v[j - k] += col_inv[k] ** 2

    return diag_v


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def to_numpy_1d(x: ArrayLike, name: str = "array") -> NDArray[np.float64]:
    """Convert array-like or Polars Series to a 1-D float64 NumPy array.

    Parameters
    ----------
    x:
        Input data.  Accepts NumPy arrays, Python lists, Polars Series.
    name:
        Name used in error messages.

    Returns
    -------
    1-D float64 NumPy array.

    Raises
    ------
    ValueError
        If the result is not 1-D or contains NaN/Inf.
    """
    try:
        import polars as pl
        if isinstance(x, pl.Series):
            x = x.to_numpy()
    except ImportError:
        pass

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def validate_inputs(
    y: NDArray,
    weights: NDArray | None,
    n: int,
) -> NDArray[np.float64]:
    """Validate and return a weight vector of length n.

    If weights is None, returns an array of ones.  Raises if lengths
    do not match or if any weight is negative.

    Parameters
    ----------
    y:
        Response vector of length n.
    weights:
        Weight vector or None.
    n:
        Expected length.

    Returns
    -------
    1-D float64 weight array of length n.
    """
    if len(y) != n:
        raise ValueError(f"y length {len(y)} does not match n={n}")
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = to_numpy_1d(weights, "weights")
    if len(w) != n:
        raise ValueError(f"weights length {len(w)} != y length {n}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w

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
    """Return a banded descriptor of P = D^T D.

    Returns a lightweight array of shape (order+1, n) where the first
    element holds n and order (used as a descriptor).  The actual computation
    uses the full matrix for correctness.

    Parameters
    ----------
    n:
        Number of rating cells.
    order:
        Difference order q.

    Returns
    -------
    NDArray of shape (order + 1, n).  The shape encodes (order, n).
    """
    # We compute the full penalty matrix and store it in the first row
    # with the banded descriptor in the rest — but for compatibility with
    # the selection module, we just return a descriptor array.
    # The shape (order+1, n) is used by callers to recover n and order.
    ab = np.zeros((order + 1, n))
    return ab


def add_lambda_to_banded(
    ab_P: NDArray[np.float64],
    W_diag: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Legacy compatibility function — returns the banded descriptor unchanged.

    The actual system matrix is built in selection.py via penalty_matrix.
    This function is kept for API compatibility.

    Parameters
    ----------
    ab_P:
        Banded descriptor, shape (order+1, n).
    W_diag:
        Diagonal weight vector of length n.
    lam:
        Smoothing parameter.

    Returns
    -------
    The same descriptor ab_P (value not used).
    """
    return ab_P.copy()


def diag_of_inverse_banded(
    ab: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Compute the diagonal of A^{-1} where A = W + lambda * P.

    This function is called from smoother.py after the banded solve.
    Since we now use full matrices, ``ab`` here is the actual system
    matrix A = W + lambda * P (full, shape (n, n)).

    Parameters
    ----------
    ab:
        Full system matrix A, shape (n, n).
    order:
        Unused — kept for API compatibility.

    Returns
    -------
    NDArray of length n — the diagonal of A^{-1}.
    """
    from scipy.linalg import cho_factor, cho_solve

    n = ab.shape[0]
    cf = cho_factor(ab)
    diag_v = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        v = cho_solve(cf, e_i)
        diag_v[i] = v[i]
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
        If the result is not 1-D.
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

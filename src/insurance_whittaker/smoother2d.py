"""
2-D Whittaker-Henderson smoother for actuarial cross-tables.

Smooths a two-dimensional table (e.g., age x vehicle group, age x
duration) by solving a Kronecker-structured penalised least-squares
problem.  The penalty separates into independent row-wise and column-wise
components, with independent smoothing parameters lambda_x and lambda_z.

Reference: Biessy (2026), ASTIN Bulletin.  arXiv:2306.06932.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import to_numpy_1d, penalty_banded
from ._smoother2d import solve_2d_full

LambdaMethod = Literal["reml"]


@dataclass
class WHResult2D:
    """Results from a fitted 2-D Whittaker-Henderson smoother.

    Attributes
    ----------
    fitted:
        Smoothed theta_hat values, shape (nx, nz).
    ci_lower:
        Lower bound of the 95% posterior credible interval, shape (nx, nz).
    ci_upper:
        Upper bound of the 95% posterior credible interval, shape (nx, nz).
    std_fitted:
        Posterior standard deviations sigma_hat * sqrt(diag(V)), shape (nx, nz).
    lambda_x:
        Smoothing parameter for the x-direction.
    lambda_z:
        Smoothing parameter for the z-direction.
    edf:
        Effective degrees of freedom.
    order_x:
        Difference order in x-direction.
    order_z:
        Difference order in z-direction.
    criterion:
        Lambda selection criterion used.
    x_labels:
        Labels for x-axis (first dimension).
    z_labels:
        Labels for z-axis (second dimension).
    sigma2:
        Estimated residual variance sigma_hat^2 = dev / (n - edf).
    """

    fitted: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    std_fitted: NDArray[np.float64]
    lambda_x: float
    lambda_z: float
    edf: float
    order_x: int
    order_z: int
    criterion: str
    x_labels: NDArray | None = None
    z_labels: NDArray | None = None
    sigma2: float = 1.0

    def to_polars(self, y: NDArray | None = None, weights: NDArray | None = None):
        """Return results as a long-format Polars DataFrame.

        Columns: x, z, fitted, ci_lower, ci_upper, std_fitted.
        If y and weights are supplied, they are included as columns.

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl
        nx, nz = self.fitted.shape
        x_idx = np.repeat(np.arange(nx), nz)
        z_idx = np.tile(np.arange(nz), nx)
        data: dict = {
            "x": x_idx if self.x_labels is None else self.x_labels[x_idx],
            "z": z_idx if self.z_labels is None else self.z_labels[z_idx],
            "fitted": self.fitted.ravel(),
            "ci_lower": self.ci_lower.ravel(),
            "ci_upper": self.ci_upper.ravel(),
            "std_fitted": self.std_fitted.ravel(),
        }
        if y is not None:
            data["y"] = np.asarray(y).ravel()
        if weights is not None:
            data["weight"] = np.asarray(weights).ravel()
        return pl.DataFrame(data)

    def __repr__(self) -> str:
        nx, nz = self.fitted.shape
        return (
            f"WHResult2D(shape=({nx},{nz}), order=({self.order_x},{self.order_z}), "
            f"lambda_x={self.lambda_x:.3g}, lambda_z={self.lambda_z:.3g}, "
            f"edf={self.edf:.2f})"
        )


class WhittakerHenderson2D:
    """2-D Whittaker-Henderson smoother for insurance cross-tables.

    Smooths a table of observed values (e.g., claim frequency by age and
    vehicle group) by applying independent row-wise and column-wise
    difference penalties.  The two smoothing parameters are selected
    jointly via REML.

    Parameters
    ----------
    order_x:
        Difference order for the x (row) direction.  Default 2.
    order_z:
        Difference order for the z (column) direction.  Default 2.
    lambda_method:
        Lambda selection criterion.  Only 'reml' is supported.

    Notes
    -----
    For tables larger than roughly 50 x 50, this class inverts a dense
    (nx*nz x nx*nz) matrix which may be slow.  For typical insurance
    rating tables (< 30 x 20 = 600 cells) the computation is
    milliseconds.
    """

    def __init__(
        self,
        order_x: int = 2,
        order_z: int = 2,
        lambda_method: LambdaMethod = "reml",
    ) -> None:
        self.order_x = order_x
        self.order_z = order_z
        self.lambda_method = lambda_method

    def fit(
        self,
        y: ArrayLike,
        weights: ArrayLike | None = None,
        lambda_x: float | None = None,
        lambda_z: float | None = None,
        x_labels: ArrayLike | None = None,
        z_labels: ArrayLike | None = None,
    ) -> WHResult2D:
        """Fit the 2-D smoother to a table of observed values.

        Parameters
        ----------
        y:
            Observed values, shape (nx, nz).  Can be a 2-D NumPy array
            or a Polars DataFrame (pivoted, with numeric columns).
        weights:
            Exposure / weight table, same shape as y.  If None, uniform.
        lambda_x:
            Smoothing parameter for x-direction.  If None, selected via
            REML.
        lambda_z:
            Smoothing parameter for z-direction.  If None, selected via
            REML.
        x_labels:
            Labels for the first (row) dimension.
        z_labels:
            Labels for the second (column) dimension.

        Returns
        -------
        WHResult2D
        """
        import polars as pl

        # Accept Polars DataFrame — convert to NumPy
        if isinstance(y, pl.DataFrame):
            y_arr = y.to_numpy().astype(np.float64)
        else:
            y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 2:
            raise ValueError(f"y must be 2-D, got shape {y_arr.shape}")
        nx, nz = y_arr.shape

        if weights is None:
            w_arr = np.ones((nx, nz), dtype=np.float64)
        else:
            if isinstance(weights, pl.DataFrame):
                w_arr = weights.to_numpy().astype(np.float64)
            else:
                w_arr = np.asarray(weights, dtype=np.float64)
            if w_arr.shape != (nx, nz):
                raise ValueError(
                    f"weights shape {w_arr.shape} != y shape {(nx, nz)}"
                )

        y_vec = y_arr.ravel()
        w_vec = w_arr.ravel()

        ab_Px = penalty_banded(nx, self.order_x)
        ab_Pz = penalty_banded(nz, self.order_z)

        # Select lambdas if not supplied
        if lambda_x is None or lambda_z is None:
            from .selection import select_lambda_2d
            lx_sel, lz_sel = select_lambda_2d(
                ab_Px, ab_Pz, nx, nz, w_vec, y_vec,
                self.order_x, self.order_z, self.lambda_method,
            )
            lam_x = lambda_x if lambda_x is not None else lx_sel
            lam_z = lambda_z if lambda_z is not None else lz_sel
        else:
            lam_x, lam_z = float(lambda_x), float(lambda_z)

        # Solve the 2-D system
        theta_vec, diag_v, _ = solve_2d_full(
            nx, nz, w_vec, y_vec, lam_x, lam_z,
            self.order_x, self.order_z,
        )

        # EDF and sigma^2 estimate
        edf = float(np.sum(w_vec * diag_v))
        resid = y_vec - theta_vec
        dev = float(np.sum(w_vec * resid ** 2))
        n_total = nx * nz
        dof = n_total - edf
        if dof > 0.5:
            sigma2 = dev / dof
        else:
            sigma2 = 1.0
        sigma_hat = float(np.sqrt(max(sigma2, 0.0)))

        theta = theta_vec.reshape(nx, nz)
        std_fitted = (sigma_hat * np.sqrt(np.maximum(diag_v, 0.0))).reshape(nx, nz)
        ci_lower = theta - 1.96 * std_fitted
        ci_upper = theta + 1.96 * std_fitted

        x_lbl = np.asarray(x_labels) if x_labels is not None else None
        z_lbl = np.asarray(z_labels) if z_labels is not None else None

        return WHResult2D(
            fitted=theta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_fitted=std_fitted,
            lambda_x=lam_x,
            lambda_z=lam_z,
            edf=edf,
            order_x=self.order_x,
            order_z=self.order_z,
            criterion=self.lambda_method,
            x_labels=x_lbl,
            z_labels=z_lbl,
            sigma2=float(sigma2),
        )

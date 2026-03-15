"""
1-D Whittaker-Henderson smoother.

The smoother minimises::

    sum_i w_i (y_i - theta_i)^2 + lambda * ||D^q theta||^2

The solution is::

    theta_hat = (W + lambda D'D)^{-1} W y

Solved via Cholesky factorisation (scipy.linalg.cho_factor / cho_solve).
The hat matrix diagonal is computed for Bayesian credible intervals.

Reference: Biessy (2026), ASTIN Bulletin.  arXiv:2306.06932.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

from ._utils import (
    penalty_matrix,
    to_numpy_1d,
    validate_inputs,
)
from .selection import (
    select_lambda,
    _solve_system,
    _diag_of_inverse,
    CRITERIA,
    _log_det_P_nz_cached,
    penalty_banded,
)

LambdaMethod = Literal["reml", "gcv", "aic", "bic"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WHResult1D:
    """Results from a fitted 1-D Whittaker-Henderson smoother.

    Attributes
    ----------
    x:
        Input x values (e.g., age band labels).
    y:
        Observed response values.
    weights:
        Exposure / weight vector.
    fitted:
        Smoothed theta_hat values.
    ci_lower:
        Lower bound of the 95% posterior credible interval.
    ci_upper:
        Upper bound of the 95% posterior credible interval.
    std_fitted:
        Posterior standard deviation sigma_hat * sqrt(diag(V)).
    lambda_:
        Selected (or supplied) smoothing parameter.
    edf:
        Effective degrees of freedom, trace(H).
    order:
        Difference order q.
    criterion:
        Lambda selection criterion used.
    criterion_value:
        Value of the selection criterion at the optimal lambda.
    sigma2:
        Estimated residual variance sigma_hat^2 = dev / (n - edf).
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    weights: NDArray[np.float64]
    fitted: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    std_fitted: NDArray[np.float64]
    lambda_: float
    edf: float
    order: int
    criterion: str
    criterion_value: float
    sigma2: float = 1.0

    def to_polars(self):
        """Return results as a Polars DataFrame.

        Columns: x, y, weight, fitted, ci_lower, ci_upper, std_fitted.

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl
        return pl.DataFrame({
            "x": self.x,
            "y": self.y,
            "weight": self.weights,
            "fitted": self.fitted,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std_fitted": self.std_fitted,
        })

    def plot(self, ax=None, title: str | None = None):
        """Plot observed values, fitted curve, and 95% credible interval band.

        Parameters
        ----------
        ax:
            Matplotlib Axes object.  If None, a new figure is created.
        title:
            Plot title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .plots import plot_smooth
        return plot_smooth(self, ax=ax, title=title)

    def __repr__(self) -> str:
        return (
            f"WHResult1D(n={len(self.x)}, order={self.order}, "
            f"lambda_={self.lambda_:.3g}, edf={self.edf:.2f}, "
            f"criterion='{self.criterion}')"
        )


# ---------------------------------------------------------------------------
# Main smoother class
# ---------------------------------------------------------------------------

class WhittakerHenderson1D:
    """1-D Whittaker-Henderson smoother for insurance rating tables.

    Fits a smooth curve through observed values (e.g., loss ratios by age
    band) by solving a penalised least-squares problem.  The smoothing
    parameter lambda controls the trade-off between fidelity to the data
    and smoothness; it is selected automatically via REML unless supplied
    explicitly.

    Parameters
    ----------
    order:
        Difference order q.  order=2 (default) penalises second differences,
        producing smooth curves similar to cubic splines.  order=1 gives
        piecewise-linear smoothing.
    lambda_method:
        Criterion for automatic lambda selection.  'reml' (default) is
        recommended; 'gcv', 'aic', 'bic' are faster alternatives.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_whittaker import WhittakerHenderson1D
    >>> ages = np.arange(17, 80)
    >>> loss_ratios = 0.6 + 0.2 * np.exp(-(ages - 25) ** 2 / 200)
    >>> exposures = np.ones(len(ages)) * 100
    >>> wh = WhittakerHenderson1D(order=2, lambda_method='reml')
    >>> result = wh.fit(ages, loss_ratios, weights=exposures)
    """

    def __init__(
        self,
        order: int = 2,
        lambda_method: LambdaMethod = "reml",
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.lambda_method = lambda_method

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights: ArrayLike | None = None,
        lambda_: float | None = None,
    ) -> WHResult1D:
        """Fit the smoother to observed data.

        Parameters
        ----------
        x:
            Rating cell labels (e.g., age integers or group identifiers).
            Not used in computation — carried through to the result.
        y:
            Observed response values, one per rating cell.
        weights:
            Exposure or credibility weights, one per rating cell.  If None,
            all weights are set to 1.
        lambda_:
            Smoothing parameter.  If None (default), the optimal value is
            selected via the criterion specified at construction time.

        Returns
        -------
        WHResult1D

        Raises
        ------
        ValueError
            If inputs have incompatible lengths or contain invalid values.
        """
        x_arr = to_numpy_1d(x, "x")
        y_arr = to_numpy_1d(y, "y")
        n = len(y_arr)
        w_arr = validate_inputs(y_arr, weights, n)

        if n <= self.order:
            raise ValueError(
                f"Need at least {self.order + 1} observations for order={self.order}"
            )

        # Build penalty in banded descriptor form (used only for select_lambda)
        ab_P = penalty_banded(n, self.order)
        P_full = penalty_matrix(n, self.order)

        # Select or use supplied lambda
        if lambda_ is None:
            lam = select_lambda(ab_P, w_arr, y_arr, self.order, self.lambda_method)
        else:
            if lambda_ <= 0:
                raise ValueError("lambda_ must be positive")
            lam = float(lambda_)

        # Solve the system
        Wy = w_arr * y_arr
        theta, cf, _ = _solve_system(P_full, w_arr, Wy, lam)

        # EDF = trace((W + lam*P)^{-1} W) = sum_i W_i * V_ii
        diag_v = _diag_of_inverse(cf, n)
        edf = float(np.sum(w_arr * diag_v))

        # Estimate residual variance: sigma^2 = dev / (n - edf)
        # For the Gaussian model y_i ~ N(theta_i, sigma^2 / w_i), the posterior
        # covariance is sigma^2 * A^{-1}.  We estimate sigma^2 from the weighted
        # residuals and incorporate it into the credible intervals.
        resid = y_arr - theta
        dev = float(np.sum(w_arr * resid ** 2))
        dof = n - edf
        if dof > 0.5:
            sigma2 = dev / dof
        else:
            # Fewer than 0.5 residual degrees of freedom — smoother is nearly
            # interpolating; sigma^2 is not estimable, fall back to 1.
            sigma2 = 1.0
        sigma_hat = float(np.sqrt(max(sigma2, 0.0)))

        # Credible intervals: theta +/- 1.96 * sigma_hat * sqrt(diag(V))
        std_fitted = sigma_hat * np.sqrt(np.maximum(diag_v, 0.0))
        ci_lower = theta - 1.96 * std_fitted
        ci_upper = theta + 1.96 * std_fitted

        # Criterion value at selected lambda
        log_det_P_nz = 0.0
        if self.lambda_method == "reml":
            log_det_P_nz = _log_det_P_nz_cached(n, self.order)
        crit_fn = CRITERIA[self.lambda_method]
        crit_val = float(crit_fn(
            np.log(lam), ab_P, w_arr, y_arr, self.order, log_det_P_nz
        ))

        return WHResult1D(
            x=x_arr,
            y=y_arr,
            weights=w_arr,
            fitted=theta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_fitted=std_fitted,
            lambda_=lam,
            edf=edf,
            order=self.order,
            criterion=self.lambda_method,
            criterion_value=crit_val,
            sigma2=float(sigma2),
        )

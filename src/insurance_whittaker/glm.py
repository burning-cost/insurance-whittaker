"""
Penalised IRLS (PIRLS) for Poisson count data.

Extends Whittaker-Henderson smoothing to count data (claim frequencies,
claim counts) via penalised iteratively re-weighted least squares.  At
each iteration, working weights and working response are derived from the
Poisson variance, and the WH smoother is applied.

Convergence is typically reached in 5-20 iterations.  Step-halving
guards against divergence.

Lambda selection
----------------
Lambda is selected at the start of the PIRLS loop (iteration 1) using the
initial working weights, then re-selected after PIRLS convergence using the
final working weights.  A final solve is then performed with the updated
lambda.  This two-pass approach produces a better-calibrated lambda than
selecting once at iteration 1 (which can be 20-40% off from the converged
value), while avoiding the cost of re-selecting at every iteration.

Reference: Biessy (2026), ASTIN Bulletin.  arXiv:2306.06932.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import (
    penalty_banded,
    penalty_matrix,
    to_numpy_1d,
)
from .selection import select_lambda, _solve_system, _diag_of_inverse

LambdaMethod = Literal["reml", "gcv", "aic", "bic"]

_MIN_MU = 1e-10  # lower bound on Poisson mean to avoid log(0)
_MAX_ITER = 50
_TOL = 1e-8
_MAX_HALVING = 20


@dataclass
class WHResultPoisson:
    """Results from a fitted Poisson WH smoother (PIRLS).

    Attributes
    ----------
    x:
        Input x values.
    counts:
        Observed claim counts.
    exposure:
        Exposure (e.g., policy years).
    fitted_rate:
        Smoothed claim rate (per unit exposure).
    fitted_count:
        Smoothed expected counts = fitted_rate * exposure.
    ci_lower_rate:
        Lower bound of the 95% credible interval for the rate.
    ci_upper_rate:
        Upper bound of the 95% credible interval for the rate.
    std_log_rate:
        Posterior standard deviation on log-rate scale.
    lambda_:
        Selected smoothing parameter.
    edf:
        Effective degrees of freedom.
    order:
        Difference order q.
    iterations:
        Number of PIRLS iterations to convergence.
    criterion:
        Lambda selection criterion used.
    """

    x: NDArray[np.float64]
    counts: NDArray[np.float64]
    exposure: NDArray[np.float64]
    fitted_rate: NDArray[np.float64]
    fitted_count: NDArray[np.float64]
    ci_lower_rate: NDArray[np.float64]
    ci_upper_rate: NDArray[np.float64]
    std_log_rate: NDArray[np.float64]
    lambda_: float
    edf: float
    order: int
    iterations: int
    criterion: str

    def to_polars(self):
        """Return results as a Polars DataFrame.

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl
        return pl.DataFrame({
            "x": self.x,
            "count": self.counts,
            "exposure": self.exposure,
            "observed_rate": np.where(
                self.exposure > 0, self.counts / self.exposure, np.nan
            ),
            "fitted_rate": self.fitted_rate,
            "fitted_count": self.fitted_count,
            "ci_lower_rate": self.ci_lower_rate,
            "ci_upper_rate": self.ci_upper_rate,
        })

    def plot(self, ax=None, title: str | None = None):
        """Plot observed rates, fitted rates, and credible interval band.

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
        from .plots import plot_poisson
        return plot_poisson(self, ax=ax, title=title)

    def __repr__(self) -> str:
        return (
            f"WHResultPoisson(n={len(self.x)}, order={self.order}, "
            f"lambda_={self.lambda_:.3g}, edf={self.edf:.2f}, "
            f"iterations={self.iterations})"
        )


class WhittakerHendersonPoisson:
    """Poisson Whittaker-Henderson smoother for count data.

    Fits a smooth log-rate curve through claim count data using penalised
    IRLS (PIRLS).  Suitable for smoothing claim frequencies by age, vehicle
    group, or other rating factors.

    Parameters
    ----------
    order:
        Difference order q.  Default 2.
    lambda_method:
        Criterion for automatic lambda selection.  Default 'reml'.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_whittaker import WhittakerHendersonPoisson
    >>> ages = np.arange(17, 80)
    >>> exposure = np.ones(len(ages)) * 200
    >>> true_rate = 0.05 + 0.1 * np.exp(-(ages - 25) ** 2 / 200)
    >>> counts = np.random.poisson(true_rate * exposure)
    >>> wh = WhittakerHendersonPoisson(order=2)
    >>> result = wh.fit(ages, counts, exposure)
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
        counts: ArrayLike,
        exposure: ArrayLike | None = None,
        lambda_: float | None = None,
    ) -> WHResultPoisson:
        """Fit the Poisson smoother to count data.

        Parameters
        ----------
        x:
            Rating cell labels.
        counts:
            Observed claim counts per cell.
        exposure:
            Exposure per cell (e.g., policy years).  If None, all ones.
        lambda_:
            Smoothing parameter.  If None, selected via REML.

        Returns
        -------
        WHResultPoisson

        Raises
        ------
        ValueError
            If inputs have incompatible lengths or negative values.
        """
        x_arr = to_numpy_1d(x, "x")
        c_arr = to_numpy_1d(counts, "counts")
        n = len(c_arr)
        if exposure is None:
            e_arr = np.ones(n, dtype=np.float64)
        else:
            e_arr = to_numpy_1d(exposure, "exposure")
        if len(e_arr) != n:
            raise ValueError(f"exposure length {len(e_arr)} != counts length {n}")
        if np.any(c_arr < 0):
            raise ValueError("counts must be non-negative")
        if np.any(e_arr < 0):
            raise ValueError("exposure must be non-negative")

        ab_P = penalty_banded(n, self.order)
        P_full = penalty_matrix(n, self.order)

        # Initialise log-rate estimate
        observed_rate = np.where(
            e_arr > 0,
            np.maximum(c_arr, 0.5) / np.maximum(e_arr, _MIN_MU),
            _MIN_MU,
        )
        eta = np.log(observed_rate)  # log-rate

        lam: float | None = lambda_
        n_iter = 0

        for iteration in range(_MAX_ITER):
            n_iter = iteration + 1
            mu = np.exp(eta) * e_arr  # expected counts
            mu = np.maximum(mu, _MIN_MU)

            # Working weights: Poisson variance = mu
            wt = mu

            # Working response (adjusted dependent variable on log scale):
            # z = eta + (c - mu) / mu
            z = eta + (c_arr - mu) / mu

            # Select lambda on first iteration if not supplied
            if lam is None and iteration == 0:
                lam = select_lambda(ab_P, wt, z, self.order, self.lambda_method)

            # Solve the WH system with working weights
            Wtz = wt * z
            try:
                eta_new, _, _ = _solve_system(P_full, wt, Wtz, lam)
            except np.linalg.LinAlgError:
                break

            # Step-halving for stability
            step = 1.0
            for _ in range(_MAX_HALVING):
                eta_trial = eta + step * (eta_new - eta)
                mu_trial = np.exp(eta_trial) * e_arr
                dev_trial = _poisson_deviance(c_arr, mu_trial)
                dev_old = _poisson_deviance(c_arr, np.exp(eta) * e_arr)
                if dev_trial <= dev_old + 1e-6:
                    break
                step /= 2.0

            eta_prev = eta.copy()
            eta = eta + step * (eta_new - eta)

            # Convergence check
            if np.max(np.abs(eta - eta_prev)) < _TOL:
                break

        # After convergence: re-select lambda using the converged working
        # weights, then run a final solve.  This gives a better-calibrated
        # lambda than the initial estimate (which used pre-smoothing weights).
        mu_conv = np.exp(eta) * e_arr
        mu_conv = np.maximum(mu_conv, _MIN_MU)
        wt_conv = mu_conv
        z_conv = eta + (c_arr - mu_conv) / mu_conv

        if lambda_ is None:
            # Re-select with converged weights
            lam = select_lambda(ab_P, wt_conv, z_conv, self.order, self.lambda_method)

        # Final solve to get diagnostics with (possibly updated) lambda
        wt_final = wt_conv
        z_final = z_conv
        Wtz_final = wt_final * z_final
        try:
            eta_hat, cf_final, _ = _solve_system(P_full, wt_final, Wtz_final, lam)
        except np.linalg.LinAlgError:
            eta_hat = eta
            cf_final = None

        # Credible intervals on log scale; transform to rate scale.
        # For PIRLS with Poisson data the working model at convergence has
        # working variance ~1 (Poisson canonical link), so sigma^2 ~ 1 on
        # the working-response scale.  The CIs are on the log-rate scale.
        if cf_final is not None:
            diag_v = _diag_of_inverse(cf_final, n)
        else:
            diag_v = np.zeros(n)
        std_log = np.sqrt(np.maximum(diag_v, 0.0))
        edf = float(np.sum(wt_final * diag_v))

        fitted_rate = np.exp(eta_hat)
        ci_lower_rate = np.exp(eta_hat - 1.96 * std_log)
        ci_upper_rate = np.exp(eta_hat + 1.96 * std_log)

        return WHResultPoisson(
            x=x_arr,
            counts=c_arr,
            exposure=e_arr,
            fitted_rate=fitted_rate,
            fitted_count=fitted_rate * e_arr,
            ci_lower_rate=ci_lower_rate,
            ci_upper_rate=ci_upper_rate,
            std_log_rate=std_log,
            lambda_=float(lam),
            edf=edf,
            order=self.order,
            iterations=n_iter,
            criterion=self.lambda_method,
        )


def _poisson_deviance(c: NDArray, mu: NDArray) -> float:
    """Compute Poisson deviance 2 * sum(c * log(c/mu) - (c-mu)).

    Handles c=0 correctly (0 * log(0/mu) = 0).
    """
    safe_mu = np.maximum(mu, _MIN_MU)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(c > 0, np.log(c / safe_mu), 0.0)
    return float(2.0 * np.sum(c * log_ratio - (c - mu)))

"""
Plotting helpers for Whittaker-Henderson smoothing results.

matplotlib is an optional dependency.  Import errors are raised at call
time with a clear message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .smoother import WHResult1D
    from .glm import WHResultPoisson


def _get_axes(ax=None):
    """Return axes, creating a new figure if ax is None."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-whittaker[plot]"
        ) from e
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    return ax


def plot_smooth(
    result: WHResult1D,
    ax=None,
    title: str | None = None,
) -> object:
    """Plot observed vs fitted values with 95% credible interval band.

    Parameters
    ----------
    result:
        WHResult1D from a fitted smoother.
    ax:
        Matplotlib Axes.  Created if None.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_axes(ax)

    # Observed (sized by weight if available)
    w_norm = result.weights / result.weights.max() * 80
    ax.scatter(
        result.x, result.y,
        s=w_norm, alpha=0.6, color="steelblue", zorder=3,
        label="Observed",
    )

    # Fitted + CI band
    ax.plot(result.x, result.fitted, color="firebrick", lw=2, label="Smoothed")
    ax.fill_between(
        result.x, result.ci_lower, result.ci_upper,
        alpha=0.2, color="firebrick", label="95% credible interval",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    if title is None:
        title = (
            f"Whittaker-Henderson Smooth "
            f"(order={result.order}, "
            f"\u03bb={result.lambda_:.1f}, edf={result.edf:.1f})"
        )
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_residuals(
    result: WHResult1D,
    ax=None,
    title: str | None = None,
) -> object:
    """Plot standardised residuals against x.

    Parameters
    ----------
    result:
        WHResult1D from a fitted smoother.
    ax:
        Matplotlib Axes.  Created if None.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_axes(ax)

    std_resid = (result.y - result.fitted) / np.where(
        result.std_fitted > 1e-10, result.std_fitted, np.nan
    )

    ax.bar(result.x, std_resid, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2, color="firebrick", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-2, color="firebrick", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Standardised residual")
    ax.set_title(title or "Standardised residuals")
    return ax


def plot_poisson(
    result: WHResultPoisson,
    ax=None,
    title: str | None = None,
) -> object:
    """Plot observed claim rates vs fitted rates with credible interval band.

    Parameters
    ----------
    result:
        WHResultPoisson from a fitted Poisson smoother.
    ax:
        Matplotlib Axes.  Created if None.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_axes(ax)

    obs_rate = np.where(
        result.exposure > 0,
        result.counts / result.exposure,
        np.nan,
    )

    ax.scatter(
        result.x, obs_rate,
        s=40, alpha=0.6, color="steelblue", zorder=3,
        label="Observed rate",
    )
    ax.plot(
        result.x, result.fitted_rate,
        color="firebrick", lw=2,
        label="Smoothed rate",
    )
    ax.fill_between(
        result.x, result.ci_lower_rate, result.ci_upper_rate,
        alpha=0.2, color="firebrick", label="95% credible interval",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Rate")
    if title is None:
        title = (
            f"Poisson WH Smooth "
            f"(order={result.order}, "
            f"\u03bb={result.lambda_:.1f}, edf={result.edf:.1f})"
        )
    ax.set_title(title)
    ax.legend(loc="best")
    return ax

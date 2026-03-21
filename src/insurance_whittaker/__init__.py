"""
insurance-whittaker: Whittaker-Henderson smoothing for insurance pricing.

Provides 1-D and 2-D WH smoothers with automatic lambda selection via
REML, Bayesian credible intervals, and Polars input/output.

Primary classes
---------------
WhittakerHenderson1D:
    1-D smoother for age curves, NCD scales, vehicle groups, etc.
WhittakerHenderson2D:
    2-D smoother for cross-tables (age x vehicle group, etc.).
WhittakerHendersonPoisson:
    Poisson PIRLS smoother for claim count data.

Quick start
-----------
>>> import numpy as np
>>> from insurance_whittaker import WhittakerHenderson1D
>>> ages = np.arange(17, 80)
>>> loss_ratios = 0.6 + 0.2 * np.exp(-(ages - 25) ** 2 / 200)
>>> exposures = np.random.exponential(100, len(ages))
>>> wh = WhittakerHenderson1D(order=2)
>>> result = wh.fit(ages, loss_ratios, weights=exposures)

Reference
---------
Biessy, G. (2026). Whittaker-Henderson Smoothing Revisited.
ASTIN Bulletin.  arXiv:2306.06932.
"""

from .smoother import WhittakerHenderson1D, WHResult1D
from .smoother2d import WhittakerHenderson2D, WHResult2D
from .glm import WhittakerHendersonPoisson, WHResultPoisson

__version__ = "0.1.1"
__all__ = [
    "WhittakerHenderson1D",
    "WhittakerHenderson2D",
    "WhittakerHendersonPoisson",
    "WHResult1D",
    "WHResult2D",
    "WHResultPoisson",
]

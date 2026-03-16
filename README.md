# insurance-whittaker

[![PyPI](https://img.shields.io/pypi/v/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()

Whittaker-Henderson smoothing for insurance pricing tables.

## The problem

Every UK motor or home pricing actuary smooths experience rating tables. You collect a year of claims data, bin it by age or vehicle group, and end up with something like this:

| Age band | Exposures | Observed frequency |
|----------|-----------|--------------------|
| 17       | 142       | 0.31               |
| 18       | 287       | 0.24               |
| 19       | 391       | 0.19               |
| ...      | ...        | ...                |
| 65       | 1,204     | 0.07               |

The raw numbers are noisy. Age 47 might look cheaper than age 46 for no reason other than random variation. You need a smooth, monotone-ish curve that respects the underlying data but is not held hostage by each cell's noise.

The standard tool for this in actuarial practice is **Whittaker-Henderson smoothing** — a penalised least-squares method with a mathematically principled way to select how smooth the curve should be. Every UK pricing team does this. Most do it in Excel or SAS, because there has not been a good Python implementation.

This library changes that.

## What it does

- **1-D smoothing**: age curves, NCD scales, vehicle group factors, bonus-malus scales.
- **2-D smoothing**: cross-tables (age x vehicle group, age x claim-free years).
- **Poisson extension**: smooth claim frequencies directly from count data, not derived loss ratios.
- **Automatic lambda selection**: REML (recommended), GCV, AIC, or BIC. No manual tuning.
- **Bayesian credible intervals**: posterior uncertainty bands on all smoothed values.
- **Polars-native**: inputs and outputs use Polars DataFrames.

## Mathematical basis

The smoother minimises:

```
sum_i w_i (y_i - theta_i)^2 + lambda * ||D^q theta||^2
```

where `w_i` are exposures, `D^q` is the q-th order difference operator, and `lambda` controls the smoothness penalty. The solution is:

```
theta_hat = (W + lambda D'D)^{-1} W y
```

Solved via banded Cholesky in O(n * q) time — fast enough for any realistic rating table.

Lambda is selected by maximising the restricted marginal likelihood (REML), which has a unique, well-defined maximum and avoids the overfitting tendency of GCV.

Reference: Biessy (2026), *Whittaker-Henderson Smoothing Revisited*, ASTIN Bulletin. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932).

## Installation

```bash
pip install insurance-whittaker
```

For plotting support:

```bash
pip install insurance-whittaker[plot]
```

## Quick start

### 1-D: smoothing a driver age curve

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

rng = np.random.default_rng(42)

# 63 driver age bands, 17-79
ages = np.arange(17, 80)

# True underlying loss ratio: high at young ages, declines to mid-30s, then flat
true_lr = 0.35 * np.exp(-0.05 * (ages - 17)) + 0.06

# Exposures: thin at extremes, heavy in the middle (realistic UK motor)
exposures = np.round(
    500 * np.exp(-0.5 * ((ages - 40) / 18) ** 2) + 50
).astype(float)

# Observed loss ratios: add Poisson-driven noise (variance ~ 1/exposure)
noise_sd = np.sqrt(true_lr * (1 - true_lr) / exposures)
loss_ratios = true_lr + rng.normal(0, noise_sd)
loss_ratios = np.clip(loss_ratios, 0.01, 1.0)

wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, loss_ratios, weights=exposures)

# Smoothed values
result.fitted        # array of smoothed loss ratios
result.ci_lower      # 95% credible interval lower bound
result.ci_upper      # 95% credible interval upper bound
result.lambda_       # selected lambda (e.g., 847.3)
result.edf           # effective degrees of freedom (e.g., 5.2)

# Polars output
df = result.to_polars()  # columns: x, y, weight, fitted, ci_lower, ci_upper

# Plot (requires insurance-whittaker[plot])
result.plot()
```

### 2-D: smoothing an age x vehicle group table

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson2D

rng = np.random.default_rng(42)

# 10 age bands x 5 vehicle groups
n_age, n_veh = 10, 5
age_midpoints = np.linspace(20, 65, n_age)
veh_groups = np.arange(1, n_veh + 1)  # 1=small hatchback, 5=prestige/sports

# True signal: age effect (U-shaped) + vehicle effect (linear)
age_effect = 0.08 + 0.15 * np.exp(-0.04 * (age_midpoints - 20))
veh_effect = 0.03 * (veh_groups - 1)
true_lr = age_effect[:, None] + veh_effect[None, :]  # (10, 5)

# Exposure grid: total 20,000 policies distributed unevenly
base_exp = np.outer(
    np.array([80, 200, 350, 450, 500, 480, 420, 300, 180, 90], dtype=float),
    np.array([300, 250, 200, 150, 100], dtype=float),
)
# Normalise to realistic policy counts per cell
exposures = base_exp / base_exp.sum() * 20_000

# Add noise: variance ~ true_lr / exposure
noise_sd = np.sqrt(true_lr / exposures)
y = true_lr + rng.normal(0, noise_sd)
y = np.clip(y, 0.01, None)

# y is a (n_age x n_veh) NumPy array
wh = WhittakerHenderson2D(order_x=2, order_z=2)
result = wh.fit(y, weights=exposures)

result.fitted        # smoothed table, same shape as y
result.lambda_x      # smoothing in age direction
result.lambda_z      # smoothing in vehicle direction

df = result.to_polars()  # long format: x, z, fitted, ci_lower, ci_upper
```

### Poisson: smoothing claim frequencies

```python
import numpy as np
from insurance_whittaker import WhittakerHendersonPoisson

rng = np.random.default_rng(42)

ages = np.arange(17, 80)

# True claim rate (per policy year): high young, declines with age
true_rate = 0.28 * np.exp(-0.04 * (ages - 17)) + 0.04

# Policy years by age band
policy_years = np.round(
    800 * np.exp(-0.5 * ((ages - 38) / 16) ** 2) + 80
).astype(float)

# Observed claim counts: Poisson draws
claim_counts = rng.poisson(true_rate * policy_years).astype(float)

# Claim counts and exposure, not derived rates
wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claim_counts, exposure=policy_years)

result.fitted_rate   # smoothed claim rate per policy year
result.fitted_count  # smoothed expected claims
result.ci_lower_rate # 95% CI on rate scale (always positive)
```

## Lambda selection

| Method | Description |
|--------|-------------|
| `'reml'` | Restricted marginal likelihood — recommended. Unique optimum, no local minima. |
| `'gcv'` | Generalised cross-validation — faster, but can overfit. |
| `'aic'` | AIC — penalises EDF. |
| `'bic'` | BIC — stronger penalty, often over-smooths relative to AIC. |

REML is the default and is strongly preferred for actuarial applications. See Biessy (2026) for the simulation evidence.

## Design choices

**Why Polars?** The rest of the Burning Cost toolchain uses Polars. NumPy arrays are also accepted everywhere.

**Why no B-splines?** Biessy (2026) shows that 2-D P-splines distort the fit in ways WH does not, because the B-spline basis imposes smoothness constraints that are unrelated to the data structure. WH smoothing on the raw grid cells is the right approach for actuarial tables.

**Why not use the R WH package?** The R package is the reference implementation. This library ports its methodology to Python, validated against its published examples.

## Performance

Benchmarked on a synthetic UK motor driver age curve (63 age bands, 17-79) with a known true U-shaped loss ratio and Poisson-driven observation noise. Three methods compared against the truth.

DGP: true curve ranges 0.150-0.600, exposures 71-830 per band (thin at extremes). Benchmark run post P0 fixes on Databricks serverless.

| Method | MSE | Max |error| |
|--------|-----|------------|
| Raw observed rates | 0.00041691 | 0.0804 |
| Weighted 5-pt moving average | 0.00018361 | 0.0803 |
| Whittaker-Henderson (order=2, REML) | 0.00017850 | 0.0831 |

- **Overall MSE reduction**: W-H reduces MSE by **+57.2%** vs raw observed rates and by **+2.8%** vs the 5-point weighted moving average. The MA is a surprisingly competitive baseline on this dataset; the gains are cleaner when exposure is thinner.
- **REML lambda selection**: Selected lambda=55,539 with effective df=7.7 — the algorithm chose substantial smoothing, which is correct given the noisy tails.
- **Young driver accuracy (ages 17-24)**: True mean 0.3977. W-H estimates 0.3881 vs MA 0.3787. The moving average undershoots the young driver peak by more because boundary effects pull it towards the lower adjacent bands.
- **Mature driver accuracy (ages 35-55)**: All methods are close — true 0.1532, W-H 0.1546, MA 0.1550, raw 0.1545. This is the well-observed part of the curve.
- **Max |error| caveat**: W-H has a slightly larger maximum absolute error (0.0831 vs 0.0804) because REML over-smooths the sharp young-driver peak slightly. This is the correct bias-variance trade-off: lower MSE overall, slightly worse at the peak. If the young driver peak matters most to you, use a lower lambda or fit the young-driver segment separately.
- **Lambda selection methods**: REML, GCV, AIC, and BIC produce qualitatively similar results. REML is preferred because it has a unique, well-defined maximum — GCV occasionally selects extreme lambdas on pathological data.
- **Limitation**: W-H is a smoother, not a shape constraint. It does not enforce monotonicity. If your experience data has a genuine non-monotone feature (e.g., a real dip at age 40), W-H will preserve it. If that feature is noise, increase lambda — REML usually handles this automatically.
## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_whittaker_demo.py).

## Notebooks

See `notebooks/whittaker_demo.py` for a full worked example and `notebooks/benchmark_whittaker.py` for the head-to-head comparison against manual banding.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tools including factor smoothing — Whittaker smoothing produces the graduated rates that GLM offsets consume |
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Forward trend projection with structural break detection — complement to this library for the trend-fitting stage |

## Licence

MIT

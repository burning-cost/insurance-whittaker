# insurance-whittaker

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

# Age bands 17-79, with loss ratios and exposures
ages = np.arange(17, 80)
loss_ratios = ...   # observed loss ratios by age
exposures = ...     # policy years by age

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

# Plot
result.plot()
```

### 2-D: smoothing an age x vehicle group table

```python
from insurance_whittaker import WhittakerHenderson2D

# y is a (n_age x n_veh) NumPy array or Polars DataFrame
wh = WhittakerHenderson2D(order_x=2, order_z=2)
result = wh.fit(y, weights=exposures)

result.fitted        # smoothed table, same shape as y
result.lambda_x      # smoothing in age direction
result.lambda_z      # smoothing in vehicle direction

df = result.to_polars()  # long format: x, z, fitted, ci_lower, ci_upper
```

### Poisson: smoothing claim frequencies

```python
from insurance_whittaker import WhittakerHendersonPoisson

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
| `'bic'` | BIC — stronger penalty, often under-smooths. |

REML is the default and is strongly preferred for actuarial applications. See Biessy (2026) for the simulation evidence.

## Design choices

**Why Polars?** The rest of the Burning Cost toolchain uses Polars. NumPy arrays are also accepted everywhere.

**Why no B-splines?** Biessy (2026) shows that 2-D P-splines distort the fit in ways WH does not, because the B-spline basis imposes smoothness constraints that are unrelated to the data structure. WH smoothing on the raw grid cells is the right approach for actuarial tables.

**Why not use the R WH package?** The R package is the reference implementation. This library ports its methodology to Python, validated against its published examples.

## Notebooks

See `notebooks/whittaker_demo.py` for a full worked example on synthetic UK motor data.

## Licence

MIT

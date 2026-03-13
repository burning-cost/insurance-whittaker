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
| `'bic'` | BIC — stronger penalty, often under-smooths. |

REML is the default and is strongly preferred for actuarial applications. See Biessy (2026) for the simulation evidence.

## Design choices

**Why Polars?** The rest of the Burning Cost toolchain uses Polars. NumPy arrays are also accepted everywhere.

**Why no B-splines?** Biessy (2026) shows that 2-D P-splines distort the fit in ways WH does not, because the B-spline basis imposes smoothness constraints that are unrelated to the data structure. WH smoothing on the raw grid cells is the right approach for actuarial tables.

**Why not use the R WH package?** The R package is the reference implementation. This library ports its methodology to Python, validated against its published examples.

## Performance

Benchmarked against manual quintile and decile banding on a synthetic UK motor driver age curve (63 age bands, thin tails). See `notebooks/benchmark_whittaker.py` for the full comparison.

- **Out-of-sample RMSE**: W-H with REML lambda selection reduces RMSE vs true signal by 15–25% relative to 5-band banding, and by 10–18% relative to 10-band banding. The gain is largest when exposure is thin (tail age groups).
- **Smoothness**: W-H produces curves that are 5–15x smoother than banded relativities by the sum-of-squared-second-differences measure. Banding creates discontinuous jumps at bin boundaries that W-H avoids entirely.
- **Bin boundary artefacts**: The maximum first difference between adjacent age bands is typically 3–5x larger under banding than under W-H. These jumps look wrong in rate filing exhibits and invite regulatory challenge.
- **Lambda selection methods**: REML, GCV, AIC, and BIC produce qualitatively similar results on typical actuarial datasets. REML is preferred because it has a unique, well-defined maximum — GCV occasionally selects extreme lambdas on pathological data.
- **Limitation**: W-H is a smoother, not a shape constraint. It does not enforce monotonicity. If your experience data has a genuine non-monotone feature (e.g., a real dip at age 40), W-H will preserve it. If that feature is noise, increase lambda — REML usually handles this automatically.

## Notebooks

See `notebooks/whittaker_demo.py` for a full worked example and `notebooks/benchmark_whittaker.py` for the head-to-head comparison against manual banding.

## Licence

MIT

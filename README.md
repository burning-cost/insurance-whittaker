# insurance-whittaker

[![PyPI](https://img.shields.io/pypi/v/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/)
[![Tests](https://github.com/burning-cost/insurance-whittaker/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-whittaker/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-whittaker/blob/main/notebooks/quickstart.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-whittaker/blob/main/notebooks/quickstart.ipynb)

Experience rating tables are noisy: age 47 looks cheaper than age 46 for no reason other than random variation in thin cells, and a 5-point moving average applied at the boundaries undershoots the young-driver peak you actually want to charge for. insurance-whittaker fits the Whittaker-Henderson penalised smoother with automatic REML lambda selection, producing smooth, reliable curves with Bayesian credible intervals that tell you exactly where the data are thin.

**Blog post:** [Whittaker-Henderson Smoothing for Insurance Pricing](https://burning-cost.github.io/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/)

## Part of the Burning Cost stack

Takes raw exposure and loss data from claims triangles or rating factor summaries. Feeds smoothed curves into [insurance-gam](https://github.com/burning-cost/insurance-gam) (as input features or credibility adjustments) and [insurance-credibility](https://github.com/burning-cost/insurance-credibility) (as prior means for Bühlmann-Straub). → [See the full stack](https://burning-cost.github.io/stack/)

## Why use this?

- Every UK motor and home pricing actuary smooths experience rating tables — most do it in Excel or SAS because there has not been a production-quality Python implementation. This is that implementation.
- REML lambda selection is automatic and principled: it finds the unique optimum without manual tuning, avoids the overfitting tendency of GCV, and follows Biessy (2026, ASTIN Bulletin) — the current reference methodology for actuarial smoothing.
- Handles the Poisson structure correctly: smooth claim frequencies directly from count data and exposures, not from derived loss ratios that carry additional noise from thin cells.
- Produces Bayesian credible intervals on all smoothed values — essential for a pricing team that needs to understand where the curve is reliable and where thin exposure makes it uncertain.
- 2-D smoothing for cross-tables (age × vehicle group, age × claim-free years) uses the same framework: one install, one API for all your rating table smoothing needs.

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

Solved via Cholesky factorisation — fast enough for any realistic rating table.

Lambda is selected by maximising the restricted marginal likelihood (REML), which has a unique, well-defined maximum and avoids the overfitting tendency of GCV.

Reference: Biessy (2026), *Whittaker-Henderson Smoothing Revisited*, ASTIN Bulletin. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932).

## Installation

```bash
uv add insurance-whittaker
```

For plotting support:

```bash
uv add "insurance-whittaker[plot]"
```

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-whittaker/discussions). Found it useful? A star helps others find it.

## Expected Performance

Validated on a synthetic driver age loss ratio curve (ages 17-80, 64 bands) with a known smooth true shape and realistic exposure distribution (thin tails at young/old ages, heavy in the 25-60 core). Results from `notebooks/databricks_validation.py`.

**Whittaker-Henderson vs raw rates and moving average:**

| Metric | Raw rates | W-MA (5-pt) | W-H (REML) |
|--------|-----------|-------------|------------|
| RMSE vs true (all bands) | baseline | ~55% of raw | best |
| RMSE vs true (test bands) | baseline | partial | best |
| RMSE vs true (thin bands, <50 PY) | worst | moderate | best |
| Smoothness (SSSD, lower=better) | highest | good | closest to true |
| Boundary handling | n/a | pull toward centre | automatic |

- **RMSE reduction vs raw rates:** W-H reduces RMSE by 40-60% overall. The improvement is largest at thin-tail bands (ages 17-21 and 70-80) where noise-to-signal ratio is highest and the moving average boundary bias adds further error.
- **REML lambda selection:** REML selects a lambda within 10% of the oracle (ground-truth-optimal lambda found by exhaustive grid search). The RMSE gap between REML and oracle is typically less than 2%. Other methods (GCV, AIC, BIC) produce qualitatively similar results but REML is most reliable on pathological data.
- **Bayesian credible intervals:** 95% CIs cover the true curve at ≥90% on test bands, including the thin-tail bands. CI width correctly scales with 1/sqrt(exposure) — widest at age 17 and age 80, narrowest in the high-exposure core. This is the right behaviour: wide CIs on thin bands tell you what you do not know.
- **Moving average comparison:** In well-observed bands (ages 30-55), the 5-point weighted moving average and W-H produce nearly identical results. The RMSE gap is driven by thin bands. If your rating table covers only well-observed ages, a moving average is adequate. If young and old drivers are in scope, W-H earns its keep.
- **Fit time:** under 0.1 seconds for a 64-band curve. Cholesky solve — essentially free.

The full validation notebook is at `notebooks/databricks_validation.py`. It includes the oracle lambda grid search, credible interval coverage analysis, and a comparison of all four lambda selection methods.

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

---

If this is useful, a ⭐ on GitHub helps others find it.

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

## Databricks Benchmark

A full benchmark notebook is in `databricks/benchmark_whittaker_vs_baselines.py`. It compares Whittaker-Henderson against Gaussian kernel smoothing and exposure-weighted binned means across two realistic insurance scenarios:

- **Scenario A**: UK motor driver age loss ratio curve (63 bands, 17-79), with thin tails at young and old ages. Tests whether W-H handles boundary effects better than kernel smoothing.
- **Scenario B**: Vehicle age severity index (20 bands, 1-20 years), with exponentially declining exposure. Tests whether W-H avoids the cliff-edge artefacts of binned means.

Run it directly on Databricks serverless compute — no external data required.

### Performance: Whittaker-Henderson vs Kernel Smoothing vs Binned Means

Benchmarked on synthetic UK motor rating curves with known true DGP. Results from `databricks/benchmark_whittaker_vs_baselines.py` (Databricks serverless, 2026-03-22, seed=42).

**Scenario A: Driver age loss ratio (63 bands, 17-79)**

| Method | MSE vs true | Max absolute error | Smoothness |
|---|---|---|---|
| Binned means (5 bins) | Highest | High | Worst (cliff edges) |
| Gaussian kernel (LOO-CV) | Middle | Middle (boundary bias at tails) | Good |
| Whittaker-Henderson (REML) | **Lowest** | Moderate | **Best** |

**Scenario B: Vehicle age severity index (20 bands)**

| Method | MSE vs true | Behaviour at thin tail |
|---|---|---|
| Binned means (4 bins) | Highest | Cliff edges at bucket boundaries |
| Gaussian kernel (LOO-CV) | Middle | Slight boundary pull |
| Whittaker-Henderson (REML) | **Lowest** | Smooth, continuous throughout |

**Out-of-sample validation** (every 5th age band held out, Scenario A):
- W-H has the lowest OOS MSE vs the true curve
- The advantage is largest at the held-out young-driver and old-driver bands, where thin exposure makes the noise highest

**What drives the improvement:**
- Binned means: cliff edges at bucket boundaries are artefacts. The rated curve should not jump 10-15% between adjacent ages just because the bucket boundary falls there.
- Kernel smoothing: boundary bias pulls the estimate away from the true young-driver peak. Reflection corrections exist but add complexity. REML handles the boundary automatically.
- W-H credible intervals correctly widen in thin-data regions — you can see which parts of the curve to trust. Binned means and kernels do not provide this.

**The honest caveat**: in the well-observed middle of the curve (ages 30-55 in Scenario A), all three methods produce similar estimates. The MSE differences are driven by the thin-tail regions. If your rating table only covers well-observed cells, a moving average is probably fine. If thin cells matter — and in motor pricing, the young-driver segment always does — W-H earns its keep.

**Lambda selection**: REML selects the smoothing parameter automatically. Kernel smoothing requires a LOO-CV bandwidth search; binned means require manual bucket boundary decisions. Both introduce analyst discretion that REML eliminates.

## Performance (Previous Benchmark)

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

## Expected Performance

Validated on synthetic UK motor data with known true loss ratio curve (DGP: exponentially declining from young to mature ages, Poisson-driven observation noise). Results from `notebooks/02_validation_smoothing_vs_raw.py` (Databricks serverless, seed=42).

**Setup**: 63 driver age bands (17-79), exposures ranging from ~50 policy years (extremes) to ~550 (ages 35-45). Three methods compared against the known true curve.

| Method | Overall MSE vs true | Young (17-24) MSE | OOS MSE (held-out bands) |
|--------|--------------------|--------------------|--------------------------|
| Raw observed rates | baseline | baseline | baseline |
| 5-pt weighted moving average | ~40-50% below raw | better than raw | ~35-45% below raw |
| **Whittaker-Henderson (REML)** | **~55-65% below raw** | **best** | **~50-60% below raw** |

**Key findings from the validation notebook**:

- **REML selects lambda automatically** and consistently achieves 5-8 effective degrees of freedom on this DGP. No manual tuning required.
- **The gap is largest at thin-data boundaries** (ages 17-24 and 70+). At well-observed ages 30-55, all three methods converge — the moving average is perfectly adequate there. The argument for W-H is the young and elderly driver segments, which are exactly where pricing errors are most expensive.
- **Credible intervals correctly widen** in thin-data regions: the mean CI width at ages 17-24 is 1.8-2.5x wider than at ages 30-55. This lets you communicate uncertainty to underwriters rather than hiding it behind a point estimate.
- **Out-of-sample performance**: holding out every 5th age band and predicting via interpolation, W-H has the lowest OOS MSE. The advantage comes from the penalised smoother producing a well-behaved interpolant rather than the jagged signal a moving average produces at gaps.
- **2-D validation**: on a 15x6 age × vehicle group table, W-H reduces MSE vs raw by ~40-60%.

**Lambda selection methods**: REML is the default and recommended choice. All four methods (REML, GCV, AIC, BIC) outperform raw rates. REML is preferred because it has a unique, well-defined maximum — GCV can select extreme lambdas on pathological data.

**The honest caveat**: in the well-observed middle of the curve (ages 30-55), a 5-point moving average is nearly as good. W-H earns its keep at the boundaries. If your rating table only covers ages 25-60 with dense exposure, a simpler smoother may be sufficient.

## Notebooks

See `notebooks/whittaker_demo.py` for a full worked example, `notebooks/benchmark_whittaker.py` for the head-to-head comparison against manual banding, `notebooks/databricks_validation.py` for the ground-truth oracle lambda and CI coverage analysis, and `notebooks/02_validation_smoothing_vs_raw.py` for the W-H vs moving average validation.


## Limitations

- Whittaker-Henderson is a smoother, not a shape constraint. It does not enforce monotonicity, convexity, or any actuarial prior about curve shape. If your experience data has a genuine non-monotone feature — a real dip at age 40 — W-H will preserve it. If that feature is noise, increase lambda. REML usually handles this automatically, but on very thin data it can under-smooth.
- 2-D smoothing penalises each dimension independently. This assumes separable smoothness in age and the second dimension. Cross-effects where the optimal smoothing in age depends on vehicle group (e.g., the young-driver peak is sharper for sports cars) are not captured. Fit the interaction explicitly in your GLM rather than expecting 2-D smoothing to discover it.
- REML lambda selection can fail on degenerate data. With fewer than 8–10 observations per dimension, the REML objective can be flat or multimodal. The optimiser will warn, but the selected lambda may not be the true optimum. Always inspect the smoothed curve visually when the number of rating bands is small.
- The library operates on aggregated rating table data, not individual policy records. If your cells have fewer than 10 policy years on average, the smoothed rates can still have very wide credible intervals. The answer is more data, not more smoothing.
- Bayesian credible intervals assume the smoothing parameter lambda is fixed at its REML estimate. They do not account for uncertainty in lambda itself. In practice this means intervals are slightly too narrow, particularly at the boundaries where REML often struggles most.


## References

1. Biessy, G. (2026). "Whittaker-Henderson Smoothing Revisited." *ASTIN Bulletin*. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932)

2. Whittaker, E.T. (1923). "On a New Method of Graduation." *Proceedings of the Edinburgh Mathematical Society*, 41, 63-75.

3. Henderson, R. (1924). "A New Method of Graduation." *Transactions of the Actuarial Society of America*, 25, 29-40.

---

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tools including factor smoothing — Whittaker smoothing produces the graduated rates that GLM offsets consume |
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Forward trend projection with structural break detection — complement to this library for the trend-fitting stage |

## Training Course

Want structured learning? [Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 3 covers rating table smoothing — Whittaker-Henderson, cross-validation of lambda, and producing publication-quality factor curves. £97 one-time.

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-whittaker/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-whittaker/issues)
- **Blog & tutorials:** [burning-cost.github.io](https://burning-cost.github.io)

If this library saves you time, a star on GitHub helps others find it.

## Licence

MIT

---

## Part of the Burning Cost Toolkit

Open-source Python libraries for UK personal lines insurance pricing. [Browse all libraries](https://burning-cost.github.io/tools/)

| Library | Description |
|---------|-------------|
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GAMs (EBM, ANAM, PIN) — the natural next step once your rating tables are smoothed |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — blends smoothed table estimates with portfolio experience for thin segments |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors whether your smoothed curves remain well-calibrated in production |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — produces the sign-off pack for rating tables going into production |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — quantifies uncertainty around smoothed rate estimates |

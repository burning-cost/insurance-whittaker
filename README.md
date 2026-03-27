# insurance-whittaker

[![PyPI](https://img.shields.io/pypi/v/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/) [![Python](https://img.shields.io/pypi/pyversions/insurance-whittaker)](https://pypi.org/project/insurance-whittaker/) [![Tests](https://github.com/burning-cost/insurance-whittaker/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-whittaker/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-whittaker/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-whittaker/blob/main/notebooks/quickstart.ipynb) [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-whittaker/blob/main/notebooks/quickstart.ipynb)

---

## The problem

Raw loss ratios by age band are noisy. Age 47 looks cheaper than age 46 for no reason other than random variation in thin cells, and a 5-point moving average applied at the boundaries undershoots the young-driver peak you actually want to charge for. Smoothing by hand introduces bias; smoothing without a principled method for choosing how much to smooth leaves you defending an arbitrary decision.

Whittaker-Henderson is the actuarial standard for this — a penalised least-squares smoother with a mathematically principled way to select the smoothing parameter. Every UK pricing team does this. Until now, most did it in Excel or SAS.

**Blog post:** [Whittaker-Henderson Smoothing for Insurance Pricing](https://burning-cost.github.io/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/)

---

## Why this library?

No production-quality Python implementation of Whittaker-Henderson existed. The R package (`WH` from CRAN) is the reference, but the Python actuarial ecosystem had nothing equivalent. This library ports the methodology, validated against the published R results.

The key design choice is automatic lambda selection via REML (restricted marginal likelihood). REML has a unique, well-defined optimum, avoids the overfitting tendency of GCV, and follows Biessy (2026, ASTIN Bulletin) — the current actuarial reference on the topic. You do not tune a smoothing parameter; the data tells you how smooth the curve should be.

---

## Compared to alternatives

| | Manual / Excel | scipy splines | R `WH` package | **insurance-whittaker** |
|---|---|---|---|---|
| Automatic lambda selection | No | Partial (CV) | Yes (REML) | Yes (REML) |
| Bayesian credible intervals | No | No | No | Yes |
| 2-D cross-tables | Tedious | No | Yes | Yes |
| Poisson count extension | No | No | No | Yes |
| Polars output | No | No | No | Yes |
| Python-native | No | Yes | No | Yes |

---

## Quickstart

```bash
uv add insurance-whittaker
```

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

ages = np.arange(17, 80)
exposures = 500 * np.exp(-0.5 * ((ages - 40) / 18) ** 2) + 50
true_lr = 0.35 * np.exp(-0.05 * (ages - 17)) + 0.06
loss_ratios = true_lr + np.random.default_rng(42).normal(0, np.sqrt(true_lr / exposures))

wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, loss_ratios, weights=exposures)

result.fitted      # smoothed loss ratios
result.ci_lower    # 95% credible interval lower bound
result.ci_upper    # 95% credible interval upper bound
result.lambda_     # selected lambda (e.g. 847.3)
result.edf         # effective degrees of freedom (e.g. 5.2)

df = result.to_polars()  # columns: x, y, weight, fitted, ci_lower, ci_upper
result.plot()            # requires insurance-whittaker[plot]
```

---

## What it does

**1-D smoothing** (`WhittakerHenderson1D`) — age curves, NCD scales, vehicle group factors, bonus-malus scales. Pass raw observations and exposures; get back a smooth curve with credible intervals.

**2-D smoothing** (`WhittakerHenderson2D`) — cross-tables (age x vehicle group, age x claim-free years). Same framework, same API, one penalty per dimension.

**Poisson extension** (`WhittakerHendersonPoisson`) — smooth claim frequencies directly from count data and exposures, not from derived loss ratios that carry additional noise from thin cells.

**Automatic lambda selection** — REML (recommended), GCV, AIC, or BIC. REML has a unique optimum and no local minima.

**Bayesian credible intervals** — posterior uncertainty bands on all smoothed values. They widen correctly in thin-data regions: wide at age 17 and age 80, narrow in the high-exposure core. This is essential for a pricing team that needs to know where the curve is reliable.

---

## Mathematical basis

The smoother minimises:

```
sum_i w_i (y_i - theta_i)^2 + lambda * ||D^q theta||^2
```

where `w_i` are exposures, `D^q` is the q-th order difference operator, and `lambda` controls the smoothness penalty. The solution is:

```
theta_hat = (W + lambda D'D)^{-1} W y
```

Solved via Cholesky factorisation — under 0.1 seconds for a 64-band rating curve. Lambda is selected by maximising the restricted marginal likelihood (REML).

Reference: Biessy (2026), *Whittaker-Henderson Smoothing Revisited*, ASTIN Bulletin. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932).

---

## Validated performance

On a synthetic driver age curve (63 bands, ages 17–79) with known true shape and realistic exposure distribution, benchmarked against a 5-point moving average and raw rates:

| Method | RMSE vs true | Thin-tail bands | Boundary behaviour |
|---|---|---|---|
| Raw rates | Highest | Worst | n/a |
| 5-point moving average | ~55% of raw | Moderate | Pulls toward centre |
| Whittaker-Henderson (REML) | **Lowest** | **Best** | Automatic |

REML selects lambda within 10% of the oracle (ground-truth-optimal) value. 95% credible intervals achieve at least 90% coverage on held-out bands, including the thin-tail ages.

In the well-observed middle of the curve (ages 30–55), a 5-point moving average and W-H produce nearly identical results. The gap is driven by the tails. If your rating table covers only well-observed ages, a moving average is fine. If young and old drivers are in scope — they always are in UK motor — W-H earns its keep.

Full validation notebook: `notebooks/databricks_validation.py`.

---

## 2-D example

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson2D

# 10 age bands x 5 vehicle groups
true_lr = (0.08 + 0.15 * np.exp(-0.04 * np.linspace(20, 65, 10)))[:, None] \
        + 0.03 * np.arange(1, 6)[None, :]
exposures = np.outer([80, 200, 350, 450, 500, 480, 420, 300, 180, 90],
                     [300, 250, 200, 150, 100])
y = np.clip(true_lr + np.random.default_rng(0).normal(0, np.sqrt(true_lr / exposures)), 0.01, None)

wh = WhittakerHenderson2D(order_x=2, order_z=2)
result = wh.fit(y, weights=exposures)

result.fitted    # smoothed table, same shape as y
result.lambda_x  # smoothing parameter in age direction
result.lambda_z  # smoothing parameter in vehicle direction
df = result.to_polars()  # long format: x, z, fitted, ci_lower, ci_upper
```

---

## Poisson example

```python
import numpy as np
from insurance_whittaker import WhittakerHendersonPoisson

ages = np.arange(17, 80)
true_rate = 0.28 * np.exp(-0.04 * (ages - 17)) + 0.04
policy_years = (800 * np.exp(-0.5 * ((ages - 38) / 16) ** 2) + 80).astype(float)
claim_counts = np.random.default_rng(42).poisson(true_rate * policy_years).astype(float)

wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claim_counts, exposure=policy_years)

result.fitted_rate    # smoothed claim rate per policy year
result.fitted_count   # smoothed expected claims
result.ci_lower_rate  # 95% CI on rate scale (always positive)
```

---

## Lambda selection

| Method | Description |
|---|---|
| `'reml'` | Restricted marginal likelihood — recommended. Unique optimum, no local minima. |
| `'gcv'` | Generalised cross-validation — faster, can overfit on small datasets. |
| `'aic'` | AIC — penalises effective degrees of freedom. |
| `'bic'` | BIC — stronger penalty, often over-smooths relative to AIC. |

REML is the default and is strongly preferred for actuarial applications. See Biessy (2026) for the simulation evidence.

---

## Limitations

- W-H is a smoother, not a shape constraint. It does not enforce monotonicity. If you want a monotone NCD curve, apply a post-fit isotonic regression pass or set a minimum lambda.
- 2-D smoothing penalises each dimension independently. Cross-effects where optimal smoothness in age depends on vehicle group are not captured. Fit the interaction explicitly in your GLM.
- REML can fail on degenerate data. With fewer than 8–10 observations per dimension, the REML objective can be flat. The optimiser will warn, but inspect the curve visually on short tables.
- Credible intervals fix lambda at its REML estimate. They do not account for uncertainty in lambda itself, so they are slightly too narrow — most visibly at boundaries.

---

## Part of the Burning Cost stack

Takes raw exposure and loss data from claims triangles or rating factor summaries. Feeds smoothed curves into [insurance-gam](https://github.com/burning-cost/insurance-gam) (as input features) and [insurance-credibility](https://github.com/burning-cost/insurance-credibility) (as prior means for Bühlmann-Straub). [See the full stack](https://burning-cost.github.io/stack/)

| Library | Description |
|---|---|
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GAMs — the natural next step once your rating tables are smoothed |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — blends smoothed table estimates with portfolio experience |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors whether smoothed curves remain calibrated in production |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — produces the sign-off pack for rating tables |

---

## References

1. Biessy, G. (2026). "Whittaker-Henderson Smoothing Revisited." *ASTIN Bulletin*. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932)
2. Whittaker, E.T. (1923). "On a New Method of Graduation." *Proceedings of the Edinburgh Mathematical Society*, 41, 63–75.
3. Henderson, R. (1924). "A New Method of Graduation." *Transactions of the Actuarial Society of America*, 25, 29–40.

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-whittaker/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-whittaker/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 3 covers rating table smoothing. £97 one-time.

## Licence

BSD-3-Clause

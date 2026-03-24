# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-whittaker: Whittaker-Henderson Smoothing for Insurance Pricing Tables
# MAGIC
# MAGIC This notebook shows the core workflow in under two minutes: generate a noisy age-frequency curve, smooth it with automatic lambda selection, and read off the smoothed relativities with Bayesian credible intervals.
# MAGIC
# MAGIC The problem this solves: every UK motor pricing actuary smooths experience tables. The Whittaker-Henderson method has been actuarial best practice since 1923. Most UK teams still do it in Excel or SAS. This library brings it to Python with automatic lambda selection (REML) and proper uncertainty quantification.

# COMMAND ----------

# MAGIC %pip install insurance-whittaker polars numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic Experience Data
# MAGIC
# MAGIC A motor claim frequency table by age band: 17 to 79. The true underlying curve is a declining exponential (young drivers are riskier). We observe it with Poisson noise, which creates the jagged step pattern you see in raw experience data before smoothing.

# COMMAND ----------

rng = np.random.default_rng(2024)
ages = np.arange(17, 80)

true_freq = 0.06 + 0.18 * np.exp(-(ages - 17) / 12)
exposures = 50 + 800 * np.exp(-((ages - 38) ** 2) / 400)
exposures = exposures.astype(float)

observed_claims = rng.poisson(true_freq * exposures)
observed_freq   = observed_claims / exposures

df = pl.DataFrame({
    "age":       ages,
    "exposure":  exposures,
    "claims":    observed_claims,
    "obs_freq":  observed_freq,
})
print(f"{len(df)} age bands | total exposure: {exposures.sum():.0f} vehicle-years")
df.head(8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: 1-D Smoothing with Automatic Lambda
# MAGIC
# MAGIC `WhittakerHenderson1D` applies penalised least squares smoothing. The penalty order `q=2` penalises second differences — it biases toward linear rather than constant fits, appropriate for age curves that should be monotone over wide ranges.
# MAGIC
# MAGIC We let the library choose `lambda` automatically via REML (Restricted Maximum Likelihood). No manual tuning.

# COMMAND ----------

from insurance_whittaker import WhittakerHenderson1D

wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, observed_freq, weights=exposures)

print(f"Selected lambda: {result.lambda_:.1f}")
print(f"Effective degrees of freedom: {result.edf_:.1f}")

smoothed_df = pl.DataFrame({
    "age":        ages,
    "obs_freq":   np.round(observed_freq, 4),
    "smoothed":   np.round(result.fitted, 4),
    "lower_95":   np.round(result.lower, 4),
    "upper_95":   np.round(result.upper, 4),
    "relativity": np.round(result.fitted / result.fitted.mean(), 3),
})
smoothed_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Poisson Smoother (for Count Data)
# MAGIC
# MAGIC `WhittakerHendersonPoisson` fits the smooth curve directly to claim counts rather than derived loss ratios. This is strictly more correct when exposures vary — it weights the likelihood properly under a Poisson model.
# MAGIC
# MAGIC Use this when your cells have low expected counts (thin data bands).

# COMMAND ----------

from insurance_whittaker import WhittakerHendersonPoisson

wh_pois = WhittakerHendersonPoisson(order=2, lambda_method="reml")
result_pois = wh_pois.fit(ages, observed_claims, exposures=exposures)

print(f"Poisson smoother — lambda: {result_pois.lambda_:.1f}, EDF: {result_pois.edf_:.1f}")

comparison = pl.DataFrame({
    "age":          ages[:10],
    "true_freq":    np.round(true_freq[:10], 4),
    "wh_gaussian":  np.round(result.fitted[:10], 4),
    "wh_poisson":   np.round(result_pois.fitted[:10], 4),
})
print("\nYoung driver bands (where data is sparse):")
print(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: 2-D Smoothing (Age x Vehicle Group)
# MAGIC
# MAGIC `WhittakerHenderson2D` smooths a cross-table simultaneously in both dimensions. The practical use case: age x vehicle group, NCD x age, or any factor interaction table you want to smooth without modelling the interaction explicitly in a GLM.

# COMMAND ----------

from insurance_whittaker import WhittakerHenderson2D

age_bands = np.arange(17, 32)  # 15 bands
veh_groups = np.arange(1, 9)   # 8 groups
rng2 = np.random.default_rng(99)

AA, VV = np.meshgrid(age_bands, veh_groups, indexing="ij")
true_surface = 0.05 + 0.15 * np.exp(-(AA - 17) / 5) + 0.01 * VV
exp_surface = rng2.poisson(200, (15, 8)).astype(float)
obs_surface = rng2.poisson(true_surface * exp_surface) / exp_surface

wh2 = WhittakerHenderson2D(order=(2, 1), lambda_method="reml")
result2 = wh2.fit(obs_surface, weights=exp_surface)

print(f"2-D smoother — lambda: ({result2.lambda_[0]:.1f}, {result2.lambda_[1]:.1f})")
print(f"Max abs error vs true: {np.abs(result2.fitted - true_surface).max():.4f}")
print(f"Smoothed surface shape: {result2.fitted.shape}  (age_bands x veh_groups)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What You Should See
# MAGIC
# MAGIC - The 1-D smoother recovers the declining age curve from noisy data. Credible intervals are wider at age 17 and 78 where exposure is thin.
# MAGIC - `WhittakerHendersonPoisson` and the Gaussian smoother agree closely in the dense centre bands; the Poisson version is more conservative at the sparse young-driver tail.
# MAGIC - The 2-D smoother handles the age x vehicle group surface in a single call, selecting different lambda values in each dimension.
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC - **`selection.py`** — compare REML, GCV, AIC, BIC lambda choices side by side
# MAGIC - **`plots.py`** — one-line smoothing diagnostic plots (requires matplotlib)
# MAGIC - **Benchmark notebook** — comparison against R's `WH` package on 100 real-world actuarial problems from Biessy (2026)
# MAGIC
# MAGIC **GitHub:** https://github.com/burning-cost/insurance-whittaker
# MAGIC **PyPI:** https://pypi.org/project/insurance-whittaker/

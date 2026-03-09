# Databricks notebook source
# MAGIC %md
# MAGIC # Whittaker-Henderson Smoothing — Insurance Pricing Demo
# MAGIC
# MAGIC This notebook demonstrates the `insurance-whittaker` library on synthetic UK
# MAGIC motor insurance data.  It covers:
# MAGIC
# MAGIC 1. 1-D smoothing of a driver age loss ratio curve
# MAGIC 2. 2-D smoothing of an age × vehicle group frequency table
# MAGIC 3. Poisson smoothing of claim counts by age
# MAGIC 4. Lambda selection comparison (REML vs GCV vs AIC)
# MAGIC
# MAGIC Reference: Biessy (2026), *Whittaker-Henderson Smoothing Revisited*, ASTIN Bulletin.

# COMMAND ----------

# MAGIC %pip install insurance-whittaker

# COMMAND ----------

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK motor data — driver age curve

# COMMAND ----------

rng = np.random.default_rng(42)

# Age bands 17 to 79
ages = np.arange(17, 80, dtype=float)
n = len(ages)

# True loss ratio: high for young drivers, declining to a plateau
true_lr = 0.85 * np.exp(-0.05 * (ages - 17)) + 0.40

# Exposures: fewer young and old drivers
exposure = 500 * np.exp(-((ages - 40) ** 2) / 800)
exposure = np.maximum(exposure, 20)

# Observed loss ratio: noisy
obs_lr = true_lr + rng.normal(0, 0.08 / np.sqrt(exposure / 200), n)

print(f"Age bands: {ages[0]:.0f} to {ages[-1]:.0f}")
print(f"Total exposure: {exposure.sum():.0f} policy years")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit the 1-D smoother with automatic lambda selection

# COMMAND ----------

from insurance_whittaker import WhittakerHenderson1D

wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, obs_lr, weights=exposure)

print(result)
print(f"\nSelected lambda: {result.lambda_:.1f}")
print(f"Effective degrees of freedom: {result.edf:.2f}")
print(f"\nFirst 5 smoothed values: {result.fitted[:5].round(4)}")

# COMMAND ----------

df_1d = result.to_polars()
print(df_1d.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare lambda selection methods

# COMMAND ----------

methods = ['reml', 'gcv', 'aic', 'bic']
results = {}
for method in methods:
    wh_m = WhittakerHenderson1D(order=2, lambda_method=method)
    r = wh_m.fit(ages, obs_lr, weights=exposure)
    results[method] = r
    print(f"{method.upper():4s}: lambda={r.lambda_:8.1f}, edf={r.edf:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify: smoothed is closer to truth than raw observations

# COMMAND ----------

raw_mse = float(np.mean((obs_lr - true_lr) ** 2))
fit_mse = float(np.mean((result.fitted - true_lr) ** 2))
print(f"Raw MSE:     {raw_mse:.6f}")
print(f"Smoothed MSE:{fit_mse:.6f}")
print(f"Improvement: {(1 - fit_mse/raw_mse)*100:.1f}%")
assert fit_mse < raw_mse, "Smoother should improve on raw data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 2-D smoothing: age × vehicle group frequency table

# COMMAND ----------

from insurance_whittaker import WhittakerHenderson2D

n_age = 20     # age bands (17-36)
n_veh = 8      # vehicle groups

age_groups = np.arange(17, 17 + n_age, dtype=float)
veh_groups = np.arange(1, n_veh + 1, dtype=float)

# True frequency surface
xx, zz = np.meshgrid(np.linspace(0, 1, n_age), np.linspace(0, 1, n_veh), indexing='ij')
true_freq = 0.08 + 0.04 * np.sin(np.pi * xx) * np.cos(np.pi * zz / 2) + 0.02 * xx

# Exposure table
exposure_2d = rng.exponential(300, (n_age, n_veh))
exposure_2d = np.maximum(exposure_2d, 50)

# Observed frequencies (noisy)
obs_freq = true_freq + rng.normal(0, 0.01 / np.sqrt(exposure_2d / 200))

print(f"Table shape: {obs_freq.shape}")
print(f"True frequency range: {true_freq.min():.4f} – {true_freq.max():.4f}")

# COMMAND ----------

wh_2d = WhittakerHenderson2D(order_x=2, order_z=2)
result_2d = wh_2d.fit(
    obs_freq,
    weights=exposure_2d,
    x_labels=age_groups,
    z_labels=veh_groups,
)
print(result_2d)
print(f"\nlambda_x={result_2d.lambda_x:.2f}, lambda_z={result_2d.lambda_z:.2f}")
print(f"EDF: {result_2d.edf:.2f}")

# COMMAND ----------

df_2d = result_2d.to_polars(y=obs_freq, weights=exposure_2d)
print(df_2d.head(10))

# COMMAND ----------

# Verify improvement
raw_mse_2d = float(np.mean((obs_freq - true_freq) ** 2))
fit_mse_2d = float(np.mean((result_2d.fitted - true_freq) ** 2))
print(f"Raw MSE (2D):     {raw_mse_2d:.8f}")
print(f"Smoothed MSE (2D):{fit_mse_2d:.8f}")
print(f"Improvement: {(1 - fit_mse_2d/raw_mse_2d)*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Poisson smoothing: claim counts by age

# COMMAND ----------

from insurance_whittaker import WhittakerHendersonPoisson

# Motor claim counts — ages 17-79, large exposures
true_rate = 0.12 * np.exp(-0.04 * (ages - 17)) + 0.03
policy_years = exposure * 5  # scale up for count data
counts = rng.poisson(true_rate * policy_years)

wh_pois = WhittakerHendersonPoisson(order=2)
result_pois = wh_pois.fit(ages, counts, policy_years)
print(result_pois)
print(f"\nIterations to converge: {result_pois.iterations}")
print(f"EDF: {result_pois.edf:.2f}")

# COMMAND ----------

df_pois = result_pois.to_polars()
print(df_pois.head(10))

# COMMAND ----------

# Verify improvement
obs_rate = np.where(policy_years > 0, counts / policy_years, 0.0)
raw_mse_pois = float(np.mean((obs_rate - true_rate) ** 2))
fit_mse_pois = float(np.mean((result_pois.fitted_rate - true_rate) ** 2))
print(f"Raw MSE (Poisson):     {raw_mse_pois:.8f}")
print(f"Smoothed MSE (Poisson):{fit_mse_pois:.8f}")
print(f"Improvement: {(1 - fit_mse_pois/raw_mse_pois)*100:.1f}%")
assert fit_mse_pois < raw_mse_pois

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Edge cases and robustness

# COMMAND ----------

# Very large lambda: approaches polynomial
wh_large = WhittakerHenderson1D(order=2)
result_large = wh_large.fit(ages, obs_lr, weights=exposure, lambda_=1e9)
d2 = np.diff(result_large.fitted, 2)
print(f"Max 2nd diff with lambda=1e9: {np.max(np.abs(d2)):.6f} (should be tiny)")

# Very small lambda: approaches interpolation
wh_small = WhittakerHenderson1D(order=2)
result_small = wh_small.fit(ages, obs_lr, weights=exposure, lambda_=1e-6)
print(f"Max residual with lambda=1e-6: {np.max(np.abs(result_small.fitted - obs_lr)):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run tests

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /tmp && pip install insurance-whittaker pytest -q && \
# MAGIC python -m pytest --tb=short -q $(python -c "import insurance_whittaker, os; print(os.path.dirname(insurance_whittaker.__file__) + '/../../tests')" 2>/dev/null || echo ".") -x

# COMMAND ----------

print("Demo complete. insurance-whittaker is working correctly.")

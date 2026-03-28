# Databricks notebook source
# MAGIC %md
# MAGIC # freMTPL2: Smoothing Driver Age Claim Frequency with Whittaker-Henderson
# MAGIC
# MAGIC Uses the freMTPL2freq dataset (677K French motor policies) from OpenML to
# MAGIC demonstrate Whittaker-Henderson smoothing on a real observed age frequency curve.
# MAGIC
# MAGIC The raw age curve is noisy (thin tails, sample variance in mid-ages). REML
# MAGIC selects lambda automatically. Bayesian credible intervals reflect residual
# MAGIC uncertainty after smoothing.

# COMMAND ----------

# MAGIC %pip install insurance-whittaker scikit-learn polars

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from sklearn.datasets import fetch_openml
from insurance_whittaker import WhittakerHendersonPoisson

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load freMTPL2 from OpenML
# MAGIC
# MAGIC Dataset ID 41214: 677,991 French motor third-party liability policies.
# MAGIC We need ClaimNb (integer count) and Exposure (policy years) per policy,
# MAGIC plus DrivAge for aggregation.

# COMMAND ----------

print("Fetching freMTPL2freq from OpenML (may take ~30s first time)...")
raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df = pl.from_pandas(raw.data).with_columns(
    pl.col("ClaimNb").cast(pl.Int32),
    pl.col("Exposure").cast(pl.Float64),
    pl.col("DrivAge").cast(pl.Int32),
)
print(f"Loaded {len(df):,} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate by driver age
# MAGIC
# MAGIC Sum claim counts and exposure within each single-year age band.
# MAGIC Restrict to ages 18-90 (very thin outside this range).

# COMMAND ----------

age_df = (
    df.group_by("DrivAge")
    .agg(
        pl.col("ClaimNb").sum().alias("claims"),
        pl.col("Exposure").sum().alias("exposure"),
    )
    .filter((pl.col("DrivAge") >= 18) & (pl.col("DrivAge") <= 90))
    .sort("DrivAge")
)

age_df = age_df.with_columns(
    (pl.col("claims") / pl.col("exposure")).alias("raw_freq")
)

print(f"Age bands: {len(age_df)}, total claims: {age_df['claims'].sum():,}, "
      f"total exposure: {age_df['exposure'].sum():,.0f} PY")
print(f"\nAge band exposure summary (policy-years):")
print(age_df.select("DrivAge", "exposure", "claims", "raw_freq").head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Whittaker-Henderson Poisson smoothing with REML
# MAGIC
# MAGIC WhittakerHendersonPoisson fits on log-rate scale via PIRLS.
# MAGIC We pass raw integer counts and exposure — no manual rate calculation needed.
# MAGIC REML selects lambda; Bayesian CIs propagate uncertainty from thin cells.

# COMMAND ----------

ages     = age_df["DrivAge"].to_numpy().astype(float)
counts   = age_df["claims"].to_numpy().astype(float)
exposure = age_df["exposure"].to_numpy()

wh = WhittakerHendersonPoisson(order=2, lambda_method="reml")
result = wh.fit(ages, counts, exposure)

print(result)
print(f"\nREML-selected lambda:       {result.lambda_:,.1f}")
print(f"Effective degrees of freedom: {result.edf:.1f} (out of {len(ages)} age bands)")
print(f"PIRLS iterations:             {result.iterations}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results table
# MAGIC
# MAGIC Shows raw vs smoothed frequency with 95% credible intervals.
# MAGIC Thin tails (young/old) show wide CIs; confident smooth in 25-65 range.

# COMMAND ----------

out = result.to_polars().with_columns(
    pl.col("x").cast(pl.Int32).alias("DrivAge"),
    (pl.col("observed_rate") * 100).alias("raw_freq_pct"),
    (pl.col("fitted_rate") * 100).alias("smoothed_freq_pct"),
    (pl.col("ci_lower_rate") * 100).alias("ci_lower_pct"),
    (pl.col("ci_upper_rate") * 100).alias("ci_upper_pct"),
).select("DrivAge", "raw_freq_pct", "smoothed_freq_pct", "ci_lower_pct", "ci_upper_pct")

print("Raw vs smoothed claim frequency (% per policy-year):\n")
print(f"{'Age':>4}  {'Raw%':>7}  {'Smooth%':>8}  {'CI lower%':>10}  {'CI upper%':>10}")
print("-" * 48)
for row in out.iter_rows():
    age, raw, smooth, lo, hi = row
    print(f"{age:>4}  {raw:>7.3f}  {smooth:>8.3f}  {lo:>10.3f}  {hi:>10.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Key metrics

# COMMAND ----------

raw_freq   = age_df["raw_freq"].to_numpy()
smoothed   = result.fitted_rate

# Smoothness: sum of squared second differences (lower = smoother)
sssd_raw      = float(np.sum(np.diff(raw_freq, 2) ** 2))
sssd_smoothed = float(np.sum(np.diff(smoothed, 2) ** 2))

# CI width as diagnostic: wide = thin exposure, narrow = credible estimate
ci_width = result.ci_upper_rate - result.ci_lower_rate
young_mask = ages < 25
old_mask   = ages > 75
mid_mask   = (ages >= 25) & (ages <= 75)

print("Smoothness (SSSD, lower = smoother):")
print(f"  Raw observed:   {sssd_raw:.6f}")
print(f"  W-H smoothed:   {sssd_smoothed:.6f}")
print(f"  Smoothing gain: {sssd_raw / sssd_smoothed:.1f}x\n")

print("Credible interval width (absolute rate):")
print(f"  Young drivers (<25):  mean CI width = {ci_width[young_mask].mean():.5f}")
print(f"  Mid-age  (25-75):     mean CI width = {ci_width[mid_mask].mean():.5f}")
print(f"  Older    (>75):       mean CI width = {ci_width[old_mask].mean():.5f}")
print(f"\nYoung-vs-mid CI ratio: {ci_width[young_mask].mean() / ci_width[mid_mask].mean():.1f}x "
      f"(reflects thin exposure in young bands)")

print(f"\nPeak smoothed frequency (age {int(ages[smoothed.argmax()])}): "
      f"{smoothed.max()*100:.3f}%")
print(f"Min  smoothed frequency (age {int(ages[smoothed.argmin()])}): "
      f"{smoothed.min()*100:.3f}%")
print(f"Peak-to-trough ratio: {smoothed.max()/smoothed.min():.2f}x")

assert result.edf < len(ages), "EDF must be less than number of bands"
assert sssd_smoothed < sssd_raw, "Smoothed curve must be smoother than raw"
assert result.lambda_ > 1, "REML should select meaningful smoothing"
print("\nAll assertions passed.")

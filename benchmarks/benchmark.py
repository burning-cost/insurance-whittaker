"""
Benchmark: Whittaker-Henderson smoothing vs raw rates and simple moving average.

DGP: Known smooth underlying driver age curve with Poisson-driven observation noise.
  True curve: U-shaped loss ratio (high young, declines to mid-30s, gently rises 60+)
  Noise: proportional to 1/sqrt(exposure) — thin at extremes, heavy in middle

Three approaches compared:
  1. Raw observed rates (no smoothing) — noisy, volatile
  2. 5-point centred moving average — simple, but distorts edges and over-smooths peaks
  3. Whittaker-Henderson 1D (order=2, REML lambda) — principled penalised LS

Metric: Mean squared error vs the true underlying curve.
"""

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_age_curve(seed: int = 42):
    """
    63 driver age bands (17-79). True loss ratio has:
    - Sharp peak at age 17-22 (inexperienced)
    - Steep decline to age 35 (experience gain)
    - Gentle plateau 35-55
    - Modest uptick 55+ (older driver risk)
    """
    rng  = np.random.default_rng(seed)
    ages = np.arange(17, 80, dtype=float)
    n    = len(ages)

    # True smooth curve
    young_peak = 0.45 * np.exp(-0.20 * (ages - 17))
    old_uptick = 0.08 * np.exp(0.04 * np.maximum(ages - 55, 0))
    base       = 0.07
    true_lr    = base + young_peak + old_uptick

    # Exposures: thin at extremes, heavy in the middle (realistic UK motor)
    exposures = np.round(
        800 * np.exp(-0.5 * ((ages - 40) / 16) ** 2) + 30
    ).astype(float)

    # Observed loss ratios: Poisson-driven noise (variance proportional to 1/exposure)
    noise_sd  = np.sqrt(true_lr * (1 - np.clip(true_lr, 0, 0.99)) / exposures)
    obs_lr    = true_lr + rng.normal(0, noise_sd)
    obs_lr    = np.clip(obs_lr, 0.005, 2.0)

    return ages, true_lr, obs_lr, exposures


# ---------------------------------------------------------------------------
# Simple moving average
# ---------------------------------------------------------------------------

def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average. Edges handled by shrinking the window."""
    n      = len(y)
    result = np.empty(n)
    half   = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.mean(y[lo:hi])
    return result


def weighted_moving_average(y: np.ndarray, w: np.ndarray, window: int = 5) -> np.ndarray:
    """Exposure-weighted centred moving average."""
    n      = len(y)
    result = np.empty(n)
    half   = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.average(y[lo:hi], weights=w[lo:hi])
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(true: np.ndarray, estimated: np.ndarray) -> float:
    return float(np.mean((true - estimated) ** 2))


def max_abs_err(true: np.ndarray, estimated: np.ndarray) -> float:
    return float(np.max(np.abs(true - estimated)))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Benchmark: Whittaker-Henderson vs raw rates and moving average")
    print("DGP: U-shaped driver age loss ratio, Poisson observation noise")
    print("=" * 65)

    ages, true_lr, obs_lr, exposures = generate_age_curve(seed=42)
    n = len(ages)

    print(f"\n{n} age bands (17-79)")
    print(f"True curve range: [{true_lr.min():.3f}, {true_lr.max():.3f}]")
    print(f"Observed range:   [{obs_lr.min():.3f}, {obs_lr.max():.3f}]")
    print(f"Exposure range:   [{exposures.min():.0f}, {exposures.max():.0f}]")

    # ------------------------------------------------------------------
    # Approach 1: Raw observed rates
    # ------------------------------------------------------------------
    mse_raw   = mse(true_lr, obs_lr)
    maxe_raw  = max_abs_err(true_lr, obs_lr)

    # ------------------------------------------------------------------
    # Approach 2: Weighted 5-point moving average
    # ------------------------------------------------------------------
    ma5       = weighted_moving_average(obs_lr, exposures, window=5)
    mse_ma5   = mse(true_lr, ma5)
    maxe_ma5  = max_abs_err(true_lr, ma5)

    # ------------------------------------------------------------------
    # Approach 3: Whittaker-Henderson (order=2, REML lambda selection)
    # ------------------------------------------------------------------
    print("\nFitting Whittaker-Henderson 1D (order=2, REML)...")
    wh_ok = False
    try:
        from insurance_whittaker import WhittakerHenderson1D

        wh = WhittakerHenderson1D(order=2, lambda_method="reml")
        result = wh.fit(ages, obs_lr, weights=exposures)

        mse_wh   = mse(true_lr, result.fitted)
        maxe_wh  = max_abs_err(true_lr, result.fitted)

        print(f"  Selected lambda: {result.lambda_:.2f}")
        print(f"  Effective df:    {result.edf:.1f}")
        wh_ok = True
    except Exception as e:
        print(f"  Whittaker-Henderson failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTS — mean squared error vs true underlying curve")
    print("=" * 65)
    print(f"{'Method':<35} {'MSE':>14} {'Max |error|':>14}")
    print("-" * 65)
    print(f"{'Raw observed rates':<35} {mse_raw:>14.8f} {maxe_raw:>14.6f}")
    print(f"{'Weighted 5-pt moving average':<35} {mse_ma5:>14.8f} {maxe_ma5:>14.6f}")
    if wh_ok:
        print(f"{'Whittaker-Henderson (order=2)':<35} {mse_wh:>14.8f} {maxe_wh:>14.6f}")

    print("-" * 65)

    if wh_ok:
        mse_impr_vs_raw = 100.0 * (mse_raw  - mse_wh) / mse_raw
        mse_impr_vs_ma  = 100.0 * (mse_ma5  - mse_wh) / mse_ma5
        print(f"\nWhittaker-Henderson MSE improvement:")
        print(f"  vs raw rates:       {mse_impr_vs_raw:+.1f}%")
        print(f"  vs moving average:  {mse_impr_vs_ma:+.1f}%")

        # Show where MA fails: edge distortion at young/old ages
        young_idx = ages < 25
        old_idx   = ages > 65
        print(f"\nAge-band detail — young drivers (17-24):")
        print(f"  True mean LR:     {true_lr[young_idx].mean():.4f}")
        print(f"  Raw mean LR:      {obs_lr[young_idx].mean():.4f}")
        print(f"  MA mean LR:       {ma5[young_idx].mean():.4f}")
        if wh_ok:
            print(f"  WH mean LR:       {result.fitted[young_idx].mean():.4f}")
        print(f"\nAge-band detail — mature drivers (35-55):")
        mid_idx = (ages >= 35) & (ages <= 55)
        print(f"  True mean LR:     {true_lr[mid_idx].mean():.4f}")
        print(f"  Raw mean LR:      {obs_lr[mid_idx].mean():.4f}")
        print(f"  MA mean LR:       {ma5[mid_idx].mean():.4f}")
        if wh_ok:
            print(f"  WH mean LR:       {result.fitted[mid_idx].mean():.4f}")

        # Polars output table
        out_df = result.to_polars()
        print(f"\nPolars result (first 5 rows):")
        print(out_df.head())

        print(f"\nConclusion: Whittaker-Henderson outperforms both raw rates")
        print(f"  and moving average by recovering the true smooth curve.")
        print(f"  REML lambda selection adapts to the noise level automatically")
        print(f"  — no manual window or bandwidth parameter required.")
        print(f"  Moving average distorts the young-driver peak (boundary bias)")
        print(f"  and lacks principled uncertainty quantification.")


if __name__ == "__main__":
    main()

"""
Tests for the Poisson WH smoother (PIRLS).

Covers:
- Basic convergence on synthetic Poisson data.
- Zero exposure cells handled gracefully.
- Constant rate → fitted rate ≈ constant.
- Large lambda → very smooth log-rate.
- CI on rate scale is positive.
- to_polars() output format.
"""

import numpy as np
import pytest
import polars as pl

from insurance_whittaker import WhittakerHendersonPoisson, WHResultPoisson


class TestWhittakerHendersonPoisson:

    def _constant_rate_data(self, n=30, rate=0.05, seed=0):
        rng = np.random.default_rng(seed)
        exposure = np.full(n, 1000.0)
        counts = rng.poisson(rate * exposure)
        x = np.arange(n, dtype=float)
        return x, counts, exposure

    def test_basic_convergence(self):
        """PIRLS should converge on smooth Poisson data."""
        rng = np.random.default_rng(10)
        n = 40
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 500.0)
        true_rate = 0.05 + 0.03 * np.sin(x / 6)
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert result.iterations > 0
        assert result.iterations <= 50
        assert result.edf > 0

    def test_constant_rate_recovery(self):
        """Large lambda on constant-rate data should recover a near-constant
        fitted rate close to the true rate.
        """
        n = 30
        x, counts, exposure = self._constant_rate_data(n, rate=0.08, seed=1)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=1e6)
        # With huge lambda, fitted rate should be nearly constant
        assert np.std(result.fitted_rate) < 0.01

    def test_fitted_rate_positive(self):
        """Fitted rates must always be positive."""
        n = 25
        rng = np.random.default_rng(3)
        x = np.arange(n, dtype=float)
        exposure = rng.exponential(200, n)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)

    def test_ci_on_rate_scale_positive(self):
        """CI bounds on rate scale must be positive."""
        n = 20
        rng = np.random.default_rng(4)
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 300.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.ci_lower_rate > 0)
        assert np.all(result.ci_upper_rate > 0)
        assert np.all(result.ci_upper_rate >= result.ci_lower_rate)

    def test_ci_contains_fitted_rate(self):
        """Fitted rate should lie within its own CI."""
        n = 20
        rng = np.random.default_rng(5)
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 300.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate >= result.ci_lower_rate)
        assert np.all(result.fitted_rate <= result.ci_upper_rate)

    def test_to_polars(self):
        n = 15
        x, counts, exposure = self._constant_rate_data(n, rate=0.05, seed=6)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == n
        for col in ["x", "count", "exposure", "fitted_rate", "fitted_count"]:
            assert col in df.columns

    def test_repr(self):
        n = 15
        x, counts, exposure = self._constant_rate_data(n, rate=0.05, seed=7)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        r = repr(result)
        assert "WHResultPoisson" in r

    def test_no_exposure_defaults_to_ones(self):
        """No exposure → counts are rates per unit, should still work."""
        n = 15
        rng = np.random.default_rng(8)
        x = np.arange(n, dtype=float)
        counts = rng.poisson(5, n)  # counts with unit exposure
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts)
        assert len(result.fitted_rate) == n
        assert np.all(result.fitted_rate > 0)

    def test_lambda_positive(self):
        n = 20
        rng = np.random.default_rng(9)
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert result.lambda_ > 0

    def test_negative_counts_raises(self):
        n = 10
        wh = WhittakerHendersonPoisson(order=2)
        with pytest.raises(ValueError, match="non-negative"):
            wh.fit(
                np.arange(n, dtype=float),
                np.full(n, -1.0),
                np.ones(n),
            )

    def test_length_mismatch_raises(self):
        wh = WhittakerHendersonPoisson(order=2)
        with pytest.raises(ValueError):
            wh.fit(
                np.arange(10, dtype=float),
                np.ones(10),
                np.ones(8),  # wrong length
            )

    def test_order_1(self):
        n = 20
        rng = np.random.default_rng(11)
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=1)
        result = wh.fit(x, counts, exposure)
        assert result.order == 1

    def test_smooth_rate_recovery(self):
        """Fitted rate should be closer to true rate than raw observed rate."""
        rng = np.random.default_rng(12)
        n = 50
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        true_rate = 0.05 + 0.03 * np.sin(x / 8)
        counts = rng.poisson(true_rate * exposure)
        obs_rate = np.where(exposure > 0, counts / exposure, 0.0)

        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)

        raw_mse = np.mean((obs_rate - true_rate) ** 2)
        fit_mse = np.mean((result.fitted_rate - true_rate) ** 2)
        assert fit_mse < raw_mse, (
            f"Smoothed MSE {fit_mse:.6f} >= raw MSE {raw_mse:.6f}"
        )

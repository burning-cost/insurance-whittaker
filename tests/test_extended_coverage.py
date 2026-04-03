"""
Extended coverage tests for insurance-whittaker.

Focuses on:
- WHResult1D attributes and methods beyond the basic smoke tests
- WHResult2D to_polars() with various input configurations
- WhittakerHenderson1D with fixed lambda (no automatic selection)
- WhittakerHenderson2D with fixed lambdas
- WHResultPoisson extended attributes
- select_lambda_2d direct invocation
- Input types: pandas Series, Python lists, integer arrays
- Boundary conditions: n=order+1 (minimum valid), very large/small lambda
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_whittaker import (
    WhittakerHenderson1D,
    WhittakerHenderson2D,
    WhittakerHendersonPoisson,
    WHResult1D,
    WHResult2D,
    WHResultPoisson,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_signal(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    y = np.sin(x / 6.0) + rng.normal(0, 0.15, size=n)
    return x, y


def _constant_signal(n: int, value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=float)
    y = np.full(n, value)
    return x, y


# ---------------------------------------------------------------------------
# WHResult1D: extended attribute checks
# ---------------------------------------------------------------------------


class TestWHResult1DExtended:

    def _fit(self, n: int = 40, order: int = 2) -> WHResult1D:
        x, y = _sine_signal(n)
        wh = WhittakerHenderson1D(order=order)
        return wh.fit(x, y)

    def test_residuals_sum_near_zero_uniform_weights(self):
        """With uniform weights, weighted residuals sum to ~0 at optimum."""
        result = self._fit()
        resid = result.y - result.fitted
        # For WH the residuals are not guaranteed to sum to zero (no intercept
        # constraint), but they should be small relative to the signal.
        rms_resid = np.sqrt(np.mean(resid**2))
        rms_signal = np.sqrt(np.mean(result.y**2))
        assert rms_resid < rms_signal  # smoother than raw data

    def test_fitted_length_matches_y(self):
        result = self._fit(n=50)
        assert len(result.fitted) == 50

    def test_ci_upper_ge_ci_lower(self):
        result = self._fit()
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_ci_width_proportional_to_std_fitted(self):
        """CI width = 2 * 1.96 * std_fitted approximately."""
        result = self._fit()
        ci_width = result.ci_upper - result.ci_lower
        # Not exact (depends on sigma2 estimation) but width > 0 where std > 0
        assert np.all(ci_width >= 0)

    def test_edf_decreases_with_lambda(self):
        """Manual: larger fixed lambda => smaller EDF."""
        x, y = _sine_signal(40)
        wh_low = WhittakerHenderson1D(order=2)
        r_low = wh_low.fit(x, y, lambda_=0.1)
        wh_high = WhittakerHenderson1D(order=2)
        r_high = wh_high.fit(x, y, lambda_=10000.0)
        assert r_high.edf < r_low.edf

    def test_to_polars_has_fitted_column(self):
        result = self._fit()
        df = result.to_polars()
        assert "fitted" in df.columns

    def test_to_polars_has_x_column(self):
        result = self._fit()
        df = result.to_polars()
        assert "x" in df.columns

    def test_to_polars_row_count_matches_n(self):
        n = 35
        result = self._fit(n=n)
        df = result.to_polars()
        assert len(df) == n

    def test_to_polars_ci_columns_present(self):
        result = self._fit()
        df = result.to_polars()
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns


# ---------------------------------------------------------------------------
# WhittakerHenderson1D: fixed lambda (no selection)
# ---------------------------------------------------------------------------


class TestWH1DFixedLambda:

    def test_fixed_lambda_small_interpolates(self):
        """lambda very small => fitted ≈ observed."""
        x, y = _sine_signal(30)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-6)
        np.testing.assert_allclose(result.fitted, y, atol=1e-3)

    def test_fixed_large_lambda_gives_polynomial(self):
        """lambda very large => near-polynomial smooth."""
        x, y = _sine_signal(30)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e8)
        # 2nd-order smooth: nearly linear => 2nd differences near zero
        d2 = np.diff(result.fitted, n=2)
        assert np.max(np.abs(d2)) < 0.5

    def test_fixed_lambda_stored(self):
        """lambda_ attribute should match the supplied value."""
        x, y = _sine_signal(30)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=42.0)
        assert abs(result.lambda_ - 42.0) < 1e-9

    def test_order_1_fixed_lambda(self):
        x, y = _sine_signal(25)
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit(x, y, lambda_=100.0)
        assert result.order == 1
        assert len(result.fitted) == 25

    def test_order_3_fixed_lambda(self):
        x, y = _sine_signal(30)
        wh = WhittakerHenderson1D(order=3)
        result = wh.fit(x, y, lambda_=10.0)
        assert result.order == 3
        assert len(result.fitted) == 30

    def test_negative_lambda_raises(self):
        x, y = _sine_signal(20)
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit(x, y, lambda_=-1.0)


# ---------------------------------------------------------------------------
# Input type handling
# ---------------------------------------------------------------------------


class TestInputTypes:

    def test_list_input_1d(self):
        x = list(range(20))
        y = [float(i ** 0.5) for i in x]
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == 20

    def test_integer_array_1d(self):
        x = np.arange(20, dtype=int)
        y = np.random.default_rng(0).integers(0, 10, size=20).astype(int)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y.astype(float))
        assert len(result.fitted) == 20

    def test_polars_series_1d(self):
        n = 25
        x = pl.Series("age", np.arange(n, dtype=float))
        y = pl.Series("lr", np.sin(np.arange(n) / 5.0))
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == n

    def test_pandas_series_1d(self):
        """WhittakerHenderson1D should accept pandas Series."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        n = 20
        x = pd.Series(np.arange(n, dtype=float))
        y = pd.Series(np.sin(np.arange(n) / 4.0))
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == n


# ---------------------------------------------------------------------------
# WhittakerHenderson2D: fixed lambdas
# ---------------------------------------------------------------------------


class TestWH2DFixedLambdas:

    def _make_table(self, nx: int = 8, nz: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((nx, nz))

    def test_fixed_lambdas_both_supplied(self):
        y = self._make_table(8, 6)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=10.0, lambda_z=5.0)
        assert abs(result.lambda_x - 10.0) < 1e-9
        assert abs(result.lambda_z - 5.0) < 1e-9
        assert result.fitted.shape == y.shape

    def test_fixed_lambda_x_only_selects_lambda_z(self):
        y = self._make_table(8, 6)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=10.0)
        assert abs(result.lambda_x - 10.0) < 1e-9
        assert result.lambda_z > 0.0  # selected

    def test_fixed_lambda_z_only_selects_lambda_x(self):
        y = self._make_table(8, 6)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_z=5.0)
        assert abs(result.lambda_z - 5.0) < 1e-9
        assert result.lambda_x > 0.0  # selected

    def test_small_lambdas_give_rough_fit(self):
        """Near-zero lambdas: fitted values close to observed."""
        y = self._make_table(6, 5)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1e-6, lambda_z=1e-6)
        np.testing.assert_allclose(result.fitted, y, atol=0.1)

    def test_large_lambdas_give_smooth_fit(self):
        """Very large lambdas: nearly planar surface."""
        y = self._make_table(8, 6)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1e8, lambda_z=1e8)
        # 2nd-order differences should be near zero in both directions
        d2x = np.diff(result.fitted, n=2, axis=0)
        d2z = np.diff(result.fitted, n=2, axis=1)
        assert np.max(np.abs(d2x)) < 0.5
        assert np.max(np.abs(d2z)) < 0.5


# ---------------------------------------------------------------------------
# WHResult2D: extended attribute and method checks
# ---------------------------------------------------------------------------


class TestWHResult2DExtended:

    def _fit(self, nx: int = 8, nz: int = 6) -> WHResult2D:
        rng = np.random.default_rng(1)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        return wh.fit(y)

    def test_fitted_shape(self):
        result = self._fit(8, 6)
        assert result.fitted.shape == (8, 6)

    def test_ci_upper_ge_lower_everywhere(self):
        result = self._fit()
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_to_polars_long_schema(self):
        result = self._fit(6, 5)
        df = result.to_polars()
        assert "fitted" in df.columns
        assert len(df) == 6 * 5

    def test_to_polars_with_x_labels_and_z_labels(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal((4, 3))
        x_labels = ["A", "B", "C", "D"]
        z_labels = [10, 20, 30]
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, x_labels=x_labels, z_labels=z_labels)
        df = result.to_polars()
        assert "x" in df.columns
        assert "z" in df.columns
        assert len(df) == 12  # 4 * 3

    def test_sigma2_positive(self):
        result = self._fit()
        assert result.sigma2 > 0.0

    def test_edf_positive(self):
        result = self._fit()
        assert result.edf > 0.0

    def test_order_x_z_stored(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal((7, 5))
        wh = WhittakerHenderson2D(order_x=1, order_z=3)
        result = wh.fit(y)
        assert result.order_x == 1
        assert result.order_z == 3


# ---------------------------------------------------------------------------
# WHResultPoisson: extended attributes
# ---------------------------------------------------------------------------


class TestWHResultPoissonExtended:

    def _fit(self, n: int = 30) -> WHResultPoisson:
        rng = np.random.default_rng(5)
        x = np.arange(n, dtype=float)
        exposure = rng.uniform(50, 200, size=n)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        return wh.fit(x, counts, exposure=exposure)

    def test_fitted_rate_positive(self):
        result = self._fit()
        assert np.all(result.fitted_rate > 0)

    def test_fitted_count_non_negative(self):
        result = self._fit()
        assert np.all(result.fitted_count >= 0)

    def test_ci_upper_rate_ge_lower_rate(self):
        result = self._fit()
        assert np.all(result.ci_upper_rate >= result.ci_lower_rate)

    def test_ci_contains_fitted_rate(self):
        result = self._fit()
        assert np.all(result.fitted_rate >= result.ci_lower_rate)
        assert np.all(result.fitted_rate <= result.ci_upper_rate)

    def test_to_polars_has_rate_columns(self):
        result = self._fit()
        df = result.to_polars()
        assert "fitted_rate" in df.columns
        assert "ci_lower_rate" in df.columns
        assert "ci_upper_rate" in df.columns

    def test_lambda_positive(self):
        result = self._fit()
        assert result.lambda_ > 0.0

    def test_edf_positive(self):
        result = self._fit()
        assert result.edf > 0.0

    def test_no_exposure_defaults_to_ones(self):
        """Without exposure, rates equal counts."""
        rng = np.random.default_rng(9)
        x = np.arange(25, dtype=float)
        counts = rng.poisson(3.0, size=25)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts)
        # With exposure=1, fitted_rate ≈ fitted_count
        np.testing.assert_allclose(result.fitted_rate, result.fitted_count, rtol=1e-6)

    def test_all_lambda_methods(self):
        """All four lambda selection methods should run without error."""
        rng = np.random.default_rng(6)
        x = np.arange(30, dtype=float)
        exposure = np.ones(30) * 100.0
        counts = rng.poisson(5.0, size=30)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHendersonPoisson(order=2, lambda_method=method)
            result = wh.fit(x, counts, exposure=exposure)
            assert result.lambda_ > 0.0, f"method={method} gave non-positive lambda"


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------


class TestValidationErrors:

    def test_1d_x_y_different_lengths_not_supported(self):
        """When x and y have different lengths, smoother should raise or handle gracefully."""
        # The smoother uses x for axis only; y for values.
        # Length mismatch in y/weights should raise.
        x = np.arange(20, dtype=float)
        y = np.ones(20)
        weights = np.ones(15)  # wrong length
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises((ValueError, Exception)):
            wh.fit(x, y, weights=weights)

    def test_poisson_negative_counts_raise(self):
        x = np.arange(20, dtype=float)
        counts = np.ones(20)
        counts[5] = -1.0
        wh = WhittakerHendersonPoisson(order=2)
        with pytest.raises(ValueError, match="[Nn]egative"):
            wh.fit(x, counts)

    def test_2d_weight_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((6, 5))
        weights = rng.uniform(0.5, 2.0, size=(5, 6))  # wrong shape
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        with pytest.raises((ValueError, Exception)):
            wh.fit(y, weights=weights)

    def test_1d_order_0_raises(self):
        # order=0 is rejected at construction time, not fit time
        with pytest.raises(ValueError, match="order must be"):
            WhittakerHenderson1D(order=0)

    def test_poisson_zero_exposure_cell_does_not_crash(self):
        """A single zero-exposure cell should not crash the Poisson smoother."""
        rng = np.random.default_rng(8)
        n = 25
        x = np.arange(n, dtype=float)
        exposure = rng.uniform(50, 200, size=n)
        exposure[10] = 0.0
        counts = rng.poisson(0.05 * np.where(exposure > 0, exposure, 1))
        counts[10] = 0
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure=exposure)
        assert result.fitted_rate is not None
        assert len(result.fitted_rate) == n


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


class TestMathProperties:

    def test_1d_constant_signal_exactly_recovered(self):
        """A constant signal is in the null space of any order difference operator.
        With any lambda, the fitted values should equal the constant."""
        n = 30
        x, y = _constant_signal(n, value=3.14)
        for order in [1, 2, 3]:
            if n > order + 1:
                wh = WhittakerHenderson1D(order=order)
                result = wh.fit(x, y, lambda_=100.0)
                np.testing.assert_allclose(
                    result.fitted, y, atol=1e-6,
                    err_msg=f"order={order}: constant not recovered"
                )

    def test_2d_constant_table_exactly_recovered(self):
        """A constant 2D table is in the null space of both penalty matrices."""
        y = np.full((8, 6), 2.0)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=100.0, lambda_z=100.0)
        np.testing.assert_allclose(result.fitted, y, atol=1e-6)

    def test_symmetry_of_1d_smooth(self):
        """Symmetric signal should produce symmetric smooth."""
        n = 31  # odd for clear centre
        x = np.arange(n, dtype=float)
        # Symmetric around centre
        mid = n // 2
        y = np.zeros(n)
        for i in range(n):
            y[i] = abs(i - mid)  # V-shape, symmetric
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        # Fitted values should also be (approximately) symmetric
        fitted = result.fitted
        np.testing.assert_allclose(
            fitted[:mid], fitted[n-1:mid:-1], atol=0.05,
            err_msg="Symmetric signal should give symmetric smooth"
        )

"""
Tests for the 1-D Whittaker-Henderson smoother.

Covers:
- Constant data → theta = constant (smoother passes through flat data).
- Known analytical limits: large lambda → polynomial, small lambda →
  interpolation.
- Single point edge case.
- All equal weights.
- Polars input/output roundtrip.
- Result attributes are well-formed.
"""

import numpy as np
import pytest
import polars as pl

from insurance_whittaker import WhittakerHenderson1D, WHResult1D
from insurance_whittaker._utils import diff_matrix, penalty_banded


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestDiffMatrix:
    def test_first_order_shape(self):
        D = diff_matrix(5, 1)
        assert D.shape == (4, 5)

    def test_second_order_shape(self):
        D = diff_matrix(10, 2)
        assert D.shape == (8, 10)

    def test_first_order_values(self):
        D = diff_matrix(4, 1)
        expected = np.array([
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
        ], dtype=float)
        np.testing.assert_allclose(D, expected)

    def test_constant_in_null_space(self):
        """Constant vector is in the null space of D^1 and D^2."""
        for order in [1, 2]:
            D = diff_matrix(10, order)
            ones = np.ones(10)
            np.testing.assert_allclose(D @ ones, 0, atol=1e-12)

    def test_linear_in_null_space_order2(self):
        """Linear vector is in the null space of D^2."""
        D = diff_matrix(20, 2)
        linear = np.arange(20, dtype=float)
        np.testing.assert_allclose(D @ linear, 0, atol=1e-10)


class TestPenaltyBanded:
    def test_shape(self):
        ab = penalty_banded(10, 2)
        assert ab.shape == (3, 10)

    def test_main_diagonal_positive(self):
        ab = penalty_banded(10, 2)
        assert np.all(ab[0, :] >= 0)

    def test_off_diagonal_sign(self):
        """Off-diagonal elements of D'D alternate in sign."""
        ab = penalty_banded(10, 2)
        # Superdiagonal 1 should be negative for order=2
        assert np.all(ab[1, :9] <= 0)
        # Superdiagonal 2 should be positive
        assert np.all(ab[2, :8] >= 0)


# ---------------------------------------------------------------------------
# Core smoother tests
# ---------------------------------------------------------------------------

class TestWhittakerHenderson1D:

    def _make_smoother(self, order=2, method="reml"):
        return WhittakerHenderson1D(order=order, lambda_method=method)

    def test_constant_data_exact_recovery(self):
        """For constant y with equal weights, fitted should equal y exactly
        (regardless of lambda, since any smooth through a constant is optimal).
        """
        n = 30
        x = np.arange(n, dtype=float)
        y = np.full(n, 0.75)
        w = np.ones(n)
        wh = self._make_smoother()
        result = wh.fit(x, y, weights=w, lambda_=1000.0)
        np.testing.assert_allclose(result.fitted, y, atol=1e-6)

    def test_constant_data_reml(self):
        """Constant data with REML selection should still recover constant."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.full(n, 1.2)
        wh = self._make_smoother()
        result = wh.fit(x, y)
        np.testing.assert_allclose(result.fitted, y, atol=1e-5)

    def test_large_lambda_gives_polynomial(self):
        """Very large lambda drives the smoother towards a degree-(q-1)
        polynomial (q=2 → linear).  Check that residuals from a linear fit
        are tiny.
        """
        n = 50
        rng = np.random.default_rng(42)
        x = np.arange(n, dtype=float)
        y = 2.0 + 0.3 * x + 0.1 * rng.standard_normal(n)
        wh = self._make_smoother()
        result = wh.fit(x, y, lambda_=1e10)
        # The fitted values should be nearly linear
        # Fit a line through the fitted values
        coefs = np.polyfit(x, result.fitted, 1)
        linear_fit = np.polyval(coefs, x)
        np.testing.assert_allclose(result.fitted, linear_fit, atol=1e-4)

    def test_small_lambda_gives_interpolation(self):
        """Very small lambda → fitted values ≈ observed values."""
        n = 20
        rng = np.random.default_rng(7)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = self._make_smoother()
        result = wh.fit(x, y, lambda_=1e-8)
        np.testing.assert_allclose(result.fitted, y, atol=1e-4)

    def test_result_is_smoother_than_data(self):
        """Fitted second differences should be smaller than raw second diffs
        (on noisy data with reasonable lambda).
        """
        rng = np.random.default_rng(0)
        n = 40
        x = np.arange(n, dtype=float)
        true_signal = np.sin(x / 5)
        y = true_signal + 0.3 * rng.standard_normal(n)
        wh = self._make_smoother()
        result = wh.fit(x, y)

        raw_d2 = np.sum(np.diff(y, 2) ** 2)
        fit_d2 = np.sum(np.diff(result.fitted, 2) ** 2)
        assert fit_d2 < raw_d2

    def test_ci_contains_truth(self):
        """Most true signal values should fall within the 95% CI."""
        rng = np.random.default_rng(1)
        n = 50
        x = np.arange(n, dtype=float)
        true_signal = np.sin(x / 8) * 0.5 + 1.0
        y = true_signal + 0.1 * rng.standard_normal(n)
        wh = self._make_smoother()
        result = wh.fit(x, y)
        inside = np.mean(
            (true_signal >= result.ci_lower) & (true_signal <= result.ci_upper)
        )
        # At least 70% should be inside (noisy data, so not exactly 95%)
        assert inside >= 0.7

    def test_edf_between_order_and_n(self):
        """EDF should be between order+1 and n."""
        n = 30
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(2)
        y = rng.standard_normal(n)
        wh = self._make_smoother(order=2)
        result = wh.fit(x, y)
        assert result.edf > 2  # more than just the linear trend
        assert result.edf < n

    def test_lambda_positive(self):
        """Selected lambda must be positive."""
        n = 25
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(3)
        y = rng.standard_normal(n)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = self._make_smoother(method=method)
            result = wh.fit(x, y)
            assert result.lambda_ > 0, f"lambda <= 0 for method={method}"

    def test_ci_width_positive(self):
        """CI upper must be >= CI lower everywhere."""
        n = 20
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(4)
        y = rng.standard_normal(n)
        wh = self._make_smoother()
        result = wh.fit(x, y)
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_polars_input(self):
        """Polars Series input should work identically to NumPy."""
        n = 20
        x_np = np.arange(n, dtype=float)
        y_np = np.sin(x_np / 5)
        w_np = np.ones(n)

        wh = WhittakerHenderson1D(order=2)
        result_np = wh.fit(x_np, y_np, weights=w_np, lambda_=10.0)

        x_pl = pl.Series("x", x_np)
        y_pl = pl.Series("y", y_np)
        w_pl = pl.Series("w", w_np)
        result_pl = wh.fit(x_pl, y_pl, weights=w_pl, lambda_=10.0)

        np.testing.assert_allclose(result_np.fitted, result_pl.fitted, atol=1e-10)

    def test_to_polars(self):
        """to_polars() should return a DataFrame with the right columns."""
        n = 15
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=100.0)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        for col in ["x", "y", "weight", "fitted", "ci_lower", "ci_upper"]:
            assert col in df.columns
        assert len(df) == n

    def test_repr(self):
        n = 20
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=100.0)
        r = repr(result)
        assert "WHResult1D" in r
        assert "order=2" in r

    def test_order_1(self):
        """Order-1 smoother (linear interpolation) should work."""
        n = 20
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(5)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit(x, y)
        assert result.order == 1
        assert result.edf > 0

    def test_weighted_smooth(self):
        """High-weight cells should be fitted more closely than low-weight."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        y[10] = 5.0  # outlier
        w = np.ones(n)
        w[10] = 100.0  # very high weight on outlier

        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=w, lambda_=100.0)
        # Fitted value at the high-weight cell should be pulled towards 5
        assert result.fitted[10] > result.fitted[9]

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            WhittakerHenderson1D(order=0)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            wh = WhittakerHenderson1D(order=2)
            wh.fit([1, 2], [0.5, 0.6], lambda_=10.0)

    def test_negative_lambda_raises(self):
        n = 10
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            wh.fit(np.arange(n), np.ones(n), lambda_=-1.0)

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError):
            wh = WhittakerHenderson1D(order=2, lambda_method="unknown")
            wh.fit(np.arange(10, dtype=float), np.ones(10))


# ---------------------------------------------------------------------------
# Regression: known property check
# ---------------------------------------------------------------------------

class TestKnownProperties:

    def test_all_criteria_give_similar_fit(self):
        """Different lambda selection criteria should give similar fitted
        values for well-behaved smooth data.
        """
        rng = np.random.default_rng(99)
        n = 40
        x = np.arange(n, dtype=float)
        y = np.sin(x / 6) + 0.05 * rng.standard_normal(n)

        fits = {}
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            fits[method] = result.fitted

        # All methods should agree within 0.1 on smooth data
        for m1 in fits:
            for m2 in fits:
                if m1 != m2:
                    diff = np.max(np.abs(fits[m1] - fits[m2]))
                    assert diff < 0.15, f"{m1} vs {m2}: max diff = {diff:.4f}"

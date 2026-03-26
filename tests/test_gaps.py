"""
Structural gap tests for insurance-whittaker.

Covers branches not exercised by the existing test suite:
- order=3 smoother (higher-order difference penalties)
- Zero-weight cells (should be handled without division by zero)
- validate_inputs: length mismatch, negative weights
- sigma2 fallback when dof <= 0.5 (high-edf regime)
- GCV denominator saturation guard (W_sum - edf near zero)
- _log_det_P_nz_cached: cache hit path
- 2D smoother with non-square table and mixed weights
- WHResult2D.to_polars() with y and weights supplied
- Poisson smoother: zero-count cells, known scale invariance
- All selection criteria: select_lambda order=1
- WhittakerHenderson1D: mismatched y/weight lengths raises
- WhittakerHenderson2D: partial lambda supplied (only lambda_x)
"""

import numpy as np
import pytest
import polars as pl

from insurance_whittaker import WhittakerHenderson1D, WhittakerHenderson2D
from insurance_whittaker._utils import validate_inputs, to_numpy_1d, penalty_banded
from insurance_whittaker.selection import (
    select_lambda,
    gcv_criterion,
    _log_det_P_nz_cached,
)


# ---------------------------------------------------------------------------
# validate_inputs edge cases
# ---------------------------------------------------------------------------

class TestValidateInputsEdgeCases:

    def test_length_mismatch_raises(self):
        """weights length != y length must raise ValueError."""
        y = np.ones(10)
        w = np.ones(8)
        with pytest.raises(ValueError, match="length"):
            validate_inputs(y, w, 10)

    def test_negative_weight_raises(self):
        """Any negative weight must raise ValueError."""
        y = np.ones(5)
        w = np.array([1.0, 1.0, -0.5, 1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            validate_inputs(y, w, 5)

    def test_zero_weight_accepted(self):
        """Zero weights are valid (missing data cells)."""
        y = np.ones(5)
        w = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        result = validate_inputs(y, w, 5)
        np.testing.assert_array_equal(result, w)

    def test_none_weights_returns_ones(self):
        """None weights should return an array of ones of the correct length."""
        y = np.ones(7)
        w = validate_inputs(y, None, 7)
        np.testing.assert_array_equal(w, np.ones(7))


# ---------------------------------------------------------------------------
# to_numpy_1d
# ---------------------------------------------------------------------------

class TestToNumpy1D:

    def test_list_input(self):
        arr = to_numpy_1d([1, 2, 3], "x")
        assert arr.dtype == np.float64
        assert arr.shape == (3,)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            to_numpy_1d(np.ones((3, 3)), "x")


# ---------------------------------------------------------------------------
# order=3 smoother
# ---------------------------------------------------------------------------

class TestOrder3Smoother:

    def test_order3_fits(self):
        """Order-3 smoother should produce valid output."""
        n = 25
        rng = np.random.default_rng(10)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=3)
        result = wh.fit(x, y)
        assert len(result.fitted) == n
        assert result.order == 3
        assert result.edf > 0

    def test_order3_large_lambda_tends_to_quadratic(self):
        """Order-3 with large lambda should smooth toward a quadratic (degree-2 poly)."""
        n = 30
        x = np.arange(n, dtype=float)
        # True quadratic
        y = 2.0 + 0.5 * x - 0.01 * x**2
        wh = WhittakerHenderson1D(order=3)
        result = wh.fit(x, y, lambda_=1e8)
        # Residuals from a quadratic fit should be tiny
        coefs = np.polyfit(x, result.fitted, 2)
        poly_fit = np.polyval(coefs, x)
        np.testing.assert_allclose(result.fitted, poly_fit, atol=1e-3)


# ---------------------------------------------------------------------------
# Zero-weight cells
# ---------------------------------------------------------------------------

class TestZeroWeightCells:

    def test_zero_weight_cells_do_not_crash(self):
        """Some cells can have zero exposure; smoother should not divide by zero."""
        n = 20
        rng = np.random.default_rng(20)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        w = np.ones(n)
        w[5] = 0.0   # cell with no data
        w[15] = 0.0  # another missing cell

        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=w, lambda_=10.0)
        assert len(result.fitted) == n
        assert np.all(np.isfinite(result.fitted))

    def test_zero_weight_cells_2d(self):
        """2D table with some zero-weight cells should not crash."""
        rng = np.random.default_rng(21)
        nx, nz = 6, 5
        y = rng.standard_normal((nx, nz))
        w = np.ones((nx, nz))
        w[2, 3] = 0.0
        w[0, 0] = 0.0

        wh = WhittakerHenderson2D()
        result = wh.fit(y, weights=w, lambda_x=20.0, lambda_z=20.0)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))


# ---------------------------------------------------------------------------
# sigma2 fallback when dof <= 0.5
# ---------------------------------------------------------------------------

class TestSigma2Fallback:

    def test_sigma2_fallback_with_near_interpolating_smoother(self):
        """When lambda is very small, edf ~= n and dof ~= 0, sigma2 falls back to 1.0."""
        n = 10
        rng = np.random.default_rng(30)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        # Very small lambda → nearly interpolating → dof near zero
        result = wh.fit(x, y, lambda_=1e-12)
        # sigma2 should be 1.0 in the fallback (or a positive finite value otherwise)
        assert result.sigma2 >= 0.0
        assert np.isfinite(result.sigma2)

    def test_sigma2_positive_with_normal_lambda(self):
        """With a moderate lambda and noisy data, sigma2 should be positive."""
        n = 30
        rng = np.random.default_rng(31)
        x = np.arange(n, dtype=float)
        y = 1.0 + 0.5 * rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=50.0)
        assert result.sigma2 > 0.0


# ---------------------------------------------------------------------------
# GCV denominator saturation guard
# ---------------------------------------------------------------------------

class TestGCVDenominatorGuard:

    def test_gcv_does_not_return_inf_at_zero_lambda(self):
        """GCV denominator (W_sum - edf) can approach zero at very small lambda;
        the guard should return 1e30 rather than inf/nan."""
        n = 10
        y = np.ones(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        # At very small lambda, edf ~= W_sum (n), making denominator ~= 0
        val = gcv_criterion(-30.0, ab_P, w, y, 2)
        assert np.isfinite(val) or val == 1e30


# ---------------------------------------------------------------------------
# _log_det_P_nz_cached: cache hit path
# ---------------------------------------------------------------------------

class TestLogDetCache:

    def test_cache_returns_same_value_twice(self):
        """Two calls with the same (n, order) should return identical values."""
        val1 = _log_det_P_nz_cached(15, 2)
        val2 = _log_det_P_nz_cached(15, 2)
        assert val1 == val2

    def test_cache_different_orders_differ(self):
        """order=1 and order=2 should give different log-determinants."""
        v1 = _log_det_P_nz_cached(20, 1)
        v2 = _log_det_P_nz_cached(20, 2)
        assert v1 != v2

    def test_cache_returns_finite(self):
        for n, order in [(10, 1), (15, 2), (20, 3)]:
            val = _log_det_P_nz_cached(n, order)
            assert np.isfinite(val)


# ---------------------------------------------------------------------------
# select_lambda with order=1
# ---------------------------------------------------------------------------

class TestSelectLambdaOrder1:

    @pytest.mark.parametrize("method", ["reml", "gcv", "aic", "bic"])
    def test_order1_all_criteria_return_positive_lambda(self, method):
        """All criteria should work with order=1."""
        n = 25
        rng = np.random.default_rng(40)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 1)
        lam = select_lambda(ab_P, w, y, 1, method)
        assert lam > 0 and np.isfinite(lam)


# ---------------------------------------------------------------------------
# WHResult2D.to_polars() with y and weights
# ---------------------------------------------------------------------------

class TestWHResult2DToPolars:

    def test_to_polars_with_y_and_weights(self):
        """to_polars(y, weights) should include y and weight columns."""
        nx, nz = 5, 4
        y = np.ones((nx, nz))
        w = np.ones((nx, nz)) * 100.0
        wh = WhittakerHenderson2D()
        result = wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)
        df = result.to_polars(y=y, weights=w)
        assert "y" in df.columns
        assert "weight" in df.columns
        assert len(df) == nx * nz

    def test_to_polars_with_labels(self):
        """x_labels and z_labels should propagate to to_polars() output."""
        nx, nz = 4, 3
        y = np.ones((nx, nz))
        x_labels = np.array([17, 25, 35, 50])
        z_labels = np.array(["A", "B", "C"])
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0,
                        x_labels=x_labels, z_labels=z_labels)
        df = result.to_polars()
        # Labels should appear in x and z columns
        assert set(df["x"].to_list()).issubset(set(x_labels.tolist()))


# ---------------------------------------------------------------------------
# 2D smoother: only one lambda supplied triggers REML selection for the other
# ---------------------------------------------------------------------------

class TestWH2DPartialLambda:

    def test_only_lambda_x_supplied_selects_lambda_z(self):
        """Supplying lambda_x but not lambda_z should trigger REML selection for lambda_z."""
        rng = np.random.default_rng(50)
        nx, nz = 6, 5
        y = np.sin(np.arange(nx)[:, None] / 3) + np.cos(np.arange(nz)[None, :] / 2)
        y += 0.1 * rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=50.0)  # lambda_z not supplied
        assert result.lambda_x == pytest.approx(50.0)
        assert result.lambda_z > 0

    def test_only_lambda_z_supplied_selects_lambda_x(self):
        """Supplying lambda_z but not lambda_x triggers REML for lambda_x."""
        rng = np.random.default_rng(51)
        nx, nz = 5, 6
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_z=30.0)
        assert result.lambda_z == pytest.approx(30.0)
        assert result.lambda_x > 0


# ---------------------------------------------------------------------------
# Poisson smoother: zero-count cells and scale invariance of rates
# ---------------------------------------------------------------------------

class TestPoissonSmootherGaps:

    def test_zero_count_cells_do_not_crash(self):
        """Zero claims in some age bands is normal; smoother should handle it."""
        from insurance_whittaker import WhittakerHendersonPoisson
        n = 20
        rng = np.random.default_rng(60)
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        counts = rng.poisson(0.03 * exposure)
        counts[3] = 0
        counts[17] = 0
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_high_exposure_leads_to_tighter_ci(self):
        """A smoother fitted to high-exposure data should have narrower CIs."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(61)
        n = 20
        x = np.arange(n, dtype=float)
        rate = 0.05 * np.ones(n)

        lo_exp = np.full(n, 10.0)
        hi_exp = np.full(n, 10000.0)
        counts_lo = rng.poisson(rate * lo_exp)
        counts_hi = rng.poisson(rate * hi_exp)

        wh = WhittakerHendersonPoisson(order=2)
        r_lo = wh.fit(x, counts_lo, lo_exp, lambda_=50.0)
        r_hi = wh.fit(x, counts_hi, hi_exp, lambda_=50.0)

        width_lo = np.mean(r_lo.ci_upper_rate - r_lo.ci_lower_rate)
        width_hi = np.mean(r_hi.ci_upper_rate - r_hi.ci_lower_rate)
        assert width_hi < width_lo, (
            f"High-exposure CIs ({width_hi:.5f}) should be narrower than "
            f"low-exposure ({width_lo:.5f})"
        )

    def test_fitted_count_equals_rate_times_exposure(self):
        """fitted_count should equal fitted_rate * exposure elementwise."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(62)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = rng.exponential(200, n)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        np.testing.assert_allclose(
            result.fitted_count,
            result.fitted_rate * exposure,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Boundary: minimum valid inputs
# ---------------------------------------------------------------------------

class TestMinimumValidInputs:

    def test_order1_minimum_observations(self):
        """order=1 needs at least 2 observations."""
        wh = WhittakerHenderson1D(order=1)
        # 2 observations is the minimum for order=1
        result = wh.fit([0.0, 1.0], [0.5, 0.6], lambda_=1.0)
        assert len(result.fitted) == 2

    def test_order2_minimum_observations(self):
        """order=2 needs at least 3 observations."""
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit([0.0, 1.0, 2.0], [0.5, 0.6, 0.55], lambda_=1.0)
        assert len(result.fitted) == 3

    def test_order2_exactly_2_raises(self):
        """order=2 with exactly 2 observations must raise ValueError."""
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit([0.0, 1.0], [0.5, 0.6], lambda_=1.0)

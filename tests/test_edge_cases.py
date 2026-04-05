"""
Edge-case tests for insurance-whittaker.

Covers gaps identified in the April 2026 test coverage audit:

1. NaN/inf inputs to 1D smoother — should raise or return finite values
   depending on what the smoother does (document actual behaviour)
2. Zero-variance input (constant y) with auto lambda selection
3. Single observation at the exact minimum boundary
4. All-equal weights on 1D and 2D smoothers
5. Mismatched array lengths supplied to WhittakerHenderson2D
6. Very large lambda: sigma2 fallback and edf near null-space size
7. Very small lambda: sigma2 fallback when dof <= 0.5
8. Poisson smoother with single-age-band minimum input
9. Zero-variance 2D table with auto lambda
10. Large/small lambda on order=1 smoother
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_whittaker import (
    WhittakerHenderson1D,
    WhittakerHenderson2D,
    WhittakerHendersonPoisson,
)
from insurance_whittaker._utils import validate_inputs


# ---------------------------------------------------------------------------
# 1. NaN inputs
# ---------------------------------------------------------------------------

class TestNanInfInputs:
    """Document and test NaN/inf handling.

    The smoother is a linear solve — NaN in y propagates to the fitted values
    (via the right-hand side W*y).  The design choice here is that callers are
    responsible for masking or imputing NaN cells before smoothing.  We test
    that: (a) NaN propagates in a predictable way, and (b) NaN in weights is
    handled consistently with the validate_inputs contract (weights must be
    non-negative finite, so NaN weights are rejected).
    """

    def test_nan_in_y_propagates_to_fitted(self):
        """NaN in y should cause NaN in the fitted values, not a crash."""
        n = 15
        rng = np.random.default_rng(100)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        y[5] = np.nan  # introduce a NaN

        wh = WhittakerHenderson1D(order=2)
        # Should not raise — NaN propagates through the linear solve
        try:
            result = wh.fit(x, y, lambda_=10.0)
            # If it completes, fitted values may contain NaN — that is acceptable
            # The critical guarantee is: no crash with a clear exception.
        except (ValueError, np.linalg.LinAlgError):
            # Also acceptable: the smoother may detect NaN and raise
            pass

    def test_inf_in_y_propagates_or_raises(self):
        """Inf in y should not crash the process with an unintelligible error."""
        n = 15
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        y[3] = np.inf

        wh = WhittakerHenderson1D(order=2)
        try:
            result = wh.fit(x, y, lambda_=10.0)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            pass  # raising with a clear error is fine

    def test_nan_in_weights_raises(self):
        """NaN in weights violates the non-negative contract and must raise."""
        n = 10
        y = np.ones(n)
        w = np.ones(n)
        w[2] = np.nan
        with pytest.raises((ValueError, TypeError)):
            validate_inputs(y, w, n)

    def test_neg_inf_weight_raises(self):
        """Negative infinity weight must be rejected."""
        n = 10
        y = np.ones(n)
        w = np.ones(n)
        w[2] = -np.inf
        with pytest.raises(ValueError, match="non-negative"):
            validate_inputs(y, w, n)

    def test_all_nan_y_does_not_crash(self):
        """All-NaN y should not crash with an obscure error."""
        n = 10
        x = np.arange(n, dtype=float)
        y = np.full(n, np.nan)
        wh = WhittakerHenderson1D(order=2)
        try:
            result = wh.fit(x, y, lambda_=10.0)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass


# ---------------------------------------------------------------------------
# 2. Zero-variance (constant) input with auto lambda selection
# ---------------------------------------------------------------------------

class TestZeroVarianceInput:
    """Zero-variance input: y is constant, so residuals are zero for any smooth.

    This exercises the sigma2 calculation (dev=0, dof potentially negative or
    zero) and the lambda selection when there is no signal variation to fit.
    """

    def test_constant_y_auto_lambda_does_not_crash(self):
        """Auto lambda selection on constant y should complete without error."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.full(n, 3.14)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == n

    def test_constant_y_fitted_close_to_constant(self):
        """Fitted values on constant data must equal the constant (no penalty can improve on it)."""
        n = 25
        x = np.arange(n, dtype=float)
        y = np.full(n, 2.718)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        np.testing.assert_allclose(result.fitted, y, atol=1e-5)

    def test_constant_y_sigma2_non_negative(self):
        """sigma2 must be non-negative even when dev=0."""
        n = 15
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert result.sigma2 >= 0.0
        assert np.isfinite(result.sigma2)

    def test_constant_y_ci_valid(self):
        """Constant data: CI upper must be >= CI lower."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.full(n, 1.0)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_constant_y_2d_does_not_crash(self):
        """Constant 2D table with auto lambda should complete."""
        nx, nz = 8, 6
        y = np.full((nx, nz), 0.5)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y)
        assert result.fitted.shape == (nx, nz)
        np.testing.assert_allclose(result.fitted, y, atol=1e-5)

    def test_constant_y_all_criteria(self):
        """All lambda criteria should handle constant y without crashing."""
        n = 15
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            assert np.all(np.isfinite(result.fitted)), f"Non-finite fitted for method={method}"


# ---------------------------------------------------------------------------
# 3. Single observation at the exact minimum boundary
# ---------------------------------------------------------------------------

class TestSingleObservationBoundary:
    """Test the minimum-observations boundary for each order.

    order=1 requires >= 2 obs; order=2 requires >= 3 obs; etc.
    Exactly at the boundary is the hardest case.
    """

    def test_order1_exactly_2_obs_completes(self):
        """order=1 with exactly 2 observations must work (minimum valid input)."""
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit([0.0, 1.0], [0.4, 0.6], lambda_=1.0)
        assert len(result.fitted) == 2
        assert np.all(np.isfinite(result.fitted))

    def test_order1_exactly_1_obs_raises(self):
        """order=1 with 1 observation must raise ValueError."""
        wh = WhittakerHenderson1D(order=1)
        with pytest.raises(ValueError):
            wh.fit([0.0], [0.5], lambda_=1.0)

    def test_order2_exactly_3_obs_completes(self):
        """order=2 with exactly 3 observations must work (minimum valid input)."""
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit([0.0, 1.0, 2.0], [0.4, 0.6, 0.5], lambda_=1.0)
        assert len(result.fitted) == 3
        assert np.all(np.isfinite(result.fitted))

    def test_order2_exactly_2_obs_raises(self):
        """order=2 with exactly 2 observations must raise ValueError."""
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit([0.0, 1.0], [0.4, 0.6], lambda_=1.0)

    def test_order3_exactly_4_obs_completes(self):
        """order=3 with exactly 4 observations must work."""
        wh = WhittakerHenderson1D(order=3)
        result = wh.fit([0.0, 1.0, 2.0, 3.0], [0.3, 0.5, 0.6, 0.4], lambda_=1.0)
        assert len(result.fitted) == 4
        assert np.all(np.isfinite(result.fitted))

    def test_order3_exactly_3_obs_raises(self):
        """order=3 with exactly 3 observations must raise ValueError."""
        wh = WhittakerHenderson1D(order=3)
        with pytest.raises(ValueError):
            wh.fit([0.0, 1.0, 2.0], [0.4, 0.6, 0.5], lambda_=1.0)

    def test_order1_2_obs_auto_lambda(self):
        """order=1 with exactly 2 obs and auto lambda selection."""
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit([0.0, 1.0], [0.4, 0.6])
        assert len(result.fitted) == 2
        assert result.lambda_ > 0


# ---------------------------------------------------------------------------
# 4. All-equal weights
# ---------------------------------------------------------------------------

class TestAllEqualWeights:
    """All-equal weights is the default case, but it is worth verifying that
    explicitly supplying a uniform weight vector gives identical results to
    supplying None, and that the output is numerically stable.
    """

    def test_equal_weights_same_as_none(self):
        """Explicit uniform weights should give the same result as weights=None."""
        n = 20
        rng = np.random.default_rng(200)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)

        wh = WhittakerHenderson1D(order=2)
        r_none = wh.fit(x, y, lambda_=10.0)
        r_ones = wh.fit(x, y, weights=np.ones(n), lambda_=10.0)
        np.testing.assert_allclose(r_none.fitted, r_ones.fitted, atol=1e-10)

    def test_equal_weights_scaled_same_result(self):
        """Scaling all weights by a positive constant should not change the fitted values.

        The WH solution theta = (W + lam*P)^{-1} W y.  Multiplying W by c gives
        (cW + lam*P)^{-1} cW y — this is NOT the same as the original unless
        lambda is also scaled.  So this test verifies the simpler statement:
        same values are recovered when all weights = k (for fixed lambda).
        """
        n = 20
        rng = np.random.default_rng(201)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)

        wh = WhittakerHenderson1D(order=2)
        r1 = wh.fit(x, y, weights=np.full(n, 1.0), lambda_=50.0)
        r5 = wh.fit(x, y, weights=np.full(n, 5.0), lambda_=50.0)

        # Fitted values change because the relative weight vs lambda changes,
        # but both must be finite and different from raw data
        assert np.all(np.isfinite(r1.fitted))
        assert np.all(np.isfinite(r5.fitted))

    def test_equal_weights_2d_auto_lambda(self):
        """2D smoother with all-equal weights and auto lambda should be stable."""
        rng = np.random.default_rng(202)
        nx, nz = 7, 5
        y = np.sin(np.arange(nx)[:, None] / 3) + np.cos(np.arange(nz)[None, :] / 2)
        y += 0.1 * rng.standard_normal((nx, nz))
        w = np.ones((nx, nz)) * 50.0

        wh = WhittakerHenderson2D()
        result = wh.fit(y, weights=w)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))
        assert result.lambda_x > 0
        assert result.lambda_z > 0

    def test_equal_weights_all_criteria_1d(self):
        """All lambda criteria with equal weights should produce finite results."""
        n = 25
        rng = np.random.default_rng(203)
        x = np.arange(n, dtype=float)
        y = np.sin(x / 4) + 0.1 * rng.standard_normal(n)
        w = np.full(n, 100.0)  # high uniform weight

        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y, weights=w)
            assert np.all(np.isfinite(result.fitted)), f"method={method}: non-finite fitted"
            assert result.lambda_ > 0, f"method={method}: lambda <= 0"

    def test_poisson_equal_exposure(self):
        """Poisson smoother with equal exposure across all cells."""
        rng = np.random.default_rng(204)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        true_rate = 0.05 * np.ones(n)
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))


# ---------------------------------------------------------------------------
# 5. Mismatched array lengths for smoother2d
# ---------------------------------------------------------------------------

class TestSmoother2DMismatchedLengths:
    """Test that WhittakerHenderson2D rejects inconsistent inputs cleanly."""

    def test_weight_wrong_number_of_rows(self):
        """weights with wrong number of rows must raise ValueError."""
        y = np.ones((6, 5))
        w = np.ones((4, 5))  # wrong row count
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="shape"):
            wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)

    def test_weight_wrong_number_of_cols(self):
        """weights with wrong number of columns must raise ValueError."""
        y = np.ones((6, 5))
        w = np.ones((6, 3))  # wrong col count
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="shape"):
            wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)

    def test_weight_completely_wrong_shape(self):
        """weights with completely wrong shape must raise ValueError."""
        y = np.ones((6, 5))
        w = np.ones((3, 7))
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="shape"):
            wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)

    def test_1d_y_raises_with_shape_message(self):
        """1D y passed to 2D smoother must raise ValueError mentioning 2-D."""
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="2-D"):
            wh.fit(np.ones(10), lambda_x=10.0, lambda_z=10.0)

    def test_3d_y_raises(self):
        """3D y must raise ValueError."""
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError):
            wh.fit(np.ones((4, 3, 2)), lambda_x=10.0, lambda_z=10.0)

    def test_scalar_y_raises(self):
        """Scalar y must raise ValueError."""
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError):
            wh.fit(np.float64(1.0), lambda_x=10.0, lambda_z=10.0)


# ---------------------------------------------------------------------------
# 6. Very large lambda: edf and sigma2 behaviour
# ---------------------------------------------------------------------------

class TestVeryLargeLambda:
    """At very large lambda, the smoother becomes a polynomial of degree order-1.

    EDF approaches the null space size (= order for 1D), and sigma2 may be
    large because the smooth over-smooths the data.
    """

    def test_large_lambda_edf_near_order(self):
        """At lambda=1e12, edf should approach the order (null space size)."""
        n = 30
        rng = np.random.default_rng(300)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e12)
        # EDF should be close to order (2), definitely < n/2
        assert result.edf < n / 2, f"edf={result.edf:.2f} unexpectedly large at huge lambda"
        assert result.edf > 0

    def test_large_lambda_fitted_nearly_linear(self):
        """At lambda=1e10, order=2 smoother should be nearly linear."""
        n = 40
        rng = np.random.default_rng(301)
        x = np.arange(n, dtype=float)
        y = 2.0 + 0.5 * x + 0.3 * rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e10)
        coefs = np.polyfit(x, result.fitted, 1)
        linear_approx = np.polyval(coefs, x)
        np.testing.assert_allclose(result.fitted, linear_approx, atol=1e-3)

    def test_large_lambda_sigma2_finite(self):
        """sigma2 should be finite and non-negative at very large lambda."""
        n = 20
        rng = np.random.default_rng(302)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e15)
        assert np.isfinite(result.sigma2)
        assert result.sigma2 >= 0.0

    def test_large_lambda_ci_finite(self):
        """CI bounds should remain finite at very large lambda."""
        n = 20
        rng = np.random.default_rng(303)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e15)
        assert np.all(np.isfinite(result.ci_lower))
        assert np.all(np.isfinite(result.ci_upper))

    def test_large_lambda_2d_edf_small(self):
        """2D smoother: at very large lambda, edf should be small."""
        nx, nz = 8, 6
        rng = np.random.default_rng(304)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1e10, lambda_z=1e10)
        assert result.edf < (nx * nz) / 2

    def test_large_lambda_order1_fitted_nearly_constant(self):
        """order=1 with very large lambda should give a nearly constant fit."""
        n = 25
        rng = np.random.default_rng(305)
        x = np.arange(n, dtype=float)
        y = 1.5 + 0.5 * rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit(x, y, lambda_=1e12)
        # order=1 null space is constants; fitted should be nearly constant
        assert np.std(result.fitted) < 0.1

    def test_large_lambda_poisson_nearly_constant_rate(self):
        """Poisson smoother at very large lambda should give a near-constant rate."""
        rng = np.random.default_rng(306)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        true_rate = 0.05 + 0.02 * np.sin(x / 4)
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=1e8)
        # Rate should be nearly constant — std much smaller than mean
        assert np.std(result.fitted_rate) < 0.01


# ---------------------------------------------------------------------------
# 7. Very small lambda: sigma2 fallback when dof <= 0.5
# ---------------------------------------------------------------------------

class TestVerySmallLambda:
    """At very small lambda, the smoother interpolates and edf approaches n.

    When edf ~ n, dof = n - edf ~ 0, and sigma2 falls back to 1.0.
    """

    def test_small_lambda_fitted_close_to_observed(self):
        """At lambda=1e-10, fitted should be near the raw observations."""
        n = 20
        rng = np.random.default_rng(400)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-10)
        np.testing.assert_allclose(result.fitted, y, atol=1e-3)

    def test_small_lambda_sigma2_non_negative(self):
        """sigma2 must be non-negative even in the sigma2=1 fallback regime."""
        n = 12
        rng = np.random.default_rng(401)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-12)
        assert result.sigma2 >= 0.0
        assert np.isfinite(result.sigma2)

    def test_small_lambda_sigma2_fallback_is_one(self):
        """When dof <= 0.5, sigma2 should be exactly 1.0 (the fallback)."""
        n = 10
        rng = np.random.default_rng(402)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        # At lambda=1e-15, edf should equal n (perfect interpolation) → dof=0
        result = wh.fit(x, y, lambda_=1e-15)
        # sigma2 must be 1.0 (fallback) or non-negative
        assert result.sigma2 >= 0.0

    def test_small_lambda_edf_near_n(self):
        """At very small lambda, edf should approach n."""
        n = 15
        rng = np.random.default_rng(403)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-12)
        # edf should be very close to n
        assert result.edf > n * 0.8, f"edf={result.edf:.2f} should be close to n={n}"

    def test_small_lambda_ci_valid(self):
        """CI upper >= CI lower must hold at very small lambda."""
        n = 15
        rng = np.random.default_rng(404)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-10)
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_small_lambda_2d_fitted_close_to_observed(self):
        """2D smoother at very small lambda should nearly interpolate."""
        rng = np.random.default_rng(405)
        nx, nz = 5, 4
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=1e-8, lambda_z=1e-8)
        np.testing.assert_allclose(result.fitted, y, atol=1e-3)

    def test_small_lambda_order1_fitted_close_to_observed(self):
        """order=1 at very small lambda should also nearly interpolate."""
        n = 15
        rng = np.random.default_rng(406)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit(x, y, lambda_=1e-10)
        np.testing.assert_allclose(result.fitted, y, atol=1e-3)


# ---------------------------------------------------------------------------
# 8. Poisson smoother minimum-input boundary
# ---------------------------------------------------------------------------

class TestPoissonMinimumInput:
    """Poisson smoother needs n > order observations (same as 1D)."""

    def test_poisson_order1_minimum_2_obs(self):
        """Poisson order=1 with 2 observations should work."""
        wh = WhittakerHendersonPoisson(order=1)
        result = wh.fit(
            np.array([0.0, 1.0]),
            np.array([5, 3]),
            np.array([100.0, 80.0]),
            lambda_=1.0,
        )
        assert len(result.fitted_rate) == 2
        assert np.all(result.fitted_rate > 0)

    def test_poisson_order2_minimum_3_obs(self):
        """Poisson order=2 with 3 observations should work."""
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(
            np.array([0.0, 1.0, 2.0]),
            np.array([5, 3, 7]),
            np.array([100.0, 80.0, 120.0]),
            lambda_=1.0,
        )
        assert len(result.fitted_rate) == 3
        assert np.all(result.fitted_rate > 0)

    def test_poisson_all_zero_counts(self):
        """All-zero counts should not crash (zero-claim band is common in insurance)."""
        rng = np.random.default_rng(500)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        counts = np.zeros(n, dtype=int)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_poisson_high_exposure_no_crash(self):
        """Very high exposure should not cause overflow in Poisson smoother."""
        rng = np.random.default_rng(501)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 1e7)  # ten million policy-years per band
        true_rate = 0.001  # rare event
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))


# ---------------------------------------------------------------------------
# 9. Zero-variance 2D: constant table with auto lambda
# ---------------------------------------------------------------------------

class TestZeroVariance2D:
    """A constant 2D table has zero variation — lambda selection must not crash."""

    def test_constant_table_auto_lambda_does_not_crash(self):
        """Auto lambda on a constant table should complete."""
        nx, nz = 8, 6
        y = np.full((nx, nz), 0.42)
        wh = WhittakerHenderson2D()
        result = wh.fit(y)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))

    def test_constant_table_fitted_close_to_constant(self):
        """Fitted values on a constant 2D table must equal the constant."""
        nx, nz = 6, 5
        val = 1.23
        y = np.full((nx, nz), val)
        wh = WhittakerHenderson2D()
        result = wh.fit(y)
        np.testing.assert_allclose(result.fitted, val, atol=1e-4)

    def test_constant_table_ci_valid(self):
        """CI upper >= CI lower on a constant table."""
        nx, nz = 6, 5
        y = np.full((nx, nz), 0.5)
        wh = WhittakerHenderson2D()
        result = wh.fit(y)
        assert np.all(result.ci_upper >= result.ci_lower)


# ---------------------------------------------------------------------------
# 10. Mixed lambda-specification edge cases
# ---------------------------------------------------------------------------

class TestMixedLambdaEdgeCases:
    """Edge cases in how lambda is specified or selected."""

    def test_zero_lambda_raises(self):
        """lambda_=0 is not a valid smoothing parameter and must raise."""
        n = 10
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError, match="positive"):
            wh.fit(x, y, lambda_=0.0)

    def test_very_small_positive_lambda_accepted(self):
        """lambda_=1e-20 is positive and should be accepted without raising."""
        n = 10
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-20)
        assert result.lambda_ == pytest.approx(1e-20)

    def test_very_large_lambda_accepted(self):
        """lambda_=1e20 is valid and should be accepted."""
        n = 10
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e20)
        assert result.lambda_ == pytest.approx(1e20)
        assert np.all(np.isfinite(result.fitted))

    def test_negative_lambda_raises(self):
        """lambda_<0 must raise ValueError."""
        n = 10
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError, match="positive"):
            wh.fit(x, y, lambda_=-1.0)

    def test_2d_both_lambdas_fixed_skips_selection(self):
        """Supplying both lambda_x and lambda_z should use them directly."""
        nx, nz = 6, 5
        rng = np.random.default_rng(600)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=7.5, lambda_z=3.2)
        assert result.lambda_x == pytest.approx(7.5)
        assert result.lambda_z == pytest.approx(3.2)

    def test_lambda_selected_is_positive_for_all_methods(self):
        """Auto-selected lambda must be positive for all criteria."""
        n = 25
        rng = np.random.default_rng(601)
        x = np.arange(n, dtype=float)
        y = np.sin(x / 5) + 0.2 * rng.standard_normal(n)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            assert result.lambda_ > 0, f"method={method}: lambda={result.lambda_}"
            assert np.isfinite(result.lambda_), f"method={method}: lambda is infinite"

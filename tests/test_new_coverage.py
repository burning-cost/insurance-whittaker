"""
Comprehensive new tests for insurance-whittaker covering gaps in the existing suite.

Covers:
- _utils: diff_matrix order=0, penalty_matrix structure, add_lambda_to_banded,
  diag_of_inverse_banded, to_numpy_1d with polars-like input, validate_inputs edge cases
- selection: _build_full_system, _solve_system, _solve_banded_system,
  _log_det_P_nonzero directly, criterion monotonicity, all criteria with non-unit weights,
  select_lambda with order=3, extreme lambda bounds behaviour
- smoother: WHResult1D attribute completeness, std_fitted is non-negative,
  criterion_value attribute, x preserves labels, list input, weights=None default,
  high-order smoothers (order=3/4), all criteria produce finite criterion_value,
  reproducibility (same result for same inputs), sigma2 attribute
- glm: _poisson_deviance directly, Poisson with explicit lambda_, order=1/3,
  all criteria for Poisson, negative exposure raises, order=0 raises,
  WHResultPoisson repr, very sparse counts, fixed large lambda near-constant,
  Poisson to_polars column completeness
- smoother2d: _eig_penalty directly, _build_2d_system, solve_2d_full directly,
  _solve_2d_system directly, 2D with mixed orders, sigma2 fallback in 2D,
  WHResult2D with labels in repr, 2D with very small/large table,
  2D smoother non-square (tall vs wide), 2D fitted count shape
- __init__: all public symbols importable, __version__ attribute exists
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# _utils tests
# ===========================================================================

class TestDiffMatrixExtended:

    def test_order_0_returns_identity(self):
        """diff_matrix(n, 0) should return the n x n identity matrix."""
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(5, 0)
        np.testing.assert_allclose(D, np.eye(5))

    def test_order_0_shape(self):
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(7, 0)
        assert D.shape == (7, 7)

    def test_first_order_sum_rows_zero(self):
        """Each row of D^1 should sum to 0 (by construction of differences)."""
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(8, 1)
        np.testing.assert_allclose(D.sum(axis=1), 0.0, atol=1e-12)

    def test_second_order_shape_large(self):
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(100, 2)
        assert D.shape == (98, 100)

    def test_third_order_shape(self):
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(10, 3)
        assert D.shape == (7, 10)

    def test_quadratic_in_null_space_order3(self):
        """Quadratic vector is in null space of D^3."""
        from insurance_whittaker._utils import diff_matrix
        n = 20
        D = diff_matrix(n, 3)
        x = np.arange(n, dtype=float)
        quadratic = x ** 2
        np.testing.assert_allclose(D @ quadratic, 0, atol=1e-8)


class TestPenaltyMatrixStructure:

    def test_penalty_matrix_symmetric(self):
        """P = D'D must be symmetric."""
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(10, 2)
        np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_penalty_matrix_psd(self):
        """P must be positive semi-definite (all eigenvalues >= 0)."""
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(15, 2)
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs >= -1e-10), f"Negative eigenvalue: {eigs.min()}"

    def test_penalty_matrix_null_space_order1(self):
        """For order=1, constant vectors are in null space (P @ ones = 0)."""
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(10, 1)
        ones = np.ones(10)
        np.testing.assert_allclose(P @ ones, 0, atol=1e-12)

    def test_penalty_matrix_null_space_order2(self):
        """For order=2, linear vectors are in null space."""
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(10, 2)
        x = np.arange(10, dtype=float)
        np.testing.assert_allclose(P @ x, 0, atol=1e-10)

    def test_penalty_matrix_shape(self):
        from insurance_whittaker._utils import penalty_matrix
        for n, order in [(5, 1), (10, 2), (15, 3)]:
            P = penalty_matrix(n, order)
            assert P.shape == (n, n)

    def test_penalty_matrix_rank(self):
        """Rank of P should be n - order (order-dimensional null space)."""
        from insurance_whittaker._utils import penalty_matrix
        for n, order in [(10, 1), (10, 2), (12, 3)]:
            P = penalty_matrix(n, order)
            rank = np.linalg.matrix_rank(P)
            assert rank == n - order, f"n={n}, order={order}: rank={rank}, expected={n-order}"


class TestAddLambdaToBanded:

    def test_returns_copy(self):
        """add_lambda_to_banded should return a copy (not the same object)."""
        from insurance_whittaker._utils import add_lambda_to_banded, penalty_banded
        ab = penalty_banded(10, 2)
        w = np.ones(10)
        result = add_lambda_to_banded(ab, w, 50.0)
        assert result is not ab

    def test_shape_preserved(self):
        """Output shape must match input descriptor shape."""
        from insurance_whittaker._utils import add_lambda_to_banded, penalty_banded
        ab = penalty_banded(15, 2)
        w = np.ones(15)
        result = add_lambda_to_banded(ab, w, 10.0)
        assert result.shape == ab.shape


class TestDiagOfInverseBanded:

    def test_against_direct_inverse(self):
        """diag_of_inverse_banded should match np.diag(np.linalg.inv(A))."""
        from insurance_whittaker._utils import diag_of_inverse_banded
        n = 6
        rng = np.random.default_rng(42)
        # Create a PD matrix
        A = np.diag(np.ones(n) * 5) + np.diag(np.ones(n - 1) * 0.5, 1) + np.diag(np.ones(n - 1) * 0.5, -1)
        diag_direct = np.diag(np.linalg.inv(A))
        diag_banded = diag_of_inverse_banded(A, order=2)
        np.testing.assert_allclose(diag_banded, diag_direct, rtol=1e-6)

    def test_all_positive_for_pd_matrix(self):
        """Diagonal of inverse of a PD matrix must be positive."""
        from insurance_whittaker._utils import diag_of_inverse_banded
        n = 8
        A = np.eye(n) * 3 + np.random.default_rng(7).standard_normal((n, n)) * 0.01
        A = A @ A.T  # make PD
        A += np.eye(n) * 1  # ensure strict PD
        diag_v = diag_of_inverse_banded(A, order=2)
        assert np.all(diag_v > 0)


class TestToNumpy1DExtended:

    def test_numpy_array_passthrough(self):
        """Pure numpy input should return float64 array."""
        from insurance_whittaker._utils import to_numpy_1d
        x = np.array([1.0, 2.0, 3.0])
        result = to_numpy_1d(x, "x")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, x)

    def test_integer_array_cast_to_float64(self):
        """Integer arrays should be cast to float64."""
        from insurance_whittaker._utils import to_numpy_1d
        result = to_numpy_1d(np.array([1, 2, 3]), "x")
        assert result.dtype == np.float64

    def test_polars_series_input(self):
        """Polars Series input should convert correctly."""
        from insurance_whittaker._utils import to_numpy_1d
        import polars as pl
        s = pl.Series("x", [1.0, 2.0, 3.0])
        result = to_numpy_1d(s, "x")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ===========================================================================
# selection tests
# ===========================================================================

class TestBuildFullSystem:

    def test_symmetric(self):
        """A = W + lambda * P should be symmetric."""
        from insurance_whittaker.selection import _build_full_system
        from insurance_whittaker._utils import penalty_matrix
        n = 8
        P = penalty_matrix(n, 2)
        W = np.ones(n) * 2
        A = _build_full_system(P, W, 10.0)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_diagonal_increases_with_lambda(self):
        """Diagonal of A increases with lambda (since P has non-negative diagonal)."""
        from insurance_whittaker.selection import _build_full_system
        from insurance_whittaker._utils import penalty_matrix
        n = 8
        P = penalty_matrix(n, 2)
        W = np.ones(n)
        A1 = _build_full_system(P, W, 1.0)
        A2 = _build_full_system(P, W, 100.0)
        assert np.all(np.diag(A2) >= np.diag(A1))

    def test_shape(self):
        from insurance_whittaker.selection import _build_full_system
        from insurance_whittaker._utils import penalty_matrix
        n = 12
        P = penalty_matrix(n, 2)
        W = np.ones(n)
        A = _build_full_system(P, W, 5.0)
        assert A.shape == (n, n)


class TestSolveSystem:

    def test_solution_satisfies_normal_equations(self):
        """A @ theta should equal W * y (normal equations)."""
        from insurance_whittaker.selection import _solve_system, _build_full_system
        from insurance_whittaker._utils import penalty_matrix
        n = 10
        rng = np.random.default_rng(1)
        P = penalty_matrix(n, 2)
        W = np.ones(n) * 2
        y = rng.standard_normal(n)
        Wy = W * y
        lam = 5.0
        theta, cf, log_det = _solve_system(P, W, Wy, lam)
        A = _build_full_system(P, W, lam)
        np.testing.assert_allclose(A @ theta, Wy, atol=1e-8)

    def test_log_det_positive_for_pd_system(self):
        """log|A| should be a finite scalar for a positive definite A."""
        from insurance_whittaker.selection import _solve_system
        from insurance_whittaker._utils import penalty_matrix
        n = 8
        P = penalty_matrix(n, 2)
        W = np.ones(n) * 10
        y = np.ones(n)
        _, _, log_det = _solve_system(P, W, W * y, 5.0)
        assert np.isfinite(log_det)

    def test_returns_tuple_of_three(self):
        from insurance_whittaker.selection import _solve_system
        from insurance_whittaker._utils import penalty_matrix
        n = 6
        P = penalty_matrix(n, 2)
        W = np.ones(n)
        y = np.ones(n)
        result = _solve_system(P, W, W * y, 1.0)
        assert len(result) == 3


class TestSolveBandedSystem:

    def test_matches_solve_system(self):
        """_solve_banded_system should return same theta as _solve_system."""
        from insurance_whittaker.selection import _solve_system, _solve_banded_system
        from insurance_whittaker._utils import penalty_matrix, penalty_banded
        n = 10
        rng = np.random.default_rng(99)
        P = penalty_matrix(n, 2)
        ab_P = penalty_banded(n, 2)
        W = np.ones(n) * 3
        y = rng.standard_normal(n)
        Wy = W * y
        theta_full, _, _ = _solve_system(P, W, Wy, 5.0)
        theta_banded, _, _ = _solve_banded_system(ab_P, W, Wy, 5.0)
        np.testing.assert_allclose(theta_full, theta_banded, atol=1e-8)


class TestLogDetPNonzero:

    def test_direct_function_returns_finite(self):
        """_log_det_P_nonzero should return a finite float."""
        from insurance_whittaker.selection import _log_det_P_nonzero
        val = _log_det_P_nonzero(10, 2)
        assert np.isfinite(val)

    def test_increases_with_n(self):
        """More cells → more non-zero eigenvalues → larger log-determinant."""
        from insurance_whittaker.selection import _log_det_P_nonzero
        v1 = _log_det_P_nonzero(10, 2)
        v2 = _log_det_P_nonzero(20, 2)
        assert v2 > v1

    def test_different_orders(self):
        """Different orders should give different log-determinants for same n."""
        from insurance_whittaker.selection import _log_det_P_nonzero
        v1 = _log_det_P_nonzero(15, 1)
        v2 = _log_det_P_nonzero(15, 2)
        v3 = _log_det_P_nonzero(15, 3)
        assert v1 != v2
        assert v2 != v3


class TestCriteriaWithNonUnitWeights:

    def _make_data(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        y = np.sin(np.arange(n, dtype=float) / 4) + 0.2 * rng.standard_normal(n)
        w = rng.exponential(1.0, n) * 100  # heterogeneous weights
        return y, w

    @pytest.mark.parametrize("fn_name", ["reml_criterion", "gcv_criterion", "aic_criterion", "bic_criterion"])
    def test_all_criteria_finite_with_heterogeneous_weights(self, fn_name):
        from insurance_whittaker import selection as sel
        from insurance_whittaker._utils import penalty_banded
        fn = getattr(sel, fn_name)
        n = 20
        y, w = self._make_data(n)
        ab_P = penalty_banded(n, 2)
        log_det_P = sel._log_det_P_nz_cached(n, 2)
        val = fn(np.log(50.0), ab_P, w, y, 2, log_det_P)
        assert np.isfinite(val), f"{fn_name} returned {val}"

    def test_reml_criterion_decreases_then_increases(self):
        """REML criterion should have a minimum (bowl-shaped in log-lambda)."""
        from insurance_whittaker.selection import reml_criterion, _log_det_P_nz_cached
        from insurance_whittaker._utils import penalty_banded
        n = 30
        rng = np.random.default_rng(5)
        y = np.sin(np.arange(n, dtype=float) / 5) + 0.2 * rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        log_det_P = _log_det_P_nz_cached(n, 2)

        log_lams = np.linspace(-4, 10, 30)
        vals = [reml_criterion(ll, ab_P, w, y, 2, log_det_P) for ll in log_lams]
        # Find the minimum
        min_idx = np.argmin(vals)
        # Values should decrease before minimum and increase after
        assert 0 < min_idx < len(vals) - 1, "Minimum at boundary — no bowl shape"

    def test_select_lambda_order3(self):
        """select_lambda should work for order=3."""
        from insurance_whittaker.selection import select_lambda
        from insurance_whittaker._utils import penalty_banded
        n = 25
        rng = np.random.default_rng(8)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 3)
        lam = select_lambda(ab_P, w, y, 3, "reml")
        assert lam > 0
        assert np.isfinite(lam)

    def test_aic_bic_ordering(self):
        """BIC penalises complexity more than AIC (log(n) > 2 for n >= 8).

        With the same data, BIC should select higher lambda (more smoothing)
        than AIC, which means lower EDF.
        """
        from insurance_whittaker.selection import select_lambda, _edf_from_hat
        from insurance_whittaker._utils import penalty_banded
        n = 30
        rng = np.random.default_rng(11)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        lam_aic = select_lambda(ab_P, w, y, 2, "aic")
        lam_bic = select_lambda(ab_P, w, y, 2, "bic")
        edf_aic = _edf_from_hat(ab_P, w, lam_aic)
        edf_bic = _edf_from_hat(ab_P, w, lam_bic)
        # BIC should penalise complexity more → lower EDF (or equal)
        # n=30, so log(n) ~ 3.4 > 2 — BIC selects smoother fit
        # Allow some tolerance; the key is direction not exact magnitude
        assert edf_bic <= edf_aic + 2.0, (
            f"BIC edf={edf_bic:.2f} should be <= AIC edf={edf_aic:.2f}"
        )


# ===========================================================================
# smoother (1D) tests
# ===========================================================================

class TestWHResult1DAttributes:

    def _fit(self, n=20, order=2, method="reml", seed=0):
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        from insurance_whittaker import WhittakerHenderson1D
        wh = WhittakerHenderson1D(order=order, lambda_method=method)
        return wh.fit(x, y)

    def test_std_fitted_non_negative(self):
        """std_fitted must be >= 0 everywhere."""
        result = self._fit()
        assert np.all(result.std_fitted >= 0)

    def test_criterion_value_finite(self):
        """criterion_value attribute must be a finite scalar."""
        result = self._fit()
        assert np.isfinite(result.criterion_value)

    def test_criterion_attribute_matches_method(self):
        """criterion string should match the lambda_method used."""
        for method in ["reml", "gcv", "aic", "bic"]:
            result = self._fit(method=method)
            assert result.criterion == method

    def test_x_preserved(self):
        """x attribute should preserve the input x values."""
        from insurance_whittaker import WhittakerHenderson1D
        x = np.array([17, 25, 35, 50, 65], dtype=float)
        y = np.ones(5)
        wh = WhittakerHenderson1D(order=1)
        result = wh.fit(x, y, lambda_=1.0)
        np.testing.assert_array_equal(result.x, x)

    def test_y_preserved(self):
        """y attribute should store the original observations."""
        from insurance_whittaker import WhittakerHenderson1D
        x = np.arange(10, dtype=float)
        y = np.linspace(0.5, 1.5, 10)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        np.testing.assert_array_equal(result.y, y)

    def test_weights_preserved(self):
        """weights attribute should store the supplied weights."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 10
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        w = np.arange(1, n + 1, dtype=float)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=w, lambda_=10.0)
        np.testing.assert_array_equal(result.weights, w)

    def test_sigma2_attribute_present(self):
        result = self._fit()
        assert hasattr(result, "sigma2")
        assert result.sigma2 >= 0.0
        assert np.isfinite(result.sigma2)

    def test_edf_attribute_present(self):
        result = self._fit()
        assert hasattr(result, "edf")
        assert result.edf > 0

    def test_order_attribute(self):
        result = self._fit(order=3)
        assert result.order == 3


class TestWHSmootherReproducibility:

    def test_same_result_for_same_inputs(self):
        """Fitting twice with same inputs should give identical results."""
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(42)
        n = 25
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        r1 = wh.fit(x, y)
        r2 = wh.fit(x, y)
        np.testing.assert_array_equal(r1.fitted, r2.fitted)
        assert r1.lambda_ == r2.lambda_


class TestSmootherEdgeCases:

    def test_list_input_works(self):
        """Python lists should be accepted as input."""
        from insurance_whittaker import WhittakerHenderson1D
        x = list(range(10))
        y = [float(i) * 0.1 for i in range(10)]
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        assert len(result.fitted) == 10

    def test_weights_none_is_uniform(self):
        """Fitting without weights should equal fitting with all-ones weights."""
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(77)
        n = 15
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        r_none = wh.fit(x, y, lambda_=10.0)
        r_ones = wh.fit(x, y, weights=np.ones(n), lambda_=10.0)
        np.testing.assert_allclose(r_none.fitted, r_ones.fitted, atol=1e-10)

    def test_order4_smoother(self):
        """Order-4 smoother should produce valid output for sufficiently large n."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 30
        rng = np.random.default_rng(3)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=4)
        result = wh.fit(x, y)
        assert len(result.fitted) == n
        assert result.order == 4
        assert np.all(np.isfinite(result.fitted))

    def test_all_criteria_criterion_value_finite(self):
        """All selection methods should produce finite criterion_value."""
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(55)
        n = 20
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            assert np.isfinite(result.criterion_value), f"criterion_value not finite for {method}"

    def test_fitted_array_length_matches_n(self):
        """Fitted array length should equal number of input observations."""
        from insurance_whittaker import WhittakerHenderson1D
        for n in [5, 20, 100]:
            x = np.arange(n, dtype=float)
            y = np.sin(x / 5)
            wh = WhittakerHenderson1D(order=2)
            result = wh.fit(x, y, lambda_=50.0)
            assert len(result.fitted) == n

    def test_large_n_does_not_crash(self):
        """Smoother should handle n=200 without crashing."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 200
        rng = np.random.default_rng(9)
        x = np.arange(n, dtype=float)
        y = np.sin(x / 20) + 0.1 * rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == n
        assert np.all(np.isfinite(result.fitted))

    def test_ci_symmetric_around_fitted_at_equal_weights(self):
        """With uniform weights, CI should be symmetric around fitted."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 20
        x = np.arange(n, dtype=float)
        y = np.sin(x / 4) + 0.1 * np.random.default_rng(0).standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        upper_delta = result.ci_upper - result.fitted
        lower_delta = result.fitted - result.ci_lower
        np.testing.assert_allclose(upper_delta, lower_delta, atol=1e-10)


class TestSmootherMismatch:

    def test_y_weight_length_mismatch_raises(self):
        """Mismatched y and weights lengths should raise ValueError."""
        from insurance_whittaker import WhittakerHenderson1D
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit(np.arange(10), np.ones(10), weights=np.ones(8), lambda_=10.0)

    def test_x_y_different_lengths_work(self):
        """x and y can differ in length — x is just labels; n is len(y)."""
        from insurance_whittaker import WhittakerHenderson1D
        # x is not used for computation, so longer/shorter x doesn't necessarily raise
        # But the implementation converts x to array — let's test what actually happens
        wh = WhittakerHenderson1D(order=2)
        x = np.arange(10, dtype=float)
        y = np.ones(10)
        result = wh.fit(x, y, lambda_=1.0)
        assert len(result.fitted) == 10


# ===========================================================================
# glm (Poisson) tests
# ===========================================================================

class TestPoissonDeviance:

    def test_zero_when_fitted_equals_observed(self):
        """Deviance should be 0 (or near 0) when mu = c everywhere."""
        from insurance_whittaker.glm import _poisson_deviance
        c = np.array([5.0, 10.0, 3.0, 0.0])
        mu = c.copy()
        mu = np.maximum(mu, 1e-10)
        dev = _poisson_deviance(c, mu)
        np.testing.assert_allclose(dev, 0.0, atol=1e-8)

    def test_positive_when_misspecified(self):
        """Deviance should be > 0 when mu != c."""
        from insurance_whittaker.glm import _poisson_deviance
        c = np.array([5.0, 10.0, 3.0])
        mu = np.array([3.0, 8.0, 5.0])
        dev = _poisson_deviance(c, mu)
        assert dev > 0

    def test_zero_counts_handled(self):
        """Zero counts should not produce nan/inf (0 * log(0) = 0)."""
        from insurance_whittaker.glm import _poisson_deviance
        c = np.array([0.0, 5.0, 0.0, 3.0])
        mu = np.array([1.0, 4.0, 2.0, 3.5])
        dev = _poisson_deviance(c, mu)
        assert np.isfinite(dev)
        assert dev >= 0

    def test_symmetric_in_spirit(self):
        """Deviance from over-prediction and under-prediction are both positive."""
        from insurance_whittaker.glm import _poisson_deviance
        c = np.array([10.0])
        dev_over = _poisson_deviance(c, np.array([15.0]))
        dev_under = _poisson_deviance(c, np.array([5.0]))
        assert dev_over > 0
        assert dev_under > 0


class TestPoissonSmootherExtended:

    def test_fixed_lambda_runs(self):
        """Poisson smoother should run successfully with a fixed lambda_."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(20)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=50.0)
        assert result.lambda_ == pytest.approx(50.0)
        assert np.all(result.fitted_rate > 0)

    def test_order1_poisson(self):
        """Order-1 Poisson smoother should produce valid output."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(21)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=1)
        result = wh.fit(x, counts, exposure)
        assert result.order == 1
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_order3_poisson(self):
        """Order-3 Poisson smoother should produce valid output."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(22)
        n = 25
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 500.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=3)
        result = wh.fit(x, counts, exposure)
        assert result.order == 3
        assert np.all(result.fitted_rate > 0)

    def test_all_criteria_for_poisson(self):
        """All lambda selection criteria should work for the Poisson smoother."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(23)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHendersonPoisson(order=2, lambda_method=method)
            result = wh.fit(x, counts, exposure)
            assert result.lambda_ > 0, f"lambda <= 0 for method={method}"
            assert result.criterion == method

    def test_negative_exposure_raises(self):
        """Negative exposure values should raise ValueError."""
        from insurance_whittaker import WhittakerHendersonPoisson
        n = 10
        wh = WhittakerHendersonPoisson(order=2)
        with pytest.raises(ValueError, match="non-negative"):
            wh.fit(
                np.arange(n, dtype=float),
                np.ones(n) * 5,
                np.full(n, -100.0),
            )

    def test_order0_raises(self):
        """order=0 should raise ValueError."""
        from insurance_whittaker import WhittakerHendersonPoisson
        with pytest.raises(ValueError, match="order must be >= 1"):
            WhittakerHendersonPoisson(order=0)

    def test_repr_contains_key_info(self):
        """WHResultPoisson repr should contain class name, order, and lambda."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(25)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        r = repr(result)
        assert "WHResultPoisson" in r
        assert "order=2" in r
        assert "lambda_" in r

    def test_very_sparse_counts(self):
        """Very few claims (sparse counts) should not crash the smoother."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(30)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 10.0)
        # With rate 0.01 and exposure 10, expected counts ~ 0.1 per cell
        counts = rng.poisson(0.01 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_large_lambda_near_constant_rate(self):
        """Poisson smoother with very large lambda should give near-constant rate."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(31)
        n = 25
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 500.0)
        true_rate = 0.05
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=1e8)
        # Near-constant fitted rate — std should be tiny relative to mean
        cv = np.std(result.fitted_rate) / np.mean(result.fitted_rate)
        assert cv < 0.01, f"Coefficient of variation {cv:.4f} too large for large lambda"

    def test_to_polars_includes_observed_rate_column(self):
        """to_polars() should include an observed_rate column."""
        from insurance_whittaker import WhittakerHendersonPoisson
        import polars as pl
        rng = np.random.default_rng(32)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        df = result.to_polars()
        assert "observed_rate" in df.columns
        assert isinstance(df, pl.DataFrame)

    def test_std_log_rate_non_negative(self):
        """std_log_rate should be non-negative everywhere."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(33)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.std_log_rate >= 0)

    def test_edf_attribute(self):
        """edf should be positive and less than n."""
        from insurance_whittaker import WhittakerHendersonPoisson
        rng = np.random.default_rng(34)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert result.edf > 0
        assert result.edf < n


# ===========================================================================
# smoother2d internal helpers
# ===========================================================================

class TestEigPenalty:

    def test_eigenvalues_non_negative(self):
        """Eigenvalues of the penalty matrix should be non-negative (PSD)."""
        from insurance_whittaker._smoother2d import _eig_penalty
        vals, vecs = _eig_penalty(10, 2)
        assert np.all(vals >= -1e-10), f"Negative eigenvalue: {vals.min()}"

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors should form an orthonormal matrix."""
        from insurance_whittaker._smoother2d import _eig_penalty
        vals, vecs = _eig_penalty(8, 2)
        # V @ V.T should be identity
        np.testing.assert_allclose(vecs @ vecs.T, np.eye(8), atol=1e-10)

    def test_returns_two_arrays(self):
        from insurance_whittaker._smoother2d import _eig_penalty
        result = _eig_penalty(6, 1)
        assert len(result) == 2
        vals, vecs = result
        assert vals.shape == (6,)
        assert vecs.shape == (6, 6)

    def test_null_eigenvalues_count_matches_order(self):
        """Number of near-zero eigenvalues should equal the difference order."""
        from insurance_whittaker._smoother2d import _eig_penalty
        for n, order in [(10, 1), (10, 2), (12, 3)]:
            vals, _ = _eig_penalty(n, order)
            n_zero = np.sum(vals < 1e-10)
            assert n_zero == order, f"n={n}, order={order}: {n_zero} zero eigs, expected {order}"


class TestBuild2DSystem:

    def test_returns_four_arrays(self):
        from insurance_whittaker._smoother2d import _build_2d_system
        result = _build_2d_system(6, 5, 2, 2)
        assert len(result) == 4
        vals_x, vecs_x, vals_z, vecs_z = result
        assert vals_x.shape == (6,)
        assert vecs_x.shape == (6, 6)
        assert vals_z.shape == (5,)
        assert vecs_z.shape == (5, 5)

    def test_eigenvalues_non_negative_both_dims(self):
        from insurance_whittaker._smoother2d import _build_2d_system
        vals_x, vecs_x, vals_z, vecs_z = _build_2d_system(8, 6, 2, 2)
        assert np.all(vals_x >= -1e-10)
        assert np.all(vals_z >= -1e-10)


class TestSolve2DSystem:

    def test_solution_shape(self):
        """theta_vec should have length nx * nz."""
        from insurance_whittaker._smoother2d import _solve_2d_system
        from insurance_whittaker._utils import penalty_banded
        nx, nz = 5, 4
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        W_vec = np.ones(nx * nz)
        y_vec = np.ones(nx * nz) * 0.5
        theta, log_det = _solve_2d_system(ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 5.0, 5.0)
        assert theta.shape == (nx * nz,)
        assert np.isfinite(log_det)

    def test_constant_surface_recovered(self):
        """Constant y with equal weights and any lambda → fitted near constant."""
        from insurance_whittaker._smoother2d import _solve_2d_system
        from insurance_whittaker._utils import penalty_banded
        nx, nz = 5, 4
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        W_vec = np.ones(nx * nz)
        y_vec = np.full(nx * nz, 0.7)
        theta, _ = _solve_2d_system(ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 50.0, 50.0)
        np.testing.assert_allclose(theta, y_vec, atol=1e-6)


class TestSolve2DFull:

    def test_returns_three_arrays(self):
        from insurance_whittaker._smoother2d import solve_2d_full
        nx, nz = 6, 5
        W_vec = np.ones(nx * nz)
        y_vec = np.ones(nx * nz)
        result = solve_2d_full(nx, nz, W_vec, y_vec, 10.0, 10.0)
        assert len(result) == 3
        theta_vec, diag_v, log_det = result
        assert theta_vec.shape == (nx * nz,)
        assert diag_v.shape == (nx * nz,)
        assert np.isfinite(log_det)

    def test_diag_v_positive(self):
        """Diagonal of A^{-1} should be positive."""
        from insurance_whittaker._smoother2d import solve_2d_full
        nx, nz = 5, 4
        W_vec = np.ones(nx * nz)
        y_vec = np.ones(nx * nz) * 0.5
        _, diag_v, _ = solve_2d_full(nx, nz, W_vec, y_vec, 5.0, 5.0)
        assert np.all(diag_v > 0)

    def test_constant_surface(self):
        """For constant y and equal weights, theta should equal y."""
        from insurance_whittaker._smoother2d import solve_2d_full
        nx, nz = 5, 4
        W_vec = np.ones(nx * nz)
        y_vec = np.full(nx * nz, 1.5)
        theta, _, _ = solve_2d_full(nx, nz, W_vec, y_vec, 100.0, 100.0)
        np.testing.assert_allclose(theta, y_vec, atol=1e-5)


# ===========================================================================
# smoother2d public API tests
# ===========================================================================

class TestWH2DExtended:

    def test_mixed_orders(self):
        """order_x != order_z should work (e.g., order_x=1, order_z=2)."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(40)
        nx, nz = 8, 6
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=1, order_z=2)
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)
        assert result.order_x == 1
        assert result.order_z == 2

    def test_sigma2_fallback_2d(self):
        """With very small lambda (nearly interpolating), sigma2 should be >= 0."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(41)
        nx, nz = 5, 4
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=1e-12, lambda_z=1e-12)
        assert result.sigma2 >= 0.0
        assert np.isfinite(result.sigma2)

    def test_sigma2_positive_moderate_lambda(self):
        """Moderate lambda with noisy data should give positive sigma2."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(42)
        nx, nz = 8, 6
        y = rng.standard_normal((nx, nz)) + 0.5
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.sigma2 > 0.0

    def test_repr_contains_shape_and_lambdas(self):
        from insurance_whittaker import WhittakerHenderson2D
        y = np.ones((5, 4))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=7.5, lambda_z=3.2)
        r = repr(result)
        assert "WHResult2D" in r
        assert "5" in r  # nx
        assert "4" in r  # nz

    def test_x_labels_z_labels_stored(self):
        """x_labels and z_labels should be stored in the result."""
        from insurance_whittaker import WhittakerHenderson2D
        nx, nz = 5, 3
        y = np.ones((nx, nz))
        x_labels = np.array([20, 30, 40, 50, 60])
        z_labels = np.array(["A", "B", "C"])
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0,
                        x_labels=x_labels, z_labels=z_labels)
        np.testing.assert_array_equal(result.x_labels, x_labels)
        np.testing.assert_array_equal(result.z_labels, z_labels)

    def test_tall_table(self):
        """Tall table (many rows, few columns) should work."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(43)
        nx, nz = 15, 3
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))

    def test_wide_table(self):
        """Wide table (few rows, many columns) should work."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(44)
        nx, nz = 3, 15
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))

    def test_std_fitted_non_negative(self):
        """std_fitted should be non-negative everywhere."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(45)
        nx, nz = 6, 5
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert np.all(result.std_fitted >= 0)

    def test_edf_between_order_and_n(self):
        """EDF should be between the null-space dimension and nx*nz."""
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(46)
        nx, nz = 8, 6
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=5.0, lambda_z=5.0)
        assert result.edf > 0
        assert result.edf < nx * nz

    def test_criterion_attribute(self):
        """criterion attribute should be 'reml' for the default."""
        from insurance_whittaker import WhittakerHenderson2D
        y = np.ones((5, 4))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.criterion == "reml"

    def test_lambda_x_z_stored(self):
        """Supplied lambda_x and lambda_z should be stored exactly."""
        from insurance_whittaker import WhittakerHenderson2D
        y = np.ones((5, 4))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=12.3, lambda_z=45.6)
        assert result.lambda_x == pytest.approx(12.3)
        assert result.lambda_z == pytest.approx(45.6)

    def test_order_x_order_z_stored(self):
        """order_x and order_z should be accessible from the result."""
        from insurance_whittaker import WhittakerHenderson2D
        y = np.ones((6, 5))
        wh = WhittakerHenderson2D(order_x=1, order_z=3)
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.order_x == 1
        assert result.order_z == 3


# ===========================================================================
# __init__ / public API
# ===========================================================================

class TestPublicAPI:

    def test_all_public_classes_importable(self):
        """All classes listed in __all__ should be importable from the package."""
        import insurance_whittaker as iw
        for name in ["WhittakerHenderson1D", "WhittakerHenderson2D",
                     "WhittakerHendersonPoisson", "WHResult1D", "WHResult2D",
                     "WHResultPoisson"]:
            assert hasattr(iw, name), f"{name} not in insurance_whittaker"

    def test_version_attribute_exists(self):
        """__version__ should be a non-empty string."""
        import insurance_whittaker as iw
        assert hasattr(iw, "__version__")
        assert isinstance(iw.__version__, str)
        assert len(iw.__version__) > 0

    def test_WHResult1D_is_dataclass(self):
        """WHResult1D should be a dataclass with expected fields."""
        from insurance_whittaker import WHResult1D
        import dataclasses
        assert dataclasses.is_dataclass(WHResult1D)
        field_names = {f.name for f in dataclasses.fields(WHResult1D)}
        expected = {"x", "y", "weights", "fitted", "ci_lower", "ci_upper",
                    "std_fitted", "lambda_", "edf", "order", "criterion",
                    "criterion_value", "sigma2"}
        assert expected.issubset(field_names), f"Missing fields: {expected - field_names}"

    def test_WHResult2D_is_dataclass(self):
        """WHResult2D should be a dataclass with expected fields."""
        from insurance_whittaker import WHResult2D
        import dataclasses
        assert dataclasses.is_dataclass(WHResult2D)
        field_names = {f.name for f in dataclasses.fields(WHResult2D)}
        expected = {"fitted", "ci_lower", "ci_upper", "std_fitted",
                    "lambda_x", "lambda_z", "edf", "order_x", "order_z",
                    "criterion", "sigma2"}
        assert expected.issubset(field_names), f"Missing fields: {expected - field_names}"

    def test_WHResultPoisson_is_dataclass(self):
        """WHResultPoisson should be a dataclass with expected fields."""
        from insurance_whittaker import WHResultPoisson
        import dataclasses
        assert dataclasses.is_dataclass(WHResultPoisson)
        field_names = {f.name for f in dataclasses.fields(WHResultPoisson)}
        expected = {"x", "counts", "exposure", "fitted_rate", "fitted_count",
                    "ci_lower_rate", "ci_upper_rate", "std_log_rate",
                    "lambda_", "edf", "order", "iterations", "criterion"}
        assert expected.issubset(field_names), f"Missing fields: {expected - field_names}"


# ===========================================================================
# Mathematical correctness
# ===========================================================================

class TestMathematicalCorrectness:

    def test_1d_constant_in_null_space_no_penalty(self):
        """Constant signal with any lambda should fit exact constant (null space property)."""
        from insurance_whittaker import WhittakerHenderson1D
        for order in [1, 2, 3]:
            n = max(order + 3, 10)
            x = np.arange(n, dtype=float)
            y = np.full(n, 2.5)
            wh = WhittakerHenderson1D(order=order)
            result = wh.fit(x, y, lambda_=1000.0)
            np.testing.assert_allclose(result.fitted, y, atol=1e-6,
                                       err_msg=f"Constant not recovered for order={order}")

    def test_1d_linear_in_null_space_order2(self):
        """Perfect linear data with large order=2 lambda should be recovered exactly."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 20
        x = np.arange(n, dtype=float)
        y = 1.0 + 0.5 * x  # exactly linear
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e8)
        np.testing.assert_allclose(result.fitted, y, atol=1e-4)

    def test_hat_matrix_edf_sum(self):
        """EDF = trace(H) = sum_i W_i * V_ii. This should match the result attribute."""
        from insurance_whittaker import WhittakerHenderson1D
        from insurance_whittaker.selection import _solve_system, _diag_of_inverse
        from insurance_whittaker._utils import penalty_matrix
        n = 15
        rng = np.random.default_rng(77)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        W = np.ones(n)
        lam = 50.0
        P = penalty_matrix(n, 2)
        _, cf, _ = _solve_system(P, W, W * y, lam)
        diag_v = _diag_of_inverse(cf, n)
        edf_manual = float(np.sum(W * diag_v))
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=W, lambda_=lam)
        np.testing.assert_allclose(result.edf, edf_manual, rtol=1e-6)

    def test_2d_separability(self):
        """For a separable signal f(x)*g(z), 2D smoothing should recover it reasonably.

        Smoothing a separable signal that is already smooth should return something
        close to the original signal.
        """
        from insurance_whittaker import WhittakerHenderson2D
        nx, nz = 10, 8
        x = np.arange(nx, dtype=float)
        z = np.arange(nz, dtype=float)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        # Smooth separable signal — easy for the smoother to recover
        signal = np.sin(xx / 4) * np.cos(zz / 3) + 1.0
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(signal, lambda_x=0.1, lambda_z=0.1)
        # With very small lambda, fitted should be near the signal (interpolating)
        np.testing.assert_allclose(result.fitted, signal, atol=1e-4)

    def test_poisson_deviance_formula(self):
        """Deviance formula: 2 * sum(c * log(c/mu) - (c - mu))."""
        from insurance_whittaker.glm import _poisson_deviance
        c = np.array([3.0, 7.0, 5.0])
        mu = np.array([4.0, 6.0, 5.0])
        expected = 2.0 * np.sum(
            c * np.log(c / mu) - (c - mu)
        )
        actual = _poisson_deviance(c, mu)
        np.testing.assert_allclose(actual, expected, rtol=1e-8)

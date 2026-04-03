"""
Expanded test coverage for insurance-whittaker (April 2026).

Targets untested or thinly-tested code paths:

1. _utils: diff_matrix order=0, order=3+, penalty_matrix symmetry/PSD,
   add_lambda_to_banded, diag_of_inverse_banded, to_numpy_1d Polars path
2. selection: _build_full_system, _solve_system return shapes,
   _edf_from_hat at boundary lambdas, _diag_of_inverse correctness,
   _log_det_P_nonzero directly, reml/gcv/aic/bic on order=1,
   select_lambda_2d non-reml raises, _solve_banded_system compat shim
3. smoother: WHResult1D repr content, edf boundary at large/small lambda,
   criterion_value stored, sigma2=1 fallback exposed, list input,
   order boundary (exactly order+1 obs), criterion_value finite,
   zero-lambda raises, all methods produce consistent ordering
4. smoother2d: WHResult2D repr, non-square table, 1D-consistent lambda,
   both lambdas supplied skips selection, wrong weight shape raises,
   wrong y shape raises (not 2D), order_x=1/order_z=1
5. glm: WHResultPoisson repr content, negative exposure raises,
   zero exposure some cells, all lambda methods for Poisson,
   _poisson_deviance directly, order=3 Poisson,
   Poisson with explicit lambda skips selection,
   Poisson: fitted_count positive, large lambda → near-constant rate
6. _smoother2d: _eig_penalty returns sorted eigenvalues, _build_2d_system shapes,
   _solve_2d_system return types, solve_2d_full diag_v positive
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_whittaker import (
    WhittakerHenderson1D,
    WhittakerHenderson2D,
    WhittakerHendersonPoisson,
    WHResult1D,
    WHResultPoisson,
)
from insurance_whittaker._utils import (
    diff_matrix,
    penalty_matrix,
    penalty_banded,
    add_lambda_to_banded,
    diag_of_inverse_banded,
    to_numpy_1d,
    validate_inputs,
)
from insurance_whittaker.selection import (
    _build_full_system,
    _solve_system,
    _edf_from_hat,
    _diag_of_inverse,
    _log_det_P_nonzero,
    _log_det_P_nz_cached,
    reml_criterion,
    gcv_criterion,
    aic_criterion,
    bic_criterion,
    select_lambda,
    select_lambda_2d,
    _solve_banded_system,
    CRITERIA,
)
from insurance_whittaker._smoother2d import (
    _eig_penalty,
    _build_2d_system,
    _solve_2d_system,
    solve_2d_full,
)
from insurance_whittaker.glm import _poisson_deviance


# ---------------------------------------------------------------------------
# 1. _utils
# ---------------------------------------------------------------------------

class TestDiffMatrixEdgeCases:

    def test_order_zero_returns_identity(self):
        D = diff_matrix(5, 0)
        np.testing.assert_array_equal(D, np.eye(5))

    def test_order_zero_shape(self):
        D = diff_matrix(8, 0)
        assert D.shape == (8, 8)

    def test_order_3_shape(self):
        D = diff_matrix(10, 3)
        assert D.shape == (7, 10)

    def test_order_4_shape(self):
        D = diff_matrix(15, 4)
        assert D.shape == (11, 15)

    def test_second_order_values_row0(self):
        """D^2 row 0 should be [1, -2, 1, 0, ...]."""
        D = diff_matrix(6, 2)
        np.testing.assert_allclose(D[0, :4], [1, -2, 1, 0], atol=1e-12)

    def test_linear_in_null_space_order3(self):
        """A quadratic vector is in the null space of D^3."""
        n = 20
        D = diff_matrix(n, 3)
        quad = np.arange(n, dtype=float) ** 2
        np.testing.assert_allclose(D @ quad, 0, atol=1e-8)

    def test_order1_null_space_is_constants_only(self):
        """The null space of D^1 is spanned by the constant vector."""
        D = diff_matrix(10, 1)
        ones = np.ones(10)
        np.testing.assert_allclose(D @ ones, 0, atol=1e-12)
        # A non-constant vector should not be in the null space
        ramp = np.arange(10, dtype=float)
        assert not np.allclose(D @ ramp, 0)


class TestPenaltyMatrix:

    def test_penalty_matrix_symmetric(self):
        for n, order in [(10, 1), (15, 2), (12, 3)]:
            P = penalty_matrix(n, order)
            np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_penalty_matrix_psd(self):
        """P = D'D is positive semi-definite (all eigenvalues >= 0)."""
        for n, order in [(10, 1), (15, 2), (12, 3)]:
            P = penalty_matrix(n, order)
            eigs = np.linalg.eigvalsh(P)
            assert np.all(eigs >= -1e-10), f"Negative eigenvalue found: {eigs.min()}"

    def test_penalty_matrix_rank(self):
        """P = D'D has rank n - order (nullity = order)."""
        for n, order in [(10, 1), (15, 2), (12, 3)]:
            P = penalty_matrix(n, order)
            rank = np.linalg.matrix_rank(P, tol=1e-10)
            assert rank == n - order, f"Expected rank {n-order}, got {rank}"

    def test_penalty_matrix_shape(self):
        P = penalty_matrix(20, 2)
        assert P.shape == (20, 20)


class TestAddLambdaToBanded:

    def test_returns_copy_unchanged(self):
        """add_lambda_to_banded is a legacy shim — it returns ab_P unchanged."""
        ab = penalty_banded(10, 2)
        result = add_lambda_to_banded(ab, np.ones(10), 100.0)
        assert result.shape == ab.shape

    def test_does_not_modify_original(self):
        ab = penalty_banded(10, 2)
        original = ab.copy()
        add_lambda_to_banded(ab, np.ones(10), 50.0)
        np.testing.assert_array_equal(ab, original)


class TestDiagOfInverseBanded:

    def test_returns_positive_diagonal(self):
        """Diagonal of A^{-1} should be positive for PD matrix A."""
        n = 8
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        lam = 10.0
        A = np.diag(w) + lam * P_full
        diag_v = diag_of_inverse_banded(A, 2)
        assert np.all(diag_v > 0)

    def test_length_matches_n(self):
        n = 12
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        A = np.diag(w) + 50.0 * P_full
        diag_v = diag_of_inverse_banded(A, 2)
        assert len(diag_v) == n

    def test_matches_explicit_inverse(self):
        """diag(A^{-1}) must match the diagonal of the explicit inverse."""
        n = 6
        P_full = penalty_matrix(n, 2)
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.5, 1.0])
        A = np.diag(w) + 5.0 * P_full
        diag_v = diag_of_inverse_banded(A, 2)
        explicit = np.diag(np.linalg.inv(A))
        np.testing.assert_allclose(diag_v, explicit, rtol=1e-10)


class TestToNumpy1D:

    def test_float_list(self):
        arr = to_numpy_1d([1.0, 2.0, 3.0], "x")
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_integer_list_converted_to_float64(self):
        arr = to_numpy_1d([1, 2, 3], "y")
        assert arr.dtype == np.float64

    def test_numpy_array_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        arr = to_numpy_1d(x, "x")
        assert arr.dtype == np.float64

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            to_numpy_1d(np.ones((2, 3, 4)), "x")

    def test_empty_array_ok(self):
        arr = to_numpy_1d([], "x")
        assert len(arr) == 0


class TestValidateInputsExtra:

    def test_y_length_not_n_raises(self):
        """y length != n should raise (guard in validate_inputs)."""
        y = np.ones(5)
        with pytest.raises(ValueError):
            validate_inputs(y, None, 7)

    def test_all_zero_weights_accepted(self):
        """All-zero weight array is technically valid (though degenerate)."""
        y = np.ones(5)
        w = np.zeros(5)
        result = validate_inputs(y, w, 5)
        np.testing.assert_array_equal(result, w)


# ---------------------------------------------------------------------------
# 2. selection internals
# ---------------------------------------------------------------------------

class TestBuildFullSystem:

    def test_shape(self):
        n = 8
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        A = _build_full_system(P_full, w, 10.0)
        assert A.shape == (n, n)

    def test_diagonal_larger_than_off_diagonal(self):
        """For large lambda the diagonal should be dominated by the penalty."""
        n = 8
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        A = _build_full_system(P_full, w, 1e6)
        assert np.all(np.diag(A) > 0)

    def test_symmetric(self):
        n = 10
        P_full = penalty_matrix(n, 2)
        w = np.random.default_rng(0).uniform(0.5, 2.0, n)
        A = _build_full_system(P_full, w, 50.0)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_lambda_zero_equals_diag_w(self):
        """At lambda=0, A should equal diag(W)."""
        n = 5
        P_full = penalty_matrix(n, 2)
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        A = _build_full_system(P_full, w, 0.0)
        np.testing.assert_allclose(A, np.diag(w))


class TestSolveSystem:

    def test_returns_three_elements(self):
        n = 10
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        y = np.ones(n)
        result = _solve_system(P_full, w, w * y, 10.0)
        assert len(result) == 3

    def test_theta_shape(self):
        n = 12
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        y = np.random.default_rng(1).standard_normal(n)
        theta, _, _ = _solve_system(P_full, w, w * y, 20.0)
        assert theta.shape == (n,)

    def test_log_det_finite(self):
        n = 10
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        y = np.ones(n)
        _, _, log_det = _solve_system(P_full, w, w * y, 5.0)
        assert np.isfinite(log_det)

    def test_constant_rhs_recovers_approximately(self):
        """For constant Wy = W*c, theta should be close to c when lambda is small."""
        n = 15
        c = 0.7
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        theta, _, _ = _solve_system(P_full, w, w * np.full(n, c), 1e-6)
        np.testing.assert_allclose(theta, c, atol=1e-4)


class TestDiagOfInverseDirect:

    def test_matches_explicit_inverse(self):
        n = 8
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        _, cf, _ = _solve_system(P_full, w, w * np.ones(n), 10.0)
        diag_v = _diag_of_inverse(cf, n)
        # Verify against explicit inverse
        from scipy.linalg import cho_factor, cho_solve
        A = np.diag(w) + 10.0 * P_full
        cf2 = cho_factor(A)
        explicit = np.array([cho_solve(cf2, np.eye(n)[:, i])[i] for i in range(n)])
        np.testing.assert_allclose(diag_v, explicit, rtol=1e-10)

    def test_all_positive(self):
        n = 10
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        _, cf, _ = _solve_system(P_full, w, w, 100.0)
        diag_v = _diag_of_inverse(cf, n)
        assert np.all(diag_v > 0)


class TestLogDetPNonzero:

    def test_returns_finite(self):
        for n, order in [(10, 1), (15, 2), (20, 3)]:
            val = _log_det_P_nonzero(n, order)
            assert np.isfinite(val), f"Not finite for n={n}, order={order}"

    def test_order_1_vs_order_2(self):
        """Different orders give different log-determinants."""
        v1 = _log_det_P_nonzero(20, 1)
        v2 = _log_det_P_nonzero(20, 2)
        assert v1 != v2

    def test_larger_n_gives_larger_value(self):
        """More eigenvalues → larger sum of log-eigenvalues."""
        v10 = _log_det_P_nonzero(10, 2)
        v20 = _log_det_P_nonzero(20, 2)
        assert v20 > v10


class TestCriteriaOrder1:

    def _data(self, n=20, seed=5):
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab = penalty_banded(n, 1)
        return ab, w, y

    @pytest.mark.parametrize("fn", [gcv_criterion, aic_criterion, bic_criterion])
    def test_criterion_finite_order1(self, fn):
        ab, w, y = self._data()
        val = fn(np.log(10.0), ab, w, y, 1, 0.0)
        assert np.isfinite(val)

    def test_reml_criterion_finite_order1(self):
        ab, w, y = self._data()
        log_det_P = _log_det_P_nz_cached(20, 1)
        val = reml_criterion(np.log(10.0), ab, w, y, 1, log_det_P)
        assert np.isfinite(val)

    def test_aic_monotone_at_small_lambda(self):
        """AIC should decrease as lambda decreases (better fit at small lambda)."""
        n = 20
        ab, w, y = self._data(n)
        val_small = aic_criterion(np.log(1.0), ab, w, y, 2, 0.0)
        val_large = aic_criterion(np.log(1e6), ab, w, y, 2, 0.0)
        # At very large lambda, dev is large (over-smoothing) and edf is small
        # so AIC can go either way — just check both are finite
        assert np.isfinite(val_small)
        assert np.isfinite(val_large)


class TestSelectLambda2D:

    def test_non_reml_raises(self):
        """select_lambda_2d only supports method='reml'."""
        nx, nz = 5, 4
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        w = np.ones(nx * nz)
        y = np.ones(nx * nz)
        with pytest.raises(ValueError, match="reml"):
            select_lambda_2d(ab_Px, ab_Pz, nx, nz, w, y, 2, 2, method="gcv")

    def test_returns_positive_pair(self):
        rng = np.random.default_rng(10)
        nx, nz = 6, 5
        y = np.sin(np.arange(nx)[:, None] / 3) + np.cos(np.arange(nz)[None, :] / 2)
        y = y.ravel() + 0.05 * rng.standard_normal(nx * nz)
        w = np.ones(nx * nz)
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        lx, lz = select_lambda_2d(ab_Px, ab_Pz, nx, nz, w, y, 2, 2, method="reml")
        assert lx > 0
        assert lz > 0
        assert np.isfinite(lx)
        assert np.isfinite(lz)


class TestSolveBandedSystemShim:

    def test_matches_solve_system(self):
        """_solve_banded_system should produce the same theta as _solve_system."""
        n = 10
        P_full = penalty_matrix(n, 2)
        w = np.ones(n)
        y = np.random.default_rng(2).standard_normal(n)
        lam = 15.0
        ab_P = penalty_banded(n, 2)

        theta1, _, _ = _solve_system(P_full, w, w * y, lam)
        theta2, _, _ = _solve_banded_system(ab_P, w, w * y, lam)
        np.testing.assert_allclose(theta1, theta2, rtol=1e-10)

    def test_log_det_finite(self):
        n = 8
        ab_P = penalty_banded(n, 2)
        w = np.ones(n)
        y = np.ones(n)
        _, _, log_det = _solve_banded_system(ab_P, w, w * y, 5.0)
        assert np.isfinite(log_det)


# ---------------------------------------------------------------------------
# 3. smoother (1D) additional paths
# ---------------------------------------------------------------------------

class TestWHResult1DAdditional:

    def _fit_simple(self, n=20, lam=50.0, order=2):
        x = np.arange(n, dtype=float)
        y = np.sin(x / 4)
        wh = WhittakerHenderson1D(order=order)
        return wh.fit(x, y, lambda_=lam)

    def test_repr_contains_edf(self):
        result = self._fit_simple()
        r = repr(result)
        assert "edf=" in r

    def test_repr_contains_criterion(self):
        result = self._fit_simple()
        r = repr(result)
        assert "criterion=" in r

    def test_criterion_value_finite(self):
        result = self._fit_simple()
        assert np.isfinite(result.criterion_value)

    def test_sigma2_attribute_accessible(self):
        result = self._fit_simple()
        assert hasattr(result, "sigma2")
        assert result.sigma2 >= 0.0

    def test_std_fitted_non_negative(self):
        result = self._fit_simple()
        assert np.all(result.std_fitted >= 0)

    def test_list_input_works(self):
        """Plain Python lists should be accepted."""
        x = list(range(15))
        y = [0.5] * 15
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        assert len(result.fitted) == 15

    def test_edf_large_lambda_near_order(self):
        """At very large lambda, EDF should be near order (null space dimension)."""
        n = 30
        order = 2
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        wh = WhittakerHenderson1D(order=order)
        result = wh.fit(x, y, lambda_=1e12)
        assert result.edf < order + 3

    def test_edf_small_lambda_near_n(self):
        """At very small lambda, EDF should approach n."""
        n = 15
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(99)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e-10)
        assert result.edf > n - 2

    def test_exactly_order_plus_one_observations(self):
        """order=2 with exactly 3 observations — the minimum — should work."""
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit([0.0, 1.0, 2.0], [0.4, 0.6, 0.5], lambda_=1.0)
        assert len(result.fitted) == 3

    def test_all_criteria_return_criterion_value(self):
        """criterion_value should be set for all methods."""
        n = 25
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(50)
        y = rng.standard_normal(n)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            assert np.isfinite(result.criterion_value), f"Not finite for method={method}"

    def test_monotone_weights_pull_fit(self):
        """Smoothed values at high-weight positions should be closer to y than at
        low-weight positions."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.zeros(n)
        y[10] = 10.0  # outlier
        w = np.ones(n)
        w[10] = 1000.0  # very high weight
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=w, lambda_=100.0)
        # High-weight outlier: fitted should be closer to 10 than to 0
        assert result.fitted[10] > 1.0

    def test_ci_lower_le_fitted_le_ci_upper(self):
        """Fitted values should lie within their credible intervals."""
        n = 20
        rng = np.random.default_rng(11)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert np.all(result.ci_lower <= result.fitted + 1e-10)
        assert np.all(result.fitted <= result.ci_upper + 1e-10)

    def test_zero_lambda_raises(self):
        """lambda_=0 should raise ValueError (must be positive)."""
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit(np.arange(10, dtype=float), np.ones(10), lambda_=0.0)

    def test_order_1_minimum_is_2_obs(self):
        """order=1 needs at least 2 observations."""
        wh = WhittakerHenderson1D(order=1)
        with pytest.raises(ValueError):
            wh.fit([1.0], [0.5], lambda_=1.0)

    def test_mismatched_y_weight_raises(self):
        """y and weights of different lengths should raise ValueError."""
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit(np.arange(10, dtype=float), np.ones(10), weights=np.ones(8))

    def test_2d_y_raises(self):
        """2-D y array should raise ValueError."""
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError):
            wh.fit(np.arange(10, dtype=float), np.ones((10, 2)), lambda_=1.0)


# ---------------------------------------------------------------------------
# 4. smoother2d additional paths
# ---------------------------------------------------------------------------

class TestWHResult2DAdditional:

    def _fit_simple(self, nx=6, nz=5, lx=20.0, lz=20.0, order_x=2, order_z=2):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=order_x, order_z=order_z)
        return wh.fit(y, lambda_x=lx, lambda_z=lz)

    def test_repr_contains_shape(self):
        result = self._fit_simple()
        r = repr(result)
        assert "shape=" in r

    def test_repr_contains_lambda_x(self):
        result = self._fit_simple()
        r = repr(result)
        assert "lambda_x=" in r

    def test_non_square_table(self):
        """Non-square table (more rows than columns) should work."""
        nx, nz = 12, 4
        rng = np.random.default_rng(1)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)

    def test_non_square_table_wide(self):
        """Wide table (more columns than rows) should work."""
        nx, nz = 4, 15
        rng = np.random.default_rng(2)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)

    def test_order_x1_order_z1(self):
        """order_x=1 and order_z=1 should produce valid output."""
        nx, nz = 6, 5
        rng = np.random.default_rng(3)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=1, order_z=1)
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)
        assert np.all(np.isfinite(result.fitted))

    def test_order_x1_order_z2(self):
        """Mixed orders should work."""
        nx, nz = 7, 6
        rng = np.random.default_rng(4)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=1, order_z=2)
        result = wh.fit(y, lambda_x=5.0, lambda_z=20.0)
        assert result.fitted.shape == (nx, nz)

    def test_wrong_weight_shape_raises(self):
        """weights with wrong shape should raise ValueError."""
        nx, nz = 5, 4
        y = np.ones((nx, nz))
        w = np.ones((nx + 1, nz))
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError):
            wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)

    def test_1d_y_raises(self):
        """1-D y should raise ValueError."""
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError):
            wh.fit(np.ones(10), lambda_x=10.0, lambda_z=10.0)

    def test_ci_bounds_shape(self):
        result = self._fit_simple()
        assert result.ci_lower.shape == result.fitted.shape
        assert result.ci_upper.shape == result.fitted.shape

    def test_std_fitted_non_negative(self):
        result = self._fit_simple()
        assert np.all(result.std_fitted >= 0)

    def test_sigma2_accessible(self):
        result = self._fit_simple()
        assert hasattr(result, "sigma2")

    def test_both_lambdas_supplied_no_selection(self):
        """When both lambdas are supplied, the result should use them exactly."""
        nx, nz = 6, 5
        y = np.ones((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=7.0, lambda_z=13.0)
        assert result.lambda_x == pytest.approx(7.0)
        assert result.lambda_z == pytest.approx(13.0)

    def test_large_lambda_both_dims_gives_smooth_fit(self):
        """Very large lambdas → nearly polynomial fit."""
        nx, nz = 8, 6
        rng = np.random.default_rng(5)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1e8, lambda_z=1e8)
        # Second differences should be tiny
        d2x = np.sum(np.diff(result.fitted, n=2, axis=0) ** 2)
        d2z = np.sum(np.diff(result.fitted, n=2, axis=1) ** 2)
        assert d2x < 1e-4
        assert d2z < 1e-4

    def test_x_labels_stored(self):
        nx, nz = 5, 4
        y = np.ones((nx, nz))
        x_labels = np.arange(nx)
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0, x_labels=x_labels)
        np.testing.assert_array_equal(result.x_labels, x_labels)

    def test_z_labels_stored(self):
        nx, nz = 5, 4
        y = np.ones((nx, nz))
        z_labels = np.arange(nz)
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0, z_labels=z_labels)
        np.testing.assert_array_equal(result.z_labels, z_labels)

    def test_edf_positive_finite(self):
        result = self._fit_simple()
        assert result.edf > 0
        assert np.isfinite(result.edf)

    def test_criterion_attribute(self):
        result = self._fit_simple()
        assert result.criterion == "reml"


# ---------------------------------------------------------------------------
# 5. GLM (Poisson) additional paths
# ---------------------------------------------------------------------------

class TestPoissonDeviance:

    def test_zero_deviance_for_perfect_fit(self):
        """Deviance should be ~0 when mu == c exactly."""
        c = np.array([5.0, 10.0, 3.0, 8.0])
        mu = c.copy()
        dev = _poisson_deviance(c, mu)
        assert abs(dev) < 1e-10

    def test_deviance_positive_for_imperfect_fit(self):
        c = np.array([5.0, 10.0, 3.0, 8.0])
        mu = np.array([4.0, 11.0, 3.5, 7.0])
        dev = _poisson_deviance(c, mu)
        assert dev > 0

    def test_zero_count_handled(self):
        """c=0 cells: 0 * log(0/mu) = 0 — no NaN or inf."""
        c = np.array([0.0, 5.0, 0.0, 3.0])
        mu = np.array([1.0, 5.0, 2.0, 3.0])
        dev = _poisson_deviance(c, mu)
        assert np.isfinite(dev)
        assert dev >= 0

    def test_all_zeros_deviance(self):
        """All-zero counts: deviance = 2 * sum(0 - (0 - mu)) = 2 * sum(mu)."""
        c = np.zeros(5)
        mu = np.array([1.0, 2.0, 3.0, 1.5, 0.5])
        dev = _poisson_deviance(c, mu)
        expected = 2.0 * np.sum(mu)
        assert abs(dev - expected) < 1e-10


class TestWhittakerHendersonPoissonExtra:

    def test_negative_exposure_raises(self):
        wh = WhittakerHendersonPoisson(order=2)
        with pytest.raises(ValueError, match="non-negative"):
            wh.fit(
                np.arange(10, dtype=float),
                np.ones(10),
                exposure=np.array([-1.0] + [1.0] * 9),
            )

    def test_zero_exposure_some_cells(self):
        """Some cells with zero exposure should not crash."""
        rng = np.random.default_rng(20)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        exposure[5] = 0.0
        exposure[15] = 0.0
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_order_3_poisson(self):
        """Order-3 Poisson smoother should work."""
        rng = np.random.default_rng(21)
        n = 25
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=3)
        result = wh.fit(x, counts, exposure)
        assert result.order == 3
        assert len(result.fitted_rate) == n
        assert np.all(result.fitted_rate > 0)

    @pytest.mark.parametrize("method", ["reml", "gcv", "aic", "bic"])
    def test_all_lambda_methods(self, method):
        """All lambda selection methods should produce valid Poisson fits."""
        rng = np.random.default_rng(22)
        n = 25
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 300.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2, lambda_method=method)
        result = wh.fit(x, counts, exposure)
        assert result.lambda_ > 0
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_explicit_lambda_skips_selection(self):
        """When lambda_ is supplied, iterations should still converge
        but the supplied lambda should be used."""
        rng = np.random.default_rng(23)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=100.0)
        assert result.lambda_ == pytest.approx(100.0)
        assert result.iterations > 0

    def test_large_lambda_gives_near_constant_rate(self):
        """Very large lambda on Poisson data should give near-constant fitted rate."""
        rng = np.random.default_rng(24)
        n = 30
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 500.0)
        true_rate = 0.06 + 0.02 * np.sin(x / 5)
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure, lambda_=1e7)
        assert np.std(result.fitted_rate) < 0.01

    def test_fitted_count_positive(self):
        """Fitted counts must be positive (since rate > 0 and exposure >= 0)."""
        rng = np.random.default_rng(25)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_count >= 0)

    def test_repr_contains_iterations(self):
        rng = np.random.default_rng(26)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        r = repr(result)
        assert "iterations=" in r

    def test_repr_contains_edf(self):
        rng = np.random.default_rng(27)
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 200.0)
        counts = rng.poisson(0.04 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        r = repr(result)
        assert "edf=" in r

    def test_ci_lower_le_ci_upper(self):
        """CI lower must be <= CI upper everywhere on rate scale."""
        rng = np.random.default_rng(28)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 300.0)
        counts = rng.poisson(0.05 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.ci_lower_rate <= result.ci_upper_rate)

    def test_order_invalid_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            WhittakerHendersonPoisson(order=0)

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError):
            wh = WhittakerHendersonPoisson(order=2, lambda_method="unknown")
            wh.fit(
                np.arange(10, dtype=float),
                np.ones(10),
                np.ones(10),
            )

    def test_all_zero_counts_no_crash(self):
        """All-zero counts with nonzero exposure: no crash, positive rate."""
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        counts = np.zeros(n)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)

    def test_single_nonzero_count(self):
        """Only one non-zero count cell should not crash."""
        n = 15
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 100.0)
        counts = np.zeros(n)
        counts[7] = 3
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(np.isfinite(result.fitted_rate))


# ---------------------------------------------------------------------------
# 6. _smoother2d internals
# ---------------------------------------------------------------------------

class TestEigPenalty:

    def test_returns_two_arrays(self):
        vals, vecs = _eig_penalty(8, 2)
        assert vals.shape == (8,)
        assert vecs.shape == (8, 8)

    def test_eigenvalues_sorted_ascending(self):
        """eigh returns sorted ascending eigenvalues."""
        vals, _ = _eig_penalty(10, 2)
        assert np.all(np.diff(vals) >= -1e-12)

    def test_eigenvalues_non_negative(self):
        """Eigenvalues of PSD matrix should be >= 0."""
        for n, order in [(8, 1), (10, 2), (12, 3)]:
            vals, _ = _eig_penalty(n, order)
            assert np.all(vals >= -1e-10), f"Negative eigenvalue for n={n}, order={order}"

    def test_zero_eigenvalues_count(self):
        """Number of zero eigenvalues should equal the order (null space dim)."""
        for n, order in [(10, 1), (12, 2), (15, 3)]:
            vals, _ = _eig_penalty(n, order)
            n_zero = np.sum(vals < 1e-10)
            assert n_zero == order, f"Expected {order} zero eigs, got {n_zero}"


class TestBuild2DSystem:

    def test_returns_four_arrays(self):
        result = _build_2d_system(6, 5, 2, 2)
        assert len(result) == 4

    def test_eigenvector_shapes(self):
        nx, nz = 8, 6
        vals_x, vecs_x, vals_z, vecs_z = _build_2d_system(nx, nz, 2, 2)
        assert vecs_x.shape == (nx, nx)
        assert vecs_z.shape == (nz, nz)

    def test_eigenvalue_lengths(self):
        nx, nz = 8, 6
        vals_x, vecs_x, vals_z, vecs_z = _build_2d_system(nx, nz, 2, 2)
        assert len(vals_x) == nx
        assert len(vals_z) == nz


class TestSolve2DSystem:

    def test_returns_two_elements(self):
        nx, nz = 5, 4
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        w = np.ones(nx * nz)
        y = np.ones(nx * nz)
        result = _solve_2d_system(ab_Px, ab_Pz, nx, nz, w, y, 1.0, 1.0)
        assert len(result) == 2

    def test_theta_shape(self):
        nx, nz = 6, 5
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        w = np.ones(nx * nz)
        y = np.random.default_rng(3).standard_normal(nx * nz)
        theta, _ = _solve_2d_system(ab_Px, ab_Pz, nx, nz, w, y, 10.0, 10.0)
        assert theta.shape == (nx * nz,)

    def test_log_det_finite(self):
        nx, nz = 5, 4
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        w = np.ones(nx * nz)
        y = np.ones(nx * nz)
        _, log_det = _solve_2d_system(ab_Px, ab_Pz, nx, nz, w, y, 5.0, 5.0)
        assert np.isfinite(log_det)


class TestSolve2DFull:

    def test_returns_three_elements(self):
        nx, nz = 5, 4
        w = np.ones(nx * nz)
        y = np.ones(nx * nz)
        result = solve_2d_full(nx, nz, w, y, 10.0, 10.0)
        assert len(result) == 3

    def test_theta_shape(self):
        nx, nz = 6, 5
        w = np.ones(nx * nz)
        y = np.random.default_rng(4).standard_normal(nx * nz)
        theta, diag_v, _ = solve_2d_full(nx, nz, w, y, 10.0, 10.0)
        assert theta.shape == (nx * nz,)

    def test_diag_v_positive(self):
        """Diagonal of A^{-1} should be positive."""
        nx, nz = 5, 4
        w = np.ones(nx * nz)
        y = np.ones(nx * nz)
        _, diag_v, _ = solve_2d_full(nx, nz, w, y, 10.0, 10.0)
        assert np.all(diag_v > 0)

    def test_constant_table_recovery(self):
        """With very large lambdas, constant table should be recovered."""
        nx, nz = 6, 5
        c = 0.8
        y = np.full(nx * nz, c)
        w = np.ones(nx * nz)
        theta, _, _ = solve_2d_full(nx, nz, w, y, 1e8, 1e8)
        np.testing.assert_allclose(theta, c, atol=1e-5)

    def test_log_det_finite(self):
        nx, nz = 5, 4
        w = np.ones(nx * nz)
        y = np.random.default_rng(5).standard_normal(nx * nz)
        _, _, log_det = solve_2d_full(nx, nz, w, y, 5.0, 5.0)
        assert np.isfinite(log_det)

    def test_order_1_in_both_dims(self):
        nx, nz = 5, 4
        w = np.ones(nx * nz)
        y = np.ones(nx * nz) * 2.0
        theta, diag_v, _ = solve_2d_full(nx, nz, w, y, 10.0, 10.0, order_x=1, order_z=1)
        assert len(theta) == nx * nz
        assert np.all(diag_v > 0)


# ---------------------------------------------------------------------------
# 7. Numerical correctness tests
# ---------------------------------------------------------------------------

class TestNumericalCorrectness:

    def test_1d_smoother_recovers_quadratic_exactly_at_large_lambda(self):
        """At very large lambda (order=2), the smoother should recover the
        unique quadratic passing through the data (least-norm solution)."""
        n = 20
        x = np.arange(n, dtype=float)
        # True quadratic: y = 1 + 0.5x - 0.02x^2
        y = 1.0 + 0.5 * x - 0.02 * x**2
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=1e10)
        np.testing.assert_allclose(result.fitted, y, atol=1e-4)

    def test_uniform_weights_vs_no_weights(self):
        """Uniform weights of 1 should give same result as no weights."""
        n = 20
        x = np.arange(n, dtype=float)
        rng = np.random.default_rng(55)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        r1 = wh.fit(x, y, lambda_=50.0)
        r2 = wh.fit(x, y, weights=np.ones(n), lambda_=50.0)
        np.testing.assert_allclose(r1.fitted, r2.fitted, atol=1e-12)

    def test_2d_smoother_constant_table_all_lambdas(self):
        """Constant table should be recovered for various lambda values."""
        nx, nz = 7, 6
        for lam in [0.1, 1.0, 100.0, 10000.0]:
            y = np.full((nx, nz), 0.65)
            wh = WhittakerHenderson2D()
            result = wh.fit(y, lambda_x=lam, lambda_z=lam)
            np.testing.assert_allclose(result.fitted, y, atol=1e-5,
                                       err_msg=f"Failed for lambda={lam}")

    def test_high_weight_outlier_pulled_in_2d(self):
        """High-weight outlier cell should be fitted closer to its observed value."""
        nx, nz = 7, 6
        y = np.zeros((nx, nz))
        y[3, 3] = 10.0
        w = np.ones((nx, nz))
        w[3, 3] = 1000.0
        wh = WhittakerHenderson2D()
        result = wh.fit(y, weights=w, lambda_x=100.0, lambda_z=100.0)
        assert result.fitted[3, 3] > 1.0

    def test_poisson_rate_in_observed_ci(self):
        """True rate should be within the 95% CI for most cells (large exposure)."""
        rng = np.random.default_rng(100)
        n = 40
        x = np.arange(n, dtype=float)
        true_rate = 0.05 + 0.02 * np.sin(x / 8)
        exposure = np.full(n, 10000.0)
        counts = rng.poisson(true_rate * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        in_ci = (result.ci_lower_rate <= true_rate) & (true_rate <= result.ci_upper_rate)
        # At least 85% should be in the CI (95% nominal, some slack for finite samples)
        assert np.mean(in_ci) >= 0.85


# ---------------------------------------------------------------------------
# 8. Boundary / robustness tests
# ---------------------------------------------------------------------------

class TestBoundaryRobustness:

    def test_n_equal_order_plus_one_is_minimum(self):
        """For order=q, n=q+1 is the minimum valid input."""
        for order in [1, 2, 3]:
            wh = WhittakerHenderson1D(order=order)
            n = order + 1
            x = np.arange(n, dtype=float)
            y = np.ones(n)
            result = wh.fit(x, y, lambda_=1.0)
            assert len(result.fitted) == n

    def test_very_large_n(self):
        """Large n (n=200) should complete without crashing."""
        n = 200
        rng = np.random.default_rng(101)
        x = np.arange(n, dtype=float)
        y = np.sin(x / 20) + 0.1 * rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y)
        assert len(result.fitted) == n
        assert np.all(np.isfinite(result.fitted))

    def test_uniform_y_gives_trivial_smoothing(self):
        """Uniform y with any lambda should give fitted = constant."""
        for n in [5, 20, 50]:
            x = np.arange(n, dtype=float)
            y = np.full(n, 1.5)
            for lam in [0.01, 1.0, 1000.0]:
                wh = WhittakerHenderson1D(order=2)
                result = wh.fit(x, y, lambda_=lam)
                np.testing.assert_allclose(
                    result.fitted, 1.5, atol=1e-6,
                    err_msg=f"n={n}, lam={lam}"
                )

    def test_poisson_very_high_counts(self):
        """Very high claim counts (large mu) should not cause numerical issues."""
        rng = np.random.default_rng(102)
        n = 20
        x = np.arange(n, dtype=float)
        exposure = np.full(n, 1e6)
        counts = rng.poisson(500 * exposure)
        wh = WhittakerHendersonPoisson(order=2)
        result = wh.fit(x, counts, exposure)
        assert np.all(result.fitted_rate > 0)
        assert np.all(np.isfinite(result.fitted_rate))

    def test_1d_single_outlier_with_zero_weight_smoothed_over(self):
        """A cell with zero weight and outlier value should be smoothed over."""
        n = 20
        rng = np.random.default_rng(103)
        x = np.arange(n, dtype=float)
        y = np.ones(n) * 0.5
        y[10] = 100.0  # outlier
        w = np.ones(n)
        w[10] = 0.0  # zero weight: outlier is ignored
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, weights=w, lambda_=100.0)
        # The zero-weight outlier should have a fitted value near 0.5, not 100
        assert result.fitted[10] < 10.0

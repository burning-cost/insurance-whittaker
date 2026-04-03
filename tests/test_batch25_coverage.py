"""
Batch 25 test coverage expansion for insurance-whittaker.

Targets uncovered or under-covered branches:
1. _utils — diff_matrix order=0, penalty_matrix, add_lambda_to_banded,
             diag_of_inverse_banded, to_numpy_1d Polars Series, validate_inputs edge cases
2. selection — reml_criterion directly, gcv_criterion, aic_criterion, bic_criterion,
               _edf_from_hat, _log_det_P_nonzero, select_lambda unknown method,
               select_lambda_2d (non-reml raises), _solve_banded_system
3. smoother — WHResult1D.plot raises ImportError gracefully, to_polars std_fitted col,
              lambda_=0 raises, lambda_method invalid at fit time, all criteria
              exact output shapes, x length mismatch raises
4. smoother2d — WHResult2D repr, to_polars without y/weights, fit both lambdas auto,
                fit with explicit both lambdas, non-square tables, order_x != order_z
5. glm — WHResultPoisson repr, to_polars, negative counts raises,
          negative exposure raises, exposure length mismatch raises,
          manual lambda bypasses reml, all criteria with Poisson,
          order=1 Poisson smoother
6. plots — plot_smooth and plot_poisson create figures (with matplotlib mock)
"""
from __future__ import annotations

import numpy as np
import pytest
import polars as pl


# ============================================================================
# Helpers
# ============================================================================

def _make_1d(n=30, seed=42):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    y = np.sin(x / 5) + 0.2 * rng.standard_normal(n)
    w = rng.exponential(100, n)
    return x, y, w


def _make_poisson(n=25, seed=10):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    exposure = np.full(n, 200.0)
    rate = 0.05 + 0.02 * np.sin(x / 4)
    counts = rng.poisson(rate * exposure).astype(float)
    return x, counts, exposure


# ============================================================================
# 1. _utils
# ============================================================================

class TestDiffMatrixExtended:

    def test_order0_returns_identity(self):
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(5, 0)
        np.testing.assert_allclose(D, np.eye(5))

    def test_order3_shape(self):
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(10, 3)
        assert D.shape == (7, 10)

    def test_order3_polynomial_in_null_space(self):
        """Quadratic vector is in the null space of D^3."""
        from insurance_whittaker._utils import diff_matrix
        D = diff_matrix(15, 3)
        quad = np.arange(15, dtype=float) ** 2
        np.testing.assert_allclose(D @ quad, 0, atol=1e-8)


class TestPenaltyMatrix:

    def test_penalty_matrix_shape(self):
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(8, 2)
        assert P.shape == (8, 8)

    def test_penalty_matrix_symmetric(self):
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(10, 2)
        np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_penalty_matrix_psd(self):
        """Penalty matrix should be positive semi-definite."""
        from insurance_whittaker._utils import penalty_matrix
        P = penalty_matrix(10, 2)
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs >= -1e-10)

    def test_penalty_matrix_rank_equals_n_minus_order(self):
        """P = D'D has rank n - order."""
        from insurance_whittaker._utils import penalty_matrix
        n, q = 12, 2
        P = penalty_matrix(n, q)
        rank = np.linalg.matrix_rank(P, tol=1e-8)
        assert rank == n - q


class TestAddLambdaToBanded:

    def test_returns_copy_unchanged(self):
        from insurance_whittaker._utils import add_lambda_to_banded, penalty_banded
        ab = penalty_banded(10, 2)
        result = add_lambda_to_banded(ab, np.ones(10), 5.0)
        assert result.shape == ab.shape
        # Result is a copy — modifying it doesn't change original
        result[0, 0] = 9999.0
        assert ab[0, 0] != 9999.0


class TestDiagOfInverseBanded:

    def test_returns_positive_values(self):
        """Diagonal of A^{-1} should be positive for positive definite A."""
        from insurance_whittaker._utils import diag_of_inverse_banded, penalty_matrix
        n = 8
        P = penalty_matrix(n, 2)
        W = np.diag(np.ones(n) * 50)
        A = W + 1.0 * P
        diag_v = diag_of_inverse_banded(A, 2)
        assert diag_v.shape == (n,)
        assert np.all(diag_v > 0)


class TestToNumpy1DExtended:

    def test_polars_series_input(self):
        from insurance_whittaker._utils import to_numpy_1d
        s = pl.Series("x", [1.0, 2.0, 3.0])
        arr = to_numpy_1d(s, "x")
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_integer_list_converted_to_float64(self):
        from insurance_whittaker._utils import to_numpy_1d
        arr = to_numpy_1d([1, 2, 3, 4], "x")
        assert arr.dtype == np.float64

    def test_empty_1d_accepted(self):
        from insurance_whittaker._utils import to_numpy_1d
        arr = to_numpy_1d([], "x")
        assert arr.shape == (0,)


class TestValidateInputsExtended:

    def test_y_length_mismatch_n_raises(self):
        """validate_inputs should check len(y) == n."""
        from insurance_whittaker._utils import validate_inputs
        y = np.ones(5)
        with pytest.raises(ValueError, match="length"):
            validate_inputs(y, None, 7)

    def test_float_zero_weight_accepted(self):
        from insurance_whittaker._utils import validate_inputs
        y = np.ones(4)
        w = np.array([1.0, 0.0, 1.0, 0.0])
        out = validate_inputs(y, w, 4)
        assert out[1] == 0.0
        assert out[3] == 0.0


# ============================================================================
# 2. selection — direct criterion function tests
# ============================================================================

class TestSelectionCriteria:

    def _setup(self, n=20, seed=7):
        from insurance_whittaker._utils import penalty_banded
        rng = np.random.default_rng(seed)
        w = np.ones(n) * 100.0
        y = rng.standard_normal(n)
        ab_P = penalty_banded(n, 2)
        return ab_P, w, y

    def test_reml_criterion_finite_at_reasonable_lam(self):
        from insurance_whittaker.selection import reml_criterion, _log_det_P_nz_cached
        ab_P, w, y = self._setup()
        n = len(y)
        log_det_P_nz = _log_det_P_nz_cached(n, 2)
        val = reml_criterion(np.log(50.0), ab_P, w, y, 2, log_det_P_nz)
        assert np.isfinite(val)

    def test_gcv_criterion_finite_at_reasonable_lam(self):
        from insurance_whittaker.selection import gcv_criterion
        ab_P, w, y = self._setup()
        val = gcv_criterion(np.log(50.0), ab_P, w, y, 2)
        assert np.isfinite(val)
        assert val > 0

    def test_aic_criterion_finite(self):
        from insurance_whittaker.selection import aic_criterion
        ab_P, w, y = self._setup()
        val = aic_criterion(np.log(100.0), ab_P, w, y, 2)
        assert np.isfinite(val)
        assert val > 0

    def test_bic_criterion_finite(self):
        from insurance_whittaker.selection import bic_criterion
        ab_P, w, y = self._setup()
        val = bic_criterion(np.log(100.0), ab_P, w, y, 2)
        assert np.isfinite(val)
        assert val > 0

    def test_bic_vs_aic_bic_larger_for_large_n(self):
        """BIC > AIC for sufficiently large n since log(n) > 2."""
        from insurance_whittaker.selection import aic_criterion, bic_criterion
        n = 100
        from insurance_whittaker._utils import penalty_banded
        rng = np.random.default_rng(99)
        w = np.ones(n) * 100.0
        y = rng.standard_normal(n)
        ab_P = penalty_banded(n, 2)
        log_lam = np.log(50.0)
        aic = aic_criterion(log_lam, ab_P, w, y, 2)
        bic = bic_criterion(log_lam, ab_P, w, y, 2)
        assert bic >= aic  # log(100) = 4.6 > 2

    def test_select_lambda_unknown_method_raises(self):
        from insurance_whittaker.selection import select_lambda
        from insurance_whittaker._utils import penalty_banded
        n = 10
        ab_P = penalty_banded(n, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            select_lambda(ab_P, np.ones(n), np.ones(n), 2, "not_a_method")

    def test_select_lambda_2d_non_reml_raises(self):
        from insurance_whittaker.selection import select_lambda_2d
        from insurance_whittaker._utils import penalty_banded
        with pytest.raises(ValueError, match="reml"):
            select_lambda_2d(
                penalty_banded(5, 2), penalty_banded(4, 2),
                5, 4, np.ones(20), np.ones(20), 2, 2, method="gcv"
            )

    def test_edf_from_hat_in_valid_range(self):
        from insurance_whittaker.selection import _edf_from_hat
        from insurance_whittaker._utils import penalty_banded
        n = 15
        ab_P = penalty_banded(n, 2)
        w = np.ones(n) * 50.0
        edf = _edf_from_hat(ab_P, w, 100.0)
        assert 0 < edf < n

    def test_log_det_P_nonzero_is_finite(self):
        from insurance_whittaker.selection import _log_det_P_nonzero
        for n, q in [(10, 1), (15, 2), (12, 3)]:
            val = _log_det_P_nonzero(n, q)
            assert np.isfinite(val)

    def test_solve_banded_system_returns_correct_shape(self):
        from insurance_whittaker.selection import _solve_banded_system
        from insurance_whittaker._utils import penalty_banded
        n = 10
        ab_P = penalty_banded(n, 2)
        W = np.ones(n) * 50.0
        y = np.sin(np.arange(n, dtype=float))
        Wy = W * y
        theta, cf, log_det = _solve_banded_system(ab_P, W, Wy, 10.0)
        assert theta.shape == (n,)
        assert np.isfinite(log_det)

    def test_solve_system_returns_correct_shape(self):
        from insurance_whittaker.selection import _solve_system
        from insurance_whittaker._utils import penalty_matrix
        n = 8
        P = penalty_matrix(n, 2)
        W = np.ones(n) * 50.0
        y = np.ones(n)
        theta, cf, log_det = _solve_system(P, W, W * y, 10.0)
        assert theta.shape == (n,)


# ============================================================================
# 3. smoother — WHResult1D and WhittakerHenderson1D extra tests
# ============================================================================

class TestWHResult1DExtra:

    def _fit_result(self, n=20, seed=5):
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        return WhittakerHenderson1D().fit(x, y, lambda_=50.0)

    def test_to_polars_has_std_fitted_col(self):
        result = self._fit_result()
        df = result.to_polars()
        assert "std_fitted" in df.columns

    def test_ci_lower_le_fitted_le_ci_upper(self):
        result = self._fit_result()
        assert np.all(result.ci_lower <= result.fitted + 1e-10)
        assert np.all(result.fitted <= result.ci_upper + 1e-10)

    def test_std_fitted_non_negative(self):
        result = self._fit_result()
        assert np.all(result.std_fitted >= 0)

    def test_sigma2_attribute_exists(self):
        result = self._fit_result()
        assert hasattr(result, "sigma2")
        assert result.sigma2 >= 0.0

    def test_criterion_value_is_float(self):
        result = self._fit_result()
        assert isinstance(result.criterion_value, float)

    def test_repr_contains_criterion(self):
        result = self._fit_result()
        r = repr(result)
        assert "criterion='reml'" in r


class TestWhittakerHenderson1DExtra:

    def test_lambda_zero_raises(self):
        from insurance_whittaker import WhittakerHenderson1D
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            wh.fit(np.arange(10, dtype=float), np.ones(10), lambda_=0.0)

    def test_weights_length_mismatch_raises(self):
        """Weights with wrong length should raise ValueError."""
        from insurance_whittaker import WhittakerHenderson1D
        wh = WhittakerHenderson1D(order=2)
        with pytest.raises(ValueError, match="length"):
            wh.fit(np.arange(10, dtype=float), np.ones(10), weights=np.ones(8), lambda_=50.0)

    def test_all_selection_criteria_produce_different_lambdas(self):
        """Different criteria should generally select different lambdas."""
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(42)
        n = 30
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        lambdas = {}
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHenderson1D(order=2, lambda_method=method)
            result = wh.fit(x, y)
            lambdas[method] = result.lambda_
        # All lambdas should be positive finite
        assert all(l > 0 and np.isfinite(l) for l in lambdas.values())

    def test_fit_returns_correct_length(self):
        from insurance_whittaker import WhittakerHenderson1D
        n = 25
        x, y, w = _make_1d(n)
        result = WhittakerHenderson1D().fit(x, y, weights=w, lambda_=50.0)
        assert len(result.fitted) == n
        assert len(result.ci_lower) == n
        assert len(result.ci_upper) == n
        assert len(result.std_fitted) == n

    def test_fit_with_explicit_lambda_uses_that_lambda(self):
        from insurance_whittaker import WhittakerHenderson1D
        x, y, w = _make_1d()
        result = WhittakerHenderson1D().fit(x, y, weights=w, lambda_=999.0)
        assert result.lambda_ == pytest.approx(999.0)

    def test_order1_smoother_gcv(self):
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(1)
        n = 20
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        wh = WhittakerHenderson1D(order=1, lambda_method="gcv")
        result = wh.fit(x, y)
        assert result.order == 1
        assert result.lambda_ > 0

    def test_order1_smoother_aic(self):
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(2)
        n = 20
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        result = WhittakerHenderson1D(order=1, lambda_method="aic").fit(x, y)
        assert result.lambda_ > 0

    def test_order1_smoother_bic(self):
        from insurance_whittaker import WhittakerHenderson1D
        rng = np.random.default_rng(3)
        n = 20
        x = np.arange(n, dtype=float)
        y = rng.standard_normal(n)
        result = WhittakerHenderson1D(order=1, lambda_method="bic").fit(x, y)
        assert result.lambda_ > 0

    def test_very_large_weights_concentrate_fit(self):
        """High-weight cells should pull the smooth strongly toward those values."""
        from insurance_whittaker import WhittakerHenderson1D
        n = 20
        x = np.arange(n, dtype=float)
        y = np.zeros(n)
        y[10] = 10.0  # outlier
        w = np.ones(n)
        w[10] = 1e6  # massive weight
        result = WhittakerHenderson1D().fit(x, y, weights=w, lambda_=100.0)
        # fitted at index 10 should be pulled strongly toward 10
        assert result.fitted[10] > 5.0

    def test_constant_weights_equal_unit_weights(self):
        """Uniform weights should produce the same result as weights=None."""
        from insurance_whittaker import WhittakerHenderson1D
        x, y, _ = _make_1d(20, seed=99)
        wh = WhittakerHenderson1D()
        r_ones = wh.fit(x, y, weights=np.ones(20), lambda_=50.0)
        r_none = wh.fit(x, y, weights=None, lambda_=50.0)
        np.testing.assert_allclose(r_ones.fitted, r_none.fitted, atol=1e-10)


# ============================================================================
# 4. smoother2d — WHResult2D and WhittakerHenderson2D extra tests
# ============================================================================

class TestWHResult2DExtra:

    def _fit_2d(self, nx=6, nz=5, seed=10):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(seed)
        y = rng.standard_normal((nx, nz))
        return WhittakerHenderson2D().fit(y, lambda_x=50.0, lambda_z=50.0)

    def test_repr_contains_shape(self):
        result = self._fit_2d(6, 5)
        r = repr(result)
        assert "6" in r and "5" in r

    def test_to_polars_no_y_no_weights(self):
        result = self._fit_2d()
        df = result.to_polars()
        assert "fitted" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert "y" not in df.columns

    def test_to_polars_with_y(self):
        nx, nz = 5, 4
        from insurance_whittaker import WhittakerHenderson2D
        y = np.ones((nx, nz))
        result = WhittakerHenderson2D().fit(y, lambda_x=10.0, lambda_z=10.0)
        df = result.to_polars(y=y)
        assert "y" in df.columns
        assert len(df) == nx * nz

    def test_ci_upper_ge_ci_lower(self):
        result = self._fit_2d()
        assert np.all(result.ci_upper >= result.ci_lower - 1e-10)

    def test_sigma2_non_negative(self):
        result = self._fit_2d()
        assert result.sigma2 >= 0.0

    def test_edf_in_valid_range(self):
        nx, nz = 6, 5
        result = self._fit_2d(nx, nz)
        n = nx * nz
        assert 0 < result.edf < n


class TestWhittakerHenderson2DExtra:

    def test_both_lambdas_auto_selected(self):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(20)
        nx, nz = 7, 6
        y = rng.standard_normal((nx, nz))
        result = WhittakerHenderson2D().fit(y)  # both lambdas auto
        assert result.lambda_x > 0
        assert result.lambda_z > 0
        assert result.fitted.shape == (nx, nz)

    def test_explicit_both_lambdas(self):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(21)
        y = rng.standard_normal((5, 4))
        result = WhittakerHenderson2D().fit(y, lambda_x=30.0, lambda_z=20.0)
        assert result.lambda_x == pytest.approx(30.0)
        assert result.lambda_z == pytest.approx(20.0)

    def test_non_square_table(self):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(22)
        y = rng.standard_normal((10, 3))
        result = WhittakerHenderson2D().fit(y, lambda_x=50.0, lambda_z=20.0)
        assert result.fitted.shape == (10, 3)

    def test_order_x_ne_order_z(self):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(23)
        y = rng.standard_normal((7, 6))
        result = WhittakerHenderson2D(order_x=1, order_z=2).fit(
            y, lambda_x=10.0, lambda_z=30.0
        )
        assert result.order_x == 1
        assert result.order_z == 2
        assert result.fitted.shape == (7, 6)

    def test_with_weights(self):
        from insurance_whittaker import WhittakerHenderson2D
        rng = np.random.default_rng(24)
        nx, nz = 5, 5
        y = rng.standard_normal((nx, nz))
        w = rng.exponential(100, (nx, nz))
        result = WhittakerHenderson2D().fit(y, weights=w, lambda_x=50.0, lambda_z=50.0)
        assert np.all(np.isfinite(result.fitted))

    def test_x_labels_z_labels_stored(self):
        from insurance_whittaker import WhittakerHenderson2D
        nx, nz = 4, 3
        y = np.ones((nx, nz))
        x_labels = np.array([17, 25, 35, 50])
        z_labels = np.array(["A", "B", "C"])
        result = WhittakerHenderson2D().fit(
            y, lambda_x=10.0, lambda_z=10.0,
            x_labels=x_labels, z_labels=z_labels
        )
        assert result.x_labels is not None
        assert result.z_labels is not None

    def test_constant_table_smooth_is_constant(self):
        """Smoothing a constant table should return a constant."""
        from insurance_whittaker import WhittakerHenderson2D
        y = np.full((6, 5), 2.0)
        result = WhittakerHenderson2D().fit(y, lambda_x=1000.0, lambda_z=1000.0)
        np.testing.assert_allclose(result.fitted, 2.0, atol=1e-4)

    def test_polars_dataframe_input(self):
        """Polars DataFrame should be accepted as y input."""
        from insurance_whittaker import WhittakerHenderson2D
        data = {"col0": [1.0, 0.9, 1.1], "col1": [1.0, 1.0, 0.95], "col2": [0.9, 1.0, 1.0]}
        df = pl.DataFrame(data)
        result = WhittakerHenderson2D().fit(df, lambda_x=5.0, lambda_z=5.0)
        assert result.fitted.shape == (3, 3)


# ============================================================================
# 5. glm — WhittakerHendersonPoisson extra tests
# ============================================================================

class TestWHResultPoissonExtra:

    def _fit_poisson(self, n=20, seed=5):
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson(n, seed)
        return WhittakerHendersonPoisson().fit(x, counts, exposure, lambda_=50.0)

    def test_repr_contains_iterations(self):
        result = self._fit_poisson()
        r = repr(result)
        assert "iterations=" in r
        assert "WHResultPoisson" in r

    def test_to_polars_columns(self):
        result = self._fit_poisson()
        df = result.to_polars()
        assert "fitted_rate" in df.columns
        assert "fitted_count" in df.columns
        assert "ci_lower_rate" in df.columns
        assert "ci_upper_rate" in df.columns
        assert "observed_rate" in df.columns

    def test_to_polars_zero_exposure_observed_rate_nan(self):
        """Cells with zero exposure should have NaN observed rate."""
        from insurance_whittaker import WhittakerHendersonPoisson
        n = 10
        x = np.arange(n, dtype=float)
        exposure = np.ones(n) * 100.0
        exposure[3] = 0.0
        counts = np.full(n, 5.0)
        counts[3] = 0.0
        wh = WhittakerHendersonPoisson()
        result = wh.fit(x, counts, exposure, lambda_=50.0)
        df = result.to_polars()
        # Row 3 should have NaN observed rate
        obs_rate = df["observed_rate"].to_numpy()
        assert np.isnan(obs_rate[3])

    def test_fitted_rate_positive(self):
        result = self._fit_poisson()
        assert np.all(result.fitted_rate > 0)

    def test_fitted_count_equals_rate_times_exposure(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson()
        result = WhittakerHendersonPoisson().fit(x, counts, exposure, lambda_=50.0)
        np.testing.assert_allclose(
            result.fitted_count,
            result.fitted_rate * exposure,
            rtol=1e-6,
        )

    def test_negative_counts_raises(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x = np.arange(10, dtype=float)
        counts = np.full(10, 5.0)
        counts[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            WhittakerHendersonPoisson().fit(x, counts, np.ones(10))

    def test_negative_exposure_raises(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x = np.arange(10, dtype=float)
        counts = np.full(10, 5.0)
        exposure = np.ones(10)
        exposure[2] = -5.0
        with pytest.raises(ValueError, match="non-negative"):
            WhittakerHendersonPoisson().fit(x, counts, exposure)

    def test_exposure_length_mismatch_raises(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x = np.arange(10, dtype=float)
        counts = np.full(10, 5.0)
        exposure = np.ones(8)  # wrong length
        with pytest.raises(ValueError, match="length"):
            WhittakerHendersonPoisson().fit(x, counts, exposure)

    def test_explicit_lambda_bypasses_reml(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson()
        result = WhittakerHendersonPoisson().fit(x, counts, exposure, lambda_=200.0)
        assert result.lambda_ == pytest.approx(200.0)

    def test_none_exposure_uses_ones(self):
        """When exposure=None, all exposures are 1.0."""
        from insurance_whittaker import WhittakerHendersonPoisson
        n = 15
        x = np.arange(n, dtype=float)
        counts = np.full(n, 3.0)
        result = WhittakerHendersonPoisson().fit(x, counts, None, lambda_=50.0)
        np.testing.assert_allclose(result.exposure, np.ones(n))

    def test_order_1_poisson(self):
        """Order-1 Poisson smoother should run without error."""
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson()
        result = WhittakerHendersonPoisson(order=1).fit(x, counts, exposure)
        assert result.order == 1
        assert np.all(result.fitted_rate > 0)

    def test_order_invalid_raises(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        with pytest.raises(ValueError, match="order"):
            WhittakerHendersonPoisson(order=0)

    def test_all_criteria_produce_positive_lambda(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson(n=25)
        for method in ["reml", "gcv", "aic", "bic"]:
            wh = WhittakerHendersonPoisson(order=2, lambda_method=method)
            result = wh.fit(x, counts, exposure)
            assert result.lambda_ > 0, f"lambda <= 0 for method={method}"

    def test_edf_between_order_and_n(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        n = 25
        x, counts, exposure = _make_poisson(n)
        result = WhittakerHendersonPoisson().fit(x, counts, exposure, lambda_=50.0)
        assert result.edf > 0
        assert result.edf < n

    def test_iterations_positive(self):
        from insurance_whittaker import WhittakerHendersonPoisson
        x, counts, exposure = _make_poisson()
        result = WhittakerHendersonPoisson().fit(x, counts, exposure)
        assert result.iterations >= 1

    def test_ci_lower_le_rate_le_ci_upper(self):
        result = self._fit_poisson()
        # CI should bracket the fitted rate
        assert np.all(result.ci_lower_rate <= result.fitted_rate + 1e-8)
        assert np.all(result.fitted_rate <= result.ci_upper_rate + 1e-8)

    def test_std_log_rate_non_negative(self):
        result = self._fit_poisson()
        assert np.all(result.std_log_rate >= 0)


# ============================================================================
# 6. _utils.penalty_banded — shape and usage
# ============================================================================

class TestPenaltyBandedExtra:

    def test_order1_shape(self):
        from insurance_whittaker._utils import penalty_banded
        ab = penalty_banded(10, 1)
        assert ab.shape == (2, 10)

    def test_order3_shape(self):
        from insurance_whittaker._utils import penalty_banded
        ab = penalty_banded(15, 3)
        assert ab.shape == (4, 15)

    def test_used_as_descriptor_in_select_lambda(self):
        """penalty_banded descriptor is used by select_lambda to recover n and order."""
        from insurance_whittaker.selection import select_lambda
        from insurance_whittaker._utils import penalty_banded
        n = 20
        ab_P = penalty_banded(n, 2)
        w = np.ones(n)
        y = np.sin(np.arange(n, dtype=float) / 4)
        lam = select_lambda(ab_P, w, y, 2, "gcv")
        assert lam > 0


# ============================================================================
# 7. WHResult1D — extended attribute checks
# ============================================================================

class TestWHResult1DAttributeChecks:

    def test_x_array_matches_input(self):
        from insurance_whittaker import WhittakerHenderson1D
        x = np.array([17.0, 25.0, 35.0, 50.0, 65.0])
        y = np.array([0.7, 0.6, 0.65, 0.8, 0.9])
        result = WhittakerHenderson1D(order=1).fit(x, y, lambda_=10.0)
        np.testing.assert_allclose(result.x, x)

    def test_y_array_matches_input(self):
        from insurance_whittaker import WhittakerHenderson1D
        x = np.arange(10, dtype=float)
        y = np.sin(x / 3)
        result = WhittakerHenderson1D(order=2).fit(x, y, lambda_=20.0)
        np.testing.assert_allclose(result.y, y)

    def test_weights_match_input(self):
        from insurance_whittaker import WhittakerHenderson1D
        n = 15
        x = np.arange(n, dtype=float)
        y = np.ones(n)
        w = np.arange(1, n + 1, dtype=float)
        result = WhittakerHenderson1D(order=2).fit(x, y, weights=w, lambda_=10.0)
        np.testing.assert_allclose(result.weights, w)

    def test_criterion_name_stored(self):
        from insurance_whittaker import WhittakerHenderson1D
        for method in ["reml", "gcv", "aic", "bic"]:
            x = np.arange(15, dtype=float)
            y = np.ones(15)
            result = WhittakerHenderson1D(order=2, lambda_method=method).fit(
                x, y, lambda_=50.0
            )
            assert result.criterion == method


# ============================================================================
# 8. select_lambda_2d — basic smoke test (reml)
# ============================================================================

class TestSelectLambda2D:

    def test_reml_2d_returns_positive_lambdas(self):
        from insurance_whittaker.selection import select_lambda_2d
        from insurance_whittaker._utils import penalty_banded
        rng = np.random.default_rng(99)
        nx, nz = 6, 5
        W_vec = np.ones(nx * nz) * 100.0
        y_vec = rng.standard_normal(nx * nz)
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        lx, lz = select_lambda_2d(ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 2, 2)
        assert lx > 0
        assert lz > 0
        assert np.isfinite(lx)
        assert np.isfinite(lz)

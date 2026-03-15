"""
Regression tests for bugs fixed in v0.1.1.

B1 — P0: Kronecker product ordering in 2D smoother was reversed.
     lam_x should smooth rows (axis 0), lam_z should smooth columns (axis 1).

B2 — P1: Bayesian CIs omitted sigma^2 — interval widths were independent
     of data scale.

B6 — P2: select_lambda_2d silently ignored method parameter.
"""

import numpy as np
import pytest

from insurance_whittaker import WhittakerHenderson1D, WhittakerHenderson2D


# ---------------------------------------------------------------------------
# B1 regression: asymmetric lambda must smooth the right direction
# ---------------------------------------------------------------------------

class TestKroneckerOrdering:
    """Verify that lam_x smooths rows (axis 0) and lam_z smooths columns
    (axis 1) in a non-square table.

    We use a 6x4 table (nx=6, nz=4) with lam_x=1000 (strong row smoothing)
    and lam_z=0.001 (almost no column smoothing).

    Expected behaviour:
      - Rows of fitted table should be nearly flat (very small row-wise
        second differences): sum-sq-diff-axis0 << 1.
      - Columns should be left rough (larger column-wise second differences)
        relative to the row direction.

    The pre-fix code had lam_x and lam_z swapped in the Kronecker product,
    which would produce the opposite: rough rows and smooth columns.
    """

    def _make_test_table(self, nx: int = 6, nz: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Noisy table with no strong signal — any smoothness is from the penalty
        y = rng.standard_normal((nx, nz))
        return y

    def test_strong_x_smoothing_makes_rows_flat(self):
        """With lam_x=1000 and lam_z=0.001, rows should be nearly flat."""
        nx, nz = 6, 4
        y = self._make_test_table(nx, nz)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1000.0, lambda_z=0.001)

        fitted = result.fitted  # shape (6, 4)

        # Row-wise second differences (along axis 0 = x direction)
        d2x = np.diff(fitted, n=2, axis=0)  # shape (4, 4)
        ss_rows = float(np.sum(d2x ** 2))

        # Column-wise second differences (along axis 1 = z direction)
        d2z = np.diff(fitted, n=2, axis=1)  # shape (6, 2)
        ss_cols = float(np.sum(d2z ** 2))

        # Row smoothing should be much stronger than column smoothing
        # With lam_x=1000 vs lam_z=0.001, rows should be ~10^5 times
        # smoother than columns.  We use a conservative threshold of 100x.
        assert ss_rows < ss_cols / 100.0, (
            f"Expected rows to be much smoother than columns with lam_x=1000, "
            f"lam_z=0.001.  Got ss_rows={ss_rows:.6f}, ss_cols={ss_cols:.6f}.  "
            f"Ratio ss_cols/ss_rows={ss_cols/max(ss_rows,1e-12):.1f} (need >100). "
            f"If this fails, the Kronecker ordering may be reversed."
        )

    def test_strong_z_smoothing_makes_columns_flat(self):
        """With lam_x=0.001 and lam_z=1000, columns should be nearly flat."""
        nx, nz = 6, 4
        y = self._make_test_table(nx, nz)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=0.001, lambda_z=1000.0)

        fitted = result.fitted

        d2x = np.diff(fitted, n=2, axis=0)
        ss_rows = float(np.sum(d2x ** 2))

        d2z = np.diff(fitted, n=2, axis=1)
        ss_cols = float(np.sum(d2z ** 2))

        # Column smoothing should be much stronger than row smoothing
        assert ss_cols < ss_rows / 100.0, (
            f"Expected columns to be much smoother than rows with lam_x=0.001, "
            f"lam_z=1000.  Got ss_rows={ss_rows:.6f}, ss_cols={ss_cols:.6f}.  "
            f"Ratio ss_rows/ss_cols={ss_rows/max(ss_cols,1e-12):.1f} (need >100). "
            f"If this fails, the Kronecker ordering may be reversed."
        )

    def test_asymmetric_lambdas_non_square_table(self):
        """Explicitly verify on a non-square table (nx != nz).

        This catches the bug only when nx != nz, because kron(Px, Iz) and
        kron(Iz, Px) have the same shape only when nx == nz.
        """
        nx, nz = 8, 3  # deliberately very non-square
        rng = np.random.default_rng(99)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)

        # Extreme asymmetry to make the test unambiguous
        result = wh.fit(y, lambda_x=1e5, lambda_z=1e-3)
        fitted = result.fitted

        d2x = np.diff(fitted, n=2, axis=0)
        d2z = np.diff(fitted, n=2, axis=1)
        ss_rows = float(np.sum(d2x ** 2))
        ss_cols = float(np.sum(d2z ** 2))

        assert ss_rows < ss_cols / 1000.0, (
            f"nx={nx}, nz={nz}: rows not smoothed by lam_x=1e5. "
            f"ss_rows={ss_rows:.8f}, ss_cols={ss_cols:.6f}"
        )


# ---------------------------------------------------------------------------
# B2 regression: CI width must scale with data magnitude
# ---------------------------------------------------------------------------

class TestCIScalesWithDataMagnitude:
    """Verify that credible interval widths scale with the residual noise level.

    The posterior variance is sigma^2 * A^{-1}.  If we scale y by a factor k,
    the residuals scale by k, sigma^2 scales by k^2, and CI widths should
    scale by k.

    The pre-fix code used A^{-1} directly (sigma^2 = 1), so CI widths were
    constant regardless of data scale.
    """

    def test_1d_ci_width_scales_with_data_scale(self):
        """Scaling y by 10 should scale CI widths by approximately 10."""
        rng = np.random.default_rng(17)
        n = 30
        x = np.arange(n, dtype=float)
        y_base = np.sin(x / 5) + 0.3 * rng.standard_normal(n)

        wh = WhittakerHenderson1D(order=2, lambda_method="reml")

        result1 = wh.fit(x, y_base, lambda_=50.0)
        result10 = wh.fit(x, 10.0 * y_base, lambda_=50.0)

        width1 = float(np.mean(result1.ci_upper - result1.ci_lower))
        width10 = float(np.mean(result10.ci_upper - result10.ci_lower))

        ratio = width10 / width1

        # With the bug, ratio would be ~1.0 (CI independent of scale).
        # Correct: ratio should be close to 10.
        assert ratio > 5.0, (
            f"CI width ratio when y scaled by 10x should be ~10, got {ratio:.2f}. "
            f"This suggests sigma^2 is not being incorporated into the CIs."
        )

    def test_2d_ci_width_scales_with_data_scale(self):
        """2D: scaling y by 10 should scale CI widths by approximately 10."""
        rng = np.random.default_rng(23)
        nx, nz = 6, 5
        y_base = rng.standard_normal((nx, nz))

        wh = WhittakerHenderson2D(order_x=2, order_z=2)

        result1 = wh.fit(y_base, lambda_x=20.0, lambda_z=20.0)
        result10 = wh.fit(10.0 * y_base, lambda_x=20.0, lambda_z=20.0)

        width1 = float(np.mean(result1.ci_upper - result1.ci_lower))
        width10 = float(np.mean(result10.ci_upper - result10.ci_lower))

        ratio = width10 / width1

        assert ratio > 5.0, (
            f"2D CI width ratio when y scaled by 10x should be ~10, got {ratio:.2f}. "
            f"This suggests sigma^2 is not incorporated into the 2D CIs."
        )

    def test_1d_sigma2_attribute(self):
        """WHResult1D should expose sigma2 attribute."""
        n = 20
        x = np.arange(n, dtype=float)
        y = np.ones(n) + 0.1 * np.random.default_rng(0).standard_normal(n)
        wh = WhittakerHenderson1D(order=2)
        result = wh.fit(x, y, lambda_=10.0)
        assert hasattr(result, "sigma2")
        assert result.sigma2 >= 0.0

    def test_2d_sigma2_attribute(self):
        """WHResult2D should expose sigma2 attribute."""
        y = np.ones((5, 4)) + 0.1 * np.random.default_rng(1).standard_normal((5, 4))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert hasattr(result, "sigma2")
        assert result.sigma2 >= 0.0


# ---------------------------------------------------------------------------
# B6 regression: select_lambda_2d must reject unsupported methods
# ---------------------------------------------------------------------------

class TestSelectLambda2DMethodValidation:
    """select_lambda_2d should raise ValueError for method != 'reml'.

    Previously, any method string was silently accepted and REML was used.
    """

    def test_gcv_raises_value_error(self):
        with pytest.raises(ValueError, match="reml"):
            from insurance_whittaker.selection import select_lambda_2d
            import numpy as np
            from insurance_whittaker._utils import penalty_banded
            nx, nz = 5, 4
            ab_Px = penalty_banded(nx, 2)
            ab_Pz = penalty_banded(nz, 2)
            W_vec = np.ones(nx * nz)
            y_vec = np.ones(nx * nz)
            select_lambda_2d(ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 2, 2, method="gcv")

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError):
            from insurance_whittaker.selection import select_lambda_2d
            import numpy as np
            from insurance_whittaker._utils import penalty_banded
            nx, nz = 5, 4
            ab_Px = penalty_banded(nx, 2)
            ab_Pz = penalty_banded(nz, 2)
            W_vec = np.ones(nx * nz)
            y_vec = np.ones(nx * nz)
            select_lambda_2d(ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 2, 2, method="aic")

    def test_reml_still_works(self):
        """Sanity check: method='reml' should not raise."""
        from insurance_whittaker.selection import select_lambda_2d
        from insurance_whittaker._utils import penalty_banded
        nx, nz = 5, 4
        rng = np.random.default_rng(0)
        ab_Px = penalty_banded(nx, 2)
        ab_Pz = penalty_banded(nz, 2)
        W_vec = np.ones(nx * nz)
        y_vec = rng.standard_normal(nx * nz)
        lx, lz = select_lambda_2d(
            ab_Px, ab_Pz, nx, nz, W_vec, y_vec, 2, 2, method="reml"
        )
        assert lx > 0
        assert lz > 0

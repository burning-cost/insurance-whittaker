"""
Tests for the 2-D Whittaker-Henderson smoother.

Covers:
- Constant table → fitted ≈ constant.
- Smooth signal recovery on a grid.
- Lambda selection returns positive values.
- Result shapes and Polars output.
- to_polars() output format.
"""

import numpy as np
import pytest
import polars as pl

from insurance_whittaker import WhittakerHenderson2D, WHResult2D


class TestWhittakerHenderson2D:

    def _constant_table(self, nx=8, nz=6, val=0.5):
        y = np.full((nx, nz), val)
        w = np.ones((nx, nz))
        return y, w

    def test_constant_table_recovery(self):
        """Constant table with fixed lambda should return constant fitted."""
        y, w = self._constant_table(nx=8, nz=6, val=0.75)
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, weights=w, lambda_x=100.0, lambda_z=100.0)
        np.testing.assert_allclose(result.fitted, y, atol=1e-6)

    def test_result_shape(self):
        """Fitted array shape must match input shape."""
        nx, nz = 10, 8
        rng = np.random.default_rng(0)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.fitted.shape == (nx, nz)
        assert result.ci_lower.shape == (nx, nz)
        assert result.ci_upper.shape == (nx, nz)

    def test_ci_upper_ge_lower(self):
        """CI upper must be >= CI lower everywhere."""
        nx, nz = 6, 5
        rng = np.random.default_rng(1)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=50.0, lambda_z=50.0)
        assert np.all(result.ci_upper >= result.ci_lower)

    def test_large_lambda_gives_polynomial_surface(self):
        """Very large lambdas should flatten the surface."""
        nx, nz = 8, 6
        rng = np.random.default_rng(2)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D(order_x=2, order_z=2)
        result = wh.fit(y, lambda_x=1e8, lambda_z=1e8)
        # With huge penalty, second differences of fitted should be tiny
        d2x = np.diff(result.fitted, n=2, axis=0)
        d2z = np.diff(result.fitted, n=2, axis=1)
        assert np.max(np.abs(d2x)) < 0.01
        assert np.max(np.abs(d2z)) < 0.01

    def test_small_lambda_gives_interpolation(self):
        """Very small lambda → fitted ≈ observed."""
        nx, nz = 6, 5
        rng = np.random.default_rng(3)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=1e-8, lambda_z=1e-8)
        np.testing.assert_allclose(result.fitted, y, atol=1e-4)

    def test_edf_positive(self):
        nx, nz = 8, 6
        rng = np.random.default_rng(4)
        y = rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        assert result.edf > 0
        assert result.edf <= nx * nz

    def test_to_polars_long_format(self):
        nx, nz = 5, 4
        y = np.ones((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=100.0, lambda_z=100.0)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == nx * nz
        for col in ["x", "z", "fitted", "ci_lower", "ci_upper"]:
            assert col in df.columns

    def test_repr(self):
        y = np.ones((5, 4))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        r = repr(result)
        assert "WHResult2D" in r

    def test_lambda_selection_runs(self):
        """Automatic lambda selection should complete and return positive values."""
        nx, nz = 6, 5
        rng = np.random.default_rng(5)
        y = np.sin(np.arange(nx)[:, None] / 3) + np.cos(np.arange(nz)[None, :] / 2)
        y += 0.1 * rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y)
        assert result.lambda_x > 0
        assert result.lambda_z > 0

    def test_smooth_signal_recovery(self):
        """Fitted values should be closer to the true signal than raw data."""
        rng = np.random.default_rng(6)
        nx, nz = 10, 8
        x = np.arange(nx)
        z = np.arange(nz)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        true_signal = np.sin(xx / 4) * np.cos(zz / 3)
        y = true_signal + 0.2 * rng.standard_normal((nx, nz))
        wh = WhittakerHenderson2D()
        result = wh.fit(y, lambda_x=10.0, lambda_z=10.0)
        raw_mse = np.mean((y - true_signal) ** 2)
        fit_mse = np.mean((result.fitted - true_signal) ** 2)
        assert fit_mse < raw_mse

    def test_polars_dataframe_input(self):
        """Polars DataFrame input should be accepted."""
        nx, nz = 5, 4
        y_np = np.ones((nx, nz))
        y_pl = pl.DataFrame({str(j): y_np[:, j].tolist() for j in range(nz)})
        wh = WhittakerHenderson2D()
        result_np = wh.fit(y_np, lambda_x=10.0, lambda_z=10.0)
        result_pl = wh.fit(y_pl, lambda_x=10.0, lambda_z=10.0)
        np.testing.assert_allclose(result_np.fitted, result_pl.fitted, atol=1e-10)

    def test_weight_shape_mismatch_raises(self):
        y = np.ones((5, 4))
        w = np.ones((5, 3))  # wrong shape
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="shape"):
            wh.fit(y, weights=w, lambda_x=10.0, lambda_z=10.0)

    def test_1d_input_raises(self):
        wh = WhittakerHenderson2D()
        with pytest.raises(ValueError, match="2-D"):
            wh.fit(np.ones(10), lambda_x=10.0, lambda_z=10.0)

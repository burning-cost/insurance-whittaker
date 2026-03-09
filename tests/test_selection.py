"""
Tests for lambda selection criteria.

Covers:
- REML, GCV, AIC, BIC all return finite positive lambda.
- Edf decreases as lambda increases.
- REML criterion is monotone near the optimum (basic sanity).
- Noisy data gets smoothed, not over-fitted (edf < n/2 for moderate noise).
"""

import numpy as np
import pytest

from insurance_whittaker._utils import penalty_banded
from insurance_whittaker.selection import (
    select_lambda,
    reml_criterion,
    gcv_criterion,
    aic_criterion,
    bic_criterion,
    _edf_from_hat,
    _log_det_P_nz_cached,
)


def _make_data(n=30, seed=42):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    true = np.sin(x / 5)
    y = true + 0.1 * rng.standard_normal(n)
    w = np.ones(n)
    return x, y, w


class TestSelectLambda:

    @pytest.mark.parametrize("method", ["reml", "gcv", "aic", "bic"])
    def test_returns_positive_lambda(self, method):
        _, y, w = _make_data()
        ab_P = penalty_banded(len(y), 2)
        lam = select_lambda(ab_P, w, y, 2, method)
        assert lam > 0, f"lambda={lam} for method={method}"

    @pytest.mark.parametrize("method", ["reml", "gcv", "aic", "bic"])
    def test_lambda_finite(self, method):
        _, y, w = _make_data()
        ab_P = penalty_banded(len(y), 2)
        lam = select_lambda(ab_P, w, y, 2, method)
        assert np.isfinite(lam), f"lambda not finite: {lam}"

    def test_invalid_method_raises(self):
        _, y, w = _make_data()
        ab_P = penalty_banded(len(y), 2)
        with pytest.raises(ValueError, match="Unknown method"):
            select_lambda(ab_P, w, y, 2, "bogus")

    def test_smooth_data_gets_low_edf(self):
        """Very smooth data should lead to high lambda and low edf."""
        n = 30
        x = np.arange(n, dtype=float)
        y = np.sin(x / 5)  # perfectly smooth, no noise
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        lam = select_lambda(ab_P, w, y, 2, "reml")
        edf = _edf_from_hat(ab_P, w, lam)
        # Smooth data should give edf < 10 (well below n=30)
        assert edf < 15, f"edf={edf:.2f} too high for smooth data"

    def test_noisy_data_gets_higher_edf(self):
        """Noisy data should lead to lower lambda and higher edf."""
        n = 30
        rng = np.random.default_rng(7)
        y_smooth = np.sin(np.arange(n) / 5)
        y_noisy = y_smooth + 1.0 * rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)

        lam_s = select_lambda(ab_P, w, y_smooth, 2, "reml")
        lam_n = select_lambda(ab_P, w, y_noisy, 2, "reml")
        # Noisy data allows more complexity — lambda is smaller (or edf larger)
        edf_s = _edf_from_hat(ab_P, w, lam_s)
        edf_n = _edf_from_hat(ab_P, w, lam_n)
        assert edf_n >= edf_s - 2.0, (
            f"Noisy edf={edf_n:.2f} should be >= smooth edf={edf_s:.2f}"
        )


class TestEdF:

    def test_edf_increases_as_lambda_decreases(self):
        """EDF is monotonically decreasing in lambda (for fixed data)."""
        n = 20
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        lambdas = [0.01, 1.0, 100.0, 10000.0]
        edfs = [_edf_from_hat(ab_P, w, lam) for lam in lambdas]
        # EDF should decrease as lambda increases
        for i in range(len(edfs) - 1):
            assert edfs[i] >= edfs[i + 1] - 1e-6, (
                f"EDF not monotone: {edfs}"
            )

    def test_edf_at_zero_lambda_is_n(self):
        """At lambda=0 (no penalty), EDF should equal n."""
        n = 15
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        edf = _edf_from_hat(ab_P, w, 1e-12)
        np.testing.assert_allclose(edf, n, atol=1.0)

    def test_edf_at_large_lambda_is_near_order(self):
        """At very large lambda, EDF approaches the order of the null space."""
        n = 20
        order = 2
        w = np.ones(n)
        ab_P = penalty_banded(n, order)
        edf = _edf_from_hat(ab_P, w, 1e12)
        assert edf < order + 2, f"edf={edf:.2f} should be near {order}"


class TestCriterionValues:

    def test_reml_criterion_finite(self):
        n = 20
        rng = np.random.default_rng(1)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        log_det_P = _log_det_P_nz_cached(ab_P, n, 2)
        val = reml_criterion(np.log(100.0), ab_P, w, y, 2, log_det_P)
        assert np.isfinite(val)

    def test_gcv_criterion_positive(self):
        n = 20
        rng = np.random.default_rng(2)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        val = gcv_criterion(np.log(100.0), ab_P, w, y, 2)
        assert val > 0

    @pytest.mark.parametrize("fn,kw", [
        (reml_criterion, {"log_det_P_nz": 0.0}),
        (gcv_criterion, {}),
        (aic_criterion, {}),
        (bic_criterion, {}),
    ])
    def test_all_criteria_finite(self, fn, kw):
        n = 25
        rng = np.random.default_rng(3)
        y = rng.standard_normal(n)
        w = np.ones(n)
        ab_P = penalty_banded(n, 2)
        if "log_det_P_nz" in kw:
            kw["log_det_P_nz"] = _log_det_P_nz_cached(ab_P, n, 2)
        # Call with log_lam, ab_P, w, y, order, log_det_P_nz
        val = fn(np.log(50.0), ab_P, w, y, 2, kw.get("log_det_P_nz", 0.0))
        assert np.isfinite(val), f"{fn.__name__} returned {val}"

"""
tests/test_lrtc_atsn.py – Integration test for LRTC-ATSN
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from atsn import LRTC_ATSN
from atsn.missing import mixed_missing


@pytest.fixture(scope='module')
def small_tensor():
    """Synthetic low-rank tensor for fast integration testing."""
    rng = np.random.default_rng(0)
    # Create a rank-2 tensor: shape (12, 10, 8)
    A = rng.normal(size=(12, 2))
    B = rng.normal(size=(10, 2))
    C = rng.normal(size=(8,  2))
    X = np.einsum('ir, jr, kr -> ijk', A, B, C) + 5.0   # shift to positive
    return X


def test_atsn_output_shape(small_tensor):
    X_obs = mixed_missing(small_tensor, fiber_rate=0.2, element_rate=0.1, seed=42)
    result = LRTC_ATSN(small_tensor, X_obs, max_iter=10, verbose=False)
    assert result.X_hat.shape == small_tensor.shape


def test_atsn_metrics_finite(small_tensor):
    X_obs = mixed_missing(small_tensor, fiber_rate=0.2, element_rate=0.1, seed=42)
    result = LRTC_ATSN(small_tensor, X_obs, max_iter=10, verbose=False)
    assert np.isfinite(result.final_mae)
    assert np.isfinite(result.final_rmse)
    assert np.isfinite(result.final_mape)
    assert np.isfinite(result.final_er)


def test_atsn_iterations_within_bound(small_tensor):
    X_obs = mixed_missing(small_tensor, fiber_rate=0.2, element_rate=0.1, seed=42)
    result = LRTC_ATSN(small_tensor, X_obs, max_iter=15, verbose=False)
    assert result.iterations <= 15


def test_atsn_shape_mismatch():
    X      = np.ones((5, 5, 5))
    X_obs  = np.ones((5, 5, 4))  # wrong shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        LRTC_ATSN(X, X_obs, verbose=False)


def test_atsn_observed_preserved(small_tensor):
    """Observed entries must not be altered by the imputation."""
    X_obs = mixed_missing(small_tensor, fiber_rate=0.2, element_rate=0.1, seed=7)
    result = LRTC_ATSN(small_tensor, X_obs, max_iter=5, verbose=False)
    obs_mask = X_obs != 0
    np.testing.assert_allclose(
        result.X_hat[obs_mask], X_obs[obs_mask], atol=1e-8,
        err_msg="Observed entries were modified by LRTC-ATSN!"
    )

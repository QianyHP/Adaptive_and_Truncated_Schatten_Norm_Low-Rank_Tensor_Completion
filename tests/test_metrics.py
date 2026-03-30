"""
tests/test_metrics.py – Unit tests for metrics module
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from atsn.metrics import compute_mae, compute_rmse, compute_mape, compute_er, evaluate_all


@pytest.fixture
def simple_tensors():
    """Simple 2×2×2 tensors for deterministic tests."""
    truth = np.array([[[2.0, 4.0], [6.0, 8.0]],
                      [[10., 12.], [14., 16.]]])
    # observed: first element missing
    observed = truth.copy()
    observed[0, 0, 0] = 0.0
    # imputed: perfect recovery
    imputed_perfect = truth.copy()
    # imputed: off by 1 on the missing position
    imputed_off = truth.copy()
    imputed_off[0, 0, 0] = 3.0   # truth = 2.0
    return truth, observed, imputed_perfect, imputed_off


def test_mae_perfect(simple_tensors):
    truth, observed, imputed_perfect, _ = simple_tensors
    assert compute_mae(observed, truth, imputed_perfect) == pytest.approx(0.0, abs=1e-12)


def test_mae_off_by_one(simple_tensors):
    truth, observed, _, imputed_off = simple_tensors
    # One missing position, |2.0 - 3.0| = 1.0
    assert compute_mae(observed, truth, imputed_off) == pytest.approx(1.0, rel=1e-9)


def test_rmse_perfect(simple_tensors):
    truth, observed, imputed_perfect, _ = simple_tensors
    assert compute_rmse(observed, truth, imputed_perfect) == pytest.approx(0.0, abs=1e-12)


def test_mape_perfect(simple_tensors):
    truth, observed, imputed_perfect, _ = simple_tensors
    assert compute_mape(observed, truth, imputed_perfect) == pytest.approx(0.0, abs=1e-12)


def test_er_perfect(simple_tensors):
    truth, observed, imputed_perfect, _ = simple_tensors
    assert compute_er(observed, truth, imputed_perfect) == pytest.approx(0.0, abs=1e-12)


def test_evaluate_all_returns_dict(simple_tensors):
    truth, observed, imputed_perfect, _ = simple_tensors
    result = evaluate_all(observed, truth, imputed_perfect)
    assert set(result.keys()) == {'MAE', 'RMSE', 'MAPE', 'ER'}


def test_no_missing_returns_nan():
    truth    = np.ones((3, 3, 3))
    observed = truth.copy()   # no missing entries
    imputed  = truth.copy()
    # With no missing, pos_test is empty → nan
    assert np.isnan(compute_mae(observed, truth, imputed))

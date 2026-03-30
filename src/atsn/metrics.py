"""
metrics.py – Evaluation metrics for traffic data imputation
============================================================

Standard metrics used throughout the paper to evaluate imputation quality.
All functions accept:

  * ``observed``  (np.ndarray) – the *masked* tensor (0 at missing positions)
  * ``truth``     (np.ndarray) – the ground-truth complete tensor
  * ``imputed``   (np.ndarray) – the recovered tensor

Evaluation is performed **only** on positions that are:
  1. Missing in the observed tensor (value == 0)
  2. Non-zero in the ground truth (avoids division-by-zero artefacts)

Metrics
-------
MAE   Mean Absolute Error
RMSE  Root Mean Squared Error
MAPE  Mean Absolute Percentage Error
ER    Relative Error (Frobenius-norm ratio)
"""

from __future__ import annotations

import numpy as np


def _test_positions(observed: np.ndarray, truth: np.ndarray):
    """Return index tuple for test positions (missing & non-zero ground truth)."""
    return np.where((truth != 0) & (observed == 0))


def compute_mae(observed: np.ndarray, truth: np.ndarray, imputed: np.ndarray) -> float:
    """Mean Absolute Error on test positions.

    Parameters
    ----------
    observed : np.ndarray
        Observed (partially missing) tensor. Missing entries are 0.
    truth : np.ndarray
        Ground-truth complete tensor.
    imputed : np.ndarray
        Imputed tensor returned by a completion algorithm.

    Returns
    -------
    float
        MAE value (km/h for traffic speed data).
    """
    pos = _test_positions(observed, truth)
    if len(pos[0]) == 0:
        return float("nan")
    return float(np.mean(np.abs(truth[pos] - imputed[pos])))


def compute_rmse(observed: np.ndarray, truth: np.ndarray, imputed: np.ndarray) -> float:
    """Root Mean Squared Error on test positions.

    Parameters
    ----------
    observed : np.ndarray
        Observed (partially missing) tensor. Missing entries are 0.
    truth : np.ndarray
        Ground-truth complete tensor.
    imputed : np.ndarray
        Imputed tensor.

    Returns
    -------
    float
        RMSE value.
    """
    pos = _test_positions(observed, truth)
    if len(pos[0]) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((truth[pos] - imputed[pos]) ** 2)))


def compute_mape(observed: np.ndarray, truth: np.ndarray, imputed: np.ndarray) -> float:
    """Mean Absolute Percentage Error on test positions.

    Parameters
    ----------
    observed : np.ndarray
        Observed (partially missing) tensor. Missing entries are 0.
    truth : np.ndarray
        Ground-truth complete tensor.
    imputed : np.ndarray
        Imputed tensor.

    Returns
    -------
    float
        MAPE value (unitless; multiply by 100 for percentage).
    """
    pos = _test_positions(observed, truth)
    if len(pos[0]) == 0:
        return float("nan")
    return float(np.mean(np.abs(truth[pos] - imputed[pos]) / (np.abs(truth[pos]) + 1e-12)))


def compute_er(observed: np.ndarray, truth: np.ndarray, imputed: np.ndarray) -> float:
    """Relative Error (Frobenius norm ratio) on test positions.

    ER = ||truth - imputed||_F / ||truth||_F

    Parameters
    ----------
    observed : np.ndarray
        Observed (partially missing) tensor. Missing entries are 0.
    truth : np.ndarray
        Ground-truth complete tensor.
    imputed : np.ndarray
        Imputed tensor.

    Returns
    -------
    float
        Relative error (dimensionless).
    """
    pos = _test_positions(observed, truth)
    if len(pos[0]) == 0:
        return float("nan")
    numerator = np.sqrt(np.sum((truth[pos] - imputed[pos]) ** 2))
    denominator = np.sqrt(np.sum(truth[pos] ** 2)) + 1e-12
    return float(numerator / denominator)


def evaluate_all(
    observed: np.ndarray,
    truth: np.ndarray,
    imputed: np.ndarray,
) -> dict:
    """Compute all four metrics and return as a dict.

    Parameters
    ----------
    observed : np.ndarray
        Observed (partially missing) tensor.
    truth : np.ndarray
        Ground-truth tensor.
    imputed : np.ndarray
        Imputed tensor.

    Returns
    -------
    dict
        Keys: 'MAE', 'RMSE', 'MAPE', 'ER'.

    Examples
    --------
    >>> metrics = evaluate_all(observed, truth, X_hat)
    >>> print(f"MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")
    """
    return {
        "MAE":  compute_mae(observed, truth, imputed),
        "RMSE": compute_rmse(observed, truth, imputed),
        "MAPE": compute_mape(observed, truth, imputed),
        "ER":   compute_er(observed, truth, imputed),
    }

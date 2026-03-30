"""
lrtc_tspn.py – LRTC via Truncated Schatten-p Norm (LRTC-TSpN)
===============================================================

The direct predecessor of LRTC-ATSN: Schatten-p norm with truncation, but
with *fixed* (non-adaptive) p, theta, and alpha weights.

Reference
---------
[Embedded in the ATSN paper as the static baseline — corresponds to the
algorithm of Section 3.1 before the adaptive extensions are introduced.]
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..atsn.tensor_ops import unfold, fold, update_M_block
from ..atsn.metrics import evaluate_all


def lrtc_tspn(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    theta: float = 0.1,
    alpha: Optional[np.ndarray] = None,
    p: float = 0.5,
    beta: float = 1e-5,
    beta_incre: float = 0.05,
    max_iter: int = 200,
    epsilon: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """LRTC-TSpN: fixed-parameter truncated Schatten-p completion.

    Parameters
    ----------
    complete_tensor : np.ndarray
        Ground-truth tensor (for evaluation only).
    observed_tensor : np.ndarray
        Partially observed tensor (zeros = missing).
    theta : float
        Truncation ratio (default 0.1).
    alpha : np.ndarray of shape (order,), optional
        Mode weights; defaults to uniform.
    p : float
        Schatten-p exponent ∈ (0, 1] (default 0.5).
    beta : float
        Initial ADMM penalty (default 1e-5).
    beta_incre : float
        Beta multiplicative increment per iteration (default 0.05).
    max_iter : int
        Maximum iterations (default 200).
    epsilon : float
        Convergence tolerance (default 1e-3).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    order = observed_tensor.ndim
    dim   = observed_tensor.shape

    if alpha is None:
        alpha = np.ones(order) / order
    else:
        alpha = np.asarray(alpha, dtype=float)
        alpha = alpha / alpha.sum()

    X_true    = complete_tensor.astype(float)
    X_missing = observed_tensor.astype(float)
    Omega     = X_missing != 0

    X = X_missing.copy()
    X[~Omega] = float(np.mean(X_missing[Omega]))

    M = np.zeros((order, *dim), dtype=float)
    Q = np.zeros((order, *dim), dtype=float)

    err_history: list = []
    t0 = time.perf_counter()

    for k in range(max_iter):
        beta *= (1.0 + beta_incre)
        if verbose:
            print(f"\r[LRTC-TSpN] iter {k+1:4d}/{max_iter}", end="", flush=True)

        for i in range(order):
            arg = unfold(X + Q[i] / beta, dim, i)
            Mi_mat = update_M_block(arg, alpha[i], beta, p, theta)
            M[i] = fold(Mi_mat, dim, i)

        X_prev = X.copy()
        X = np.sum(beta * M - Q, axis=0) / (beta * order)
        X[Omega] = X_missing[Omega]

        for i in range(order):
            Q[i] += beta * (X - M[i])

        err = float(np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12))
        err_history.append(err)

        if err < epsilon:
            if verbose:
                print(f"\n[LRTC-TSpN] Converged at iter {k+1}.")
            break

    if verbose and err_history[-1] >= epsilon:
        print(f"\n[LRTC-TSpN] Reached max_iter.")

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(X_missing, X_true, X)
    if verbose:
        print(f"[LRTC-TSpN] iter={k+1}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": X, "iterations": k + 1, "elapsed_sec": elapsed, **metrics}

"""
halrtc.py – High-accuracy Low-Rank Tensor Completion (HaLRTC)
==============================================================

Reference
---------
Liu, J., Musialski, P., Wonka, P., & Ye, J. (2013).
  Tensor Completion for Estimating Missing Values in Visual Data.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1), 208-220.

Algorithm
---------
Minimise  sum_k alpha_k * ||X_(k)||_*
s.t.      X_Omega = Y_Omega

via ADMM / split-Bregman iterations with singular value soft-thresholding.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..atsn.tensor_ops import svt
from ..atsn.metrics import evaluate_all


def _ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding (Fortran order)."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def _mat2ten(mat: np.ndarray, shape: np.ndarray, mode: int) -> np.ndarray:
    """Inverse mode-n unfolding."""
    index = [mode] + [i for i in range(len(shape)) if i != mode]
    return np.moveaxis(
        np.reshape(mat, list(shape[index]), order='F'), 0, mode
    )


def halrtc(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    alpha: Optional[np.ndarray] = None,
    rho: float = 1e-4,
    epsilon: float = 1e-4,
    max_iter: int = 200,
    verbose: bool = True,
) -> dict:
    """HaLRTC: nuclear-norm tensor completion via split-Bregman ADMM.

    Parameters
    ----------
    complete_tensor : np.ndarray
        Ground-truth tensor (for evaluation only).
    observed_tensor : np.ndarray
        Partially observed tensor (zeros = missing).
    alpha : np.ndarray of shape (order,), optional
        Mode weights; defaults to uniform (1/order).
    rho : float
        Initial penalty parameter (default 1e-4).
    epsilon : float
        Convergence tolerance (default 1e-4).
    max_iter : int
        Maximum iterations (default 200).
    verbose : bool
        Print progress every 50 iterations.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    order = observed_tensor.ndim
    dim   = np.array(observed_tensor.shape)

    if alpha is None:
        alpha = np.ones(order) / order

    pos_miss = np.where(observed_tensor == 0)
    pos_test = np.where((complete_tensor != 0) & (observed_tensor == 0))

    tensor_hat = observed_tensor.copy().astype(float)
    B = [np.zeros(observed_tensor.shape) for _ in range(order)]
    Y = [np.zeros(observed_tensor.shape) for _ in range(order)]
    last_ten = observed_tensor.copy().astype(float)
    snorm = np.linalg.norm(observed_tensor)

    t0 = time.perf_counter()
    it = 0

    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(order):
            B[k] = _mat2ten(
                svt(_ten2mat(tensor_hat + Y[k] / rho, k), alpha[k] / rho),
                dim, k
            )
        tensor_hat[pos_miss] = ((sum(B) - sum(Y) / rho) / order)[pos_miss]
        for k in range(order):
            Y[k] = Y[k] - rho * (B[k] - tensor_hat)

        tol = np.linalg.norm(tensor_hat - last_ten) / (snorm + 1e-12)
        last_ten = tensor_hat.copy()
        it += 1

        if verbose and it % 50 == 0:
            metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
            print(f"[HaLRTC] iter {it:4d}  tol={tol:.2e}  "
                  f"MAPE={metrics['MAPE']:.4f}  RMSE={metrics['RMSE']:.4f}")

        if tol < epsilon or it >= max_iter:
            break

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[HaLRTC] Done  iter={it}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": it, "elapsed_sec": elapsed, **metrics}

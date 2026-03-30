"""
lrtc_tnn.py – LRTC with Truncated Nuclear Norm (LRTC-TNN)
==========================================================

Reference
---------
Lu, C., Feng, J., Chen, Y., Liu, W., Lin, Z., & Yan, S. (2019).
  Tensor Robust Principal Component Analysis with a New Tensor Nuclear Norm.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 925–938.

Algorithm
---------
Minimise  sum_k alpha_k * ||X_(k)||_{*,theta_k}
s.t.      X_Omega = Y_Omega

where ||·||_{*,theta} is the Truncated Nuclear Norm that preserves the first
theta singular values and penalises the remainder with L1.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..atsn.tensor_ops import svt_tnn
from ..atsn.metrics import evaluate_all


def _ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def _mat2ten(mat: np.ndarray, shape: np.ndarray, mode: int) -> np.ndarray:
    index = [mode] + [i for i in range(len(shape)) if i != mode]
    return np.moveaxis(np.reshape(mat, list(shape[index]), order='F'), 0, mode)


def lrtc_tnn(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    alpha: Optional[np.ndarray] = None,
    rho: float = 1e-4,
    theta: float = 0.30,
    epsilon: float = 1e-4,
    max_iter: int = 200,
    verbose: bool = True,
) -> dict:
    """LRTC-TNN: tensor completion via Truncated Nuclear Norm.

    Parameters
    ----------
    complete_tensor : np.ndarray
        Ground-truth tensor (for evaluation only).
    observed_tensor : np.ndarray
        Partially observed tensor (zeros = missing).
    alpha : np.ndarray of shape (order,), optional
        Mode weights; defaults to uniform.
    rho : float
        Initial ADMM penalty (default 1e-4).
    theta : float
        Truncation ratio ∈ [0, 1] (default 0.30).
    epsilon : float
        Convergence tolerance (default 1e-4).
    max_iter : int
        Maximum iterations (default 200).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    order = observed_tensor.ndim
    dim   = np.array(observed_tensor.shape)

    if alpha is None:
        alpha = np.ones(order) / order

    pos_missing = np.where(observed_tensor == 0)

    X = np.zeros(np.insert(dim, 0, order))   # (order, *dim)
    T = np.zeros(np.insert(dim, 0, order))   # dual variables
    Z = observed_tensor.copy().astype(float)
    last_tensor = Z.copy()
    snorm = np.sqrt(np.sum(Z ** 2)) + 1e-12

    t0 = time.perf_counter()
    it = 0

    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(order):
            X[k] = _mat2ten(
                svt_tnn(
                    _ten2mat(Z - T[k] / rho, k),
                    alpha[k] / rho,
                    int(np.ceil(theta * dim[k]))
                ),
                dim, k
            )
        Z[pos_missing] = np.mean(X + T / rho, axis=0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, order)))
        tensor_hat = np.einsum('k, k...-> ...', alpha, X)

        tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        last_tensor = tensor_hat.copy()
        it += 1

        if verbose and it % 50 == 0:
            metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
            print(f"[LRTC-TNN] iter {it:4d}  tol={tol:.2e}  "
                  f"MAPE={metrics['MAPE']:.4f}  RMSE={metrics['RMSE']:.4f}")

        if tol < epsilon or it >= max_iter:
            break

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[LRTC-TNN] Done  iter={it}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": it, "elapsed_sec": elapsed, **metrics}

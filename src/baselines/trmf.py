"""
trmf.py – Temporal Regularized Matrix Factorization (TRMF)
===========================================================

Reference
---------
Yu, H.-F., Rao, N., & Dhillon, I. S. (2016).
  Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction.
  NeurIPS 2016.

Algorithm
---------
Minimises  ||P_Omega(M - W X^T)||_F^2
         + lambda_w * ||W||_F^2
         + lambda_x * ||A(X)||_F^2      (temporal AR regularisation)
         + lambda_theta * ||Theta||_F^2

where A(X) = X_{t} - sum_l theta_l * X_{t-l}  is the AR residual.
Solved via block-coordinate descent.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.linalg import inv, solve

from ..atsn.metrics import evaluate_all


def trmf(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    rank: int = 10,
    time_lags: Optional[np.ndarray] = None,
    lambda_w: float = 500.0,
    lambda_x: float = 500.0,
    lambda_theta: float = 500.0,
    eta: float = 1.0,
    max_iter: int = 200,
    verbose: bool = True,
) -> dict:
    """TRMF: temporal regularised matrix factorisation for tensor completion.

    The 3-D tensor is flattened to a 2-D matrix (mode-0 × rest) before fitting.

    Parameters
    ----------
    complete_tensor : np.ndarray, shape (I, J, K)
        Ground-truth tensor (evaluation only).
    observed_tensor : np.ndarray, shape (I, J, K)
        Partially observed tensor (zeros = missing).
    rank : int
        Factorisation rank (default 10).
    time_lags : np.ndarray of int, optional
        AR lag indices (default traffic-specific lags matching 10-min intervals).
    lambda_w : float
        Regularisation weight for W (default 500).
    lambda_x : float
        Regularisation weight for X (default 500).
    lambda_theta : float
        Regularisation weight for AR coefficients (default 500).
    eta : float
        Additional identity regularisation for X (default 1.0).
    max_iter : int
        Block-coordinate descent iterations (default 200).
    verbose : bool
        Print progress every 100 iterations.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.

    Notes
    -----
    The default time_lags encode daily and weekly periodicity for 10-min
    sampling rate (144 intervals/day):
      [1, 2, 3, 144, 145, 146, 1008, 1009, 1010]
    Adjust for your specific dataset.
    """
    dim = complete_tensor.shape
    dense_mat  = complete_tensor.astype(float).reshape(dim[0], -1)
    sparse_mat = observed_tensor.astype(float).reshape(dim[0], -1)
    N, T = sparse_mat.shape

    if time_lags is None:
        # Default: short-term + daily + weekly lags (10-min resolution)
        time_lags = np.array([1, 2, 3, 144, 145, 146, 7 * 144, 7 * 144 + 1, 7 * 144 + 2])

    d = len(time_lags)
    W     = 0.1 * np.random.randn(N, rank)
    X     = 0.1 * np.random.randn(T, rank)
    theta = 0.1 * np.random.randn(d, rank)

    pos_train = np.where(sparse_mat != 0)
    pos_test  = np.where((dense_mat != 0) & (sparse_mat == 0))

    t0 = time.perf_counter()

    for it in range(max_iter):
        # --- Update W -------------------------------------------------------
        for i in range(N):
            pos_i = np.where(sparse_mat[i, :] != 0)[0]
            Xt = X[pos_i, :]
            vec = Xt.T @ sparse_mat[i, pos_i]
            mat = inv(Xt.T @ Xt + lambda_w * np.eye(rank))
            W[i, :] = mat @ vec

        # --- Update X -------------------------------------------------------
        for t in range(T):
            pos_t = np.where(sparse_mat[:, t] != 0)[0]
            Wt = W[pos_t, :]

            if t < int(np.max(time_lags)):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('kr, kr -> r', theta, X[t - time_lags, :])

            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < T - int(np.min(time_lags)):
                if t >= int(np.max(time_lags)) and t < T - int(np.max(time_lags)):
                    index = list(range(d))
                else:
                    index = list(np.where(
                        (t + time_lags >= int(np.max(time_lags))) & (t + time_lags < T)
                    )[0])
                for k_idx in index:
                    Ak = theta[k_idx, :]
                    Mt += np.diag(Ak ** 2)
                    theta0 = theta.copy()
                    theta0[k_idx, :] = 0.0
                    Nt += np.multiply(
                        Ak,
                        X[t + time_lags[k_idx], :]
                        - np.einsum('ij, ij -> j', theta0, X[t + time_lags[k_idx] - time_lags, :])
                    )

            vec0 = Wt.T @ sparse_mat[pos_t, t] + lambda_x * Nt + lambda_x * Qt
            mat0 = inv(Wt.T @ Wt + lambda_x * Mt + lambda_x * Pt + lambda_x * eta * np.eye(rank))
            X[t, :] = mat0 @ vec0

        # --- Update theta ---------------------------------------------------
        for k_idx in range(d):
            theta0 = theta.copy()
            theta0[k_idx, :] = 0.0
            lag_max = int(np.max(time_lags))
            mat_tmp = np.zeros((T - lag_max, rank))
            for L in range(d):
                mat_tmp += (
                    X[lag_max - time_lags[L]: T - time_lags[L], :]
                    @ np.diag(theta0[L, :])
                )
            VarPi = X[lag_max:T, :] - mat_tmp
            var1  = np.zeros((rank, rank))
            var2  = np.zeros(rank)
            for t in range(lag_max, T):
                B = X[t - time_lags[k_idx], :]
                var1 += np.diag(B * B)
                var2 += np.diag(B) @ VarPi[t - lag_max, :]
            theta[k_idx, :] = inv(var1 + lambda_theta * np.eye(rank) / lambda_x) @ var2

        if verbose and (it + 1) % 100 == 0:
            mat_hat = W @ X.T
            mape = np.sum(np.abs(dense_mat[pos_test] - mat_hat[pos_test])
                          / (dense_mat[pos_test] + 1e-12)) / dense_mat[pos_test].shape[0]
            rmse = np.sqrt(np.sum((dense_mat[pos_test] - mat_hat[pos_test]) ** 2)
                           / dense_mat[pos_test].shape[0])
            print(f"[TRMF] iter {it+1:4d}  MAPE={mape:.4f}  RMSE={rmse:.4f}")

    mat_hat    = W @ X.T
    tensor_hat = mat_hat.reshape(dim)

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[TRMF] Done  iter={max_iter}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": max_iter, "elapsed_sec": elapsed, **metrics}

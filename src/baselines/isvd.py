"""
isvd.py – Iterative SVD for Matrix Completion (ISVD)
=====================================================

A simple iterative algorithm that alternates between:
  1. Rank-r SVD approximation of the current estimate.
  2. Re-imputing the missing entries with the low-rank estimate.

Used as a simple matrix-based baseline.

Reference
---------
Troyanskaya, O., et al. (2001).
  Missing value estimation methods for DNA microarrays.
  Bioinformatics, 17(6), 520–525.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..atsn.metrics import evaluate_all


def isvd(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    rank: int = 10,
    max_iter: int = 100,
    verbose: bool = True,
) -> dict:
    """Iterative SVD matrix completion (tensor flattened to 2-D).

    Parameters
    ----------
    complete_tensor : np.ndarray, shape (I, J, K)
        Ground-truth tensor (evaluation only).
    observed_tensor : np.ndarray, shape (I, J, K)
        Partially observed tensor (zeros = missing).
    rank : int
        Target rank for truncated SVD (default 10).
    max_iter : int
        Maximum number of iterations (default 100).
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    dim        = complete_tensor.shape
    dense_mat  = complete_tensor.astype(float).reshape(dim[0], -1)
    sparse_mat = observed_tensor.astype(float).reshape(dim[0], -1)
    N, T = sparse_mat.shape

    pos_miss = np.where(sparse_mat == 0)

    # Initialise with bias correction
    mu = float(np.mean(sparse_mat[sparse_mat != 0]))
    bias_row = np.zeros(N)
    bias_col = np.zeros(T)
    temp = sparse_mat - mu
    for n in range(N):
        vals = temp[n, :][sparse_mat[n, :] != 0]
        bias_row[n] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    for t in range(T):
        vals = temp[:, t][sparse_mat[:, t] != 0]
        bias_col[t] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    mat = sparse_mat.copy()
    mat[pos_miss] = (mu + bias_row[:, None] + bias_col[None, :])[pos_miss]

    t0 = time.perf_counter()
    for it in range(max_iter):
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        mat_hat = u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :]
        mat[pos_miss] = mat_hat[pos_miss]

        if verbose and (it + 1) % 10 == 0:
            tmp = mat.reshape(dim)
            metrics = evaluate_all(observed_tensor, complete_tensor, tmp)
            print(f"[ISVD] iter {it+1:4d}  "
                  f"MAPE={metrics['MAPE']:.5f}  RMSE={metrics['RMSE']:.5f}")

    tensor_hat = mat.reshape(dim)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[ISVD] Done  iter={it+1}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": it + 1, "elapsed_sec": elapsed, **metrics}

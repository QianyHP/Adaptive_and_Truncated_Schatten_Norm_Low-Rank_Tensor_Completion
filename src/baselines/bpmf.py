"""
bpmf.py – Bayesian Probabilistic Matrix Factorization (BPMF)
=============================================================

Reference
---------
Salakhutdinov, R., & Mnih, A. (2008).
  Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo.
  ICML 2008.

Algorithm
---------
Gibbs sampler for matrix factorisation:  M ≈ W X^T
with Normal-Wishart hyperpriors on W and X.
Used as a baseline for *matrix-form* completion (tensor flattened to 2-D).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from numpy.linalg import inv, solve
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod, cholesky, solve_triangular
from scipy.stats import wishart

from ..atsn.metrics import evaluate_all


def _mvnrnd_pre(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    src = normrnd(size=(mu.shape[0],))
    return solve_triangular(
        cholesky(Lambda, overwrite_a=True, check_finite=False),
        src, lower=False, check_finite=False, overwrite_b=True
    ) + mu


def _cov_mat(mat: np.ndarray, mat_bar: np.ndarray) -> np.ndarray:
    return (mat - mat_bar).T @ (mat - mat_bar)


def _sample_W(
    tau_sparse: np.ndarray,
    tau_ind: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    tau: float,
    beta0: float = 1.0,
) -> np.ndarray:
    """Gibbs sample rows of factor matrix W."""
    N, rank = W.shape
    W_bar = np.mean(W, axis=0)
    temp = N / (N + beta0)
    var_mu    = temp * W_bar
    var_W_hyp = inv(np.eye(rank) + _cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda = wishart.rvs(df=N + rank, scale=var_W_hyp)
    var_mu     = _mvnrnd_pre(var_mu, (N + beta0) * var_Lambda)

    if N * rank ** 2 > 1e8:
        # Row-by-row sampling for memory efficiency
        for i in range(N):
            pos = np.where(tau_sparse[i, :] != 0)[0]
            Xt = X[pos, :]
            var_mu_i = tau * Xt.T @ tau_sparse[i, pos] + var_Lambda @ var_mu
            var_Lam_i = tau * Xt.T @ Xt + var_Lambda
            W[i, :] = _mvnrnd_pre(solve(var_Lam_i, var_mu_i), var_Lam_i)
    else:
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, N]) + var_Lambda[:, :, np.newaxis]
        var4 = var1 @ tau_sparse.T + (var_Lambda @ var_mu)[:, np.newaxis]
        for i in range(N):
            W[i, :] = _mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return W


def _sample_X(
    tau_sparse: np.ndarray,
    tau_ind: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    beta0: float = 1.0,
) -> np.ndarray:
    """Gibbs sample rows of factor matrix X."""
    T, rank = X.shape
    X_bar = np.mean(X, axis=0)
    temp = T / (T + beta0)
    var_mu    = temp * X_bar
    var_X_hyp = inv(np.eye(rank) + _cov_mat(X, X_bar) + temp * beta0 * np.outer(X_bar, X_bar))
    var_Lambda = wishart.rvs(df=T + rank, scale=var_X_hyp)
    var_mu     = _mvnrnd_pre(var_mu, (T + beta0) * var_Lambda)

    var1 = W.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind).reshape([rank, rank, T]) + var_Lambda[:, :, np.newaxis]
    var4 = var1 @ tau_sparse + (var_Lambda @ var_mu)[:, np.newaxis]
    for t in range(T):
        X[t, :] = _mvnrnd_pre(solve(var3[:, :, t], var4[:, t]), var3[:, :, t])
    return X


def _sample_tau(
    sparse: np.ndarray,
    mat_hat: np.ndarray,
    ind: np.ndarray,
) -> float:
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta  = 1e-6 + 0.5 * np.sum(((sparse - mat_hat) ** 2) * ind)
    return float(np.random.gamma(var_alpha, 1.0 / var_beta))


def bpmf(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    rank: int = 10,
    burn_iter: int = 200,
    gibbs_iter: int = 50,
    verbose: bool = True,
) -> dict:
    """BPMF matrix factorisation for tensor completion (matrix mode).

    The tensor is flattened to a 2-D matrix along mode-0 before fitting.

    Parameters
    ----------
    complete_tensor : np.ndarray, shape (I, J, K)
        Ground-truth tensor (for evaluation only).
    observed_tensor : np.ndarray, shape (I, J, K)
        Partially observed tensor (zeros = missing).
    rank : int
        Matrix factorisation rank (default 10).
    burn_iter : int
        Burn-in iterations (default 200).
    gibbs_iter : int
        Posterior averaging iterations (default 50).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: 'X_hat' (tensor shape restored), 'iterations',
        'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    dim = complete_tensor.shape
    dense_mat  = complete_tensor.astype(float).reshape(dim[0], -1)
    sparse_mat = observed_tensor.astype(float).reshape(dim[0], -1)
    ind = sparse_mat != 0

    N, T = sparse_mat.shape
    W = 0.01 * np.random.randn(N, rank)
    X = 0.01 * np.random.randn(T, rank)
    tau = 1.0

    W_plus = np.zeros_like(W)
    X_plus = np.zeros_like(X)
    mat_hat_plus = np.zeros_like(sparse_mat)
    temp_hat = np.zeros_like(sparse_mat)

    show_iter = 200
    total_iter = burn_iter + gibbs_iter

    t0 = time.perf_counter()
    for it in range(total_iter):
        tau_ind    = tau * ind.astype(float)
        tau_sparse = tau * sparse_mat
        W   = _sample_W(tau_sparse, tau_ind, W, X, tau)
        X   = _sample_X(tau_sparse, tau_ind, W, X)
        mat_hat = W @ X.T
        tau = _sample_tau(sparse_mat, mat_hat, ind)
        temp_hat += mat_hat

        if verbose and (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat /= show_iter
            tmp_tensor = temp_hat.reshape(dim)
            metrics = evaluate_all(observed_tensor, complete_tensor, tmp_tensor)
            print(f"[BPMF] iter {it+1:5d}  "
                  f"MAPE={metrics['MAPE']:.4f}  RMSE={metrics['RMSE']:.4f}")
            temp_hat = np.zeros_like(sparse_mat)

        if it + 1 > burn_iter:
            W_plus += W
            X_plus += X
            mat_hat_plus += mat_hat

    mat_hat = mat_hat_plus / gibbs_iter
    tensor_hat = mat_hat.reshape(dim)

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[BPMF] Done  iter={total_iter}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": total_iter, "elapsed_sec": elapsed, **metrics}

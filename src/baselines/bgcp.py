"""
bgcp.py – Bayesian Gaussian CP Decomposition (BGCP)
====================================================

Reference
---------
Chen, X., & Sun, L. (2022).
  Bayesian Temporal Factorization for Multidimensional Time Series Prediction.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9), 4659–4673.

Algorithm
---------
Bayesian CP decomposition with Gibbs sampling. Recovers a tensor X via a
rank-R CP factorisation:  X ≈ sum_{r} a_r ⊗ b_r ⊗ c_r
with conjugate Wishart / Gaussian priors on factor matrices.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from numpy.linalg import inv, solve
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod, cholesky, solve_triangular
from scipy.stats import wishart

from ..atsn.metrics import evaluate_all


def _mvnrnd_pre(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    """Sample from N(mu, Lambda^{-1}) using upper-Cholesky."""
    src = normrnd(size=(mu.shape[0],))
    return solve_triangular(
        cholesky(Lambda, overwrite_a=True, check_finite=False),
        src, lower=False, check_finite=False, overwrite_b=True
    ) + mu


def _ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def _cp_combine(factor: List[np.ndarray]) -> np.ndarray:
    """Reconstruct tensor from CP factors: X ≈ einsum('is,js,ts->ijt', ...)."""
    return np.einsum('is, js, ts -> ijt', factor[0], factor[1], factor[2])


def _cov_mat(mat: np.ndarray, mat_bar: np.ndarray) -> np.ndarray:
    mat = mat - mat_bar
    return mat.T @ mat


def _sample_factor(
    tau_sparse: np.ndarray,
    tau_ind: np.ndarray,
    factor: List[np.ndarray],
    k: int,
    beta0: float = 1.0,
) -> np.ndarray:
    """Gibbs sample for factor matrix k with Normal-Wishart hyperprior."""
    dim, rank = factor[k].shape
    factor_bar = np.mean(factor[k], axis=0)
    temp = dim / (dim + beta0)
    var_mu = temp * factor_bar
    var_W = inv(np.eye(rank) + _cov_mat(factor[k], factor_bar)
                + temp * beta0 * np.outer(factor_bar, factor_bar))
    var_Lambda = wishart.rvs(df=dim + rank, scale=var_W)
    var_mu = _mvnrnd_pre(var_mu, (dim + beta0) * var_Lambda)

    idx = [j for j in range(len(factor)) if j != k]
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ _ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda[:, :, np.newaxis]
    var4 = var1 @ _ten2mat(tau_sparse, k).T + (var_Lambda @ var_mu)[:, np.newaxis]
    for i in range(dim):
        factor[k][i, :] = _mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]


def _sample_tau(
    sparse: np.ndarray,
    tensor_hat: np.ndarray,
    ind: np.ndarray,
) -> float:
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta  = 1e-6 + 0.5 * np.sum(((sparse - tensor_hat) ** 2) * ind)
    return float(np.random.gamma(var_alpha, 1.0 / var_beta))


def bgcp(
    complete_tensor: np.ndarray,
    observed_tensor: np.ndarray,
    *,
    rank: int = 10,
    burn_iter: int = 1000,
    gibbs_iter: int = 200,
    verbose: bool = True,
) -> dict:
    """Bayesian Gaussian CP decomposition for tensor completion.

    Parameters
    ----------
    complete_tensor : np.ndarray
        Ground-truth tensor (for evaluation only).
    observed_tensor : np.ndarray
        Partially observed tensor (zeros = missing).
    rank : int
        CP rank (default 10).
    burn_iter : int
        Burn-in Gibbs iterations (default 1000).
    gibbs_iter : int
        Post-burn-in Gibbs samples to average (default 200).
    verbose : bool
        Print progress every 200 iterations.

    Returns
    -------
    dict
        Keys: 'X_hat', 'iterations', 'elapsed_sec', 'MAE', 'RMSE', 'MAPE', 'ER'.
    """
    dim = np.array(observed_tensor.shape)
    ind = observed_tensor != 0

    factor = [0.1 * np.random.randn(int(d), rank) for d in dim]
    factor_plus = [np.zeros_like(f) for f in factor]
    tensor_hat_plus = np.zeros(observed_tensor.shape)
    temp_hat = np.zeros(observed_tensor.shape)

    tau = 1.0
    show_iter = 200
    total_iter = burn_iter + gibbs_iter

    t0 = time.perf_counter()
    for it in range(total_iter):
        tau_ind     = tau * ind.astype(float)
        tau_sparse  = tau * observed_tensor.astype(float)
        for k in range(len(dim)):
            factor[k] = _sample_factor(tau_sparse, tau_ind, factor, k)
        tensor_hat = _cp_combine(factor)
        temp_hat  += tensor_hat
        tau        = _sample_tau(observed_tensor, tensor_hat, ind)

        if it + 1 > burn_iter:
            factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
            tensor_hat_plus += tensor_hat

        if verbose and (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat /= show_iter
            metrics = evaluate_all(observed_tensor, complete_tensor, temp_hat)
            print(f"[BGCP] iter {it+1:5d}  "
                  f"MAPE={metrics['MAPE']:.4f}  RMSE={metrics['RMSE']:.4f}")
            temp_hat = np.zeros(observed_tensor.shape)

    factor     = [f / gibbs_iter for f in factor_plus]
    tensor_hat = tensor_hat_plus / gibbs_iter

    elapsed = time.perf_counter() - t0
    metrics = evaluate_all(observed_tensor, complete_tensor, tensor_hat)
    if verbose:
        print(f"[BGCP] Done  iter={total_iter}  time={elapsed:.1f}s  "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.4f}  ER={metrics['ER']:.4f}")

    return {"X_hat": tensor_hat, "iterations": total_iter, "elapsed_sec": elapsed, **metrics}

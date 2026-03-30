"""
tensor_ops.py – Low-level tensor algebra utilities
===================================================

This module provides the fundamental tensor operations required by all
low-rank tensor completion (LRTC) algorithms in this package:

  * Mode-n unfolding / folding  (Kolda & Bader 2009 convention)
  * Generalised Soft-Thresholding (GST) for Schatten-p proximal operator
  * Truncation weight vector for the Truncated Nuclear / Schatten norm
  * Soft-singular-value thresholding (SVT) for the nuclear norm

All functions operate on plain NumPy arrays.  No external dependencies
beyond NumPy are required.

References
----------
Kolda, T. G., & Bader, B. W. (2009).
  Tensor Decompositions and Applications. SIAM Review, 51(3), 455–500.

Lu, C., et al. (2019).
  Tensor Robust Principal Component Analysis with a New Tensor Nuclear Norm.
  IEEE TPAMI.

Nie, F., et al. (2012).
  Low-Rank Matrix Recovery via Efficient Schatten p-Norm Minimization.
  AAAI Conference.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple


# ---------------------------------------------------------------------------
# Tensor fold / unfold
# ---------------------------------------------------------------------------

def _shiftdim(array: np.ndarray, n: int) -> np.ndarray:
    """Cyclic dimension shift (analogous to MATLAB's ``shiftdim``).

    Parameters
    ----------
    array : np.ndarray
        Input array of any order.
    n : int
        Number of positions to rotate axes to the *right* (positive) or
        *left* (negative).

    Returns
    -------
    np.ndarray
        View with permuted axes.
    """
    axes = tuple(range(array.ndim))
    new_axes = deque(axes)
    new_axes.rotate(n)
    return np.moveaxis(array, axes, tuple(new_axes))


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding of a tensor (Fortran / column-major order).

    The result is a 2-D matrix of shape
    ``(tensor.shape[mode], prod(tensor.shape) // tensor.shape[mode])``.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor of arbitrary order *N*.
    mode : int
        Mode index (0-based) along which to unfold.

    Returns
    -------
    np.ndarray, shape (tensor.shape[mode], -1)
        The mode-n unfolded matrix.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(24).reshape(2, 3, 4)
    >>> unfold(X, mode=0).shape
    (2, 12)
    >>> unfold(X, mode=1).shape
    (3, 8)
    """
    return np.reshape(_shiftdim(tensor, mode), (tensor.shape[mode], -1), order='F')


def fold(matrix: np.ndarray, shape: Tuple[int, ...], mode: int) -> np.ndarray:
    """Inverse of ``unfold``: fold a mode-n matrix back into a tensor.

    Parameters
    ----------
    matrix : np.ndarray, shape (shape[mode], -1)
        Mode-n unfolded matrix to fold.
    shape : tuple of int
        Target tensor shape.
    mode : int
        Mode index used during unfolding.

    Returns
    -------
    np.ndarray
        Tensor of shape ``shape``.
    """
    rolled_shape = tuple(np.roll(shape, -mode))
    tensor = _shiftdim(np.reshape(matrix, rolled_shape, order='F'), len(shape) - mode)
    return tensor


# ---------------------------------------------------------------------------
# Proximal operators for Schatten-p norm
# ---------------------------------------------------------------------------

def gst(sigma: float, w: float, p: float, max_inner_iter: int = 5) -> float:
    """Generalised Soft-Thresholding (GST) proximal operator.

    Solves the scalar sub-problem of the Schatten-p (0 < p ≤ 1) proximal
    operator via a fixed-point iteration (Cao et al., 2013):

        prox_{w |·|^p}(sigma)  =  argmin_x  (1/2)(x - sigma)^2 + w |x|^p

    The unique positive solution is found with ``max_inner_iter`` Newton
    steps; empirically J = 5 is sufficient for convergence to machine
    precision.

    Parameters
    ----------
    sigma : float
        Singular value to threshold.
    w : float
        Penalty weight (``alpha_i / beta`` in the ADMM context).
        When *w* = 0 the identity is returned.
    p : float
        Schatten-p exponent, 0 < p ≤ 1.
    max_inner_iter : int, optional
        Number of Newton iterations (default 5).

    Returns
    -------
    float
        Thresholded singular value (always non-negative).

    References
    ----------
    Cao, W., Sun, J., & Xu, Z. (2013).
      Fast image deconvolution using closed-form thresholding formulas of
      Lq (q = 1/2, 2/3) regularization.  J. Visual Communication and
      Image Representation.
    """
    if w == 0:
        return sigma

    # Threshold below which the proximal solution is exactly 0
    tau = (2.0 * w * (1.0 - p)) ** (1.0 / (2.0 - p)) + w * p * (2.0 * w * (1.0 - p)) ** ((p - 1.0) / (2.0 - p))

    abs_sigma = abs(sigma)
    if abs_sigma <= tau:
        return 0.0

    # Fixed-point Newton iteration
    x = abs_sigma
    for _ in range(max_inner_iter):
        x = abs_sigma - w * p * (x ** (p - 1.0))

    return float(np.sign(sigma) * x)


def truncation_weights(matrix: np.ndarray, theta: float) -> np.ndarray:
    """Compute the truncation weight vector for a given matrix.

    In the Truncated Schatten-p norm, the first *r* = ceil(theta * min(m, n))
    singular values are *kept unchanged* (weight 0), and the remainder are
    penalised (weight 1).

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        The unfolded matrix whose truncation weights are needed.
    theta : float
        Truncation ratio in [0, 1].  A value of 0 recovers the full
        Schatten-p norm; 1 turns off all penalisation (identity map).

    Returns
    -------
    np.ndarray, shape (min(m, n),)
        Weight vector: first *r* entries are 0, rest are 1.
    """
    m, n = matrix.shape
    k = min(m, n)
    r = int(np.ceil(theta * k))
    w = np.ones(k, dtype=float)
    w[:r] = 0.0
    return w


def svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Singular Value soft-Thresholding (nuclear norm proximal operator).

    Computes  U * max(S - tau, 0) * V^T, where M = U S V^T.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Input matrix.
    tau : float
        Threshold value (non-negative).

    Returns
    -------
    np.ndarray, shape (m, n)
        Low-rank approximation after soft-thresholding.
    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return u @ np.diag(s_thresh) @ vt


def svt_tnn(matrix: np.ndarray, tau: float, theta: int) -> np.ndarray:
    """SVT for the Truncated Nuclear Norm (TNN).

    The first *theta* singular values are kept intact; the remaining ones
    are soft-thresholded by *tau*.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Input matrix.
    tau : float
        Soft-threshold value.
    theta : int
        Number of leading singular values to preserve unchanged.

    Returns
    -------
    np.ndarray, shape (m, n)
        Proximal result.
    """
    m, n = matrix.shape

    # Use the economy SVD trick for very rectangular matrices
    if 2 * m < n:
        u, s, _ = np.linalg.svd(matrix @ matrix.T, full_matrices=False)
        s = np.sqrt(np.maximum(s, 0.0))
        idx = int(np.sum(s > tau))
        coef = np.zeros(idx)
        coef[:theta] = 1.0
        coef[theta:idx] = (s[theta:idx] - tau) / (s[theta:idx] + 1e-12)
        return (u[:, :idx] @ np.diag(coef)) @ (u[:, :idx].T @ matrix)

    if m > 2 * n:
        return svt_tnn(matrix.T, tau, theta).T

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    idx = int(np.sum(s > tau))
    vec = s[:idx].copy()
    vec[theta:] = np.maximum(vec[theta:] - tau, 0.0)
    return u[:, :idx] @ np.diag(vec) @ vt[:idx, :]


def update_M_block(
    matrix: np.ndarray,
    alpha_i: float,
    beta: float,
    p: float,
    theta: float,
) -> np.ndarray:
    """Compute the closed-form M-update in the ADMM sub-problem.

    Applies the Schatten-p proximal operator with truncation to a matrix:

        M_i = prox_{(alpha_i / beta) * w * |·|_p} ( matrix )

    where w is the truncation weight vector defined by *theta*.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Augmented Lagrangian argument: X_unfold + (1/beta) * Q_unfold.
    alpha_i : float
        Mode weight for dimension *i*.
    beta : float
        Current ADMM penalty parameter.
    p : float
        Schatten-p exponent.
    theta : float
        Truncation ratio.

    Returns
    -------
    np.ndarray, shape (m, n)
        Updated M_i (matrix form).
    """
    u, d, vt = np.linalg.svd(matrix, full_matrices=False)
    w = truncation_weights(matrix, theta)
    lam = alpha_i / beta
    d_new = np.array([gst(di, lam * wi, p) for di, wi in zip(d, w)])
    return u @ np.diag(d_new) @ vt

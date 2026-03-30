"""
missing.py – Synthetic missing-pattern generators for traffic tensors
======================================================================

This module provides three canonical missing-pattern generators used in
the paper experiments:

  1. ``random_missing``   – uniformly random element-wise missing (RM)
  2. ``fiber_missing``    – structured fiber / tube missing (FM)
  3. ``mixed_missing``    – combination of fiber + random missing (MM)

All functions use zero to denote missing entries, consistent with the
convention adopted by the ADMM solvers.

Design notes
------------
* All generators accept a ``seed`` parameter for full reproducibility.
* The *mixed* generator first places fibers, then overlays random
  element missing on top (two separate seeds are applied internally
  so results are stable across partial usage).
* Tensor shape is assumed to be (intervals, links, days) for traffic data,
  but the functions are dimension-agnostic for order-3 tensors.
"""

from __future__ import annotations

import random as _random
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_3d(tensor: np.ndarray) -> None:
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3-D tensor, got shape {tensor.shape}. "
            "All missing generators require order-3 tensors."
        )


# ---------------------------------------------------------------------------
# 1. Random (element-wise) missing
# ---------------------------------------------------------------------------

def random_missing(
    tensor: np.ndarray,
    rate: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Uniformly random element-wise missing pattern (RM).

    Independently samples a fraction *rate* of all elements and sets
    them to zero.

    Parameters
    ----------
    tensor : np.ndarray, shape (I, J, K)
        Complete ground-truth tensor.
    rate : float
        Fraction of elements to mask, in (0, 1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Corrupted tensor with ``rate * tensor.size`` entries set to 0.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(144, 214, 61)   # Guangzhou-like shape
    >>> X_obs = random_missing(X, rate=0.4, seed=42)
    >>> np.mean(X_obs == 0)   # ≈ 0.40
    """
    _check_3d(tensor)
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1), got {rate}.")

    rng = np.random.default_rng(seed)
    mask = np.ones(tensor.size, dtype=bool)
    miss_idx = rng.choice(tensor.size, size=int(rate * tensor.size), replace=False)
    mask[miss_idx] = False

    corrupted = tensor.copy()
    corrupted.flat[miss_idx] = 0.0
    return corrupted


# ---------------------------------------------------------------------------
# 2. Fiber / tube missing
# ---------------------------------------------------------------------------

def fiber_missing(
    tensor: np.ndarray,
    rate: float,
    mode: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Structured fiber missing along a single mode (FM).

    A fiber is a 1-D section obtained by fixing all indices except one.
    For traffic tensors of shape (intervals, links, days):

      * mode 0 → temporal fibers: whole time-series of a (link, day) pair
      * mode 1 → spatial fibers:  all links at a fixed (interval, day)
      * mode 2 → daily fibers:    all days at a fixed (interval, link)

    The missing fraction *rate* is defined over the 2-D slice orthogonal
    to the fiber axis (i.e., fraction of (link, day) pairs set missing).

    Parameters
    ----------
    tensor : np.ndarray, shape (I, J, K)
        Complete ground-truth tensor.
    rate : float
        Fraction of fibers to remove, in (0, 1).
    mode : int
        Fiber axis (0, 1, or 2).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Corrupted tensor.
    """
    _check_3d(tensor)
    if mode not in (0, 1, 2):
        raise ValueError(f"mode must be 0, 1, or 2 for a 3-D tensor, got {mode}.")

    corrupted = tensor.copy()
    n = tensor.shape
    # Indices of the two "selector" dimensions (orthogonal to the fiber)
    selector_shape = tuple(n[i] for i in range(3) if i != mode)
    coords = [(i, j) for i in range(selector_shape[0]) for j in range(selector_shape[1])]

    rng = _random.Random(seed)
    n_miss = int(rate * len(coords))
    selected = rng.sample(coords, n_miss)

    other_dims = [i for i in range(3) if i != mode]
    for (a, b) in selected:
        idx = [slice(None), slice(None), slice(None)]
        idx[other_dims[0]] = a
        idx[other_dims[1]] = b
        corrupted[tuple(idx)] = 0.0

    missing_rate = np.mean(corrupted == 0)
    print(f"[fiber_missing] mode={mode}, target_rate={rate:.2%}, "
          f"actual_rate={missing_rate:.2%}")
    return corrupted


# ---------------------------------------------------------------------------
# 3. Mixed missing (fiber + random)
# ---------------------------------------------------------------------------

def mixed_missing(
    tensor: np.ndarray,
    fiber_rate: float,
    element_rate: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Complex mixed missing: fiber-structured + random element missing (MM).

    This is the primary missing pattern studied in the paper.  Fiber
    segments are placed first across all three modes, then random
    element-wise corruption is overlaid.

    The fiber placement loop randomly selects:
      - a mode direction (0, 1, or 2)
      - a starting coordinate
      - a segment length

    and iterates until the total number of zeroed entries reaches the
    fiber target.  Random elements are then masked independently.

    Note: the actual combined missing rate will generally exceed
    ``fiber_rate + element_rate`` due to overlaps.

    Parameters
    ----------
    tensor : np.ndarray, shape (I, J, K)
        Complete ground-truth tensor.
    fiber_rate : float
        Target fraction of elements to zero via fiber segments, in (0, 1).
    element_rate : float
        Additional fraction of elements to zero randomly, in (0, 1).
    seed : int, optional
        Master random seed. Fiber and element stages each derive a
        reproducible sub-seed from this value.

    Returns
    -------
    np.ndarray
        Corrupted tensor with combined missing pattern.

    Examples
    --------
    >>> X_obs = mixed_missing(X, fiber_rate=0.3, element_rate=0.2, seed=1000)
    """
    _check_3d(tensor)
    corrupted = tensor.copy()
    total = corrupted.size

    # --- Stage 1: fiber segments -----------------------------------------
    rng_fiber = np.random.default_rng(seed)
    py_rng = _random.Random(seed)

    target_fiber = int(total * fiber_rate)
    placed = 0

    while placed < target_fiber:
        direction = py_rng.choice([0, 1, 2])
        start = [
            int(rng_fiber.integers(0, corrupted.shape[0])),
            int(rng_fiber.integers(0, corrupted.shape[1])),
            int(rng_fiber.integers(0, corrupted.shape[2])),
        ]
        max_len = min(
            corrupted.shape[direction] - start[direction],
            target_fiber - placed,
        )
        if max_len < 1:
            continue
        length = int(rng_fiber.integers(1, max_len + 1))

        sl = [slice(None), slice(None), slice(None)]
        sl[direction] = slice(start[direction], start[direction] + length)
        # Fix the two other dimensions at the starting point
        for d in range(3):
            if d != direction:
                sl[d] = start[d]
        corrupted[tuple(sl)] = 0.0
        placed += length

    fiber_actual = int(np.sum(corrupted == 0))
    print(f"[mixed_missing] fiber stage done: "
          f"{fiber_actual / total:.2%} missing after fibers.")

    # --- Stage 2: random element missing ---------------------------------
    seed2 = None if seed is None else seed + 1
    rng_elem = np.random.default_rng(seed2)
    n_elem = int(total * element_rate)
    elem_idx = rng_elem.choice(total, size=n_elem, replace=False)
    corrupted.flat[elem_idx] = 0.0

    total_missing = np.mean(corrupted == 0)
    print(f"[mixed_missing] final missing rate: {total_missing:.2%} "
          f"(fiber_rate={fiber_rate:.2%}, element_rate={element_rate:.2%})")
    return corrupted


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_missing_rate(tensor: np.ndarray) -> float:
    """Compute the fraction of zero (missing) entries in *tensor*.

    Parameters
    ----------
    tensor : np.ndarray
        Possibly-corrupted tensor.

    Returns
    -------
    float
        Missing rate in [0, 1].
    """
    return float(np.mean(tensor == 0))

"""
run_experiment.py – Reproducible experiment runner
====================================================

Run a single algorithm on a single dataset with a specified missing pattern.

Usage
-----
From the project root:

    python experiments/run_experiment.py \\
        --data data/raw/guangzhou_tensor.npy \\
        --algorithm atsn \\
        --missing mixed \\
        --fiber_rate 0.3 \\
        --element_rate 0.2 \\
        --seed 1000 \\
        --save_result results/atsn_gz_mixed.npz

Supported algorithms : atsn, halrtc, lrtc_tnn, lrtc_tspn, bgcp, bpmf, trmf, isvd
Supported missing    : random, fiber, mixed
Supported datasets   : Any .npy or .npz (key 'arr_0') or .mat (key 'tensor') file.
"""

import argparse
import time
import sys
import os
import numpy as np
import scipy.io

# ---- Ensure src/ is on the path when run from project root ----------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atsn           import LRTC_ATSN
from atsn.missing   import random_missing, fiber_missing, mixed_missing, get_missing_rate
from atsn.metrics   import evaluate_all
from baselines      import halrtc, lrtc_tnn, lrtc_tspn, bgcp, bpmf, trmf, isvd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tensor(path: str) -> np.ndarray:
    """Load a tensor from .npy, .npz (key 'arr_0'), or .mat (key 'tensor')."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        key  = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
        return data[key]
    elif ext == '.mat':
        data = scipy.io.loadmat(path)
        key  = 'tensor' if 'tensor' in data else [k for k in data if not k.startswith('_')][0]
        return data[key]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Missing pattern generation
# ---------------------------------------------------------------------------

def generate_missing(
    tensor: np.ndarray,
    pattern: str,
    fiber_rate: float,
    element_rate: float,
    seed: int,
) -> np.ndarray:
    if pattern == 'random':
        return random_missing(tensor, rate=element_rate, seed=seed)
    elif pattern == 'fiber':
        return fiber_missing(tensor, rate=fiber_rate, mode=0, seed=seed)
    elif pattern == 'mixed':
        return mixed_missing(tensor, fiber_rate=fiber_rate, element_rate=element_rate, seed=seed)
    else:
        raise ValueError(f"Unknown missing pattern: {pattern}. "
                         "Choose from: random, fiber, mixed.")


# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------

ALGORITHM_MAP = {
    'atsn':     lambda c, o, kw: LRTC_ATSN(c, o, **kw),
    'halrtc':   lambda c, o, kw: halrtc(c, o, **kw),
    'lrtc_tnn': lambda c, o, kw: lrtc_tnn(c, o, **kw),
    'lrtc_tspn':lambda c, o, kw: lrtc_tspn(c, o, **kw),
    'bgcp':     lambda c, o, kw: bgcp(c, o, **kw),
    'bpmf':     lambda c, o, kw: bpmf(c, o, **kw),
    'trmf':     lambda c, o, kw: trmf(c, o, **kw),
    'isvd':     lambda c, o, kw: isvd(c, o, **kw),
}


def run(args: argparse.Namespace) -> dict:
    """Load data, generate missing, run algorithm, return results."""
    print(f"\n{'='*60}")
    print(f"  Dataset   : {args.data}")
    print(f"  Algorithm : {args.algorithm.upper()}")
    print(f"  Missing   : {args.missing}  "
          f"(fiber={args.fiber_rate}, elem={args.element_rate}, seed={args.seed})")
    print(f"{'='*60}\n")

    # Load
    X = load_tensor(args.data)
    print(f"Tensor shape: {X.shape}  dtype: {X.dtype}")

    # Generate missing
    np.random.seed(args.seed)
    X_obs = generate_missing(X, args.missing, args.fiber_rate, args.element_rate, args.seed)
    print(f"Actual missing rate: {100 * get_missing_rate(X_obs):.2f}%\n")

    # Run algorithm
    algo_fn = ALGORITHM_MAP.get(args.algorithm)
    if algo_fn is None:
        raise ValueError(f"Unknown algorithm '{args.algorithm}'. "
                         f"Choose from: {list(ALGORITHM_MAP.keys())}")

    algo_kwargs = {}  # All algorithms use default hyperparameters
    result = algo_fn(X, X_obs, algo_kwargs)

    # Unpack result (ATSNResult dataclass or dict)
    if hasattr(result, 'final_mae'):
        # ATSNResult dataclass
        metrics = {
            'MAE':  result.final_mae,
            'RMSE': result.final_rmse,
            'MAPE': result.final_mape,
            'ER':   result.final_er,
        }
        X_hat    = result.X_hat
        elapsed  = result.elapsed_sec
        iters    = result.iterations
    else:
        # dict (baseline algorithms)
        metrics = {k: result[k] for k in ('MAE', 'RMSE', 'MAPE', 'ER')}
        X_hat   = result['X_hat']
        elapsed = result['elapsed_sec']
        iters   = result['iterations']

    print(f"\n{'─'*45}")
    print(f"  Final Results")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    print(f"  MAPE = {metrics['MAPE']:.4f}  ({100*metrics['MAPE']:.2f}%)")
    print(f"  ER   = {metrics['ER']:.4f}")
    print(f"  Iter = {iters}")
    print(f"  Time = {elapsed:.2f} s")
    print(f"{'─'*45}\n")

    # Save result
    if args.save_result:
        np.savez(
            args.save_result,
            X_hat=X_hat,
            MAE=metrics['MAE'],
            RMSE=metrics['RMSE'],
            MAPE=metrics['MAPE'],
            ER=metrics['ER'],
            elapsed=elapsed,
            iterations=iters,
        )
        print(f"Results saved to: {args.save_result}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a traffic data imputation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',          required=True,  help='Path to tensor data file')
    p.add_argument('--algorithm',     default='atsn',
                   choices=list(ALGORITHM_MAP.keys()),
                   help='Algorithm to run')
    p.add_argument('--missing',       default='mixed',
                   choices=['random', 'fiber', 'mixed'],
                   help='Missing pattern type')
    p.add_argument('--fiber_rate',    type=float, default=0.3,
                   help='Fiber missing rate (0–1)')
    p.add_argument('--element_rate',  type=float, default=0.2,
                   help='Random element missing rate (0–1)')
    p.add_argument('--seed',          type=int,   default=1000,
                   help='Random seed for reproducibility')
    p.add_argument('--save_result',   default=None,
                   help='Path to save .npz result file (optional)')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    run(args)

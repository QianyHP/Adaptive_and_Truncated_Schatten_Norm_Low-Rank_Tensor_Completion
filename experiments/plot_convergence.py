"""
plot_convergence.py – Visualise ADMM convergence curves
=========================================================

Reproduces the convergence figures from the paper.

Usage
-----
    python experiments/plot_convergence.py \\
        --data  data/raw/guangzhou_tensor.npy \\
        --algo  atsn \\
        --out   results/figures/convergence_gz.svg
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atsn          import LRTC_ATSN
from atsn.missing  import mixed_missing
from baselines     import lrtc_tspn, lrtc_tnn, halrtc


def plot_convergence(
    mae_history: list,
    rmse_history: list,
    err_history: list,
    title: str,
    out_path: str,
) -> None:
    """Plot MAE, RMSE, and relative error convergence curves."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    iters = range(1, len(mae_history) + 1)

    for ax, values, ylabel, color in zip(
        axes,
        [mae_history, rmse_history, err_history],
        ['MAE', 'RMSE', 'Relative Error'],
        ['royalblue', 'tomato', 'seagreen'],
    ):
        ax.plot(iters, values, color=color, linewidth=2.0, alpha=0.85)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Figure saved to: {out_path}")
    plt.close()


def main():
    import scipy.io

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  required=True)
    parser.add_argument('--out',   default='results/figures/convergence.svg')
    parser.add_argument('--fiber_rate',   type=float, default=0.3)
    parser.add_argument('--element_rate', type=float, default=0.2)
    parser.add_argument('--seed',         type=int,   default=1000)
    args = parser.parse_args()

    # Load
    ext = os.path.splitext(args.data)[1].lower()
    if ext == '.npy':
        X = np.load(args.data)
    elif ext == '.npz':
        d = np.load(args.data)
        X = d['arr_0' if 'arr_0' in d else list(d.keys())[0]]
    elif ext == '.mat':
        d = scipy.io.loadmat(args.data)
        key = 'tensor' if 'tensor' in d else [k for k in d if not k.startswith('_')][0]
        X = d[key]

    np.random.seed(args.seed)
    X_obs = mixed_missing(X, fiber_rate=args.fiber_rate,
                          element_rate=args.element_rate, seed=args.seed)

    result = LRTC_ATSN(X, X_obs, max_iter=200, verbose=True)
    plot_convergence(
        result.mae_history,
        result.rmse_history,
        result.err_history,
        title=f'LRTC-ATSN Convergence (FM={args.fiber_rate}, EM={args.element_rate})',
        out_path=args.out,
    )


if __name__ == '__main__':
    main()

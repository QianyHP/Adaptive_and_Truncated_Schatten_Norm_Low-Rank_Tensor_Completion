"""
compare_baselines.py – Full comparison table (Table 1 in the paper)
====================================================================

Reproduces the main comparison table across datasets, missing patterns,
and all algorithms.

Usage
-----
    python experiments/compare_baselines.py \\
        --guangzhou  data/raw/guangzhou_tensor.npy \\
        --seattle    data/raw/seattle_tensor.npz \\
        --output     results/comparison_table.csv

Results are printed in a formatted table and optionally saved as CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atsn            import LRTC_ATSN
from atsn.missing    import mixed_missing, get_missing_rate
from baselines       import halrtc, lrtc_tnn, lrtc_tspn, bgcp, bpmf, trmf, isvd


ALGORITHMS = {
    'LRTC-ATSN':  lambda c, o: LRTC_ATSN(c, o, verbose=False),
    'LRTC-TSpN':  lambda c, o: lrtc_tspn(c, o, verbose=False),
    'LRTC-TNN':   lambda c, o: lrtc_tnn(c, o, verbose=False),
    'HaLRTC':     lambda c, o: halrtc(c, o, verbose=False),
    'BGCP':       lambda c, o: bgcp(c, o, verbose=False),
    'BPMF':       lambda c, o: bpmf(c, o, verbose=False),
    'TRMF':       lambda c, o: trmf(c, o, verbose=False),
    'ISVD':       lambda c, o: isvd(c, o, verbose=False),
}

SCENARIOS = [
    dict(fiber_rate=0.2, element_rate=0.3, label='EM=0.2/FM=0.3'),
    dict(fiber_rate=0.5, element_rate=0.6, label='EM=0.5/FM=0.6'),
    dict(fiber_rate=0.9, element_rate=0.9, label='EM=0.9/FM=0.9'),
]


def load_tensor(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.npz':
        d = np.load(path)
        return d['arr_0' if 'arr_0' in d else list(d.keys())[0]]
    elif ext == '.mat':
        d = scipy.io.loadmat(path)
        key = 'tensor' if 'tensor' in d else [k for k in d if not k.startswith('_')][0]
        return d[key]
    raise ValueError(f"Unsupported format: {ext}")


def _get_metrics(result) -> dict:
    if hasattr(result, 'final_mae'):
        return {'MAE': result.final_mae, 'RMSE': result.final_rmse,
                'MAPE': result.final_mape, 'ER': result.final_er,
                'time': result.elapsed_sec}
    return {k: result[k] for k in ('MAE', 'RMSE', 'MAPE', 'ER')} | {'time': result['elapsed_sec']}


def run_comparison(datasets: Dict[str, str], seed: int = 1000) -> pd.DataFrame:
    rows = []
    for dataset_name, data_path in datasets.items():
        if not os.path.exists(data_path):
            print(f"[WARN] {data_path} not found, skipping.")
            continue
        X = load_tensor(data_path).astype(float)
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}  shape={X.shape}")
        print(f"{'='*70}")

        for scenario in SCENARIOS:
            np.random.seed(seed)
            X_obs = mixed_missing(X, fiber_rate=scenario['fiber_rate'],
                                  element_rate=scenario['element_rate'], seed=seed)
            actual_rate = get_missing_rate(X_obs)
            label = scenario['label']

            for algo_name, algo_fn in ALGORITHMS.items():
                print(f"  [{algo_name:12s}] {dataset_name} / {label} ... ", end='', flush=True)
                try:
                    t0 = time.perf_counter()
                    result = algo_fn(X, X_obs)
                    elapsed = time.perf_counter() - t0
                    m = _get_metrics(result)
                    status = 'OK'
                except Exception as e:
                    m = {'MAE': float('nan'), 'RMSE': float('nan'),
                         'MAPE': float('nan'), 'ER': float('nan'), 'time': 0.0}
                    elapsed = 0.0
                    status = f'ERROR: {e}'

                print(f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
                      f"MAPE={m['MAPE']:.4f}  [{elapsed:.0f}s]  {status}")

                rows.append({
                    'Dataset':       dataset_name,
                    'Scenario':      label,
                    'Missing Rate':  f"{100*actual_rate:.1f}%",
                    'Algorithm':     algo_name,
                    'MAE':           round(m['MAE'],  4),
                    'RMSE':          round(m['RMSE'], 4),
                    'MAPE':          round(m['MAPE'], 4),
                    'ER':            round(m['ER'],   4),
                    'Time (s)':      round(elapsed,   1),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare all baselines on traffic datasets.")
    parser.add_argument('--guangzhou', default='data/raw/guangzhou_tensor.mat',
                        help='Guangzhou dataset path (.mat, key=tensor)')
    parser.add_argument('--seattle',   default='data/raw/seattle_tensor.npz',
                        help='Seattle dataset path')
    parser.add_argument('--output',    default='results/comparison_table.csv',
                        help='Output CSV path')
    parser.add_argument('--seed',      type=int, default=1000)
    args = parser.parse_args()

    datasets = {
        'Guangzhou': args.guangzhou,
        'Seattle':   args.seattle,
    }

    df = run_comparison(datasets, seed=args.seed)

    # Pretty print
    print("\n\n" + "="*90)
    print("COMPARISON TABLE")
    print("="*90)
    print(df.to_string(index=False))

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nTable saved to: {args.output}")


if __name__ == '__main__':
    main()

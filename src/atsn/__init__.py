"""
ATSN: Adaptive Truncated Schatten Norm for Traffic Data Imputation
==================================================================

A Python package implementing the LRTC-ATSN algorithm and related baselines
for traffic speed/flow data recovery under complex missing patterns.

Reference
---------
[Your Name], "Adaptive Truncated Schatten Norm for Traffic Data Imputation
with Complex Missing Patterns", [Journal/Conference], [Year].

Modules
-------
tensor_ops  : Low-level tensor algebra utilities (fold, unfold, SVD proximal ops)
metrics     : Evaluation metrics (MAE, RMSE, MAPE, ER)
missing     : Synthetic missing-pattern generators
lrtc_atsn   : Proposed LRTC-ATSN algorithm
"""

from .tensor_ops import fold, unfold, gst, truncation_weights, svt
from .metrics import compute_mae, compute_rmse, compute_mape, compute_er
from .missing import (
    random_missing,
    fiber_missing,
    mixed_missing,
)
from .lrtc_atsn import LRTC_ATSN

__version__ = "1.0.0"
__all__ = [
    "fold", "unfold", "gst", "truncation_weights", "svt",
    "compute_mae", "compute_rmse", "compute_mape", "compute_er",
    "random_missing", "fiber_missing", "mixed_missing",
    "LRTC_ATSN",
]

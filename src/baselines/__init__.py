"""
baselines/__init__.py – Baseline algorithms for traffic tensor completion
=========================================================================

Available baselines
-------------------
HaLRTC    High-accuracy Low-Rank Tensor Completion (Liu et al., 2013)
LRTC_TNN  Low-Rank Tensor Completion via Truncated Nuclear Norm (Lu et al., 2019)
LRTC_TSpN Low-Rank Tensor Completion via Truncated Schatten-p Norm
BGCP      Bayesian Gaussian CP Decomposition (Chen & Sun, 2022)
BPMF      Bayesian Probabilistic Matrix Factorisation (Salakhutdinov & Mnih, 2008)
TRMF      Temporal Regularized Matrix Factorization (Yu et al., 2016)
ISVD      Iterative SVD for matrix completion
"""

from .halrtc   import halrtc
from .lrtc_tnn import lrtc_tnn
from .lrtc_tspn import lrtc_tspn
from .bgcp     import bgcp
from .bpmf     import bpmf
from .trmf     import trmf
from .isvd     import isvd

__all__ = ["halrtc", "lrtc_tnn", "lrtc_tspn", "bgcp", "bpmf", "trmf", "isvd"]

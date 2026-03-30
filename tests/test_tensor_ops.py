"""
tests/test_tensor_ops.py – Unit tests for tensor_ops module
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from atsn.tensor_ops import fold, unfold, gst, truncation_weights, svt, update_M_block


class TestFoldUnfold:
    """Round-trip fold/unfold must recover the original tensor."""

    def test_roundtrip_mode0(self):
        X = np.random.randn(4, 5, 6)
        M = unfold(X, 0)
        assert M.shape == (4, 30)
        X_rec = fold(M, X.shape, 0)
        np.testing.assert_allclose(X, X_rec, atol=1e-12)

    def test_roundtrip_mode1(self):
        X = np.random.randn(4, 5, 6)
        M = unfold(X, 1)
        assert M.shape == (5, 24)
        X_rec = fold(M, X.shape, 1)
        np.testing.assert_allclose(X, X_rec, atol=1e-12)

    def test_roundtrip_mode2(self):
        X = np.random.randn(4, 5, 6)
        M = unfold(X, 2)
        assert M.shape == (6, 20)
        X_rec = fold(M, X.shape, 2)
        np.testing.assert_allclose(X, X_rec, atol=1e-12)


class TestGST:
    """GST should return 0 below threshold and identity when w=0."""

    def test_identity_when_w_zero(self):
        assert gst(5.0, 0.0, 0.5) == pytest.approx(5.0)

    def test_zero_below_threshold(self):
        # For very small sigma, output should be 0
        result = gst(0.001, w=1.0, p=0.5)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_above_threshold(self):
        result = gst(100.0, w=0.1, p=0.5)
        assert result > 0.0
        assert result < 100.0

    def test_sign_preservation(self):
        pos = gst(5.0, w=0.01, p=0.5)
        neg = gst(-5.0, w=0.01, p=0.5)
        assert pos >= 0
        assert neg <= 0
        assert abs(pos) == pytest.approx(abs(neg), rel=1e-9)


class TestTruncationWeights:
    """Truncation weights: first r entries 0, rest 1."""

    def test_shape(self):
        M = np.random.randn(10, 8)
        w = truncation_weights(M, theta=0.3)
        assert w.shape == (8,)

    def test_zero_prefix(self):
        M = np.random.randn(10, 8)
        w = truncation_weights(M, theta=0.25)
        r = int(np.ceil(0.25 * 8))   # = 2
        assert np.all(w[:r] == 0.0)
        assert np.all(w[r:] == 1.0)

    def test_theta_zero(self):
        M = np.random.randn(6, 6)
        w = truncation_weights(M, theta=0.0)
        assert np.all(w == 1.0)


class TestSVT:
    """SVT output should be a soft-thresholded reconstruction."""

    def test_reduces_rank(self):
        u, _, vt = np.linalg.svd(np.random.randn(10, 8), full_matrices=False)
        s = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
        M = u[:, :5] @ np.diag(s) @ vt[:5, :]
        result = svt(M, tau=2.0)
        _, s2, _ = np.linalg.svd(result, full_matrices=False)
        # Singular values above tau=2 should be reduced, below zeroed
        assert s2[0] < s[0]

    def test_tau_zero(self):
        M = np.random.randn(5, 5)
        np.testing.assert_allclose(svt(M, tau=0.0), M, atol=1e-12)


class TestUpdateMBlock:
    """update_M_block should return a matrix of same shape with reduced singular values."""

    def test_output_shape(self):
        M = np.random.randn(12, 10)
        result = update_M_block(M, alpha_i=1.0, beta=1.0, p=0.5, theta=0.1)
        assert result.shape == M.shape

    def test_shrinkage(self):
        M = np.random.randn(8, 6) * 10
        result = update_M_block(M, alpha_i=1.0, beta=1.0, p=0.5, theta=0.0)
        _, s_in,  _ = np.linalg.svd(M,      full_matrices=False)
        _, s_out, _ = np.linalg.svd(result, full_matrices=False)
        # Singular values should not increase
        assert np.all(s_out <= s_in + 1e-10)

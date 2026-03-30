"""
tests/test_missing.py – Unit tests for missing-pattern generators
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from atsn.missing import random_missing, fiber_missing, mixed_missing, get_missing_rate


@pytest.fixture
def tensor():
    rng = np.random.default_rng(42)
    return rng.uniform(1.0, 10.0, size=(10, 8, 6))


class TestRandomMissing:
    def test_approximate_rate(self, tensor):
        obs = random_missing(tensor, rate=0.3, seed=0)
        actual = get_missing_rate(obs)
        assert abs(actual - 0.3) < 0.05

    def test_observed_unchanged(self, tensor):
        obs = random_missing(tensor, rate=0.3, seed=0)
        mask = obs != 0
        np.testing.assert_array_equal(obs[mask], tensor[mask])

    def test_invalid_rate(self, tensor):
        with pytest.raises(ValueError):
            random_missing(tensor, rate=1.5)

    def test_reproducibility(self, tensor):
        a = random_missing(tensor, rate=0.2, seed=99)
        b = random_missing(tensor, rate=0.2, seed=99)
        np.testing.assert_array_equal(a, b)


class TestFiberMissing:
    def test_output_shape(self, tensor):
        obs = fiber_missing(tensor, rate=0.4, mode=0, seed=0)
        assert obs.shape == tensor.shape

    def test_mode_invalid(self, tensor):
        with pytest.raises(ValueError):
            fiber_missing(tensor, rate=0.3, mode=5)

    def test_nonzero_elements_unchanged(self, tensor):
        obs = fiber_missing(tensor, rate=0.2, mode=1, seed=7)
        mask = obs != 0
        np.testing.assert_array_equal(obs[mask], tensor[mask])


class TestMixedMissing:
    def test_output_shape(self, tensor):
        obs = mixed_missing(tensor, fiber_rate=0.2, element_rate=0.1, seed=0)
        assert obs.shape == tensor.shape

    def test_missing_rate_positive(self, tensor):
        obs = mixed_missing(tensor, fiber_rate=0.2, element_rate=0.1, seed=0)
        assert get_missing_rate(obs) > 0.0

    def test_reproducibility(self, tensor):
        a = mixed_missing(tensor, fiber_rate=0.3, element_rate=0.2, seed=42)
        b = mixed_missing(tensor, fiber_rate=0.3, element_rate=0.2, seed=42)
        np.testing.assert_array_equal(a, b)


class TestGetMissingRate:
    def test_all_observed(self):
        X = np.ones((5, 5, 5))
        assert get_missing_rate(X) == pytest.approx(0.0)

    def test_all_missing(self):
        X = np.zeros((5, 5, 5))
        assert get_missing_rate(X) == pytest.approx(1.0)

    def test_half_missing(self):
        X = np.array([0.0, 1.0, 0.0, 1.0]).reshape(2, 2, 1)
        assert get_missing_rate(X) == pytest.approx(0.5)

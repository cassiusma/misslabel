"""
tests/test_noise.py
-------------------
Tests for misslabel.noise module.

Properties tested:
- t=0 produces no corruption
- output shape and type preservation
- string labels supported and returned
- flip_mask correctness
- reproducibility via random_state
- different random_state gives different results
- empirical flip rate converges to theoretical at large n
- pi supplied explicitly vs estimated from y
- minority class higher flip rate than majority class
"""

import numpy as np
import pytest
from misslabel.noise import inject, InjectionResult
from misslabel.matrix import estimate_pi, f81_transition_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def y_int():
    """Balanced integer label vector, 3 classes, 300 samples."""
    return np.repeat([0, 1, 2], 100)


@pytest.fixture
def y_skewed():
    """Skewed integer label vector: class 0 dominant."""
    return np.array([0] * 500 + [1] * 300 + [2] * 200)


@pytest.fixture
def y_str():
    """String label vector."""
    return np.array(["cat"] * 100 + ["dog"] * 100 + ["bird"] * 100)


# ---------------------------------------------------------------------------
# Return type and shape
# ---------------------------------------------------------------------------

class TestReturnType:

    def test_returns_injection_result(self, y_int):
        result = inject(y_int, t=0.5, random_state=0)
        assert isinstance(result, InjectionResult)

    def test_y_noisy_shape(self, y_int):
        result = inject(y_int, t=0.5, random_state=0)
        assert result.y_noisy.shape == y_int.shape

    def test_flip_mask_shape(self, y_int):
        result = inject(y_int, t=0.5, random_state=0)
        assert result.flip_mask.shape == y_int.shape

    def test_flip_mask_dtype_bool(self, y_int):
        result = inject(y_int, t=0.5, random_state=0)
        assert result.flip_mask.dtype == bool

    def test_y_noisy_values_in_label_set(self, y_int):
        result = inject(y_int, t=0.5, random_state=0)
        assert set(result.y_noisy).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# t = 0 edge case
# ---------------------------------------------------------------------------

class TestTZero:

    def test_no_corruption_at_t_zero(self, y_int):
        result = inject(y_int, t=0.0, random_state=0)
        assert np.array_equal(result.y_noisy, y_int)

    def test_flip_mask_all_false_at_t_zero(self, y_int):
        result = inject(y_int, t=0.0, random_state=0)
        assert result.flip_mask.sum() == 0

    def test_no_corruption_string_labels_at_t_zero(self, y_str):
        result = inject(y_str, t=0.0, random_state=0)
        assert np.array_equal(result.y_noisy, y_str)


# ---------------------------------------------------------------------------
# String label support
# ---------------------------------------------------------------------------

class TestStringLabels:

    def test_output_is_strings(self, y_str):
        result = inject(y_str, t=0.5, random_state=0)
        assert result.y_noisy.dtype.kind in ("U", "O")  # unicode or object

    def test_output_values_in_original_classes(self, y_str):
        result = inject(y_str, t=0.5, random_state=0)
        assert set(result.y_noisy).issubset({"cat", "dog", "bird"})

    def test_flip_mask_consistent_with_string_comparison(self, y_str):
        result = inject(y_str, t=1.0, random_state=0)
        expected_mask = result.y_noisy != y_str
        assert np.array_equal(result.flip_mask, expected_mask)


# ---------------------------------------------------------------------------
# flip_mask correctness
# ---------------------------------------------------------------------------

class TestFlipMask:

    def test_flip_mask_consistent_with_y_noisy(self, y_int):
        result = inject(y_int, t=1.0, random_state=0)
        expected = result.y_noisy != y_int
        assert np.array_equal(result.flip_mask, expected)

    def test_flip_mask_false_where_label_unchanged(self, y_int):
        result = inject(y_int, t=1.0, random_state=0)
        unchanged = ~result.flip_mask
        assert np.array_equal(result.y_noisy[unchanged], y_int[unchanged])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_result(self, y_int):
        r1 = inject(y_int, t=1.0, random_state=42)
        r2 = inject(y_int, t=1.0, random_state=42)
        assert np.array_equal(r1.y_noisy, r2.y_noisy)

    def test_different_seed_different_result(self, y_int):
        r1 = inject(y_int, t=1.0, random_state=0)
        r2 = inject(y_int, t=1.0, random_state=1)
        # Extremely unlikely to be identical for n=300, t=1
        assert not np.array_equal(r1.y_noisy, r2.y_noisy)

    def test_none_seed_runs_without_error(self, y_int):
        result = inject(y_int, t=0.5, random_state=None)
        assert result.y_noisy.shape == y_int.shape


# ---------------------------------------------------------------------------
# Empirical flip rate vs theoretical
# ---------------------------------------------------------------------------

class TestEmpiricalFlipRate:

    def test_empirical_flip_rate_close_to_theoretical(self):
        """
        At large n, the empirical per-class flip rate should converge to
        1 - P(t)[c, c] = (1 - exp(-mu*t)) * (1 - pi_c).
        We use n=10000 and a loose tolerance (3 standard deviations).
        """
        rng = np.random.default_rng(0)
        n = 10_000
        pi = np.array([0.5, 0.3, 0.2])
        k = len(pi)
        # Draw true labels from pi
        y = rng.choice(k, size=n, p=pi)
        t = 0.5

        result = inject(y, t=t, pi=pi, random_state=1)
        P = f81_transition_matrix(pi, t)

        for c in range(k):
            theoretical = 1.0 - P[c, c]
            n_c = (y == c).sum()
            empirical = result.flip_mask[y == c].mean()
            # Standard deviation of empirical rate ~ sqrt(p(1-p)/n_c)
            std = np.sqrt(theoretical * (1 - theoretical) / n_c)
            assert abs(empirical - theoretical) < 3 * std, (
                f"Class {c}: empirical={empirical:.4f}, "
                f"theoretical={theoretical:.4f}, 3*std={3*std:.4f}"
            )

    def test_minority_class_higher_flip_rate(self, y_skewed):
        """
        Minority classes should experience higher flip rates than majority
        classes at the same t, as predicted by the model.
        """
        pi = estimate_pi(y_skewed, k=3)
        result = inject(y_skewed, t=1.0, pi=pi, random_state=0)

        flip_rates = np.array([
            result.flip_mask[y_skewed == c].mean()
            for c in range(3)
        ])
        # Class 0 is majority (pi_0 ~ 0.5), class 2 is minority (pi_2 ~ 0.2)
        # Expected: flip_rate[2] > flip_rate[0]
        assert flip_rates[2] > flip_rates[0]


# ---------------------------------------------------------------------------
# Explicit pi vs estimated pi
# ---------------------------------------------------------------------------

class TestPiHandling:

    def test_explicit_pi_accepted(self, y_int):
        pi = np.array([1/3, 1/3, 1/3])
        result = inject(y_int, t=0.5, pi=pi, random_state=0)
        assert result.y_noisy.shape == y_int.shape

    def test_explicit_vs_estimated_pi_close_for_large_n(self):
        """
        For large balanced y, explicit uniform pi and estimated pi should
        produce similar flip rates.
        """
        rng = np.random.default_rng(0)
        y = rng.choice(3, size=10_000, p=[1/3, 1/3, 1/3])
        pi_explicit = np.array([1/3, 1/3, 1/3])

        r_explicit = inject(y, t=0.5, pi=pi_explicit, random_state=42)
        r_estimated = inject(y, t=0.5, pi=None, random_state=42)

        # Flip rates should be very close
        assert abs(r_explicit.flip_mask.mean() - r_estimated.flip_mask.mean()) < 0.02

    def test_invalid_t_raises(self, y_int):
        with pytest.raises(ValueError):
            inject(y_int, t=-1.0)
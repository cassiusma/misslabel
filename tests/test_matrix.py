"""
tests/test_matrix.py
--------------------
Tests for misslabel.matrix module.

Mathematical properties tested:
- estimate_pi: correct frequencies, sums to 1, Laplace smoothing
- normalise_rate_matrix: avg rate = 1, shape preserved
- f81_rate_matrix: rows sum to zero, correct normalisation, off-diagonal = mu*pi_j
- f81_transition_matrix: row-stochastic, boundary conditions (t=0, t->inf),
  monotone decay of diagonal, stationary distribution is fixed point
"""

import numpy as np
import pytest
from misslabel.matrix import (
    estimate_pi,
    f81_rate_matrix,
    f81_transition_matrix,
    normalise_rate_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_pi():
    """Uniform distribution over 3 classes."""
    return np.array([1/3, 1/3, 1/3])


@pytest.fixture
def skewed_pi():
    """Non-uniform distribution — the main case of interest."""
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def labels_skewed():
    """Label vector consistent with skewed_pi."""
    return np.array([0, 0, 0, 1, 1, 2])


# ---------------------------------------------------------------------------
# estimate_pi
# ---------------------------------------------------------------------------

class TestEstimatePi:

    def test_sums_to_one(self, labels_skewed):
        pi = estimate_pi(labels_skewed, k=3)
        assert np.isclose(pi.sum(), 1.0)

    def test_correct_shape(self, labels_skewed):
        pi = estimate_pi(labels_skewed, k=3)
        assert pi.shape == (3,)

    def test_approximate_frequencies(self, labels_skewed):
        # With Laplace smoothing the values are close but not exact
        pi = estimate_pi(labels_skewed, k=3)
        assert pi[0] > pi[1] > pi[2]  # ordering preserved

    def test_laplace_smoothing_no_zero(self):
        # Class 2 absent from y — should still get non-zero frequency
        y = np.array([0, 0, 1, 1])
        pi = estimate_pi(y, k=3)
        assert pi[2] > 0.0

    def test_invalid_label_out_of_range(self):
        y = np.array([0, 1, 3])  # 3 is out of range for k=3
        with pytest.raises(ValueError):
            estimate_pi(y, k=3)

    def test_invalid_label_negative(self):
        y = np.array([-1, 0, 1])
        with pytest.raises(ValueError):
            estimate_pi(y, k=3)

    def test_invalid_label_float(self):
        y = np.array([0.0, 1.0, 2.0])
        with pytest.raises(TypeError):
            estimate_pi(y, k=3)


# ---------------------------------------------------------------------------
# normalise_rate_matrix
# ---------------------------------------------------------------------------

class TestNormaliseRateMatrix:

    def test_avg_rate_equals_one(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)  # already normalised
        # Build unnormalised version by scaling up, then re-normalise
        Q_unnorm = Q * 3.7
        Q_renorm = normalise_rate_matrix(Q_unnorm, skewed_pi)
        avg_rate = -np.dot(skewed_pi, np.diag(Q_renorm))
        assert np.isclose(avg_rate, 1.0)

    def test_shape_preserved(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        Q_norm = normalise_rate_matrix(Q * 2.0, skewed_pi)
        assert Q_norm.shape == Q.shape

    def test_degenerate_pi_raises(self):
        # pi puts all mass on one class
        pi = np.array([1.0 - 3e-11, 1e-11, 2e-11])
        pi /= pi.sum()
        k = len(pi)
        Q = np.outer(np.ones(k), pi)
        np.fill_diagonal(Q, -(1.0 - pi))
        with pytest.raises(ValueError):
            normalise_rate_matrix(Q, pi)


# ---------------------------------------------------------------------------
# f81_rate_matrix
# ---------------------------------------------------------------------------

class TestF81RateMatrix:

    def test_rows_sum_to_zero(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        assert np.allclose(Q.sum(axis=1), 0.0)

    def test_normalised(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        avg_rate = -np.dot(skewed_pi, np.diag(Q))
        assert np.isclose(avg_rate, 1.0)

    def test_off_diagonal_proportional_to_pi(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        k = len(skewed_pi)
        # For fixed i, off-diagonal entries Q[i,j] should be proportional to pi[j]
        for i in range(k):
            off_diag_indices = [j for j in range(k) if j != i]
            ratios = Q[i, off_diag_indices] / skewed_pi[off_diag_indices]
            assert np.allclose(ratios, ratios[0]), \
                f"Off-diagonal row {i} not proportional to pi"

    def test_diagonal_negative(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        assert np.all(np.diag(Q) < 0)

    def test_shape(self, skewed_pi):
        Q = f81_rate_matrix(skewed_pi)
        k = len(skewed_pi)
        assert Q.shape == (k, k)

    def test_uniform_pi(self, uniform_pi):
        # Uniform pi is a valid special case
        Q = f81_rate_matrix(uniform_pi)
        assert np.allclose(Q.sum(axis=1), 0.0)
        avg_rate = -np.dot(uniform_pi, np.diag(Q))
        assert np.isclose(avg_rate, 1.0)

    def test_invalid_pi_negative(self):
        with pytest.raises(ValueError):
            f81_rate_matrix(np.array([-0.1, 0.6, 0.5]))

    def test_invalid_pi_does_not_sum_to_one(self):
        with pytest.raises(ValueError):
            f81_rate_matrix(np.array([0.4, 0.4, 0.4]))


# ---------------------------------------------------------------------------
# f81_transition_matrix
# ---------------------------------------------------------------------------

class TestF81TransitionMatrix:

    def test_row_stochastic(self, skewed_pi):
        for t in [0.0, 0.1, 0.5, 1.0, 5.0]:
            P = f81_transition_matrix(skewed_pi, t)
            assert np.allclose(P.sum(axis=1), 1.0), f"Not row-stochastic at t={t}"

    def test_non_negative(self, skewed_pi):
        for t in [0.0, 0.1, 1.0, 10.0]:
            P = f81_transition_matrix(skewed_pi, t)
            assert np.all(P >= -1e-12), f"Negative entries at t={t}"

    def test_identity_at_t_zero(self, skewed_pi):
        P = f81_transition_matrix(skewed_pi, t=0.0)
        assert np.allclose(P, np.eye(len(skewed_pi)))

    def test_converges_to_pi_at_large_t(self, skewed_pi):
        # At t=1000 every row should equal pi
        P = f81_transition_matrix(skewed_pi, t=1000.0)
        expected = np.outer(np.ones(len(skewed_pi)), skewed_pi)
        assert np.allclose(P, expected, atol=1e-6)

    def test_stationary_distribution_is_fixed_point(self, skewed_pi):
        # pi @ P(t) = pi for all t (pi is left eigenvector with eigenvalue 1)
        for t in [0.1, 1.0, 5.0]:
            P = f81_transition_matrix(skewed_pi, t)
            assert np.allclose(skewed_pi @ P, skewed_pi), \
                f"pi not fixed point at t={t}"

    def test_diagonal_monotone_decreasing_in_t(self, skewed_pi):
        # P(t)_ii should decrease as t increases
        ts = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        diagonals = [np.diag(f81_transition_matrix(skewed_pi, t)) for t in ts]
        for i in range(len(ts) - 1):
            assert np.all(diagonals[i] >= diagonals[i+1] - 1e-12), \
                f"Diagonal not monotone between t={ts[i]} and t={ts[i+1]}"

    def test_diagonal_lower_bounded_by_pi(self, skewed_pi):
        # P(t)_ii >= pi_i for all t (never goes below stationary)
        for t in [0.1, 1.0, 5.0, 100.0]:
            P = f81_transition_matrix(skewed_pi, t)
            assert np.all(np.diag(P) >= skewed_pi - 1e-9), \
                f"Diagonal below pi at t={t}"

    def test_shape(self, skewed_pi):
        P = f81_transition_matrix(skewed_pi, t=1.0)
        k = len(skewed_pi)
        assert P.shape == (k, k)

    def test_negative_t_raises(self, skewed_pi):
        with pytest.raises(ValueError):
            f81_transition_matrix(skewed_pi, t=-0.1)

    def test_chapman_kolmogorov(self, skewed_pi):
        # P(s+t) = P(s) @ P(t)
        s, t = 0.3, 0.7
        P_s = f81_transition_matrix(skewed_pi, s)
        P_t = f81_transition_matrix(skewed_pi, t)
        P_st = f81_transition_matrix(skewed_pi, s + t)
        assert np.allclose(P_s @ P_t, P_st, atol=1e-10), \
            "Chapman-Kolmogorov equation violated"
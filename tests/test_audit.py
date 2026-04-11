"""
tests/test_audit.py
-------------------
Tests for misslabel.audit module.

Properties tested:
- empirical_T: shape, row-stochastic, correct counts, identity at t=0
- empirical_T: string labels supported
- summary: keys present, types correct, per_class keys match label vocabulary
- summary: theoretical quantities present iff t is provided
- summary: max_T_error converges to zero at large n
- summary: flip_rate consistent with flip_mask
- summary: string labels supported
"""

import numpy as np
import pytest
from misslabel.noise import inject
from misslabel.audit import empirical_T, summary
from misslabel.matrix import estimate_pi, f81_transition_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_int():
    """Paired true/noisy integer labels from a controlled injection."""
    rng = np.random.default_rng(0)
    pi = np.array([0.5, 0.3, 0.2])
    y_true = rng.choice(3, size=1000, p=pi)
    result = inject(y_true, t=0.5, pi=pi, random_state=1)
    return y_true, result.y_noisy, pi


@pytest.fixture
def paired_str():
    """Paired true/noisy string labels."""
    y_true = np.array(["cat"] * 200 + ["dog"] * 150 + ["bird"] * 100)
    result = inject(y_true, t=0.5, random_state=0)
    return y_true, result.y_noisy


@pytest.fixture
def paired_no_noise():
    """Paired labels with t=0: no corruption."""
    y_true = np.repeat([0, 1, 2], 100)
    result = inject(y_true, t=0.0, random_state=0)
    return y_true, result.y_noisy


# ---------------------------------------------------------------------------
# empirical_T
# ---------------------------------------------------------------------------

class TestEmpiricalT:

    def test_shape(self, paired_int):
        y_true, y_noisy, _ = paired_int
        T_hat = empirical_T(y_true, y_noisy)
        assert T_hat.shape == (3, 3)

    def test_row_stochastic(self, paired_int):
        y_true, y_noisy, _ = paired_int
        T_hat = empirical_T(y_true, y_noisy)
        assert np.allclose(T_hat.sum(axis=1), 1.0)

    def test_non_negative(self, paired_int):
        y_true, y_noisy, _ = paired_int
        T_hat = empirical_T(y_true, y_noisy)
        assert np.all(T_hat >= 0.0)

    def test_identity_at_t_zero(self, paired_no_noise):
        y_true, y_noisy = paired_no_noise
        T_hat = empirical_T(y_true, y_noisy)
        assert np.allclose(T_hat, np.eye(3), atol=1e-10)

    def test_string_labels_shape(self, paired_str):
        y_true, y_noisy = paired_str
        T_hat = empirical_T(y_true, y_noisy)
        assert T_hat.shape == (3, 3)

    def test_string_labels_row_stochastic(self, paired_str):
        y_true, y_noisy = paired_str
        T_hat = empirical_T(y_true, y_noisy)
        assert np.allclose(T_hat.sum(axis=1), 1.0)

    def test_mismatched_shapes_raises(self):
        y_true = np.array([0, 1, 2])
        y_noisy = np.array([0, 1])
        with pytest.raises(ValueError):
            empirical_T(y_true, y_noisy)

    def test_diagonal_dominant_at_low_t(self):
        """At small t, T_hat should be diagonally dominant."""
        y_true = np.repeat([0, 1, 2], 500)
        result = inject(y_true, t=0.1, random_state=0)
        T_hat = empirical_T(y_true, result.y_noisy)
        assert np.all(np.diag(T_hat) > 0.5)

    def test_converges_to_P_t_at_large_n(self):
        """
        At large n, T_hat should be close to P(t).
        We use n=20000 and a loose tolerance.
        """
        rng = np.random.default_rng(0)
        pi = np.array([0.5, 0.3, 0.2])
        t = 0.5
        y_true = rng.choice(3, size=20_000, p=pi)
        result = inject(y_true, t=t, pi=pi, random_state=1)
        T_hat = empirical_T(y_true, result.y_noisy)
        P_t = f81_transition_matrix(pi, t)
        assert np.allclose(T_hat, P_t, atol=0.05)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:

    def test_required_keys_present(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        for key in ["n", "k", "n_flipped", "flip_rate", "per_class",
                    "T_hat", "P_t", "max_T_error"]:
            assert key in report, f"Missing key: {key}"

    def test_n_correct(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert report["n"] == len(y_true)

    def test_k_correct(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert report["k"] == 3

    def test_flip_rate_consistent(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        expected = report["n_flipped"] / report["n"]
        assert np.isclose(report["flip_rate"], expected)

    def test_per_class_keys_are_integer_labels(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert set(report["per_class"].keys()) == {0, 1, 2}

    def test_per_class_keys_are_string_labels(self, paired_str):
        y_true, y_noisy = paired_str
        report = summary(y_true, y_noisy)
        assert set(report["per_class"].keys()) == {"cat", "dog", "bird"}

    def test_per_class_n_sums_to_n(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        total = sum(v["n"] for v in report["per_class"].values())
        assert total == report["n"]

    def test_per_class_n_flipped_sums_to_n_flipped(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        total = sum(v["n_flipped"] for v in report["per_class"].values())
        assert total == report["n_flipped"]

    def test_theoretical_present_when_t_given(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert report["P_t"] is not None
        assert report["max_T_error"] is not None
        for v in report["per_class"].values():
            assert v["flip_rate_theoretical"] is not None

    def test_theoretical_absent_when_t_not_given(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi)
        assert report["P_t"] is None
        assert report["max_T_error"] is None
        for v in report["per_class"].values():
            assert v["flip_rate_theoretical"] is None

    def test_max_T_error_converges_at_large_n(self):
        """max|T_hat - P(t)| should be small at large n."""
        rng = np.random.default_rng(0)
        pi = np.array([0.5, 0.3, 0.2])
        t = 0.5
        y_true = rng.choice(3, size=20_000, p=pi)
        result = inject(y_true, t=t, pi=pi, random_state=1)
        report = summary(y_true, result.y_noisy, pi=pi, t=t)
        assert report["max_T_error"] < 0.05

    def test_zero_flips_at_t_zero(self, paired_no_noise):
        y_true, y_noisy = paired_no_noise
        report = summary(y_true, y_noisy)
        assert report["n_flipped"] == 0
        assert report["flip_rate"] == 0.0

    def test_T_hat_shape(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert report["T_hat"].shape == (3, 3)

    def test_P_t_row_stochastic(self, paired_int):
        y_true, y_noisy, pi = paired_int
        report = summary(y_true, y_noisy, pi=pi, t=0.5)
        assert np.allclose(report["P_t"].sum(axis=1), 1.0)
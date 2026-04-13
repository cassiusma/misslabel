"""
misslabel.matrix
----------------
F81 substitution model mathematics.

The F81 model (Felsenstein 1981) is a continuous-time Markov model on k states
with stationary distribution pi. It is the simplest model allowing non-uniform
equilibrium frequencies.

Rate matrix (unnormalised)
--------------------------
    Q_ij = pi_j          for i != j
    Q_ii = -1 + pi_i   diagonal (ensures rows sum to zero)

Normalisation
-------------
Any rate matrix Q is normalised so that the expected number of substitutions
per unit time equals one:

    mu = 1 / (sum_i pi_i * |Q_ii|)
    Q_normalised = mu * Q

For F81 this simplifies to:

    mu = 1 / (1 - sum_i pi_i^2)

where the denominator is the heterozygosity (probability that two
independently drawn labels differ under pi).

Transition matrix (closed form for F81)
----------------------------------------
    P(t) = exp(-mu*t) * I + (1 - exp(-mu*t)) * outer(1, pi)

Eigenstructure of Q:
  - eigenvalue 0 with eigenvector pi (stationary distribution)
  - eigenvalue -mu with multiplicity k-1
giving the matrix exponential directly without diagonalisation.

    P(t)_ij -> delta_ij   as t -> 0   (identity matrix)
    P(t)_ij -> pi_j       as t -> inf (tends to the equilibrium distribution)
"""

import numpy as np
from misslabel._validators import validate_pi, validate_labels, validate_t


def estimate_pi(y: np.ndarray, k: int) -> np.ndarray:
    """
    Estimate stationary distribution from empirical label frequencies.

    Parameters
    ----------
    y : ndarray of shape (n,), dtype int
        True label vector with values in {0, ..., k-1}.
    k : int
        Number of classes.

    Returns
    -------
    pi : ndarray of shape (k,), dtype float64
        Empirical label frequencies. Sums to 1.

    Notes
    -----
    Uses additive (Laplace) smoothing of 1e-10 to avoid zero frequencies,
    which would make Q singular
    """
    validate_labels(y, k)
    counts = np.bincount(y, minlength=k).astype(np.float64)
    counts += 1e-10  # avoid zero frequencies
    pi = counts / counts.sum()
    return pi


def normalise_rate_matrix(Q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Normalise a rate matrix Q to one expected substitution per unit time.

    The normalisation scalar is:

        mu = 1 / (sum_i pi_i * |Q_ii|)

    and the normalised matrix is mu * Q.


    Parameters
    ----------
    Q : ndarray of shape (k, k)
        Unnormalised rate matrix. Must have rows summing to zero and
        non-positive diagonal.
    pi : ndarray of shape (k,)
        Stationary distribution with respect to which normalisation is defined.

    Returns
    -------
    Q_normalised : ndarray of shape (k, k), dtype float64
        Rate matrix satisfying sum_i pi_i * |Q_ii| = 1.

    Raises
    ------
    ValueError
        If the average rate is effectively zero (degenerate input).
    """
    validate_pi(pi)
    avg_rate = -np.dot(pi, np.diag(Q))  # sum_i pi_i * |Q_ii|
    if avg_rate < 1e-10:
        raise ValueError(
            "Average substitution rate is effectively zero. "
            "Check that pi is not degenerate and Q is not the zero matrix."
        )
    return Q / avg_rate


def f81_rate_matrix(pi: np.ndarray) -> np.ndarray:
    """
    Construct the normalised F81 instantaneous rate matrix Q.

    Builds the unnormalised F81 matrix and normalises it via
    normalise_rate_matrix.

    Parameters
    ----------
    pi : ndarray of shape (k,)
        Stationary distribution. Must be non-negative and sum to 1.

    Returns
    -------
    Q : ndarray of shape (k, k), dtype float64
        Normalised rate matrix satisfying:
          - Q_ij = mu * pi_j  for i != j
          - rows sum to zero
          - sum_i pi_i * |Q_ii| = 1
    """
    validate_pi(pi)
    k = len(pi)

    # Unnormalised F81: off-diagonal Q_ij = pi_j, diagonal Q_ii = -1 + pi_i
    Q = np.outer(np.ones(k), pi)
    np.fill_diagonal(Q, -1.0 + pi)

    return normalise_rate_matrix(Q, pi)


def f81_transition_matrix(pi: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the F81 transition matrix P(t) = exp(Q*t) in closed form.

    Parameters
    ----------
    pi : ndarray of shape (k,)
        Stationary distribution.
    t : float
        Time parameter (non-negative). Controls the amount of label noise:
          t -> 0   : P(t) -> I         (no corruption)
          t -> inf : P(t)_ij -> pi_j   (labels fully randomised under pi)

    Returns
    -------
    P : ndarray of shape (k, k), dtype float64
        Row-stochastic transition matrix.
        P_ij = probability that label i becomes label j after time t.
    """
    validate_pi(pi)
    validate_t(t)
    k = len(pi)

    heterozygosity = 1.0 - np.dot(pi, pi)
    if heterozygosity < 1e-12:
        raise ValueError(
            "pi is effectively degenerate (one class has frequency ~1). "
            "Normalisation is undefined."
        )
    mu = 1.0 / heterozygosity
    decay = np.exp(-mu * t)

    # P(t) = decay * I + (1 - decay) * outer(1, pi)
    P = (1.0 - decay) * np.outer(np.ones(k), pi)
    P += decay * np.eye(k)

    return P
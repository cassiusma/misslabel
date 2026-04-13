"""
misslabel._validators
---------------------
Internal input validation. Not part of the public API.
"""

import numpy as np


def validate_labels(y: np.ndarray, k: int) -> None:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}.")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"y must be an integer array, got dtype {y.dtype}.")
    if y.min() < 0 or y.max() >= k:
        raise ValueError(
            f"Labels must be in {{0, ..., {k-1}}}, "
            f"got range [{y.min()}, {y.max()}]."
        )


def validate_pi(pi: np.ndarray) -> None:
    pi = np.asarray(pi)
    if pi.ndim != 1:
        raise ValueError(f"pi must be 1-dimensional, got shape {pi.shape}.")
    if np.any(pi < 0):
        raise ValueError("pi must be non-negative.")
    if not np.isclose(pi.sum(), 1.0, atol=1e-6):
        raise ValueError(f"pi must sum to 1, got sum={pi.sum():.6f}.")


def validate_t(t: float) -> None:
    if not np.isscalar(t):
        raise TypeError(f"t must be a scalar, got {type(t)}.")
    if t < 0:
        raise ValueError(f"t must be non-negative, got t={t}.")


def validate_transition_matrix(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square, got shape {P.shape}.")
    if np.any(P < -1e-9):
        raise ValueError("P contains negative entries.")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"P is not row-stochastic. Row sums: {row_sums}.")
"""
misslabel.audit
---------------
Diagnostics for label noise injection experiments.

Given paired vectors of true and noisy labels, this module estimates the
empirical transition matrix T_hat and compares it to the theoretical F81
transition matrix P(t).

The empirical transition matrix is defined as:

    T_hat[i, j] = #{samples with y_true=i and y_noisy=j} / #{samples with y_true=i}

This is the maximum likelihood estimate of the row-conditional distribution
P(y_noisy | y_true) from finite samples. It converges to P(t) as n -> inf.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from misslabel._validators import validate_t
from misslabel.matrix import estimate_pi, f81_transition_matrix


def empirical_T(
    y_true: np.ndarray,
    y_noisy: np.ndarray,
    k: int = None,
) -> np.ndarray:
    """
    Estimate the empirical noise transition matrix from paired label vectors.

    T_hat[i, j] = P(y_noisy = j | y_true = i), estimated by frequency counts.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True label vector. May be integers or strings.
    y_noisy : array-like of shape (n,)
        Corrupted label vector. Must contain the same label vocabulary as
        y_true.
    k : int, optional
        Number of classes. If None, inferred from the union of y_true and
        y_noisy. Provide explicitly if some classes may be absent from the
        sample.

    Returns
    -------
    T_hat : ndarray of shape (k, k), dtype float64
        Row-stochastic empirical transition matrix.
        Rows with zero counts are set to the uniform distribution 1/k
        (this should not occur in practice for reasonable n).

    Notes
    -----
    Uses the same LabelEncoder convention as inject(): classes are ordered
    lexicographically. If y_true contains integers, lexicographic order
    coincides with numeric order.
    """
    y_true = np.asarray(y_true)
    y_noisy = np.asarray(y_noisy)

    if y_true.shape != y_noisy.shape:
        raise ValueError(
            f"y_true and y_noisy must have the same shape, "
            f"got {y_true.shape} and {y_noisy.shape}."
        )

    # Encode to integers using the union of both label sets
    le = LabelEncoder()
    le.fit(np.concatenate([y_true, y_noisy]))
    y_true_int = le.transform(y_true)
    y_noisy_int = le.transform(y_noisy)
    k = k or len(le.classes_)

    T_hat = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        mask = y_true_int == i
        n_i = mask.sum()
        if n_i == 0:
            T_hat[i, :] = 1.0 / k  # fallback: uniform
        else:
            counts = np.bincount(y_noisy_int[mask], minlength=k)
            T_hat[i, :] = counts / n_i

    return T_hat


def summary(
    y_true: np.ndarray,
    y_noisy: np.ndarray,
    pi: np.ndarray = None,
    t: float = None,
) -> dict:
    """
    Produce a diagnostic summary comparing empirical and theoretical noise.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True label vector.
    y_noisy : array-like of shape (n,)
        Corrupted label vector.
    pi : ndarray of shape (k,), optional
        Stationary distribution used during injection. If None, estimated
        from y_true.
    t : float, optional
        Time parameter used during injection. If provided, theoretical
        flip rates are included in the summary. If None, only empirical
        quantities are reported.

    Returns
    -------
    report : dict with the following keys:

        n : int
            Total number of samples.
        k : int
            Number of classes.
        n_flipped : int
            Total number of labels that changed.
        flip_rate : float
            Overall empirical flip rate = n_flipped / n.
        per_class : dict mapping class label -> dict with keys:
            n           : int   — number of samples in this class
            n_flipped   : int   — number flipped
            flip_rate   : float — empirical flip rate for this class
            flip_rate_theoretical : float or None
                Expected flip rate 1 - P(t)[c,c] under F81.
                None if t was not provided.
        T_hat : ndarray of shape (k, k)
            Empirical transition matrix.
        P_t : ndarray of shape (k, k) or None
            Theoretical F81 transition matrix. None if t not provided.
        max_T_error : float or None
            max|T_hat - P(t)| (elementwise). None if t not provided.
    """
    y_true = np.asarray(y_true)
    y_noisy = np.asarray(y_noisy)

    if t is not None:
        validate_t(t)

    # Encode to integers
    le = LabelEncoder()
    le.fit(np.concatenate([y_true, y_noisy]))
    y_true_int = le.transform(y_true)
    y_noisy_int = le.transform(y_noisy)
    k = len(le.classes_)

    if pi is None:
        pi = estimate_pi(y_true_int, k)

    T_hat = empirical_T(y_true, y_noisy)

    P_t = f81_transition_matrix(pi, t) if t is not None else None
    max_T_error = float(np.abs(T_hat - P_t).max()) if P_t is not None else None

    flip_mask = y_noisy_int != y_true_int
    n = len(y_true)

    per_class = {}
    for c in range(k):
        class_mask = y_true_int == c
        n_c = int(class_mask.sum())
        n_flipped_c = int(flip_mask[class_mask].sum())
        flip_rate_c = n_flipped_c / n_c if n_c > 0 else float("nan")

        if P_t is not None:
            theoretical_c = float(1.0 - P_t[c, c])
        else:
            theoretical_c = None

        per_class[le.classes_[c]] = {
            "n": n_c,
            "n_flipped": n_flipped_c,
            "flip_rate": flip_rate_c,
            "flip_rate_theoretical": theoretical_c,
        }

    return {
        "n": n,
        "k": k,
        "n_flipped": int(flip_mask.sum()),
        "flip_rate": float(flip_mask.mean()),
        "per_class": per_class,
        "T_hat": T_hat,
        "P_t": P_t,
        "max_T_error": max_T_error,
    }
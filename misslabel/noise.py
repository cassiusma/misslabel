"""
misslabel.noise
---------------
Label noise injection engine.

The injection model is a continuous-time Markov model on k label states,
parametrised by a single time parameter t >= 0. The transition matrix P(t)
is computed from the F81 substitution model (see misslabel.matrix).

For each sample i with true label y[i] = c, the corrupted label is drawn as:

    y_noisy[i] ~ Categorical(P(t)[c, :])

The process is a stochastic channel: a sample may be drawn back to
its original class (not a flip) with probability P(t)[c, c]. The flip_mask
records only actual changes: flip_mask[i] = (y_noisy[i] != y[i]).

Expected flip rate for class c:

    E[flip rate | class c] = 1 - P(t)[c, c]
                           = (1 - exp(-mu*t)) * (1 - pi_c)

Minority classes (small pi_c) experience higher expected corruption than
majority classes at the same t. This is a property of the model, not a bug.
"""

from collections import namedtuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from misslabel._validators import validate_labels, validate_t
from misslabel.matrix import estimate_pi, f81_transition_matrix


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

InjectionResult = namedtuple("InjectionResult", ["y_noisy", "flip_mask"])
InjectionResult.__doc__ = """
Result of a label noise injection.

Attributes
----------
y_noisy : ndarray of shape (n,), dtype int
    Corrupted label vector.
flip_mask : ndarray of shape (n,), dtype bool
    True at positions where the label was changed (y_noisy[i] != y[i]).
"""


# ---------------------------------------------------------------------------
# Internal encoding layer
# ---------------------------------------------------------------------------

def _encode_labels(y: np.ndarray):
    """
    Encode an arbitrary label array to contiguous integers {0, ..., k-1}.

    If y is already integer-typed with values in {0, ..., k-1}, encoding is
    still applied for uniformity. The encoder can always invert the result.

    Parameters
    ----------
    y : array-like of shape (n,)
        Label vector. May contain integers, strings, or any type accepted
        by sklearn.preprocessing.LabelEncoder.

    Returns
    -------
    y_int : ndarray of shape (n,), dtype int64
        Integer-encoded labels.
    le : LabelEncoder
        Fitted encoder. Call le.inverse_transform(y_int) to recover original
        labels. le.classes_ gives the ordered class vocabulary.
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    return y_int, le


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject(
    y: np.ndarray,
    t: float,
    pi: np.ndarray = None,
    random_state: int = None,
) -> InjectionResult:
    """
    Inject F81 label noise into a label vector.

    Accepts integer, string, or any label type supported by
    sklearn.preprocessing.LabelEncoder. Returns corrupted labels in the
    same type as the input.

    Parameters
    ----------
    y : array-like of shape (n,)
        True label vector. Values may be integers, strings, or any
        comparable type. Internally encoded to {0, ..., k-1}.
    t : float
        Time parameter (non-negative). Controls noise intensity:
          t = 0   : no corruption, y_noisy == y
          t -> inf: labels fully randomised under pi
    pi : ndarray of shape (k,), optional
        Stationary distribution over the k classes, in the order determined
        by LabelEncoder (lexicographic). If None, estimated empirically
        from y.
        Supplying pi explicitly is useful when y is a subset of a larger
        dataset and the global class frequencies are known.
    random_state : int, optional
        Seed for the random number generator. Pass an integer for
        reproducible results.

    Returns
    -------
    result : InjectionResult
        Named tuple with fields:
          y_noisy  : corrupted label vector, same type and shape as y
          flip_mask: boolean ndarray of shape (n,), True where label changed

    Notes
    -----
    The injection loops over k classes (not n samples). For each class c,
    all n_c samples sharing that label are drawn jointly from
    Categorical(P(t)[c, :]) using numpy, making the inner loop O(k) in
    Python and O(n) in C.

    The expected flip rate for class c is:

        E[flip rate | class c] = 1 - P(t)[c,c] = (1 - exp(-mu*t)) * (1 - pi_c)

    Minority classes experience higher expected corruption than majority
    classes at the same t.

    Examples
    --------
    >>> import numpy as np
    >>> from misslabel.noise import inject
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> result = inject(y, t=0.5, random_state=0)
    >>> result.flip_mask.sum()  # number of corrupted labels
    ...

    >>> # String labels work identically
    >>> y_str = np.array(["cat", "cat", "dog", "dog", "bird", "bird"])
    >>> result = inject(y_str, t=0.5, random_state=0)
    >>> result.y_noisy  # returns strings
    array([...], dtype='<U4')
    """
    validate_t(t)

    # Encode to contiguous integers, remember encoder for inversion
    y_int, le = _encode_labels(y)
    k = len(le.classes_)

    validate_labels(y_int, k)

    # Estimate pi from encoded y if not supplied
    if pi is None:
        pi = estimate_pi(y_int, k)

    # Compute F81 transition matrix
    P = f81_transition_matrix(pi, t)

    # Draw corrupted labels class by class
    rng = np.random.default_rng(random_state)
    y_noisy_int = np.empty(len(y_int), dtype=y_int.dtype)

    for c in range(k):
        mask = (y_int == c)
        n_c = mask.sum()
        if n_c > 0:
            y_noisy_int[mask] = rng.choice(k, size=n_c, p=P[c, :])

    # Decode back to original label type
    y_noisy = le.inverse_transform(y_noisy_int)
    flip_mask = y_noisy != np.asarray(y)

    return InjectionResult(y_noisy=y_noisy, flip_mask=flip_mask)
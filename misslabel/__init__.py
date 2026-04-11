"""
misslabel
---------
Controlled label noise injection for classifier robustness studies.

The noise model is the F81 continuous-time Markov chain (Felsenstein 1981),
parametrised by a single time parameter t >= 0. The stationary distribution
pi is estimated from the empirical label frequencies of the dataset.

Basic usage
-----------
>>> import numpy as np
>>> from misslabel import inject, summary

>>> y = np.array([0, 0, 1, 1, 2, 2] * 100)
>>> result = inject(y, t=0.5, random_state=42)
>>> result.y_noisy        # corrupted labels
>>> result.flip_mask      # True where label changed

>>> report = summary(y, result.y_noisy, t=0.5)
>>> report["flip_rate"]   # overall empirical flip rate
>>> report["per_class"]   # per-class breakdown

String labels are supported:
>>> y_str = np.array(["cat", "dog", "bird"] * 100)
>>> result = inject(y_str, t=0.5, random_state=42)
"""

from misslabel.noise import inject, InjectionResult
from misslabel.matrix import (
    estimate_pi,
    f81_rate_matrix,
    f81_transition_matrix,
    normalise_rate_matrix,
)
from misslabel.audit import empirical_T, summary

__version__ = "0.1.0"

__all__ = [
    # core
    "inject",
    "InjectionResult",
    # matrix
    "estimate_pi",
    "f81_rate_matrix",
    "f81_transition_matrix",
    "normalise_rate_matrix",
    # audit
    "empirical_T",
    "summary",
]
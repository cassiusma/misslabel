# misslabel

Misslabel datasets for classifier robustness studies.

`misslabel` corrupts a label vector using the **F81 continuous-time Markov model** (Felsenstein 1981), a well-studied noise model from phylogenetics. A single parameter $t \geq 0$ controls noise intensity. The stationary distribution $\pi$ is estimated from the empirical label frequencies of your dataset.

---

## Why misslabel

Most label noise tools inject noise uniformly. `misslabel` derives the transition matrix from first principles:

- The noise model is a **reversible continuous-time Markov chain** on $k$ label states
- The stationary distribution $\pi$ is the empirical class frequency vector of your dataset
- The single parameter $t$ controls how far labels evolve along the chain


The transition matrix has a closed form:

$$P(t) = e^{-\mu t} I + (1 - e^{-\mu t}) \mathbf{1}\pi^\top$$

where $\mu = 1 / (1 - \sum_i \pi_i^2)$ is the normalisation scalar (inverse heterozygosity).

---

## Installation

```bash
pip install misslabel
```

Or from source:

```bash
git clone https://github.com/cassiusma/misslabel.git
cd misslabel
pip install -e .
```

---

## Quick start

```python
import numpy as np
from misslabel import inject, summary

# Integer labels
y = np.array([0, 0, 1, 1, 2, 2] * 50)
result = inject(y, t=0.5, random_state=42)

result.y_noisy    # corrupted label vector
result.flip_mask  # True where label was changed

# Diagnostic report
report = summary(y, result.y_noisy, t=0.5)
print(f"Overall flip rate: {report['flip_rate']:.3f}")
print(f"Theoretical flip rate: {report['per_class']}")
print(f"max|T_hat - P(t)|: {report['max_T_error']:.4f}")
```

String labels are supported:

```python
y_str = np.array(["cat", "dog", "bird"] * 50)
result = inject(y_str, t=0.5, random_state=42)
result.y_noisy  # returns strings: array(['cat', 'dog', 'cat', ...])
```

---

## Robustness experiment — Iris

Running `examples/iris_robustness.py` trains a logistic regression classifier
on increasingly corrupted Iris labels:

```
Baseline accuracy (clean labels): 0.9667

     t   flip_rate_theoretical   flip_rate_empirical    accuracy
--------------------------------------------------------------------
  0.00                  0.0000                0.0000      0.9667
  0.10                  0.0929                0.0867      0.8533
  0.25                  0.2085                0.1933      0.7267
  0.50                  0.3518                0.3600      0.5867
  0.75                  0.4502                0.4467      0.5000
  1.00                  0.5179                0.5000      0.4533
  1.50                  0.5964                0.6200      0.4200
  2.00                  0.6335                0.6667      0.3467
  3.00                  0.6593                0.6800      0.4200
  5.00                  0.6663                0.6867      0.3933

Random baseline (1/k): 0.3333
```

The flip rate saturates at $1 - \sum_c \pi_c^2 \approx 0.667$ (the heterozygosity),
regardless of $t$. This is a mathematical property of the F81 model with uniform $\pi$.

---

## API reference

### `inject(y, t, pi=None, random_state=None)`

Inject F81 label noise into a label vector.

| Parameter | Type | Description |
|---|---|---|
| `y` | array-like | True labels. Integers or strings. |
| `t` | float | Noise parameter $t \geq 0$. |
| `pi` | ndarray, optional | Stationary distribution. Estimated from `y` if `None`. |
| `random_state` | int, optional | Random seed for reproducibility. |

Returns `InjectionResult(y_noisy, flip_mask)`.

---

### `summary(y_true, y_noisy, pi=None, t=None)`

Diagnostic report comparing empirical and theoretical noise.

Returns a dict with keys: `n`, `k`, `n_flipped`, `flip_rate`, `per_class`,
`T_hat`, `P_t`, `max_T_error`.

---

### `empirical_T(y_true, y_noisy)`

Estimate the empirical noise transition matrix $\hat{T}$ from paired label vectors.

$$\hat{T}_{ij} = \frac{\{y_\text{true}=i,\ y_\text{noisy}=j\}}{\{y_\text{true}=i\}}$$

---

### Matrix utilities

```python
from misslabel import estimate_pi, f81_rate_matrix, f81_transition_matrix, normalise_rate_matrix
```

| Function | Description |
|---|---|
| `estimate_pi(y, k)` | Empirical label frequencies with Laplace smoothing |
| `f81_rate_matrix(pi)` | Normalised F81 rate matrix $Q$ |
| `f81_transition_matrix(pi, t)` | Closed-form $P(t) = e^{Qt}$ |
| `normalise_rate_matrix(Q, pi)` | Model-agnostic normalisation to 1 substitution/unit time |

---

## Noise model details

The F81 rate matrix is:

$$Q_{ij} = \mu \pi_j \quad (i \neq j), \qquad Q_{ii} = -\mu(1 - \pi_i)$$

normalised so that $\sum_i \pi_i |Q_{ii}| = 1$, giving:

$$\mu = \frac{1}{1 - \sum_i \pi_i^2}$$

The expected flip rate for class $c$ at time $t$ is:

$$1 - P(t)_{cc} = (1 - e^{-\mu t})(1 - \pi_c)$$

Minority classes (small $\pi_c$) have higher expected flip rates than majority
classes at the same $t$.

---

## Roadmap

- [ ] GTR model (6 exchangeability parameters)
- [ ] Per-class noise parameter $t_c$
- [ ] MLflow integration hook
- [ ] CLI runner (`noisebench`-style)

---

## License

MIT

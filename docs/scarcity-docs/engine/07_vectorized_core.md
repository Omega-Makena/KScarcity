# Engine — Vectorized Core

---

## vectorized_core.py — Massively Parallel RLS

Replaces the object-per-hypothesis architecture with batched matrix operations over all M hypotheses simultaneously.

### Shape conventions

| Symbol | Meaning |
|--------|---------|
| M | Number of hypotheses (e.g. 10 000) |
| F | Number of features (e.g. 2 for linear: [1, x]) |
| `W` | Weight matrix — shape `(M, F)` |
| `P` | Covariance tensor — shape `(M, F, F)` |

### `VectorizedRLS`

Runs M independent Recursive Least Squares models in a single set of NumPy operations — O(1) Python overhead, O(M) native C overhead.

```python
from scarcity.engine.vectorized_core import VectorizedRLS

rls = VectorizedRLS(n_models=10_000, n_features=2, lambda_forget=0.99)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_models` | — | Number of parallel models (M) |
| `n_features` | 2 | Feature dimension F |
| `lambda_forget` | 0.99 | Forgetting factor λ — lower = faster adaptation |

**State tensors**:
- `W`: initialized to zeros `(M, F)`
- `P`: initialized to `10 × I` per model — `(M, F, F)`

#### `predict(X)`

```
X: (M, F) → returns (M,) predictions
prediction_i = W_i · X_i = einsum('ij,ij->i', W, X)
```

#### `update(X, Y, active_mask=None)`

Batched RLS update:

```
For each active model i:
  error_i = Y_i − W_i · X_i
  K_i = P_i X_i / (λ + X_i^T P_i X_i)       # Kalman gain
  W_i += K_i × error_i                        # Weight update
  P_i = (P_i − K_i X_i^T P_i) / λ           # Covariance update
```

`active_mask` (boolean `(M,)`) selects which models update — avoids wasted computation on dead hypotheses. Boolean-indexed subsets are updated and scattered back.

#### `update_subset(indices, X_sub, Y_sub)`

Direct index-based update for a pre-selected subset.

### robustness.py — Robust Online Statistics

#### `OnlineWinsorizer`

Clips inputs to [p_lower, p_upper] using a sliding window of recent values.

```python
winsorizer = OnlineWinsorizer(window_size=1000, lower_p=1.0, upper_p=99.0)
clipped = winsorizer.update(raw_value)
```

- Bounds recomputed every 10 steps (after ≥ 100 samples)
- No clipping for first 20 samples (insufficient data)
- Non-finite values passed through unchanged

#### `OnlineMAD`

Tracks median and Median Absolute Deviation (MAD) — robust to heavy-tailed distributions.

```python
mad = OnlineMAD(window_size=1000)
mad.update(x)
print(mad.median, mad.mad)
```

#### Huber loss

`HUBER_DELTA = 1.345` (module-level constant).

Huber loss provides gradient clipping for RLS: behaves as L2 for `|error| < δ` and L1 (linear) for `|error| ≥ δ`, preventing large outliers from destabilising weights.

### Integration with HypothesisPool

`VectorizedRLS` instances are managed by `VectorizedHypothesisPool`, which maintains the global weight/covariance tensors and maps hypothesis indices to rows. The per-hypothesis `VectorizedFunctionalHypothesis` objects hold only their row index — the actual computation is batched.

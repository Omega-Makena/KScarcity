# Utility Files

This page documents the smaller utility files in the engine module.

---

## types.py — Type Definitions

Common type definitions used across the engine.

### `Candidate` (Dataclass)

Represents a candidate path to evaluate:

```python
@dataclass
class Candidate:
    source_idx: int    # Source variable index
    target_idx: int    # Target variable index
    lags: List[int]    # Lag structure (e.g., [1, 2])
    arm_id: int        # Bandit arm ID
    path_id: str       # Human-readable path ID
```

### `EvalResult` (Dataclass)

Result from evaluating a candidate:

```python
@dataclass
class EvalResult:
    candidate: Candidate
    accepted: bool
    gain: float
    ci_lo: float
    ci_hi: float
    stability: float
    error: Optional[str]  # If rejected, why
```

### `Reward` (Dataclass)

Shaped reward for bandit learning:

```python
@dataclass
class Reward:
    arm_id: int
    raw_reward: float
    shaped_reward: float
```

---

## utils.py — Mathematical Utilities

Common mathematical operations used throughout the engine.

### Numerical Safety

```python
def clip(x, lo, hi):
    """Bound value to range."""
    return max(lo, min(hi, x))

def safe_div(a, b, default=0.0):
    """Division with fallback for zero denominator."""
    return a / b if abs(b) > 1e-10 else default

def softplus(x, beta=1.0):
    """Smooth approximation to ReLU."""
    return np.log1p(np.exp(beta * x)) / beta
```

### Robust Statistics

```python
def robust_zscore(x, center, mad):
    """Z-score using median absolute deviation."""
    return (x - center) / (mad + 1e-10)

def robust_quantiles(arr, qs=[0.1, 0.5, 0.9]):
    """Quantile computation."""
    return np.quantile(arr, qs)

def compute_median_mad(arr):
    """Median and median absolute deviation."""
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return median, mad
```

---

## robustness.py — Online Robust Statistics

### `OnlineMAD`

Tracks median absolute deviation in streaming fashion:

```python
class OnlineMAD:
    def __init__(self, window_size=100):
        self.buffer = []
        self.window_size = window_size
        self.median = 0.0
        self.mad = 0.0
    
    def update(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        self.median = np.median(self.buffer)
        self.mad = np.median(np.abs(np.array(self.buffer) - self.median))
```

**Why MAD over variance?** More robust to outliers — single extreme value doesn't break stability tracking.

---

## vectorized_core.py — High-Performance Operations

Batched NumPy operations for hypothesis updates.

### `VectorizedHypothesisPool`

Accelerates batch updates when many hypotheses exist:

```python
class VectorizedHypothesisPool:
    def __init__(self, capacity):
        # Pre-allocated arrays for hypothesis states
        self.confidence = np.zeros(capacity)
        self.stability = np.zeros(capacity)
        self.fit_scores = np.zeros(capacity)
        
    def batch_update(self, row_array, indices):
        # Vectorized update of multiple hypotheses
        # Much faster than Python loops
        pass
```

### Key Optimizations

- **Pre-allocated arrays**: Avoid allocation in hot loop
- **SIMD operations**: NumPy uses vectorized CPU instructions
- **Cache-friendly**: Contiguous memory layout

---

## resource_profile.py — DRG Configuration

Default resource profiles for the Dynamic Resource Governor.

### `clone_default_profile()`

Returns a fresh default profile:

```python
def clone_default_profile():
    return {
        "n_proposals": 10,
        "n_resamples": 10,
        "max_lag": 6,
        "holdout_frac": 0.2,
        "gain_min": 0.02,
        "stability_min": 0.6,
        "lambda_ci": 1.0,
        "allocation_limit": 1000
    }
```

These can be overridden by the Governor based on system load.

---

## relationship_config.py — Hypothesis Configuration

Configuration dataclasses for each relationship type.

### Example: `CausalConfig`

```python
@dataclass
class CausalConfig:
    min_buffer: int = 20
    granger_threshold: float = 0.02
    confidence_growth_rate: float = 0.01
```

### Example: `TemporalConfig`

```python
@dataclass
class TemporalConfig:
    rls_lambda: float = 0.99  # RLS forgetting factor
    min_lag: int = 1
    max_lag: int = 5
```

Pass to hypothesis constructors to customize behavior.

---

## exporter.py — Knowledge Graph Export

Formats discovered relationships for output.

### `Exporter`

```python
class Exporter:
    def export_edges(self, store: HypergraphStore) -> List[Dict]:
        """Export edges as dictionaries."""
        edges = []
        for (src, dst), rec in store.edges.items():
            edges.append({
                "source": store.nodes[src],
                "target": store.nodes[dst],
                "effect": rec.weight,
                "confidence": rec.stability,  # or compute from hits
                "ci_lo": rec.ci_lo,
                "ci_hi": rec.ci_hi
            })
        return edges
```

### Output Formats

- JSON for API responses
- CSV for analysis
- NetworkX graph for visualization

---

## simulation.py — Engine Simulation Mode

Enables running the engine in simulation mode for testing.

### Key Functionality

```python
def simulate_from_hypotheses(hypotheses, initial_state, n_steps):
    """
    Run forward simulation using learned hypotheses.
    
    For each timestep:
    1. Get predictions from all hypotheses
    2. Aggregate predictions (e.g., weighted average)
    3. Update state
    4. Return trajectory
    """
    pass
```

**Use case**: Given discovered relationships, predict future values.

---

## algorithms_online.py — Online Learning Algorithms

Core statistical algorithms for streaming data.

### Recursive Least Squares (RLS)

```python
def rls_update(P, theta, x, y, lambda_):
    """
    Online regression update.
    
    P: Covariance matrix
    theta: Weight vector
    x: New feature vector
    y: New target value
    lambda_: Forgetting factor (0.99 typical)
    """
    k = P @ x / (lambda_ + x.T @ P @ x)
    theta = theta + k * (y - x.T @ theta)
    P = (P - np.outer(k, x.T @ P)) / lambda_
    return P, theta
```

### Exponential Moving Average

```python
def ema_update(current, new, alpha=0.1):
    """Exponential moving average."""
    return alpha * new + (1 - alpha) * current
```

### Welford's Online Variance

```python
class WelfordVariance:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    
    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0
```

Numerically stable single-pass variance computation.

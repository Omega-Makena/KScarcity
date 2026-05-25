# Engine Operators — Overview

The `engine/operators/` directory contains **specialized mathematical operations** used by the encoding and evaluation pipelines. Each module focuses on a specific category of computation.

---

## Directory Structure

```
engine/operators/
├── __init__.py              # Exports all operators
├── attention_ops.py         # Attention mechanisms
├── causal_semantic_ops.py   # Causal reasoning primitives
├── evaluation_ops.py        # Scoring and metrics
├── integrative_ops.py       # Cross-module integration
├── relational_ops.py        # Relationship computations
├── sketch_ops.py            # Dimensionality reduction
├── stability_ops.py         # Stability metrics
└── structural_ops.py        # Graph structure operations
```

---

## attention_ops.py — Attention Mechanisms

Implements attention-based aggregation for time series.

### Core Functions

```python
def attn_linear(Q, K, V, n_heads=8):
    """
    Linear attention mechanism.
    
    Faster than softmax attention: O(n) vs O(n²).
    Used for aggregating time series features.
    """
    pass

def pooling_avg(x, axis=-1):
    """Simple mean pooling."""
    return np.mean(x, axis=axis)

def pooling_lastk(x, k=5, axis=-1):
    """Average of last k timesteps."""
    return np.mean(x[..., -k:], axis=axis)
```

### Normalization

```python
def layernorm(x, eps=1e-6):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def rmsnorm(x, eps=1e-6):
    """Root mean square normalization (faster than LayerNorm)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
    return x / (rms + eps)
```

---

## sketch_ops.py — Dimensionality Reduction

Random projection methods for compact representations.

### Core Functions

```python
def poly_sketch(x, target_dim, degree=2, seed=42):
    """
    Polynomial sketch projection.
    
    Creates approximate polynomial features in reduced dimension.
    Deterministic given seed.
    """
    pass

def tensor_sketch(x, y, target_dim, seed=42):
    """
    Tensor sketch for outer product approximation.
    
    Approximates x ⊗ y in low dimensions.
    Used for interaction features.
    """
    pass

def countsketch(x, target_dim, seed=42):
    """
    Count-Min sketch variant.
    
    Random sign flips + hashing for compression.
    Very fast, moderate accuracy.
    """
    pass

def latent_clip(x, max_norm=10.0):
    """Clip latent vectors to bounded norm."""
    norm = np.linalg.norm(x)
    if norm > max_norm:
        x = x * (max_norm / norm)
    return x
```

**Why sketching?** Reduces dimensionality while preserving dot products (useful for similarity comparisons).

---

## evaluation_ops.py — Scoring Operations

Statistical operations for hypothesis evaluation.

### Core Functions

```python
def r2_gain(y_true, y_pred, baseline_pred=None):
    """
    Compute R² gain over baseline.
    
    If baseline_pred is None, uses mean baseline.
    Returns: R²_model - R²_baseline
    """
    pass

def ols_solve(X, y):
    """
    Ordinary least squares solution.
    
    Returns: (coefficients, residuals)
    Uses numerically stable SVD-based solver.
    """
    pass

def bootstrap_indices(n, n_boot, rng):
    """Generate bootstrap sample indices."""
    return rng.choice(n, size=(n_boot, n), replace=True)
```

---

## stability_ops.py — Stability Metrics

Track consistency of effects over time.

### Core Functions

```python
def sign_consistency(values):
    """
    Fraction of values with same sign as median.
    
    Returns: float in [0, 1]
    1.0 = perfectly consistent signs
    0.5 = random signs
    """
    median_sign = np.sign(np.median(values))
    return np.mean(np.sign(values) == median_sign)

def effect_stability(current, history, window=10):
    """
    How stable is current effect vs recent history?
    
    Uses MAD-based z-score.
    """
    pass

def regime_detection(series, n_regimes=2):
    """
    Simple regime detection via k-means on returns.
    
    Returns: regime labels per timestep
    """
    pass
```

---

## relational_ops.py — Relationship Computations

Core computations for relationship hypotheses.

### Core Functions

```python
def granger_statistic(X_augmented, X_restricted, y):
    """
    Compute Granger causality F-statistic.
    
    Compares fit of augmented model (with lagged X)
    to restricted model (Y lags only).
    """
    pass

def online_correlation(x, y, state):
    """
    Update running Pearson correlation.
    
    Uses Welford-style online updates.
    Returns: (correlation, updated_state)
    """
    pass

def partial_correlation(x, y, z, data):
    """
    Correlation between x and y, controlling for z.
    
    Used for confounding detection.
    """
    pass
```

---

## causal_semantic_ops.py — Causal Reasoning

Higher-level causal inference operations.

### Core Functions

```python
def backdoor_adjustment(data, treatment, outcome, confounders):
    """
    Estimate causal effect controlling for confounders.
    """
    pass

def instrument_validity_check(instrument, treatment, outcome):
    """
    Check if instrument is valid for IV estimation.
    
    Returns: (relevance_score, exclusion_score)
    """
    pass

def counterfactual_prediction(model, factual, intervention):
    """
    Predict outcome under counterfactual intervention.
    """
    pass
```

---

## structural_ops.py — Graph Operations

Operations on the relationship graph structure.

### Core Functions

```python
def detect_cycles(adjacency):
    """
    Find cycles in directed graph.
    
    Returns: List of cycle paths
    """
    pass

def compute_centrality(adjacency, method="degree"):
    """
    Node centrality scores.
    
    Methods: degree, betweenness, pagerank
    """
    pass

def find_hubs(adjacency, threshold=0.8):
    """
    Identify hub nodes (high out-degree).
    """
    pass

def topological_sort(adjacency):
    """
    Order nodes by dependency.
    
    Required for causal ordering in simulation.
    """
    pass
```

---

## integrative_ops.py — Cross-Module Operations

Operations that integrate multiple subsystems.

### Core Functions

```python
def merge_hypotheses(local, remote, strategy="weighted"):
    """
    Merge hypothesis states from distributed nodes.
    
    Strategies: weighted (by evidence), union, intersection
    """
    pass

def knowledge_graph_consistency(edges):
    """
    Check knowledge graph for logical consistency.
    
    Returns: List of violations
    """
    pass
```

---

## Usage Notes

Most operators are called internally by the Encoder and Evaluator. Direct use:

```python
from scarcity.engine.operators.sketch_ops import poly_sketch
from scarcity.engine.operators.evaluation_ops import r2_gain

# Project features
latent = poly_sketch(features, target_dim=64)

# Compute gain
gain = r2_gain(y_true, y_pred)
```

Operators are **pure functions** (no state), making them easy to test and parallelize.

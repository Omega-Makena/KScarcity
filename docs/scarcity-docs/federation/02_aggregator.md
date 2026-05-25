# aggregator.py — Byzantine-Robust Aggregation

The `FederatedAggregator` provides **robust aggregation methods** that resist poisoning attacks and Byzantine failures when combining updates from multiple clients.

---

## The Problem

In federated learning, a malicious client can send:
- **Gradient scaling attacks**: Multiply updates by large factor
- **Sign flipping**: Reverse the direction of updates
- **Random noise**: Submit garbage to slow convergence
- **Targeted poisoning**: Craft updates to bias the model

Simple averaging (FedAvg) is vulnerable to all of these.

---

## Aggregation Methods

### `AggregationMethod` (Enum)

| Method | Description | Byzantine Tolerance |
|--------|-------------|---------------------|
| `FEDAVG` | Simple averaging | None |
| `WEIGHTED` | Weight by sample count | None |
| `ADAPTIVE` | Weight by inverse loss | Low |
| `MEDIAN` | Coordinate-wise median | ~25% |
| `TRIMMED_MEAN` | Remove extremes, then mean | ~α% |
| `KRUM` | Select most central vector | ~(n-f-2)/n |
| `MULTI_KRUM` | Select m most central, average | ~(n-f-2)/n |
| `BULYAN` | Krum + Trimmed Mean | ~(n-f-2)/n |

---

## How Each Method Works

### FedAvg / Weighted Mean

Simple baseline — no robustness:

```
aggregate = Σ(w_i * update_i) / Σ(w_i)
```

**Use when**: All clients are trusted.

### Median

Take coordinate-wise median:

```
aggregate[j] = median(update_1[j], update_2[j], ..., update_n[j])
```

**Robustness**: Tolerates up to 25% Byzantine clients (they can only affect extreme quantiles).

### Trimmed Mean

Remove α% from each tail, then average:

```
For each coordinate j:
  1. Sort values: update_1[j], update_2[j], ...
  2. Remove bottom α% and top α%
  3. Average remaining
```

**Robustness**: Tolerates up to α% Byzantine clients.

### Krum

Select the single vector that's most "central":

```
For each vector v_i:
  score_i = sum of distances to (n-f-2) nearest neighbors
  
Return v_i with lowest score
```

**Intuition**: Byzantine updates are likely outliers, so the most central vector is probably honest.

**Robustness**: Tolerates f < n/2 - 2 Byzantine clients.

### Multi-Krum

Select m most central vectors, then average:

```
1. Compute Krum scores for all vectors
2. Select top m by lowest score
3. Average selected vectors
```

**Better convergence** than single Krum while maintaining robustness.

### Bulyan

Combines Krum selection with Trimmed Mean:

```
1. Use Multi-Krum to select trustworthy subset
2. Apply Trimmed Mean to the selected subset
```

**Strongest robustness** but requires more clients.

---

## Class: `FederatedAggregator`

### Configuration

```python
@dataclass
class AggregationConfig:
    method: AggregationMethod = AggregationMethod.TRIMMED_MEAN
    trim_alpha: float = 0.1           # For trimmed mean
    multi_krum_m: int = 5             # For multi-krum
    trust_min: float = 0.2            # Minimum trust score
    adaptive_metric_is_loss: bool = True  # For adaptive weighting
```

### Initialization

```python
config = AggregationConfig(
    method=AggregationMethod.BULYAN,
    trim_alpha=0.2
)
aggregator = FederatedAggregator(config)
```

### `aggregate(updates) -> Tuple[np.ndarray, Dict]`

Main aggregation method:

```python
updates = [
    client_1_update,  # np.ndarray
    client_2_update,
    ...
]

result, meta = aggregator.aggregate(updates)
# result: aggregated vector
# meta: {"method": "bulyan", "n_participants": 10, ...}
```

**Fallback logic**:
- If Bulyan fails (not enough participants), falls back to Krum
- If Krum fails, falls back to Trimmed Mean
- If Trimmed Mean fails, falls back to Median

### `detect_outliers(updates, reference, z_thresh=4.0) -> List[int]`

Identify suspicious updates:

```python
outliers = aggregator.detect_outliers(
    updates=all_updates,
    reference=aggregated_result,
    z_thresh=4.0
)
# Returns: [3, 7]  # Indices of outlier updates
```

**Algorithm**:
1. Compute distance of each update from reference
2. Compute z-score of distances
3. Flag updates with z > threshold

---

## Internal Functions

### `_pairwise_distances(array)`

Compute Euclidean distances between all pairs of vectors. Used by Krum.

### `_krum_select(array, m)`

Select m vectors with lowest Krum scores.

### `_trimmed_mean(array, alpha)`

Coordinate-wise trimmed mean.

### `_parse_updates(updates)`

Handle input formats:
- List of arrays
- List of tuples (vector, weight)
- List of dicts with "vector" and "weight" keys

---

## Choosing an Aggregation Method

| Scenario | Recommended Method |
|----------|-------------------|
| All clients trusted | FEDAVG |
| Some dropout expected | WEIGHTED (by samples) |
| <10% Byzantine possible | TRIMMED_MEAN (α=0.1) |
| 10-20% Byzantine possible | KRUM or MULTI_KRUM |
| 20-30% Byzantine possible | BULYAN |
| Unknown threat model | BULYAN (conservative) |

---

## Computational Cost

| Method | Complexity |
|--------|------------|
| FedAvg | O(n × d) |
| Median | O(n × d × log n) |
| Trimmed Mean | O(n × d × log n) |
| Krum | O(n² × d) |
| Multi-Krum | O(n² × d) |
| Bulyan | O(n² × d) |

Where:
- n = number of clients
- d = vector dimension

For large n, consider sampling before aggregation.

---

## Edge Cases

### Too Few Updates

- Krum requires at least 3 updates
- Bulyan requires at least 2f + 3 updates
- Falls back to simpler methods when insufficient

### All-Zero Updates

Returns zero vector — no error.

### NaN/Inf in Updates

Updates with invalid values are filtered before aggregation.

---

## Example: Defending Against Attack

```python
from scarcity.federation import FederatedAggregator, AggregationConfig, AggregationMethod
import numpy as np

# Setup
config = AggregationConfig(method=AggregationMethod.BULYAN)
aggregator = FederatedAggregator(config)

# Honest updates
honest = [np.random.randn(100) * 0.1 for _ in range(8)]

# Malicious updates (gradient scaling)
malicious = [np.random.randn(100) * 100 for _ in range(2)]

all_updates = honest + malicious

# Aggregate
result, meta = aggregator.aggregate(all_updates)
print(f"Method: {meta['method']}, Participants used: {meta['n_participants']}")

# Verify outliers detected
outliers = aggregator.detect_outliers(all_updates, result)
print(f"Outliers detected at indices: {outliers}")
# Expected: [8, 9] (the malicious ones)
```

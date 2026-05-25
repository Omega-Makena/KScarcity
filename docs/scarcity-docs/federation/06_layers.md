# layers.py — Two-Layer Aggregation

The `layers.py` file implements the **two-layer aggregation** system: Layer 1 for within-basket aggregation and Layer 2 for cross-basket global aggregation.

---

## Architecture

```
        Layer 2: Global Aggregation
    ┌─────────────────────────────────────┐
    │     GlobalMetaModel                  │
    │     (weighted sum of baskets)       │
    └─────────────────────────────────────┘
                     ▲
                     │ SecureAggregator
                     │ + CentralDPMechanism
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐       ┌───▼───┐       ┌───▼───┐
│Basket │       │Basket │       │Basket │
│Model A│       │Model B│       │Model C│
└───────┘       └───────┘       └───────┘
    ▲                ▲                ▲
    │ Layer1Aggregator               │
    │ (per-basket)                   │
    │                                │
┌───┴───┐       ┌───────┐       ┌───┴───┐
│Updates│       │Updates│       │Updates│
└───────┘       └───────┘       └───────┘
```

---

## Layer 1: Intra-Basket Aggregation

### `Layer1Config`

```python
@dataclass
class Layer1Config:
    aggregation_method: AggregationMethod = AggregationMethod.TRIMMED_MEAN
    trim_alpha: float = 0.1
    min_updates: int = 3
    apply_local_noise: bool = True
```

### `Layer1Aggregator`

Aggregates updates within a single basket:

```python
aggregator = Layer1Aggregator(config=Layer1Config())

basket_model = aggregator.aggregate(
    updates=[update_1, update_2, update_3, ...],
    weights=[1.0, 1.0, 0.8, ...]  # Optional staleness weights
)
```

**Process**:
1. Apply Byzantine-robust aggregation (from `aggregator.py`)
2. Optionally add local DP noise
3. Produce `BasketModel`

### `BasketModel`

```python
@dataclass
class BasketModel:
    basket_id: str
    aggregate: np.ndarray
    n_contributors: int
    round_id: int
    timestamp: float
    confidence: float  # Based on contributor count
```

---

## Layer 2: Cross-Basket Aggregation

### `Layer2Config`

```python
@dataclass
class Layer2Config:
    use_secure_aggregation: bool = True
    central_dp_epsilon: float = 1.0
    central_dp_delta: float = 1e-5
    clip_norm: float = 1.0
    min_baskets: int = 2
    weight_by_size: bool = True
```

### `Layer2Aggregator`

Combines basket models into a global model:

```python
aggregator = Layer2Aggregator(config=Layer2Config())

global_model = aggregator.aggregate(
    basket_models=[model_a, model_b, model_c],
    secure_agg_coordinator=coordinator  # Optional
)
```

**Process**:
1. Collect basket aggregates
2. If `use_secure_aggregation`: run cryptographic protocol
3. Weight by basket size (if enabled)
4. Apply central DP noise
5. Produce `GlobalMetaModel`

### `GlobalMetaModel`

```python
@dataclass
class GlobalMetaModel:
    aggregate: np.ndarray
    n_baskets: int
    n_clients_total: int
    round_id: int
    timestamp: float
    privacy_spent: Tuple[float, float]  # (ε, δ)
```

---

## Secure Aggregator

### `SecureAggregator`

Cryptographic secure aggregation wrapper:

```python
secure_agg = SecureAggregator(config=Layer2Config())

# Clients submit encrypted shares
for client_id, share in shares:
    secure_agg.submit_share(client_id, share)

# Reveal aggregate (individual inputs hidden)
aggregate = secure_agg.finalize()
```

**Properties**:
- Sum is revealed, individual contributions hidden
- Dropout-tolerant (works if t out of n clients participate)
- Based on secret sharing

### When Secure Aggregation is Skipped

If `use_secure_aggregation=False`:
- Simple weighted average of basket models
- Faster but reveals individual basket aggregates
- Appropriate when baskets themselves provide sufficient privacy

---

## Central DP Mechanism

### `CentralDPMechanism`

Adds noise to the global aggregate:

```python
dp = CentralDPMechanism(
    epsilon=1.0,
    delta=1e-5,
    sensitivity=1.0
)

noised = dp.add_noise(aggregate)
```

**Why central DP?**
- Additional privacy layer beyond local DP
- Protects against inference attacks on the global model
- Accounts for composition across rounds

---

## Weight Schemes

### By Basket Size

```python
# weight_by_size=True (default)
weights = [model.n_contributors for model in basket_models]
aggregate = weighted_average(basket_models, weights)
```

Larger baskets have more influence.

### Uniform

```python
# weight_by_size=False
aggregate = simple_average(basket_models)
```

All baskets equal regardless of size.

### By Trust Score

Future enhancement: weight by basket trust/reliability.

---

## Aggregation Flow

### Layer 1 Trigger

When a basket's trigger fires:

```python
def aggregate_basket(basket_id):
    updates = buffer.pop_basket_updates(basket_id)
    
    # Compute staleness weights
    weights = [0.9 ** u.staleness for u in updates]
    
    # Aggregate
    basket_model = layer1.aggregate(
        [u.update for u in updates],
        weights
    )
    
    store_basket_model(basket_id, basket_model)
```

### Layer 2 Trigger

When global trigger fires:

```python
def aggregate_global():
    basket_models = get_all_basket_models()
    
    if len(basket_models) < config.min_baskets:
        return None
    
    global_model = layer2.aggregate(basket_models)
    distribute_to_clients(global_model)
    
    return global_model
```

---

## Confidence Scoring

Basket models carry confidence:

```python
confidence = min(1.0, n_contributors / expected_contributors)
```

Low confidence → fewer contributors than expected → less reliable.

Layer 2 can use confidence as weight:

```python
weights = [m.confidence for m in basket_models]
```

---

## Edge Cases

### Single Basket

Layer 2 with one basket:
- Secure aggregation is overkill (nothing to hide from)
- Still applies central DP
- Global model equals basket model (noised)

### Empty Basket Model

Basket with zero updates produces None:
- Skipped in Layer 2 aggregation
- Doesn't count toward min_baskets

### Extreme Staleness

If all basket models are very old:
- Confidence is low
- May trigger re-aggregation
- Global model heavily weighted toward recent baskets

---

## Statistics

### Layer 1 Stats

```python
{
    "basket_id": "healthcare_0",
    "n_updates": 15,
    "aggregation_method": "bulyan",
    "outliers_removed": 2,
    "local_noise_added": True
}
```

### Layer 2 Stats

```python
{
    "n_baskets": 5,
    "n_clients_total": 47,
    "secure_agg_used": True,
    "central_dp_epsilon": 1.0,
    "round_id": 42
}
```

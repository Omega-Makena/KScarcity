# DomainServerMeta — Federation-to-Meta Bridge

`scarcity/meta/domain_server_meta.py`

---

## Purpose

`DomainServerMeta` **bridges the federation layer and the meta-learning pipeline**. It observes `DomainServer` instances (from `scarcity/federation/`) and converts their runtime state — hit rate, memory size, base parameters, and round-over-round changes — into `DomainMetaUpdate` objects that `CrossDomainMetaLearner` can consume.

This class is the **Phase 5a component** of the federated meta-learning architecture. It answers the question: *given what a domain server has learned, what update signal should travel upward to the cross-domain learner?*

### Key Design Decisions

- **Duck typing**: `DomainServerMeta` accepts any object with `.basket_id`, `.domain_id`, `.base_params`, `.hit_rate`, `.memory_size`, and `.round_id` attributes. It never imports `DomainServer` directly, breaking the potential `meta ↔ federation` circular import.
- **Stateful tracking**: Maintains a snapshot of each server's base params and hit rate between calls to compute deltas.
- **Confidence-scaled deltas**: The learning rate applied to the delta vector is proportional to observed confidence, so noisy or low-quality domains contribute smaller updates.

---

## Configuration

### `DomainServerMetaConfig`

| Param | Default | Purpose |
|-------|---------|---------|
| `hit_rate_weight` | `0.6` | Weight on hit_rate in confidence formula |
| `memory_weight` | `0.4` | Weight on log-scaled memory_size in confidence formula |
| `memory_reference` | `64` | Reference memory size (denominator for memory contribution) |
| `min_confidence` | `0.05` | Floor on confidence to avoid zero deltas |
| `meta_lr_min` | `0.05` | Minimum meta learning rate |
| `meta_lr_max` | `0.2` | Maximum meta learning rate |
| `performance_gain_boost` | `0.05` | Additional confidence per unit of positive `performance["gain"]` |

---

## Confidence Formula

```
raw_confidence = hit_rate_weight × hit_rate
              + memory_weight × log1p(memory_size) / log1p(memory_reference)

if performance is provided and performance["gain"] > 0:
    raw_confidence += performance_gain_boost × performance["gain"]

confidence = max(raw_confidence, min_confidence)   # floor applied
```

`confidence` is bounded to `[min_confidence, 1.0]`.

---

## Delta Vector Formula

The delta vector captures the parameter change since the last observation, scaled by a confidence-proportional learning rate:

```
meta_lr = meta_lr_min + (meta_lr_max − meta_lr_min) × confidence
delta[key] = meta_lr × (current_base_params[key] − prev_base_params[key])
```

On the first observation for a domain server, `prev_base_params` is all zeros (cold start).

---

## API

### `DomainServerMeta(config=None)`

| Attribute | Description |
|-----------|-------------|
| `n_domains_tracked` | Number of unique basket IDs observed at least once |

#### `observe(server, performance=None) → DomainMetaUpdate`

Observe a single domain server and return a meta update.

```python
update = meta.observe(server, performance={"gain": 0.8})
```

The returned `DomainMetaUpdate` contains:
- `domain_id` — from `server.domain_id`
- `vector` — confidence-scaled parameter delta (empty array if `base_params` is empty)
- `keys` — sorted parameter names
- `confidence` — computed from hit_rate + memory_size + optional performance
- `score_delta` — change in `hit_rate` since last observe
- `timestamp` — `time.time()` at call time

#### `observe_registry(registry, performance_map=None) → List[DomainMetaUpdate]`

Convenience wrapper that observes all servers in a `DomainServerRegistry`:

```python
updates = meta.observe_registry(registry, performance_map={"basket_hc": {"gain": 0.85}})
```

`performance_map` is keyed by `basket_id`. Servers not in the map receive `performance=None`.

#### `status() → Dict[str, Any]`

Returns telemetry:

```python
{
    "n_domains_tracked": 3,
    "basket_ids": ["basket_hc", "basket_fin", "basket_retail"],
}
```

---

## Usage Example

```python
from scarcity.meta.domain_server_meta import DomainServerMeta, DomainServerMetaConfig
from scarcity.federation.domain_server import DomainServerRegistry
import numpy as np

# Set up servers (via HierarchicalFederation or directly)
registry = DomainServerRegistry()
s1 = registry.get_or_create("healthcare", "b1")
s1.update_base_params(np.array([0.4, 0.8], dtype=np.float32), ["gain", "tau"], reward=0.8)

# Create the observer
meta = DomainServerMeta(DomainServerMetaConfig(min_confidence=0.05))

# First observation — delta from zero baseline
update = meta.observe(s1)
print(update.domain_id)    # "healthcare"
print(update.confidence)   # ≥ 0.05
print(update.keys)         # ["gain", "tau"]
print(update.vector)       # non-zero delta array

# Update the server, observe again — delta reflects the change
s1.update_base_params(np.array([0.7, 0.9], dtype=np.float32), ["gain", "tau"], reward=0.9)
update2 = meta.observe(s1)
print(update2.vector)      # delta captures the 0.3, 0.1 change (scaled by meta_lr)

# Registry-wide observation
updates = meta.observe_registry(registry)
print(len(updates))        # 1

print(meta.n_domains_tracked)   # 1
print(meta.status())
```

---

## Integration

`DomainServerMeta` is instantiated inside `HierarchicalFederation` as `self.domain_server_meta`. It is called once per `run_full_meta_round()` call:

```
HierarchicalFederation.run_full_meta_round()
        │
        └─ DomainServerMeta.observe_registry(self.domain_registry, performance_map)
                │
                └─► List[DomainMetaUpdate]
                        │
                        └─► CrossDomainMetaLearner.aggregate(updates)
```

No circular imports: `DomainServerMeta` uses duck typing and never imports from `scarcity.federation`.

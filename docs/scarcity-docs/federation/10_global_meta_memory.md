# GlobalMetaMemory — Cross-Domain Episodic Prior Store

`scarcity/federation/global_meta_memory.py`

---

## Purpose

`GlobalMetaMemory` is the **cross-domain episodic memory** used by the federated meta-learner. After each federation round, it:

1. Reads the base models from all registered `DomainServer` instances
2. Computes an aggregate parameter vector via REPTILE blending
3. Stores the aggregation as an episode (context + outcome)
4. Answers prior queries from `CrossDomainMetaLearner` and new domains via `suggest_prior()`

This is what makes the system "learn to learn": each round's aggregation outcome is remembered, and future aggregations are initialized from the best matching historical configuration rather than cold defaults.

---

## Configuration

### `GlobalMetaMemoryConfig`

| Param | Default | Purpose |
|-------|---------|---------|
| `memory_capacity` | `512` | Max episodes stored (FIFO eviction) |
| `min_domains_for_aggregate` | `2` | Minimum active domains required to store an episode |
| `retrieval_top_k` | `5` | Top-K nearest episodes returned for prior blending |
| `context_dim` | `8` | Internal context embedding dimension |
| `blend_alpha` | `0.7` | Weight given to retrieved episodic prior vs current aggregate |

---

## API

### `GlobalMetaMemory(config=None)`

| Attribute | Type | Description |
|-----------|------|-------------|
| `memory_size` | `int` | Number of episodes currently stored |

#### `aggregate(registry, performance_map=None) → Dict[str, float]`

Main aggregation call. Reads `base_params` from each `DomainServer` in `registry`, computes a REPTILE-blended global parameter vector, stores the episode, and returns the global params as a flat dict.

```
global_params[key] = blend_alpha × retrieved_prior[key]
                   + (1 − blend_alpha) × mean(domain_base_params[key])
```

When `performance_map` is provided (`{basket_id: {metric: value}}`), domain contributions are weighted by their performance metrics.

Returns an empty dict if fewer than `min_domains_for_aggregate` domains are active.

#### `suggest_prior(domain_id, context) → Optional[Dict[str, float]]`

Retrieves a parameter prior for a given domain and context. Used by `CrossDomainMetaLearner._query_prior()`.

- Queries episodic memory for the `retrieval_top_k` most similar past episodes
- Blends their parameter vectors weighted by similarity
- Returns `None` when `memory_size == 0`

The context dict typically contains: `{"n_domains": float, "confidence_mean": float, "score_delta_mean": float}` plus any extra keys forwarded from the caller.

---

## Episode Structure

Each stored episode has:

```python
{
    "context": {
        "n_domains": 3.0,
        "confidence_mean": 0.72,
        "score_delta_mean": 0.04,
    },
    "params": {
        "gain": 0.45,
        "tau": 0.87,
        # ... other aggregated keys
    },
}
```

Episodes are stored in an internal `EpisodicMemory` buffer (same module used by `DomainServer`). `memory_size` delegates to `len(internal_memory)`.

---

## Usage Example

```python
from scarcity.federation.global_meta_memory import GlobalMetaMemory, GlobalMetaMemoryConfig
from scarcity.federation.domain_server import DomainServer, DomainServerRegistry
import numpy as np

# Set up some domain servers with trained base params
registry = DomainServerRegistry()
s1 = registry.get_or_create("healthcare", "b1")
s2 = registry.get_or_create("finance", "b2")

s1.update_base_params(np.array([0.4, 0.8], dtype=np.float32), ["gain", "tau"], reward=0.8)
s2.update_base_params(np.array([0.6, 0.9], dtype=np.float32), ["gain", "tau"], reward=0.7)

# Create global memory and run an aggregation
gmm = GlobalMetaMemory(GlobalMetaMemoryConfig(memory_capacity=128))
global_params = gmm.aggregate(registry)
# {"gain": 0.35, "tau": 0.595}  (blended across domains)

print(gmm.memory_size)  # 1 (one episode stored)

# After many rounds, query for a prior for a new domain
prior = gmm.suggest_prior("retail", {"n_domains": 2.0, "confidence_mean": 0.65})
# Returns dict of param → float, or None if memory is empty
```

---

## Integration

`GlobalMetaMemory` is instantiated inside `HierarchicalFederation` and wired to:

- `CrossDomainMetaLearner` — receives it as `global_meta_memory` constructor arg; calls `suggest_prior()` each round
- `HierarchicalFederation.run_full_meta_round()` — calls `aggregate(registry, performance_map)` and returns the result as `"global_params"`
- `HierarchicalFederation.suggest_prior()` — thin wrapper over `gmm.suggest_prior()`

No circular imports: `GlobalMetaMemory` imports `DomainServerRegistry` from the same federation package, not from `meta`.

# run_full_meta_round — Integrated Meta-Learning Round

`scarcity/federation/hierarchical.py` · `HierarchicalFederation.run_full_meta_round()`

---

## Purpose

`run_full_meta_round()` is the single-call entry point for the **Phase 5 federated meta-learning pipeline**. It combines three previously separate operations into one atomic round:

1. **Observe** — `DomainServerMeta` reads all registered domain servers and produces `DomainMetaUpdate` objects
2. **Cross-domain aggregate** — `CrossDomainMetaLearner` blends the updates with a memory-backed prior to produce a global update vector
3. **Global memory update** — `GlobalMetaMemory` aggregates domain base params and stores the episode

Each call advances the system's meta-learning state: the global memory grows, memory quality rises, and subsequent rounds produce increasingly well-informed cross-domain priors.

---

## Signature

```python
def run_full_meta_round(
    self,
    performance_map: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
```

### Parameters

| Param | Type | Description |
|-------|------|-------------|
| `performance_map` | `dict` or `None` | Per-basket performance metrics, e.g. `{"basket_hc": {"gain": 0.8, "stability": 0.9}}`. Forwarded to both `DomainServerMeta.observe_registry()` and `GlobalMetaMemory.aggregate()`. |

### Return Value

```python
{
    "global_params": Dict[str, float],        # Global parameter aggregate from GlobalMetaMemory
    "cross_domain":  Tuple[                   # Output of CrossDomainMetaLearner
        np.ndarray,                           #   blended update vector
        List[str],                            #   parameter keys
        Dict[str, Any],                       #   meta: source, memory_quality, prior_keys_matched, ...
    ],
    "n_updates": int,                         # Number of domain servers that produced updates
}
```

---

## Internal Pipeline

```
HierarchicalFederation.run_full_meta_round(performance_map)
        │
        ├─ Step 1: DomainServerMeta.observe_registry(domain_registry, performance_map)
        │          → List[DomainMetaUpdate]   (one per active domain server)
        │          Each update carries: vector, keys, confidence, score_delta
        │
        ├─ Step 2: CrossDomainMetaLearner.aggregate(updates)
        │          → (cross_vec, cross_keys, cross_meta)
        │
        │          If GlobalMetaMemory has episodes:
        │            quality = min(memory_size / reference_capacity, max_memory_quality)
        │            prior   = GlobalMetaMemory.suggest_prior("cross_domain", context)
        │            result  = (1 − quality) × fallback_vec + quality × prior_vec
        │            meta["source"] = "memory_backed"
        │          Else:
        │            result  = trimmed-mean of update vectors
        │            meta["source"] = "fallback"
        │
        └─ Step 3: GlobalMetaMemory.aggregate(domain_registry, performance_map)
                   → Dict[str, float]   global_params
                   Stores episode in memory for future suggest_prior() calls
```

---

## Memory Quality Progression

Memory quality (`meta["memory_quality"]`) grows from 0 to `max_memory_quality` (default 0.8) as episodes accumulate:

```
Round  memory_size  quality (cap=256)   source
─────────────────────────────────────────────────
  1        0           0.000            fallback
  5        4           0.016            fallback (quality > 0 but no prior match)
 25       24           0.094            memory_backed
 50       48           0.188            memory_backed
128      128           0.500            memory_backed
256+     256           0.800            memory_backed (capped)
```

At quality 0, the result is identical to the fallback trimmed-mean. At quality 0.8, the prior contributes 80% of the blended output.

---

## New Instance Attributes

`run_full_meta_round()` uses two attributes added to `HierarchicalFederation`:

| Attribute | Type | Config field |
|-----------|------|--------------|
| `self.domain_server_meta` | `DomainServerMeta` | `HierarchicalFederationConfig.domain_server_meta` |
| `self.cross_domain_learner` | `CrossDomainMetaLearner` | `HierarchicalFederationConfig.cross_domain_meta` |

Both are instantiated in `__init__` with defaults if no config is provided. `cross_domain_learner` receives `self.global_meta_memory` so it can query episodes from the current federation instance.

---

## Usage Example

```python
from scarcity.federation import HierarchicalFederation
import numpy as np

fed = HierarchicalFederation()

# Register domain servers and load them with data
s1 = fed.get_domain_server("healthcare", "basket_hc")
s2 = fed.get_domain_server("finance",    "basket_fin")

s1.update_base_params(np.array([0.4, 0.8], dtype=np.float32), ["gain", "tau"], reward=0.8)
s2.update_base_params(np.array([0.6, 0.9], dtype=np.float32), ["gain", "tau"], reward=0.7)

# Round 1 — no memory yet; cross_domain falls back to trimmed mean
result = fed.run_full_meta_round()
print(result["n_updates"])                   # 2
print(result["cross_domain"][2]["source"])   # "fallback"
print(result["cross_domain"][2]["memory_quality"])  # 0.0

# Rounds 2–N — memory accumulates; cross_domain becomes memory-backed
for _ in range(10):
    result = fed.run_full_meta_round()

print(result["cross_domain"][2]["source"])           # "memory_backed"
print(result["cross_domain"][2]["memory_quality"])   # > 0.0
print(result["global_params"])                       # {"gain": ..., "tau": ...}

# With performance map
perf = {
    "basket_hc":  {"gain": 0.85, "stability": 0.92},
    "basket_fin": {"gain": 0.71, "stability": 0.88},
}
result = fed.run_full_meta_round(performance_map=perf)
```

---

## Backwards Compatibility

`run_full_meta_round()` is **additive** — the existing `run_meta_round()` method is unchanged and continues to work. The two methods are independent:

| Method | Introduced | Uses |
|--------|-----------|------|
| `run_meta_round()` | Pre-Phase 5 | `GlobalMetaMemory.aggregate()` only |
| `run_full_meta_round()` | Phase 5 | All three components: `DomainServerMeta` + `CrossDomainMetaLearner` + `GlobalMetaMemory` |

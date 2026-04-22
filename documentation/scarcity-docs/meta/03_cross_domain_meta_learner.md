# CrossDomainMetaLearner — Memory-Backed Cross-Domain Aggregation

`scarcity/meta/cross_meta.py` (Phase 5b addition)

---

## Purpose

`CrossDomainMetaLearner` is the **Phase 5b meta-learner** — an upgrade to the existing `CrossDomainMetaAggregator` that adds **episodic memory backing**. It implements "learning to learn" at the cross-domain level: as the system accumulates aggregation episodes in `GlobalMetaMemory`, subsequent aggregations are initialized from the best historical configuration rather than a cold statistical baseline.

### Two-Class Design

`cross_meta.py` contains two aggregators with different roles:

| Class | Phase | Memory | Description |
|-------|-------|--------|-------------|
| `CrossDomainMetaAggregator` | 1 / fallback | None | Pure statistical aggregation (trimmed mean or median). Always available. |
| `CrossDomainMetaLearner` | 5b | `GlobalMetaMemory` | Memory-backed blending. Falls back to `CrossDomainMetaAggregator` when memory is empty. |

---

## Configuration

### `CrossDomainMetaLearnerConfig`

| Param | Default | Purpose |
|-------|---------|---------|
| `fallback_method` | `"trimmed_mean"` | Aggregation method for the fallback (`"trimmed_mean"` or `"median"`) |
| `fallback_trim_alpha` | `0.1` | Trim fraction for trimmed-mean fallback |
| `fallback_min_confidence` | `0.05` | Minimum confidence for a domain update to be included |
| `memory_reference_capacity` | `256` | Episode count at which `memory_quality` saturates toward `max_memory_quality` |
| `min_memory_quality` | `0.0` | Floor on memory quality (0 = pure fallback when memory empty) |
| `max_memory_quality` | `0.8` | Ceiling on memory quality — fallback always contributes ≥ 20% |

---

## Memory Quality

Memory quality controls how much the historical prior influences the result:

```
memory_size  = global_meta_memory.memory_size   (0 when empty)
quality      = clip(memory_size / memory_reference_capacity,
                    min_memory_quality,
                    max_memory_quality)
```

| memory_size | quality (cap=256, max=0.8) | Effect |
|-------------|---------------------------|--------|
| 0 | 0.0 | Pure fallback — identical to `CrossDomainMetaAggregator` |
| 32 | 0.125 | 87.5% fallback + 12.5% prior |
| 128 | 0.5 | 50% fallback + 50% prior |
| 256+ | 0.8 | 20% fallback + 80% prior (capped) |

The ceiling (`max_memory_quality = 0.8`) ensures the system retains some live signal from current updates even after saturating memory.

---

## Blend Formula

```
fallback_vec = CrossDomainMetaAggregator.aggregate(updates)   # trimmed mean
prior_vec    = GlobalMetaMemory.suggest_prior("cross_domain", context)

result_vec   = (1 − quality) × fallback_vec
             + quality       × prior_vec_aligned_to_keys
```

`prior_vec` is aligned to the current key set by zero-filling any keys absent from the prior.

---

## API

### `CrossDomainMetaLearner(config=None, global_meta_memory=None)`

`global_meta_memory` is typed as `Optional[Any]` to avoid circular imports. It accepts any object with a `.memory_size` property (or `__len__`) and a `.suggest_prior(domain_id, context)` method — both satisfied by `GlobalMetaMemory`.

#### `aggregate(updates, context=None) → Tuple[np.ndarray, List[str], Dict[str, Any]]`

```python
result_vec, keys, meta = learner.aggregate(updates, context={"custom": 0.5})
```

**Parameters:**
- `updates` — `Sequence[DomainMetaUpdate]`, one per active domain server
- `context` — optional extra dict merged into the memory query context

**Returns:**
- `result_vec` — blended update vector (`np.ndarray`, `float32`)
- `keys` — parameter names corresponding to each vector position
- `meta` — dict with aggregation diagnostics:

| Key | Type | Description |
|-----|------|-------------|
| `source` | `str` | `"memory_backed"` or `"fallback"` |
| `memory_quality` | `float` | Blend weight used |
| `prior_keys_matched` | `int` | Keys shared between prior and current key set |
| `participants` | `int` | Number of updates that passed confidence filter |
| `method` | `str` | `"trimmed_mean"` or `"median"` |
| `confidence_mean` | `float` | Mean confidence of participating updates |

**Fallback conditions** (result identical to `CrossDomainMetaAggregator`):
- `global_meta_memory is None`
- `quality == 0.0` (memory is empty)
- `suggest_prior()` returns `None`
- `keys` is empty (no updates passed confidence filter)

---

## Usage Example

```python
from scarcity.meta.cross_meta import CrossDomainMetaLearner, CrossDomainMetaLearnerConfig
from scarcity.federation.global_meta_memory import GlobalMetaMemory, GlobalMetaMemoryConfig
from scarcity.federation.domain_server import DomainServerRegistry
import numpy as np

# Build memory with some episodes first
registry = DomainServerRegistry()
s1 = registry.get_or_create("healthcare", "b1")
s2 = registry.get_or_create("finance",    "b2")
s1.update_base_params(np.array([0.4, 0.8], dtype=np.float32), ["gain", "tau"], reward=0.8)
s2.update_base_params(np.array([0.6, 0.9], dtype=np.float32), ["gain", "tau"], reward=0.7)

gmm = GlobalMetaMemory(GlobalMetaMemoryConfig(memory_capacity=64))
for _ in range(10):
    gmm.aggregate(registry)

# Learner with low reference capacity so memory fills quickly
cfg = CrossDomainMetaLearnerConfig(memory_reference_capacity=10)
learner = CrossDomainMetaLearner(cfg, global_meta_memory=gmm)

# Simulate domain updates
from scarcity.meta.domain_server_meta import DomainServerMeta
observer = DomainServerMeta()
updates = observer.observe_registry(registry)

result_vec, keys, meta = learner.aggregate(updates)
print(meta["source"])          # "memory_backed" (10 episodes > 0)
print(meta["memory_quality"])  # 1.0 → clipped to 0.8
print(meta["prior_keys_matched"])  # number of keys shared with prior

# No memory — identical to CrossDomainMetaAggregator
bare_learner = CrossDomainMetaLearner()
vec2, keys2, meta2 = bare_learner.aggregate(updates)
print(meta2["source"])         # "fallback"
print(meta2["memory_quality"]) # 0.0
```

---

## Learning Progression

As rounds accumulate, the learner gets progressively better at initializing from historical optima:

```
Round 1:   memory empty  → pure trimmed-mean of current deltas
Round 10:  memory sparse → 95% trimmed-mean + 5% prior
Round 50:  memory filling → 75% trimmed-mean + 25% prior
Round 256: memory full   → 20% trimmed-mean + 80% prior (stable warm-start)
```

This is "learning to learn": the cross-domain aggregation rule improves over time without changing model architecture — it simply blends current observations with accumulated wisdom from past episodes.

---

## Integration

`CrossDomainMetaLearner` is instantiated in `HierarchicalFederation` as `self.cross_domain_learner`, receiving `self.global_meta_memory` at construction time. It is called in `run_full_meta_round()` after `DomainServerMeta.observe_registry()`.

See [12_run_full_meta_round.md](./../../federation/12_run_full_meta_round.md) for the full pipeline.

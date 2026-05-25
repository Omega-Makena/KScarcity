# Meta Module Utilities

This page documents the individual components of the meta module.

---

## integrative_meta.py — Tier-5 Meta Governance

### `MetaState`

State vector for the meta layer:

```python
@dataclass
class MetaState:
    tau: float = 0.9           # Decay scaling
    gamma_diversity: float = 0.3  # Exploration weight
    g_min: float = 0.01        # Minimum gain
    lambda_ci: float = 0.5     # CI width
    tier2_enabled: bool = True
    tier3_topk: int = 5
    ema_reward: float = 0.0
    last_reward: float = 0.0
    cooldowns: Dict[str, int] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    decision_count: int = 0
    rollback_count: int = 0
```

### `MetaIntegrativeLayer`

Main governance class:

```python
layer = MetaIntegrativeLayer(config=IntegrativeMetaConfig())

result = layer.update({
    "gain_p50": 0.15,
    "stability_mean": 0.8,
    "vram_high": False
})
# result["policy_updates"]: Adjusted parameters
# result["resource_hint"]: Resource scaling hints
# result["scoring"]: Meta-reward breakdown
```

**Key methods**:
- `update(telemetry)`: Process telemetry, return adjustments
- `_compute_reward(telemetry)`: Calculate meta-reward
- `_apply_policies(...)`: Adjust knobs based on reward
- `_safety_checks(...)`: Check for rollback conditions

### `MetaSupervisor`

Runtime bridge to EventBus:

```python
supervisor = MetaSupervisor(bus=event_bus)
supervisor.start()
# Subscribes to telemetry, calls MetaIntegrativeLayer
```

---

## meta_learning.py — Agent Orchestrator

### `MetaLearningConfig`

Unified configuration:

```python
@dataclass
class MetaLearningConfig:
    domain: DomainMetaConfig
    cross: CrossMetaConfig
    optimizer: MetaOptimizerConfig
    scheduler: MetaSchedulerConfig
    validator: MetaValidatorConfig
    storage: MetaStorageConfig
```

### `MetaLearningAgent`

Central orchestrator:

```python
agent = MetaLearningAgent(bus=event_bus, config=config)
await agent.start()

# Subscribes to:
# - "processing_metrics": Trigger meta updates
# - "federation.policy_pack": Domain knowledge
```

**Workflow**:
1. Receive domain policy packs
2. Validate and queue updates
3. On trigger: aggregate across domains
4. Apply Reptile optimization
5. Broadcast updated priors

---

## domain_meta.py — Domain Learning

### `DomainMetaLearner`

Tracks domain-specific priors:

```python
learner = DomainMetaLearner(config=DomainMetaConfig())

update = learner.observe(
    domain_id="healthcare",
    metrics={"gain": 0.2, "stability": 0.9},
    params={"tau": 0.91}
)
```

### `DomainMetaUpdate`

Structure for domain updates:

```python
@dataclass
class DomainMetaUpdate:
    domain_id: str
    vector: np.ndarray
    reward: float
    confidence: float
    timestamp: float
```

---

## cross_meta.py — Cross-Domain Aggregation

### `CrossDomainMetaAggregator`

Combines domain updates:

```python
aggregator = CrossDomainMetaAggregator(config=CrossMetaConfig())

agg_vector, keys, meta = aggregator.aggregate(updates=[...])
```

**Features**:
- Weighted by domain confidence
- Entropy-based diversity balancing
- Outlier filtering

---

## optimizer.py — Reptile Optimizer

### `OnlineReptileOptimizer`

Meta-optimization:

```python
optimizer = OnlineReptileOptimizer(config=MetaOptimizerConfig())

updated_prior = optimizer.apply(
    aggregated_vector=np.array([...]),
    keys=["tau", "g_min", "gamma"],
    reward=0.85,
    drg_profile={"vram_high": False}
)

if optimizer.should_rollback(0.3):  # Low reward
    prior = optimizer.rollback()
```

**Reptile algorithm**:
```
prior = prior + lr * (aggregated - prior)
```

---

## scheduler.py — Update Scheduling

### `MetaScheduler`

Controls update timing:

```python
scheduler = MetaScheduler(config=MetaSchedulerConfig(
    update_interval_windows=10,
    latency_target_ms=50.0
))

scheduler.record_window()

if scheduler.should_update(metrics):
    # Trigger meta update
```

---

## storage.py — Prior Persistence

### `MetaStorageManager`

Saves and loads priors:

```python
storage = MetaStorageManager(config=MetaStorageConfig(
    root=Path("./meta_storage")
))

# Save
storage.save_prior(prior_dict)

# Load
prior = storage.load_prior()
```

---

## validator.py — Update Validation

### `MetaPacketValidator`

Validates updates before aggregation:

```python
validator = MetaPacketValidator(config=MetaValidatorConfig())

is_valid = validator.validate_update(update)
```

**Checks**:
- Vector dimensions match
- Values in valid ranges
- No NaN/Inf

---

## telemetry_hooks.py — Metrics Publishing

### `build_meta_metrics_snapshot`

Constructs metrics payload:

```python
snapshot = build_meta_metrics_snapshot(
    reward=0.85,
    update_rate=0.15,
    gain=0.12,
    confidence=0.9,
    drift_score=0.05,
    latency_ms=45.0,
    storage_mb=12.5
)
```

### `publish_meta_metrics`

Sends to EventBus:

```python
await publish_meta_metrics(bus, snapshot)
# Publishes to "meta.telemetry" topic
```

---

## Index

| File | Purpose |
|------|---------|
| `integrative_meta.py` | Tier-5 meta governance |
| `integrative_config.py` | Configuration classes |
| `meta_learning.py` | Agent orchestrator |
| `domain_meta.py` | Domain-specific learning |
| `cross_meta.py` | Cross-domain aggregation (fallback + memory-backed) |
| `optimizer.py` | Reptile meta-optimization |
| `scheduler.py` | Update timing |
| `storage.py` | Prior persistence |
| `validator.py` | Update validation |
| `telemetry_hooks.py` | Metrics publishing |
| `domain_server_meta.py` | DomainServerMeta — Phase 5a bridge |
| `encoder.py` | ContextEncoder |
| `memory.py` | EpisodicMemory |
| `adaptation.py` | AdaptationEngine |

---

## domain_server_meta.py — Federation-to-Meta Bridge (Phase 5a)

### `DomainServerMetaConfig`

```python
@dataclass
class DomainServerMetaConfig:
    hit_rate_weight: float = 0.6
    memory_weight: float = 0.4
    memory_reference: int = 64
    min_confidence: float = 0.05
    meta_lr_min: float = 0.05
    meta_lr_max: float = 0.2
    performance_gain_boost: float = 0.05
```

### `DomainServerMeta`

Observes `DomainServer` instances via duck typing and converts their state to `DomainMetaUpdate` objects:

```python
meta = DomainServerMeta(config=DomainServerMetaConfig())

# Single server
update = meta.observe(server, performance={"gain": 0.8})
# update.confidence, update.vector, update.keys, update.score_delta

# Whole registry
updates = meta.observe_registry(registry, performance_map={"basket_hc": {"gain": 0.8}})

# Telemetry
print(meta.n_domains_tracked)
print(meta.status())  # {"n_domains_tracked": N, "basket_ids": [...]}
```

**Confidence formula:**
```
confidence = hit_rate_w × hit_rate
           + mem_w × log1p(memory_size) / log1p(memory_reference)
           + gain_boost × max(0, performance["gain"])
confidence = max(confidence, min_confidence)
```

**Delta formula:**
```
meta_lr = meta_lr_min + (meta_lr_max − meta_lr_min) × confidence
delta   = meta_lr × (current_base_params − prev_base_params)
```

---

## cross_meta.py — Phase 5b: CrossDomainMetaLearner

### `CrossDomainMetaLearnerConfig`

```python
@dataclass
class CrossDomainMetaLearnerConfig:
    fallback_method: str = "trimmed_mean"
    fallback_trim_alpha: float = 0.1
    fallback_min_confidence: float = 0.05
    memory_reference_capacity: int = 256
    min_memory_quality: float = 0.0
    max_memory_quality: float = 0.8
```

### `CrossDomainMetaLearner`

Memory-backed wrapper around `CrossDomainMetaAggregator`:

```python
learner = CrossDomainMetaLearner(
    config=CrossDomainMetaLearnerConfig(memory_reference_capacity=64),
    global_meta_memory=gmm,  # GlobalMetaMemory instance
)

result_vec, keys, meta = learner.aggregate(updates)
# meta["source"]            → "memory_backed" | "fallback"
# meta["memory_quality"]    → blend weight used (0.0 – 0.8)
# meta["prior_keys_matched"] → int
```

**Blend formula:**
```
quality     = clip(memory_size / reference_capacity, 0, max_memory_quality)
prior_vec   = memory.suggest_prior("cross_domain", context)
result_vec  = (1 − quality) × fallback_vec + quality × prior_vec
```

Falls back to pure `CrossDomainMetaAggregator` when `global_meta_memory` is `None`, memory is empty, or `suggest_prior()` returns `None`.

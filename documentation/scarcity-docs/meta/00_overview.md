# Scarcity Meta Module — Overview

The **meta module** provides **meta-learning** and **meta-governance** capabilities. It learns to learn — adapting hyperparameters, aggregating knowledge across domains, and optimizing the discovery system itself.

---

## Purpose

As the system runs across multiple domains and time periods, patterns emerge:
- Some hyperparameters work better for certain data types
- Domain-specific insights can transfer to other domains
- System behavior should adapt based on performance

The meta module:
1. **Learns hyperparameter priors** across domains
2. **Aggregates meta-knowledge** for transfer learning
3. **Governs system behavior** with rule-based policies
4. **Optimizes discovery** parameters online

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │      MetaIntegrativeLayer       │
                    │    (Tier-5 Meta Governance)     │
                    │                                 │
                    │  ┌─────────────────────────┐   │
                    │  │   Policy Engine         │   │
                    │  │   - Reward computation  │   │
                    │  │   - Hyperparameter      │   │
                    │  │     adjustment          │   │
                    │  │   - Rollback safety     │   │
                    │  └─────────────────────────┘   │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      MetaLearningAgent          │
                    │                                 │
                    │  ┌───────┐  ┌───────┐  ┌─────┐ │
                    │  │Domain │  │Cross  │  │Optim│ │
                    │  │Meta   │◄─│Domain │◄─│izer │ │
                    │  │Learner│  │Agg    │  │     │ │
                    │  └───────┘  └───────┘  └─────┘ │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Domain A  │   │ Domain B  │   │ Domain C  │
            │ (Engine)  │   │ (Engine)  │   │ (Engine)  │
            └───────────┘   └───────────┘   └───────────┘
```

---

## Key Components

### MetaIntegrativeLayer (`integrative_meta.py`)

The **Tier-5 meta-governance layer**:
- Computes meta-reward from system telemetry
- Adjusts hyperparameters (tau, gamma, g_min, etc.)
- Applies cooldowns to prevent oscillation
- Performs safety rollbacks when performance drops

### MetaLearningAgent (`meta_learning.py`)

The **central orchestrator** that:
- Subscribes to domain policy packs
- Aggregates updates from multiple domains
- Applies Reptile meta-optimization
- Broadcasts updated global priors

### DomainMetaLearner (`domain_meta.py`)

Learns domain-specific hyperparameter preferences:
- Tracks performance per domain
- Builds domain-specific priors
- Identifies good starting points for new domains

### CrossDomainMetaAggregator (`cross_meta.py`)

Aggregates knowledge across domains:
- Weighted combination of domain updates
- Entropy-based diversity balancing
- Outlier filtering

### DomainServerMeta (`domain_server_meta.py`) — Phase 5a

Bridges the federation layer and the meta pipeline:
- Observes `DomainServer` state via duck typing (no circular imports)
- Converts hit_rate, memory_size, and base_params into `DomainMetaUpdate` objects
- Confidence-scaled delta vectors: higher-quality servers send stronger signals

### CrossDomainMetaLearner (`cross_meta.py`) — Phase 5b

Memory-backed upgrade to `CrossDomainMetaAggregator`:
- Queries `GlobalMetaMemory` for a cross-domain prior each round
- Blends statistical fallback with historical prior weighted by memory quality
- Memory quality grows 0 → 0.8 as episodes accumulate (learning to learn)

### OnlineReptileOptimizer (`optimizer.py`)

Meta-optimization algorithm:
- Reptile-style gradient updates
- EMA reward tracking
- Automatic rollback on performance drops

---

## Core Concepts

### Meta-Reward

Computed from system telemetry:

```python
reward = (
    w_gain * normalized_gain +
    w_stability * stability_score -
    w_resource * resource_penalty
)
```

**Components**:
- `gain`: How much predictive gain hypotheses are producing
- `stability`: How stable relationships are over time
- `resource`: Cost of computation (VRAM, latency)

### Hyperparameter Knobs

The meta layer adjusts:

| Knob | Purpose | Bounds |
|------|---------|--------|
| `tau` | Decay rate scaling | [0.8, 0.99] |
| `gamma_diversity` | Exploration vs. exploitation | [0.1, 0.5] |
| `g_min` | Minimum gain threshold | [0.001, 0.1] |
| `lambda_ci` | Confidence interval width | [0.3, 0.7] |
| `tier2_enabled` | Enable layer 2 | Boolean |
| `tier3_topk` | Top-K paths to explore | [3, 10] |

### Cooldowns

After adjusting a knob, it enters cooldown:
- Prevents oscillation
- Allows change to take effect
- Decay over N decisions

### Rollback

If reward drops significantly after a change:
1. Detect via EMA comparison
2. Revert to previous snapshot
3. Increment rollback counter
4. Apply longer cooldown

---

## File Guide

| File | Purpose |
|------|---------|
| `integrative_meta.py` | MetaIntegrativeLayer — Tier-5 governance |
| `integrative_config.py` | Configuration dataclasses |
| `meta_learning.py` | MetaLearningAgent orchestrator |
| `domain_meta.py` | Domain-specific learning |
| `cross_meta.py` | CrossDomainMetaAggregator + CrossDomainMetaLearner (Phase 5b) |
| `optimizer.py` | OnlineReptileOptimizer |
| `scheduler.py` | Meta update scheduling |
| `storage.py` | Prior persistence |
| `validator.py` | Update validation |
| `telemetry_hooks.py` | Metrics publishing |
| `domain_server_meta.py` | DomainServerMeta — federation-to-meta bridge (Phase 5a) |
| `encoder.py` | ContextEncoder — context embedding |
| `memory.py` | EpisodicMemory — local adaptation episode buffer |
| `adaptation.py` | AdaptationEngine — context-driven parameter retrieval |

---

## Integration

### With Engine Module

```python
# Engine publishes telemetry
bus.publish("processing_metrics", {
    "gain_p50": 0.15,
    "stability_mean": 0.8,
    "vram_utilization": 0.6
})

# Meta layer subscribes and adjusts
meta_layer.update(telemetry)
# Returns: {"policy_updates": {"tau": 0.92}, ...}
```

### With Federation Module

```python
# Federation publishes policy packs
bus.publish("federation.policy_pack", {
    "domain_id": "healthcare",
    "metrics": {...},
    "controller": {"tau": 0.91}
})

# MetaLearningAgent aggregates across domains
agent._handle_policy_pack(topic, payload)
```

---

## Usage Example

```python
from scarcity.meta import MetaIntegrativeLayer, MetaLearningAgent
from scarcity.runtime import get_bus

# Setup
bus = get_bus()
meta_layer = MetaIntegrativeLayer()
agent = MetaLearningAgent(bus=bus)

# Start
await agent.start()

# Meta layer updates (called periodically)
result = meta_layer.update({
    "gain_p50": 0.15,
    "stability_mean": 0.82,
    "vram_high": False
})

print(result["policy_updates"])
# {"tau": 0.92, "g_min": 0.008}
```

---

## Edge Cases

### Cold Start

No prior data:
- Use default conservative parameters
- High exploration (gamma_diversity)
- Quick adaptation expected

### Performance Drop

Sustained low reward:
- Multiple rollbacks triggered
- Eventually converges to safe defaults
- Logs warnings for investigation

### Domain Mismatch

Very different domains:
- Cross-domain aggregation weighted low
- Domain-specific priors dominate
- Gradual transfer as similarity increases

---

## Phase 5 — Federated Meta-Learning Pipeline

The Phase 5 components connect the federation layer to the meta pipeline via `HierarchicalFederation.run_full_meta_round()`:

```
DomainServerRegistry  (held by HierarchicalFederation)
        │  one DomainServer per basket
        │  each holds: base_params, hit_rate, memory_size, round_id
        │
        ▼
DomainServerMeta.observe_registry(registry, performance_map)
        │  duck-typed observation (no circular imports)
        │  confidence = hit_rate_w × hit_rate + mem_w × log1p(mem) / log1p(ref)
        │  delta = meta_lr(confidence) × (curr_params − prev_params)
        │
        ▼ List[DomainMetaUpdate]   (one per domain server)
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
CrossDomainMetaLearner.aggregate(updates)         GlobalMetaMemory.aggregate(registry)
        │                                                      │
        │  quality = clip(memory_size / ref_cap, 0, 0.8)      │
        │  prior   = memory.suggest_prior("cross_domain", ctx) │
        │  result  = (1−quality)×fallback + quality×prior      │
        │                                                      │
        ▼ (blended_vec, keys, meta)                           ▼ global_params dict
        │                                                      │
        └──────────────────────┬───────────────────────────────┘
                               │
                               ▼
        run_full_meta_round() returns:
        {
            "global_params":  {param → float},
            "cross_domain":   (vec, keys, {source, memory_quality, ...}),
            "n_updates":      N,
        }
```

### Memory Quality Progression

Over successive rounds, `memory_quality` rises from 0 (pure fallback) toward 0.8 (80% prior-driven):

```
Rounds completed    memory_quality    cross_domain source
─────────────────────────────────────────────────────────
1                   0.00              fallback
~10                 0.04–0.10         fallback / transitional
~50                 0.20–0.40         memory_backed
256+                0.80 (capped)     memory_backed
```

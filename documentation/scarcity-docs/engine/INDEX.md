# Scarcity Engine Module — Documentation Index

Complete documentation for the `scarcity.engine` module — the core relationship discovery system.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and concepts |
| [01_engine.md](./01_engine.md) | MPIEOrchestrator — Event-driven pipeline |
| [02_engine_v2.md](./02_engine_v2.md) | OnlineDiscoveryEngine — Main entry point |
| [03_discovery.md](./03_discovery.md) | Hypothesis and HypothesisPool abstractions |
| [04_evaluator.md](./04_evaluator.md) | Path scoring with bootstrap CIs |
| [05_relationships.md](./05_relationships.md) | 10 core relationship types |
| [06_relationships_extended.md](./06_relationships_extended.md) | 5 advanced relationship types |
| [08_store.md](./08_store.md) | HypergraphStore — Persistent edge storage |
| [09_encoder.md](./09_encoder.md) | Feature encoding pipeline |
| [11_bandit_router.md](./11_bandit_router.md) | Multi-armed bandit path selection |
| [12_arbitration.md](./12_arbitration.md) | Hypothesis conflict resolution |
| [13_controller.md](./13_controller.md) | MetaController lifecycle state machine |
| [14_grouping.md](./14_grouping.md) | Adaptive variable clustering |
| [15_utilities.md](./15_utilities.md) | Types, utils, algorithms, and more |
| [operators/00_overview.md](./operators/00_overview.md) | Mathematical operators |

---

## Reading Order

### For Understanding (Top-Down)

1. **[Overview](./00_overview.md)** — Understand the architecture
2. **[Discovery](./03_discovery.md)** — Core abstractions
3. **[Relationships](./05_relationships.md)** — What gets discovered
4. **[Engine V2](./02_engine_v2.md)** — How to use it

### For Implementation (Bottom-Up)

1. **[Utilities](./15_utilities.md)** — Basic building blocks
2. **[Operators](./operators/00_overview.md)** — Mathematical primitives
3. **[Encoder](./09_encoder.md)** — Feature processing
4. **[Evaluator](./04_evaluator.md)** — Scoring logic
5. **[Store](./08_store.md)** — Persistence

### For Operations

1. **[Controller](./13_controller.md)** — Lifecycle management
2. **[Arbitration](./12_arbitration.md)** — Conflict handling
3. **[Bandit Router](./11_bandit_router.md)** — Exploration/exploitation
4. **[Engine](./01_engine.md)** — Production pipeline

---

## Key Concepts Summary

### The Hypothesis Survival Paradigm

Relationships are treated as **hypotheses** that must survive the stream of data:

- Each hypothesis is a statistical model (e.g., "A causes B")
- Hypotheses compete for survival based on fit, stability, confidence
- Winners are persisted as edges in the knowledge graph

### Lifecycle States

```
TENTATIVE → ACTIVE → DECAYING → DEAD
```

### Relationship Types

15 types from simple to complex:

| Category | Types |
|----------|-------|
| **Core** | Causal, Correlational, Temporal, Functional |
| **Probabilistic** | Probabilistic, Equilibrium |
| **Structural** | Compositional, Competitive, Synergistic, Structural |
| **Advanced** | Mediating, Moderating, Graph, Similarity, Logical |

### Main Entry Points

```python
# Simple use
from scarcity.engine import OnlineDiscoveryEngine
engine = OnlineDiscoveryEngine()

# Production use
from scarcity.engine.engine import MPIEOrchestrator
orchestrator = MPIEOrchestrator(bus=event_bus)
```

---

## What's NOT Documented Here

- **`tests/`** — Test files (separate test documentation)
- **Federation** — See `federation/` module docs
- **Simulation** — See `simulation/` module docs
- **Dashboard** — See backend documentation

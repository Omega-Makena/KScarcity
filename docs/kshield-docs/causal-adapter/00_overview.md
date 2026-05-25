# Causal Adapter — Overview

> `kshiked.causal_adapter` — Bridges the Scarcity causal inference engine with KShield intelligence.

---

## Purpose

The causal adapter translates knowledge graph edges discovered by the Scarcity engine into formal causal inference tasks that can be validated using DoWhy. Results are fed back as simulation parameter updates, closing the loop between discovery and action.

---

## Pipeline

```
Knowledge Graph Edge (Scarcity)
    │
    ▼
spec_builder.py  →  CausalTaskSpec
    │                 (treatment, outcome, covariates)
    ▼
policy.py        →  EstimandDecision
    │                 (select which estimands to run)
    ▼
runner.py        →  AdapterRunResult
    │                 (execute DoWhy pipeline)
    ▼
artifacts.py     →  KShieldArtifactStore
    │                 (persist results)
    ▼
integration.py   →  SimulationParameterUpdate
                     (feed back to SFCEconomy)
```

---

## File Guide

| File | Size | Purpose |
|------|------|---------|
| `__init__.py` | 1.2 KB | Public exports — all types and functions |
| `config.py` | 4.0 KB | `AdapterConfig` + sub-configs (Edge, Policy, Runtime, Selection, Simulation) |
| `types.py` | 2.7 KB | Core types: `CausalTaskSpec`, `KnowledgeGraphEdge`, `SimulationParameterUpdate`, `TaskWindow`, `BatchContext` |
| `spec_builder.py` | 3.1 KB | `build_estimand_specs()`, `build_task_specs()` — converts edges to DoWhy task specs |
| `policy.py` | 2.6 KB | `select_estimands()` — policy-based filtering of which causal tests to run |
| `runner.py` | 4.3 KB | Executes the DoWhy pipeline and produces `AdapterRunResult` |
| `dataset.py` | 2.6 KB | `load_unified_dataset()`, `segment_dataset()` — data loading for causal analysis |
| `integration.py` | 4.8 KB | `artifact_to_edge()`, `edge_to_simulation_update()` — converts results to simulation params |
| `artifacts.py` | 3.6 KB | `KShieldArtifactStore` — persists causal analysis results |

---

## Key Types

### `AdapterConfig`

Hierarchical configuration for the entire adapter:

```python
AdapterConfig
├── edges: AdapterEdgeConfig        # Edge filtering rules
├── policy: AdapterPolicyConfig     # Estimand selection policy
├── runtime: AdapterRuntimeConfig   # Execution limits (timeouts, retries)
├── selection: AdapterSelectionConfig  # Feature selection
└── simulation: AdapterSimulationConfig  # How to map results to SFC params
```

### `CausalTaskSpec`

Defines a single causal inference task:

- `treatment`: variable being intervened upon
- `outcome`: variable to measure effect on
- `covariates`: confounders to control for
- `estimand_type`: backdoor, IV, or frontdoor

### `SimulationParameterUpdate`

The final output — a parameter change to apply to `SFCEconomy`:

- `parameter_name`: e.g. `"fiscal_multiplier"`
- `new_value`: estimated causal effect
- `confidence`: p-value or confidence interval
- `source_edge`: originating knowledge graph edge

---

## Integration Example

```python
from kshiked.causal_adapter import (
    AdapterConfig,
    build_task_specs,
    select_estimands,
)

config = AdapterConfig()
edges = engine.get_knowledge_graph().edges  # from Scarcity

# Build causal specs from edges
specs = build_task_specs(edges, config)

# Filter by policy
selected = select_estimands(specs, config.policy)

# Run (via runner.py) and get simulation updates
# updates = run_causal_batch(selected, dataset, config.runtime)
```

---

*Source: `kshiked/causal_adapter/` · Last updated: 2026-02-11*

# KShield Causal Module — Overview

The `kshiked.causal` module wraps Scarcity’s causal discovery engine to generate a **directed causal graph** from the Kenya World Bank macro dataset. The output powers:

- The **Causal Network** graph in the dashboard
- The **Granger Causality Tests** panel

This module is discovery‑oriented (Granger-style predictive gain), not DoWhy inference. It is designed to update dynamically as new data arrives.

---

## What It Does

- Loads `API_KEN_DS2_en_csv_v2_14659.csv` (World Bank Kenya macro data).
- Filters indicators by coverage to avoid sparse noise.
- Runs Scarcity’s `OnlineDiscoveryEngine.initialize_v2(use_causal=True)`.
- Extracts causal edges (direction, confidence, lag, strength).
- Emits graph payloads for visualization and Granger‑style results.

---

## Dynamic Behavior

`run_dynamic_causal_discovery()` maintains an in‑memory engine:

- **No changes** → returns cached results.
- **New years appended** → incremental update (no full retrain).
- **Schema changes / backfill edits** → full retrain.

This keeps the dashboard fast while remaining up to date.

---

## Key APIs

```python
from kshiked.causal import (
    EconomicCausalConfig,
    run_dynamic_causal_discovery,
    build_granger_results_from_edges,
)

cfg = EconomicCausalConfig(
    min_coverage=0.6,
    max_indicators=40,
    min_confidence=0.7,
)

results = run_dynamic_causal_discovery(cfg)
nodes = results["nodes"]
links = results["links"]
edges = results["edges"]

granger = build_granger_results_from_edges(edges)
```

---

## Output Payload

```json
{
  "nodes": [{ "id": "GDP", "group": "Real", "val": 4 }],
  "links": [{ "source": "Inflation", "target": "GDP", "value": 0.18, "width": 2.7 }],
  "edges": [{ "cause": "Inflation", "effect": "GDP", "confidence": 0.74, "strength": 0.18, "lag": 2 }],
  "columns": ["GDP", "Inflation", "Unemployment", "..."]
}
```

---

## Configuration Notes

- **Coverage filter**: `min_coverage` drops sparse indicators.
- **Scale**: `max_indicators` caps graph size for performance.
- **Confidence**: `min_confidence` controls which edges appear.

---

## Dashboard Integration

The dashboard calls this module via `kshiked.ui.data_connector`:

- Causal graph renders from `results["nodes/links"]`
- Granger tests render from `build_granger_results_from_edges(edges)`

---

## Dependencies

- `pandas` (required for CSV handling)
- Scarcity engine (already in the repo)


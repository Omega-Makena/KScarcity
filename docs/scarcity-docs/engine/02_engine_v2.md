# engine_v2.py — Online Discovery Engine

The `OnlineDiscoveryEngine` is the **main entry point** for the relationship discovery system. It provides a simple, row-by-row API for processing streaming data and building up a knowledge graph of relationships.

---

## Purpose

This is the class you'll interact with most often. It:

- Takes a data schema and initializes a population of hypothesis candidates
- Processes rows one at a time, updating all hypotheses
- Manages the lifecycle (birth, promotion, decay, death) of hypotheses
- Exports the current best understanding as a knowledge graph

---

## Core Concepts

### The "Tick" Model

Each call to `process_row(row)` is one "tick" of the system:

```
Tick N:
  1. Sanitize incoming row (convert to floats, handle NaN)
  2. Update all hypotheses with the new data
  3. MetaController manages lifecycle transitions
  4. Periodically run arbitration (conflict resolution)
  5. Periodically run exploration (generate new hypotheses)
  6. Return status summary
```

The engine is designed for **anytime** use — you can query the knowledge graph after any number of ticks.

### Hypothesis Pool Initialization

Two initialization methods exist:

| Method | Use Case |
|--------|----------|
| `initialize(schema)` | Basic initialization with simpler hypothesis types |
| `initialize_v2(schema, use_causal=True)` | Full initialization with all 15 relationship types |

**`initialize_v2`** is the recommended method. It creates:

- CausalHypothesis (Granger causality) for all variable pairs — O(n²)
- TemporalHypothesis (AR model) for each variable — O(n)
- CorrelationalHypothesis for all pairs — O(n²)
- StructuralHypothesis (graph patterns) — O(1)
- Extended types: Mediating, Moderating, Graph, Similarity, Logical

---

## Class: `OnlineDiscoveryEngine`

### Initialization

```python
def __init__(self, explore_interval: int = 10):
```

- **`explore_interval`**: How often to run the exploration phase (every N ticks)

Creates:
- `HypothesisPool` — Container for all hypotheses
- `MetaController` — State machine arbiter
- `HypothesisArbiter` — Conflict resolver
- `AdaptiveGrouper` — Variable clustering

### `initialize_v2(schema, use_causal=True)`

The full initialization routine:

```python
schema = {
    "fields": ["gdp", "inflation", "unemployment", "interest_rate"],
    "types": {"gdp": "float", "inflation": "float", ...}
}
engine.initialize_v2(schema)
```

For a schema with N variables:
- Creates approximately **n² + n** hypotheses initially
- Each hypothesis starts in `TENTATIVE` state
- Grouper creates atomic groups (one per variable)

Set `use_causal=False` to skip the expensive O(n²) causal hypotheses.

### `process_row(row)`

The main processing method:

```python
status = engine.process_row({
    "gdp": 2.3,
    "inflation": 3.1,
    "unemployment": 4.5,
    "interest_rate": 5.0
})
```

Returns:

```python
{
    "step": 1234,
    "active_hypotheses": 42,
    "meta_summary": {
        "active": 42,
        "tentative": 120,
        "decaying": 8,
        "dead": 15
    },
    "grouping_stats": {...}
}
```

### `get_knowledge_graph()`

Exports the current best understanding:

```python
knowledge = engine.get_knowledge_graph()
# Returns list of edge dictionaries:
[
    {
        "source": "interest_rate",
        "target": "inflation",
        "type": "causal",
        "effect": -0.23,
        "confidence": 0.87,
        "stability": 0.92,
        "lag": 2
    },
    ...
]
```

---

## Internal Workflow

### Row Sanitization (`_sanitize_row`)

Converts all values to floats:
- `None` → `NaN`
- `"3.14"` → `3.14`
- Non-numeric strings → `NaN`
- Keys filtered to strings only

### Hypothesis Update Cycle

For each hypothesis in the pool:

1. **`fit_step(row)`**: Update internal model parameters (learning)
2. **`evaluate(row)`**: Compute fit metrics without changing model
3. **State check**: MetaController updates lifecycle state

The pool uses **vectorized operations** where possible for performance.

### Arbitration Phase (`_arbitrate_step`)

Runs periodically (every 50 ticks by default):

1. Collect all `ACTIVE` hypotheses
2. `HypothesisArbiter.arbitrate()` filters redundancies
3. **Hierarchy enforcement**: Causal > Temporal > Correlational
4. Conflicting/weaker hypotheses killed

### Exploration Phase (`_explore_step`)

Runs every `explore_interval` ticks:

1. Sample new variable pairs not in current pool
2. Create exploratory hypotheses with minimal prior
3. Promote weak hypotheses that showed recent improvement
4. Inject diversity to escape local optima

---

## Lifecycle State Machine

Hypotheses transition through states managed by `MetaController`:

```
    ┌────────────────────────────────────────┐
    │                                        │
    ▼                                        │
TENTATIVE                                    │
    │                                        │
    │ (enough evidence +                     │
    │  high confidence +                     │
    │  high stability)                       │
    ▼                                        │
ACTIVE ◀─────────────────────────────────────┤
    │          (metrics recover)             │
    │                                        │
    │ (metrics drop)                         │
    ▼                                        │
DECAYING ─────────────────────────────────────┘
    │
    │ (metrics collapse)
    ▼
  DEAD → moved to graveyard
```

---

## Edge Cases and Gotchas

### Sparse Data

If a variable has many missing values:
- Hypotheses involving it accumulate fewer observations
- May never reach `min_evidence` threshold
- Solution: Pre-impute data or increase patience thresholds

### Cold Start

First ~20-50 rows:
- All hypotheses are `TENTATIVE`
- Knowledge graph may be empty
- Normal — system needs burn-in period

### Computational Cost

For N variables:
- Initial hypotheses: O(n²)
- Per-row update: O(n²) in worst case
- Mitigation: Use `use_causal=False` or sample variable pairs

---

## When to Use This vs. MPIEOrchestrator

| Scenario | Use |
|----------|-----|
| Simple streaming, row-by-row | `OnlineDiscoveryEngine` |
| Event-driven architecture | `MPIEOrchestrator` |
| Need DRG integration | `MPIEOrchestrator` |
| Prototyping / testing | `OnlineDiscoveryEngine` |
| Production with telemetry | `MPIEOrchestrator` |

---

## Example: Full Workflow

```python
from scarcity.engine import OnlineDiscoveryEngine
import pandas as pd

# Load data
df = pd.read_csv("economic_indicators.csv")

# Initialize
engine = OnlineDiscoveryEngine(explore_interval=20)
engine.initialize_v2({
    "fields": list(df.columns),
    "types": {col: "float" for col in df.columns}
})

# Process streaming data
for idx, row in df.iterrows():
    status = engine.process_row(row.to_dict())
    
    if idx % 100 == 0:
        print(f"Step {idx}: {status['meta_summary']}")

# Get final knowledge graph
kg = engine.get_knowledge_graph()
print(f"Discovered {len(kg)} relationships")

# Examine top relationships
for edge in kg[:5]:
    print(f"{edge['source']} → {edge['target']}: "
          f"effect={edge['effect']:.3f}, "
          f"confidence={edge['confidence']:.2f}")
```

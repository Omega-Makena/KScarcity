# discovery.py — Core Abstractions

This file defines the **foundational abstractions** for the relationship discovery system: the `Hypothesis` base class, the `HypothesisPool` container, and supporting types.

---

## Philosophy: Hypothesis as Active Constraint

Traditional relationship discovery works in batch mode: collect all data, run analysis, produce results. The scarcity engine takes a fundamentally different approach:

> **A hypothesis is an active constraint that must continuously survive the stream of data.**

Each hypothesis is a living entity that:
- Learns from every new observation
- Competes with other hypotheses for "survival"
- Can be promoted, demoted, or killed based on performance

---

## Key Types

### `RelationshipType` (Enum)

Defines all 15 relationship types the engine can discover:

| Type | Description |
|------|-------------|
| `CAUSAL` | X Granger-causes Y |
| `CORRELATIONAL` | X and Y move together |
| `TEMPORAL` | Autoregressive (Y depends on past Y) |
| `FUNCTIONAL` | Y = f(X) relationship |
| `PROBABILISTIC` | Statistical dependence |
| `COMPOSITIONAL` | Multiple inputs explain Y |
| `COMPETITIVE` | Inputs compete for influence |
| `SYNERGISTIC` | Inputs have interaction effects |
| `MEDIATING` | X → M → Y chain |
| `MODERATING` | M changes X→Y strength |
| `GRAPH` | Network/graph patterns |
| `SIMILARITY` | Variables behave alike |
| `EQUILIBRIUM` | Long-run balance |
| `STRUCTURAL` | Graph topology patterns |
| `LOGICAL` | Boolean constraints |

### `HypothesisState` (Enum)

The lifecycle states:

| State | Meaning |
|-------|---------|
| `TENTATIVE` | New, insufficient evidence |
| `ACTIVE` | Strong, stable, confirmed |
| `DECAYING` | Was active, now degrading |
| `DEAD` | Killed, archived in graveyard |

### `HypothesisMetadata` (Dataclass)

MLOps metadata for each hypothesis:

```python
@dataclass
class HypothesisMetadata:
    id: str              # UUID
    created_at: float    # Unix timestamp
    state: HypothesisState
    generation: int      # For evolutionary tracking
    parents: List[str]   # Lineage (if spawned from another)
```

---

## Base Class: `Hypothesis`

Abstract base class that all relationship types extend.

### Constructor

```python
def __init__(self, variables: List[str], rel_type: RelationshipType):
```

- **`variables`**: List of variable names involved (e.g., `["gdp", "inflation"]`)
- **`rel_type`**: The relationship type

Initializes:
- Metadata with unique ID
- Default metrics (confidence=0.5, stability=0.5, fit_score=0, evidence=0)

### Core Methods

Every hypothesis subclass must implement:

#### `fit_step(row: Dict[str, float])`

**Purpose**: Update internal model parameters with new observation.

This is the "learning" step. Examples:
- CausalHypothesis: Update lag buffer, recalculate Granger coefficients
- CorrelationalHypothesis: Update running correlation statistics
- TemporalHypothesis: RLS update of AR weights

```python
def fit_step(self, row: Dict[str, float]) -> None:
    # Subclass implements specific learning logic
    pass
```

#### `evaluate(row: Dict[str, float]) -> Dict[str, float]`

**Purpose**: Measure how well the new data aligns with the hypothesis, *without* modifying the model.

Returns metrics dictionary:

```python
{
    "fit_score": 0.85,      # How well data fits (0-1)
    "confidence": 0.73,     # Bayesian belief (0-1)
    "evidence": 150,        # Observation count
    "stability": 0.91       # Metric consistency (0-1)
}
```

#### `predict_value(row: Dict[str, float]) -> Optional[Tuple[str, float]]`

**Purpose**: Predict the target variable's value for simulation.

Returns `(target_name, predicted_value)` or `None` if the hypothesis isn't predictive.

Example:
```python
# CausalHypothesis: "interest_rate causes inflation"
prediction = hyp.predict_value({"interest_rate": 5.0, "inflation": 3.0})
# Returns: ("inflation", 2.8)  # Predicted inflation
```

### Composite Method: `update(row)`

Combines fit and evaluate in the correct sequence:

```python
def update(self, row: Dict[str, float]) -> Dict[str, float]:
    # 1. Evaluate (measure fit before learning)
    metrics = self.evaluate(row)
    
    # 2. Learn from the new data
    self.fit_step(row)
    
    # 3. Update stored metrics
    self.fit_score = metrics.get("fit_score", self.fit_score)
    self.confidence = metrics.get("confidence", self.confidence)
    # ...
    
    return metrics
```

### Serialization: `to_dict()`

Exports hypothesis state for storage or transmission:

```python
{
    "id": "a1b2c3d4-...",
    "type": "causal",
    "variables": ["gdp", "inflation"],
    "state": "active",
    "fit_score": 0.85,
    "confidence": 0.73,
    "stability": 0.91,
    "evidence": 150,
    "effect_size": 0.23,
    "created_at": 1700000000.0
}
```

---

## Container: `HypothesisPool`

Manages the population of hypotheses.

### Initialization

```python
def __init__(self, capacity: int = 1000):
```

- **`capacity`**: Maximum hypotheses before aggressive pruning
- Also creates a `VectorizedHypothesisPool` for batch operations

### Properties

- `population: Dict[str, Hypothesis]` — Living hypotheses by ID
- `graveyard: List[Dict]` — Dead hypotheses (for debugging/analysis)

### Methods

#### `add(hypothesis: Hypothesis)`

Registers a new hypothesis:
```python
pool.add(CausalHypothesis("gdp", "inflation"))
```

Handles:
- Duplicate ID detection
- Capacity overflow (triggers pruning)

#### `update_all(row: Dict[str, float])`

Updates every hypothesis in the pool:

```python
pool.update_all({"gdp": 2.3, "inflation": 3.1, ...})
```

Uses a **hybrid execution model**:
1. Vectorized hypotheses processed in batch via `VectorizedHypothesisPool`
2. Standard OOP hypotheses processed individually

This is the main per-tick operation.

#### `get_strongest(top_k: int = 10) -> List[Hypothesis]`

Returns the top-k hypotheses by confidence × stability:

```python
best = pool.get_strongest(10)
# Returns: [Hypothesis, Hypothesis, ...]
```

### Private Methods

#### `_kill(hid: str)`

Permanently removes a hypothesis:
- Moves serialized state to `graveyard`
- Removes from `population`

#### `_prune_weakest(force: bool = False)`

Removes lowest-performing hypotheses when capacity exceeded:
- Targets `DECAYING` and `TENTATIVE` hypotheses first
- Keeps all `ACTIVE` hypotheses
- `force=True` prunes immediately without waiting for capacity overflow

---

## Vectorized Hypothesis Pool

For performance, `discovery.py` imports `VectorizedHypothesisPool` from `vectorized_core.py`. This enables:

- Batch matrix operations on hypothesis states
- NumPy-accelerated update loops
- Reduced Python overhead for large populations

The standard `HypothesisPool` automatically delegates to the vectorized pool when beneficial.

---

## Creating Custom Hypotheses

To define a new relationship type:

```python
class MyCustomHypothesis(Hypothesis):
    def __init__(self, var1: str, var2: str, **kwargs):
        super().__init__([var1, var2], RelationshipType.CUSTOM)
        # Initialize internal state
        
    def fit_step(self, row: Dict[str, float]) -> None:
        # Learning logic
        
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        # Evaluation logic
        return {"fit_score": ..., "confidence": ..., ...}
        
    def predict_value(self, row: Dict[str, float]):
        # Optional: prediction logic
        return None  # If not predictive
```

Then add to the pool during initialization.

---

## Integration Points

- **`relationships.py`**: Concrete implementations of all 15 relationship types
- **`controller.py`**: MetaController uses hypothesis metrics to manage lifecycle
- **`arbitration.py`**: HypothesisArbiter uses relationship types for conflict resolution
- **`store.py`**: HypergraphStore persists successful hypotheses as edges

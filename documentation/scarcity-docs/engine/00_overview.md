# Engine Module Overview

The `scarcity.engine` module is the **heart of the relationship discovery system**. It takes streaming data and continuously learns which variables influence which others, building up a knowledge graph of causal, temporal, and statistical relationships over time.

---

## The Core Problem

Imagine you have dozens of economic indicators arriving as a time series: GDP, inflation, unemployment, interest rates, commodity prices. You want to automatically discover:

- **Which variables cause changes in others** (e.g., does inflation *Granger-cause* unemployment?)
- **How strong are these relationships** (effect sizes, confidence intervals)
- **Are the relationships stable** (or do they shift under different economic regimes?)

Traditional approaches require offline batch analysis. The engine does this **online** — as data streams in, it maintains a population of "hypotheses" about relationships, tests them against new observations, and gradually converges on the truth.

---

## Architecture: The Hypothesis Survival Paradigm

The engine treats relationship discovery as a **survival-of-the-fittest competition** among hypotheses:

```
Stream of Data
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Hypothesis Pool                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ A→B     │ │ B→C     │ │ A↔C     │ │ X→Y     │  ...   │
│  │ Causal  │ │ Temporal│ │ Correl. │ │ Func.   │        │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
│       ▲           ▲           ▲           ▲             │
│       └───────────┴───────────┴───────────┘             │
│                     UPDATE                               │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Meta-Controller                        │
│   State Machine: TENTATIVE → ACTIVE → DECAYING → DEAD  │
│   Promotes strong hypotheses, kills weak ones           │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Knowledge Graph                        │
│   Edges: (Source, Target, Effect, Confidence, Stability)│
│   Stored in HypergraphStore with decay and pruning      │
└─────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Discovery Layer (`discovery.py`, `engine_v2.py`)

This is where hypotheses are created and managed:

- **`Hypothesis`**: Abstract base class for all relationship types. Each hypothesis knows how to:
  - `fit_step(row)`: Update its internal model with new data
  - `evaluate(row)`: Measure how well the new data fits the hypothesis
  - `predict_value(row)`: Predict a target variable (for simulation)

- **`HypothesisPool`**: The population of living hypotheses. Manages capacity limits and pruning.

- **`OnlineDiscoveryEngine`**: The main entry point. Initializes hypotheses based on the data schema, processes rows through the pool, and exports the knowledge graph.

### 2. Relationship Types (`relationships.py`, `relationships_extended.py`)

The engine supports **15 relationship types**:

| Type | What It Detects | Algorithm |
|------|-----------------|-----------|
| Causal | X Granger-causes Y | Incremental regression with lagged features |
| Correlational | X and Y move together | Online Pearson correlation |
| Temporal | Y depends on its own past | Autoregressive (AR) model |
| Functional | Y = f(X) | Polynomial regression |
| Probabilistic | Statistical dependence | Mutual information estimation |
| Compositional | X + W jointly explain Y | Multi-input regression |
| Competitive | X and W compete for Y | Residual comparison |
| Synergistic | X and W interact | Interaction terms |
| Mediating | X → M → Y | Two-stage regression |
| Moderating | X's effect depends on M | Stratified analysis |
| Structural | Graph-level patterns | Cycle detection, hierarchy |
| Equilibrium | Long-run balance | Cointegration-like tests |
| Graph | Network relationships | Adjacency patterns |
| Similarity | Variables behave alike | Distance metrics |
| Logical | Boolean constraints | Rule checking |

### 3. Evaluation Layer (`evaluator.py`, `bandit_router.py`)

**Evaluator**: Scores hypotheses using:
- **Predictive Gain**: Does the hypothesis improve R² over a naive baseline?
- **Uncertainty**: Bootstrap confidence intervals
- **Stability**: Consistency of effect signs across resamples

**BanditRouter**: Multi-armed bandit for exploration-exploitation. Instead of testing all hypotheses equally, it allocates more testing budget to promising ones using Thompson Sampling.

### 4. Storage (`store.py`)

**HypergraphStore**: Persists discovered relationships in a compact graph structure:
- Edges decay over time (old relationships fade unless re-confirmed)
- Bounded memory with LRU pruning
- Regime-aware (can track different relationship sets for different market conditions)

### 5. Meta-Control (`controller.py`, `arbitration.py`)

**MetaController**: State machine that transitions hypotheses through lifecycle stages:
- `TENTATIVE` → Not enough evidence yet
- `ACTIVE` → Strong, stable, supported by data
- `DECAYING` → Was active, now degrading
- `DEAD` → Killed, moved to graveyard

**HypothesisArbiter**: Resolves conflicts when multiple hypotheses compete for the same relationship:
- Causal beats temporal beats correlational
- Within the same type, higher confidence wins

### 6. Support Components

- **`encoder.py`**: Converts raw data into compact latent representations for fast processing
- **`grouping.py`**: Adaptive clustering of variables (can treat "Macro Factors" as a group)
- **`vectorized_core.py`**: High-performance batch operations using NumPy
- **`operators/`**: Specialized mathematical operations (attention, sketching, etc.)

---

## Data Flow

Here's how data flows through the system during a single timestep:

```
1. New data row arrives
       │
       ▼
2. Sanitization: Convert to floats, handle missing values
       │
       ▼
3. Pool Update: Each hypothesis calls update(row)
   - fit_step: Learn from data
   - evaluate: Measure fit quality
   - State transitions via MetaController
       │
       ▼
4. Arbitration: Resolve conflicts between hypotheses
       │
       ▼
5. Store Update: Confirmed relationships written to HypergraphStore
       │
       ▼
6. Exploration: BanditRouter may spawn new hypotheses to test
       │
       ▼
7. Knowledge Graph Export: Top-k strongest relationships returned
```

---

## Integration Points

The engine integrates with other scarcity modules:

- **`runtime.bus`**: Event-driven communication between components
- **`governor`**: Dynamic Resource Governor controls computational budget
- **`federation`**: Distributed learning across multiple data sources
- **`meta`**: Meta-learning for hyperparameter adaptation
- **`simulation`**: Uses discovered relationships to run economic forecasts

---

## Usage Example

```python
from scarcity.engine import OnlineDiscoveryEngine

# Initialize engine
engine = OnlineDiscoveryEngine()

# Define data schema
schema = {
    "fields": ["gdp", "inflation", "unemployment", "interest_rate"],
    "types": {"gdp": "float", "inflation": "float", ...}
}
engine.initialize_v2(schema)

# Process streaming data
for row in data_stream:
    status = engine.process_row(row)
    
# Get discovered relationships
knowledge = engine.get_knowledge_graph()
# Returns: [{"source": "interest_rate", "target": "inflation", 
#            "type": "causal", "effect": -0.23, "confidence": 0.87}, ...]
```

---

## Performance Characteristics

- **Memory Bounded**: HypothesisPool and HypergraphStore both have capacity limits
- **Anytime Output**: Knowledge graph available after any number of timesteps
- **Adaptive**: MetaController adjusts thresholds if system is under-producing hypotheses
- **Parallelizable**: Vectorized operations on hypothesis batches

---

## Next Steps

For detailed documentation of each file, see:

- [engine.py](./01_engine.md) — MPIE Orchestrator
- [engine_v2.py](./02_engine_v2.md) — Online Discovery Engine
- [discovery.py](./03_discovery.md) — Core Abstractions
- [evaluator.py](./04_evaluator.md) — Path Scoring
- [relationships.py](./05_relationships.md) — Relationship Implementations
- ... (see index)

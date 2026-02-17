# Engine (MPIE) - Multi-Path Inference Engine

## Overview

The Engine module implements the Multi-Path Inference Engine (MPIE), the core causal discovery system. It discovers causal relationships from streaming data using online learning and statistical validation.

**Location**: `scarcity/engine/`

**Purpose**: Real-time causal discovery from streaming data

**Key Innovation**: Online causal inference with resource awareness and bandit-based exploration

## Architecture

### Pipeline Flow

```
Data Window
    |
    v
Controller (BanditRouter)
    | (proposes candidate paths)
    v
Encoder
    | (encodes paths to latent space)
    v
Evaluator
    | (scores paths statistically)
    v
Store (HypergraphStore)
    | (maintains causal graph)
    v
Exporter
    | (emits insights)
```

### Component Interaction

```
MPIEOrchestrator
    |
    +-- subscribes to: data_window, resource_profile, meta_policy_update
    |
    +-- coordinates:
        |
        +-- Controller.propose() -> List[Candidate]
        +-- Encoder.step() -> EncodedBatch
        +-- Evaluator.score() -> List[EvalResult]
        +-- Evaluator.make_rewards() -> List[Reward]
        +-- Controller.update(rewards)
        +-- Store.update_edges()
        +-- Exporter.emit_insights()
```

## Files

### Core Files
- `engine.py` - MPIEOrchestrator (main coordinator)
- `controller.py` - BanditRouter (path proposal)
- `encoder.py` - Encoder (feature encoding)
- `evaluator.py` - Evaluator (statistical validation)
- `store.py` - HypergraphStore (graph storage)
- `exporter.py` - Exporter (insight emission)
- `types.py` - Data types (Candidate, EvalResult, Reward)
- `utils.py` - Utility functions
- `resource_profile.py` - Resource profiling

### Operator Files (`operators/`)
- `attention_ops.py` - Attention mechanisms
- `causal_semantic_ops.py` - Causal semantics
- `evaluation_ops.py` - Evaluation metrics
- `integrative_ops.py` - Integration operations
- `relational_ops.py` - Relational operations
- `sketch_ops.py` - Sketch-based compression
- `stability_ops.py` - Stability metrics
- `structural_ops.py` - Structural operations

## Data Types

### Candidate

Represents a candidate causal path for evaluation.

```python
@dataclass
class Candidate:
    path_id: str              # Unique identifier
    vars: Tuple[int, ...]     # Variable indices in path
    lags: Tuple[int, ...]     # Temporal lags per variable
    ops: Tuple[str, ...]      # Operations per edge
    root: int                 # Root variable (for bandit arm)
    depth: int                # Path length
    domain: int               # Domain identifier
    gen_reason: str           # Generation reason (UCB/diversity/random)
```

**Example**:
```python
Candidate(
    path_id="abc123",
    vars=(0, 3, 7),           # X0 -> X3 -> X7
    lags=(0, 1, 0),           # No lag, lag 1, no lag
    ops=("sketch", "attn", "attn"),
    root=0,
    depth=3,
    domain=0,
    gen_reason="UCB"
)
```

### EvalResult

Result of evaluating a candidate path.

```python
@dataclass
class EvalResult:
    path_id: str              # Matches Candidate.path_id
    gain: float               # Predictive gain (R² improvement)
    ci_lo: float              # Lower confidence bound
    ci_hi: float              # Upper confidence bound
    stability: float          # Stability score [0,1]
    cost_ms: float            # Evaluation latency
    accepted: bool            # Whether path was accepted
    extras: Dict[str, Any]    # Additional metadata
```

### Reward

Shaped reward for bandit learning.

```python
@dataclass
class Reward:
    path_id: str              # Matches Candidate.path_id
    arm_key: Tuple[int, int]  # (root, depth) for bandit
    value: float              # Shaped reward [-1, 1]
    latency_penalty: float    # Latency penalty component
    diversity_bonus: float    # Diversity bonus component
    accepted: bool            # Whether path was accepted
```

## MPIEOrchestrator

### Class Definition

```python
class MPIEOrchestrator:
    def __init__(self, bus: Optional[EventBus] = None)
    async def start() -> None
    async def stop() -> None
    def get_stats() -> Dict[str, Any]
```

### Initialization

```python
orchestrator = MPIEOrchestrator(bus=event_bus)
await orchestrator.start()
```

**Initializes**:
- BanditRouter (controller)
- Encoder
- Evaluator
- HypergraphStore
- Exporter

### Event Handling

**Subscribes to**:
- `data_window` - Incoming data for processing
- `resource_profile` - Resource updates from DRG
- `meta_policy_update` - Policy updates from meta-learning

**Publishes**:
- `processing_metrics` - Performance metrics
- `engine.insight` - Discovered causal edges

### Processing Pipeline

**Step-by-step execution** (from `_handle_data_window`):

1. **Propose paths**
   ```python
   candidates = controller.propose(
       window_meta={'length': len(data), 'timestamp': time.time()},
       schema=data['schema'],
       budget=resource_profile['n_paths']
   )
   ```

2. **Extract window tensor**
   ```python
   window_tensor = np.array(data['data'])
   ```

3. **Score candidates**
   ```python
   results = evaluator.score(window_tensor, candidates)
   ```

4. **Compute diversity**
   ```python
   diversity_dict = {
       cand.path_id: controller.diversity_score(cand) 
       for cand in candidates
   }
   ```

5. **Produce rewards**
   ```python
   rewards = evaluator.make_rewards(
       results, 
       lambda pid: diversity_dict[pid],
       candidates=candidates
   )
   ```

6. **Update controller**
   ```python
   controller.update(rewards)
   ```

7. **Update store**
   ```python
   store.update_edges(accepted_payloads)
   ```

8. **Publish metrics**
   ```python
   await bus.publish("processing_metrics", metrics)
   ```

### Metrics Published

```python
{
    'engine_latency_ms': float,        # Total processing time
    'n_candidates': int,                # Candidates proposed
    'accepted_count': int,              # Candidates accepted
    'accept_rate': float,               # Acceptance rate
    'edges_active': int,                # Active edges in store
    'oom_flag': bool,                   # Out-of-memory flag
    
    # Controller metrics
    'proposal_entropy': float,          # Entropy of arm selection
    'diversity_index': float,           # Diversity of proposals
    'arm_mean_r_topk': List[float],    # Top-K arm rewards
    'drift_detections': int,            # Drift events detected
    'thompson_mode': bool,              # Thompson sampling active
    
    # Evaluator metrics
    'eval_accept_rate': float,          # Evaluator acceptance rate
    'gain_p50': float,                  # Median gain
    'gain_p90': float,                  # 90th percentile gain
    'ci_width_avg': float,              # Average CI width
    'stability_avg': float,             # Average stability
    'total_evaluated': int              # Total paths evaluated
}
```

## Controller (BanditRouter)

### Purpose

Proposes candidate causal paths using multi-armed bandit algorithms (UCB/Thompson Sampling) with diversity tracking.

### Algorithm

**Multi-armed bandit with diversity**:

1. **Arm Definition**: Each root variable is an arm
2. **UCB Score**: 
   ```
   UCB(arm) = mean_reward + τ·sqrt(2·log(T)/n) + γ·D(arm) - η·cost(arm)
   ```
   Where:
   - τ = exploration temperature
   - γ = diversity weight
   - η = cost weight
   - D(arm) = diversity score
   - T = total pulls
   - n = arm pulls

3. **Diversity Score**:
   ```
   D(path) = mean(1 / sqrt(1 + coverage[v]) for v in path.vars)
   ```

4. **Root Selection**:
   - 60% from top UCB arms (exploitation)
   - 25% from low-coverage arms (diversity)
   - 15% random (exploration)

5. **Path Expansion**: Each root expanded to full paths with random walks

### Key Methods

```python
def propose(window_meta, schema, budget) -> List[Candidate]
def diversity_score(candidate: Candidate) -> float
def update(rewards: List[Reward]) -> None
def update_resource_profile(profile: Dict) -> None
```

### Drift Detection

**Page-Hinkley Test**:
```python
diff = reward - old_mean - threshold
page_hinkley_sum += diff
page_hinkley_sum = max(0, page_hinkley_sum)

if page_hinkley_sum > 100:
    # Drift detected
    temperature += 0.2
    thompson_mode = True
```

### Statistics

```python
{
    'n_arms': int,                      # Number of arms
    'temperature': float,               # Exploration temperature
    'diversity_weight': float,          # Diversity weight
    'proposal_entropy': float,          # Selection entropy
    'arm_mean_r_topk': List[float],    # Top-K arm rewards
    'drift_detections': int,            # Drift count
    'thompson_mode': bool,              # Thompson active
    'total_windows': int                # Windows processed
}
```

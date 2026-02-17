# SCARCITY: Complete System Documentation

**Version**: 1.0.0 
**Last Updated**: December 3, 2025 
**Status**: Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
- [Runtime Bus](#runtime-bus)
- [MPIE Engine](#mpie-engine)
- [Dynamic Resource Governor (DRG)](#dynamic-resource-governor-drg)
- [Meta-Learning System](#meta-learning-system)
- [Federation](#federation)
- [Simulation Engine](#simulation-engine)
- [Stream Processing](#stream-processing)
4. [Backend API](#backend-api)
5. [Frontend Dashboard](#frontend-dashboard)
6. [Deployment](#deployment)
7. [Development Guide](#development-guide)
8. [API Reference](#api-reference)

---

## Executive Summary

**SCARCITY** is an advanced machine learning platform for **causal discovery**, **adaptive resource management**, and **federated learning** across distributed domains. The system combines cutting-edge algorithms with production-ready infrastructure to enable:

- **Real-time causal inference** from streaming data
- **Adaptive resource management** with predictive control
- **Cross-domain meta-learning** with safety guarantees
- **Federated learning** with privacy preservation
- **3D visualization** of causal relationships
- **Multi-domain orchestration** with automatic scaling

### Key Features

**Multi-Path Inference Engine (MPIE)** - Discovers causal relationships using multi-armed bandits and bootstrap validation 
**Dynamic Resource Governor (DRG)** - Maintains system stability through EMA/Kalman filtering and policy-based control 
**5-Tier Meta-Learning** - Learns optimal hyperparameters across domains with automatic rollback 
**Federated Learning** - Secure aggregation with differential privacy and Byzantine-robust algorithms 
**3D Simulation** - Real-time visualization of causal graphs with adaptive LOD 
**Stream Processing** - Online windowing with Welford statistics and PI-controller rate regulation 

### Technology Stack

**Core Library**: Python 3.11+, NumPy, asyncio 
**Backend**: FastAPI, Uvicorn, Pydantic 
**Frontend**: React 18, TypeScript, Vite, TailwindCSS 
**Visualization**: Three.js, React Three Fiber 
**Optional**: PyTorch, PyTorch3D, psutil, pynvml 

---

## System Architecture

### High-Level Architecture

```

SCARCITY Platform 



Frontend Backend Core Lib 
Dashboard FastAPI (Python) 
(React) REST API 



Event Bus (Runtime) 
Pub/Sub Communication Fabric (asyncio) 



MPIE DRG Meta Federa 
Engine Governor Learning tion 



Stream Simula 
Process tion 



```

### Data Flow

```
Raw Data → Stream Source → Window Builder → Schema Manager
↓
MPIE Controller
↓

↓ ↓
Encoder Evaluator
↓ ↓
Latent Space Bootstrap Stats

↓
Hypergraph Store
↓

↓ ↓
Simulation Federation
↓ ↓
3D Viz Meta-Learning
↓ ↓
Dashboard ←
```

### Component Interaction

```

Stream Publishes: data_window
Source 
↓

Event Bus 
MPIE Publishes: (Runtime) 
Engine - processing_metrics 
- engine.insight 
↑


DRG Publishes: 
Governor - drg.telemetry 
- drg.signal.* 



Meta Publishes: 
Learning - meta_prior_update 
- meta_metrics 



Simulation Publishes: 
Engine - simulation.state 
- simulation.frame 



Federation Publishes: 
Coordinator - federation.policy_pack 

```

---

## Core Components


### Runtime Bus

**Location**: `scarcity/runtime/`

The Runtime Bus is the central nervous system of SCARCITY, providing event-driven communication between all components.

#### Architecture

**Publish-Subscribe Pattern**:
- Asynchronous message delivery
- Topic-based routing
- Thread-safe operations
- Automatic error handling

#### Core Topics

| Topic | Publisher | Subscribers | Payload |
|-------|-----------|-------------|---------|
| `data_window` | Stream Source | MPIE Engine | Raw data window |
| `processing_metrics` | MPIE Engine | Meta-Learning, DRG | Performance metrics |
| `engine.insight` | MPIE Exporter | Simulation, Dashboard | Causal insights |
| `drg.telemetry` | DRG Governor | Dashboard, Meta | Resource metrics |
| `drg.signal.*` | DRG Governor | Subsystems | Control signals |
| `meta_prior_update` | Meta-Learning | MPIE, Dashboard | Global hyperparameters |
| `simulation.state` | Simulation | Dashboard | Simulation state |
| `simulation.frame` | Simulation | Dashboard | 3D visualization frame |
| `federation.policy_pack` | Federation | Meta-Learning | Cross-domain updates |

#### EventBus API

```python
from scarcity.runtime import EventBus, get_bus

# Get singleton bus instance
bus = get_bus()

# Subscribe to topic
def handler(topic: str, data: Any):
print(f"Received {topic}: {data}")

bus.subscribe("data_window", handler)

# Publish message
await bus.publish("data_window", {
"data": window_data,
"timestamp": time.time()
})

# Unsubscribe
bus.unsubscribe("data_window", handler)

# Get statistics
stats = bus.get_stats()
# {
# 'messages_published': 1000,
# 'messages_delivered': 1000,
# 'delivery_errors': 0,
# 'topics_active': 8,
# 'total_subscribers': 15
# }
```

#### Implementation Details

**Thread Safety**: Uses `asyncio.Lock` for concurrent access 
**Error Handling**: Catches and logs subscriber exceptions without stopping delivery 
**Performance**: O(1) publish, O(n) delivery where n = subscribers per topic 
**Memory**: Bounded by number of topics and subscribers (no message queue) 

---

### MPIE Engine

**Location**: `scarcity/engine/`

The Multi-Path Inference Engine (MPIE) is the core causal discovery system that learns relationships from streaming data using multi-armed bandits and statistical validation.

#### Architecture

**Pipeline**: Controller → Encoder → Evaluator → Store → Exporter

```

Controller Multi-armed bandit path proposal
(Bandit) UCB/Thompson sampling
Diversity scoring
Drift detection
↓

Encoder FP16 latent space encoding
Attention/Sketch operators
Dimensionality reduction

↓

Evaluator Bootstrap statistical validation
R² gain computation
Confidence intervals

↓

Store Incremental hypergraph
(Hypergraph) EMA edge weights
Decay and pruning

↓

Exporter Insight emission
Top-K edges
Regime tracking
```

#### Controller (BanditRouter)

**Algorithm**: Multi-armed bandit with UCB and diversity bonuses

**UCB Score**:
```
UCB(arm) = μ + τ·√(2·log(T)/n) + γ·D(arm) - η·cost(arm)
```

Where:
- `μ` = mean reward
- `τ` = exploration temperature (default: 0.9)
- `T` = total trials
- `n` = arm trials
- `γ` = diversity weight (default: 0.3)
- `D(arm)` = diversity score
- `η` = cost penalty weight

**Diversity Score**:
```python
D(path) = mean(1 / √(1 + coverage[v]) for v in path.vars)
```

**Root Selection Strategy**:
- 60% top UCB arms (exploitation)
- 25% low-coverage arms (diversity)
- 15% random (exploration)

**Drift Detection**: Page-Hinkley test
```python
diff = reward - old_mean - threshold
page_hinkley_sum += diff
if page_hinkley_sum > 100:
# Increase exploration
temperature *= 1.2
```

**Configuration**:
```python
@dataclass
class ControllerConfig:
n_arms_max: int = 200
tau: float = 0.9 # Exploration temperature
gamma_diversity: float = 0.3 # Diversity weight
thompson_mode: bool = False # Use Thompson sampling
drift_threshold: float = 0.5 # Drift detection threshold
```

#### Encoder

**Purpose**: Transform paths to latent space using FP16 operations

**Pipeline**:
1. Variable embeddings + lag encoding
2. Sequence tensor construction
3. Attention/pooling
4. Normalization (LayerNorm/RMSNorm)
5. Sketch projection
6. Safety clipping

**Sketch Methods**:
- **Polynomial sketch**: For dimensionality reduction
- **Tensor sketch**: For high-dimensional tensor products
- **Count sketch**: For sparse data

**Configuration**:
```python
@dataclass
class EncoderConfig:
sketch_dim: int = 64
sketch_method: str = "polynomial"
use_fp16: bool = True
attention_heads: int = 4
normalization: str = "layer_norm"
```

**Output**:
```python
@dataclass
class EncodedBatch:
latents: List[np.ndarray] # FP16 latents
meta: List[Dict] # Metadata
stats: Dict # Statistics
telemetry: Dict # Performance
```

#### Evaluator

**Purpose**: Score paths using bootstrap resampling

**Algorithm**:
1. Build design matrix with lags
2. Bootstrap resampling (default 8 resamples)
3. Train/holdout split per resample
4. Compute R² gain vs baseline
5. Robust quantiles for CI
6. Stability from sign agreement

**Acceptance Criteria**:
```python
accepted = (
gain >= gain_min and
stability >= stability_min and
ci_width <= lambda * abs(gain)
)
```

**Reward Shaping**:
```python
r_core = w_g·tanh(ĝ) + w_s·ŝ + w_c·(ĉ - 0.5)
r_total = r_core - α_L·latency_penalty + β_D·diversity_bonus
```

**Configuration**:
```python
@dataclass
class EvaluatorConfig:
n_resamples: int = 8
gain_min: float = 0.01
stability_min: float = 0.3
lambda_ci: float = 0.5
reward_weights: Dict[str, float] = field(default_factory=lambda: {
'gain': 0.6,
'stability': 0.3,
'ci': 0.1
})
```

#### HypergraphStore

**Purpose**: Maintain causal graph with incremental updates

**Data Structures**:
```python
nodes: Dict[int, Dict] # node_id → metadata
edges: Dict[Tuple[int,int], EdgeRec] # (src,dst) → edge
hyperedges: Dict[frozenset, HyperRec] # sources → hyperedge
out_index: Dict[int, List] # node → neighbors
in_index: Dict[int, List] # node → predecessors
```

**EdgeRec**:
```python
@dataclass
class EdgeRec:
weight: float # EMA of effect
var: float # Variance
stability: float # EMA of stability
ci_lo: float # Lower CI
ci_hi: float # Upper CI
regime_id: int # Regime ID
last_seen: int # Window ID
hits: int # Acceptance count
```

**Update Algorithm**:
```python
# EMA update
edge.weight = (1-α)·edge.weight + α·new_effect
edge.stability = (1-α)·edge.stability + α·new_stability

# Decay
edge.weight *= decay_factor

# Prune
if abs(edge.weight) < threshold or age > max_age:
remove_edge()
```

**Configuration**:
```python
@dataclass
class StoreConfig:
ema_alpha: float = 0.3
decay_factor: float = 0.99
prune_threshold: float = 0.01
max_age_windows: int = 1000
```

#### Complete MPIE Capabilities

**1. Causal Discovery**
- PC Algorithm implementation
- Conditional independence testing
- Multi-path causal inference
- Temporal causal relationships

**2. Attention Mechanisms**
- Multi-head attention
- Self-attention for sequences
- Cross-attention for multi-modal data

**3. Sketch-Based Learning**
- Polynomial sketching
- Tensor sketching
- Count sketching
- Feature hashing

**4. Stability Analysis**
- Concept drift detection
- Model stability metrics
- Bootstrap stability estimation
- Regime change detection

**5. Structural Pattern Recognition**
- Graph topology analysis
- Structural motif detection
- Community detection
- Network centrality measures

**6. Relational Learning**
- Graph neural network operations
- Node embedding learning
- Edge prediction

**7. Integrative Multi-Modal Analysis**
- Multi-source data fusion
- Cross-modal attention
- Joint representation learning

**8. Semantic Causal Reasoning**
- Concept graph construction
- Semantic similarity computation
- Knowledge graph integration

**9. Advanced Evaluation**
- Multiple evaluation metrics
- Model selection criteria
- Cross-validation strategies

#### Usage Example

```python
from scarcity.engine import MPIEOrchestrator, MPIEConfig
from scarcity.runtime import get_bus

# Create MPIE with default config
config = MPIEConfig()
mpie = MPIEOrchestrator(config, get_bus())

# Start processing
await mpie.start()

# Publish data
await bus.publish("data_window", {
"data": window_data,
"schema": schema_info,
"timestamp": time.time()
})

# MPIE automatically:
# 1. Proposes candidate paths
# 2. Encodes to latent space
# 3. Evaluates with bootstrap
# 4. Updates hypergraph
# 5. Exports insights

# Stop processing
await mpie.stop()
```

---


### Dynamic Resource Governor (DRG)

**Location**: `scarcity/governor/`

The Dynamic Resource Governor maintains system stability through real-time monitoring and adaptive control using EMA smoothing, Kalman filtering, and policy-based actuation.

#### Architecture

**Control Loop**: Sensors → Profiler → Policies → Actuators → Monitor

```

Sensors Collect telemetry
CPU, GPU, Memory, I/O


↓

Profiler EMA smoothing
Kalman forecasting


↓

Policies Threshold-based rules
Priority ordering


↓

Actuators Execute control actions
Subsystem registration


↓

Monitor Record metrics
Diagnostics

```

#### Control Loop

**Main Loop** (runs every 0.5 seconds):
```python
async def _loop(self):
while self._running:
# 1. Sample current metrics
metrics = self.sensors.sample()

# 2. Smooth and forecast
ema, forecast = self.profiler.update(metrics)

# 3. Evaluate policies
decisions = self._evaluate_policies(metrics, ema, forecast)

# 4. Dispatch signals
await self._dispatch_signals(metrics, decisions)

# 5. Record metrics
self.monitor.record({**metrics, **ema})

await asyncio.sleep(self.config.control_interval)
```

#### Resource Sensors

**Collected Metrics**:
```python
{
# CPU Metrics
'cpu_util': float, # [0, 1]
'cpu_freq': float, # MHz

# Memory Metrics
'mem_util': float, # [0, 1]
'mem_available_gb': float, # GB
'swap_util': float, # [0, 1]

# GPU Metrics
'gpu_util': float, # [0, 1]
'vram_util': float, # [0, 1]

# I/O Metrics
'disk_read_mb': float, # MB
'disk_write_mb': float, # MB
'net_sent_mb': float, # MB
'net_recv_mb': float, # MB
}
```

**Dependencies**:
- `psutil` - CPU, memory, I/O
- `torch.cuda` - GPU metrics
- `pynvml` - Advanced GPU metrics (optional)

#### Resource Profiler

**Dual Smoothing Strategy**:

**1. Exponential Moving Average (EMA)**:
```python
ema[t] = (1 - α)·ema[t-1] + α·value[t]
```
- Default α = 0.3
- Smooths noisy measurements
- Fast response to changes

**2. Kalman Filter Forecasting**:
```python
# Predict
x̂[t|t-1] = x̂[t-1|t-1]
P[t|t-1] = P[t-1|t-1] + Q

# Update
K = P[t|t-1] / (P[t|t-1] + R)
x̂[t|t] = x̂[t|t-1] + K·(z[t] - x̂[t|t-1])
P[t|t] = (1 - K)·P[t|t-1]
```
- Q = 0.01 (process noise)
- R = 0.1 (measurement noise)
- Predicts next value
- Optimal for linear systems

#### Policy Rules

**PolicyRule Structure**:
```python
@dataclass
class PolicyRule:
metric: str # Metric to monitor
threshold: float # Trigger threshold
action: str # Action to execute
direction: str # ">" or "<"
factor: float # Action intensity [0, 1]
priority: int # Execution priority
```

**Default Policies**:
```python
{
"simulation": [
PolicyRule(
metric="vram_util",
threshold=0.90,
action="scale_down",
factor=0.5,
priority=3
),
PolicyRule(
metric="fps",
threshold=25.0,
action="increase_lod",
direction="<",
factor=0.75,
priority=2
),
],
"mpie": [
PolicyRule(
metric="cpu_util",
threshold=0.85,
action="reduce_batch",
factor=0.5,
priority=2
),
],
"meta": [
PolicyRule(
metric="vram_util",
threshold=0.85,
action="drop_low_priority",
priority=1
),
],
"federation": [
PolicyRule(
metric="latency_ms",
threshold=150.0,
action="delay_sync",
priority=1
),
],
"memory": [
PolicyRule(
metric="mem_util",
threshold=0.90,
action="flush_cache",
priority=1
),
],
}
```

**Available Actions**:
- `scale_down` - Reduce resource usage
- `scale_up` - Increase resource allocation
- `reduce_batch` - Decrease batch size
- `drop_low_priority` - Drop low-priority tasks
- `delay_sync` - Delay synchronization
- `flush_cache` - Clear caches
- `increase_lod` - Reduce quality (increase level of detail)

#### Resource Actuators

**Execution**:
```python
class ResourceActuators:
def execute(self, subsystem: str, action: str, factor: float) -> bool:
handle = self.registry.get(subsystem)
if handle is None:
return False

method = self._method_map.get(action)
if method is None:
return False

return handle.call(method, factor=factor)
```

**Subsystem Registration**:
```python
# Register subsystems for DRG control
drg.register_subsystem("simulation", simulation_engine)
drg.register_subsystem("mpie", mpie_orchestrator)
drg.register_subsystem("meta", meta_agent)
```

#### Event Bus Integration

**Published Events**:
```python
# Telemetry broadcast (every control interval)
await bus.publish("drg.telemetry", {
"metrics": metrics, # Raw metrics
"ema": ema, # Smoothed metrics
"forecast": forecast # Forecasted metrics
})

# Control signal (when policy triggers)
await bus.publish("drg.signal.{action}", {
"subsystem": subsystem,
"action": action,
"metric": metric,
"threshold": threshold,
"factor": factor
})
```

#### Configuration

```python
@dataclass
class DRGConfig:
sensor: SensorConfig = field(default_factory=SensorConfig)
profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
control_interval: float = 0.5 # seconds
policies: Dict[str, List[PolicyRule]] = None
monitor: MonitorConfig = field(default_factory=MonitorConfig)

@dataclass
class SensorConfig:
interval_ms: int = 250

@dataclass
class ProfilerConfig:
ema_alpha: float = 0.3
kalman_Q: float = 0.01
kalman_R: float = 0.1

@dataclass
class MonitorConfig:
log_dir: Path = Path("logs/drg")
level: str = "INFO"
```

#### Usage Example

```python
from scarcity.governor import DynamicResourceGovernor, DRGConfig
from scarcity.runtime import get_bus

# Create DRG
drg = DynamicResourceGovernor(DRGConfig(), get_bus())

# Register subsystems
drg.register_subsystem("simulation", simulation_engine)
drg.register_subsystem("mpie", mpie_orchestrator)

# Start control loop
await drg.start()

# System runs with adaptive resource management
# DRG automatically:
# 1. Monitors resources every 250ms
# 2. Smooths with EMA and forecasts with Kalman
# 3. Evaluates policies against forecasts
# 4. Executes control actions
# 5. Records metrics for diagnostics

# Stop control loop
await drg.stop()
```

---

### Meta-Learning System

**Location**: `scarcity/meta/`

The Meta-Learning System learns optimal hyperparameters across domains through a 5-tier architecture with online adaptation, cross-domain aggregation, Reptile-style optimization, and integrative governance.

#### 5-Tier Architecture

```
Tier 1: Domain-Level Meta-Learning
↓
Tier 2: Cross-Domain Aggregation
↓
Tier 3: Online Reptile Optimization
↓
Tier 4: Meta Scheduling
↓
Tier 5: Integrative Governance
```

#### Tier 1: Domain-Level Meta-Learning

**Purpose**: Track per-domain learning signals with adaptive confidence

**Algorithm**:
```python
def observe(domain_id, metrics, parameters):
# 1. Extract score and stability
score = metrics.get("meta_score", metrics.get("gain_p50", 0.0))
stability = metrics.get("stability_avg", 0.0)

# 2. Update EMA of score
ema_score = (1 - α)·ema_score + α·score

# 3. Compute score delta
score_delta = score - last_score

# 4. Update confidence
stability_term = max(stability, stability_floor)
sign_agreement = sign(score_delta) == sign(confidence)

confidence = decay·confidence + (1 - decay)·stability_term
if sign_agreement and score_delta > 0:
confidence += 0.05 # Bonus for consistent improvement

confidence = clip(confidence, 0.0, 1.0)

# 5. Compute adaptive meta learning rate
meta_lr = lr_min + (lr_max - lr_min)·confidence

# 6. Compute parameter delta
delta_vector = meta_lr·(param_vector - prev_vector)

return DomainMetaUpdate(
domain_id=domain_id,
vector=delta_vector,
keys=keys,
confidence=confidence,
timestamp=time.time(),
score_delta=score_delta
)
```

**Configuration**:
```python
@dataclass
class DomainMetaConfig:
ema_alpha: float = 0.3
meta_lr_min: float = 0.05
meta_lr_max: float = 0.2
stability_floor: float = 0.1
confidence_decay: float = 0.9
max_history: int = 20
```

#### Tier 2: Cross-Domain Aggregation

**Purpose**: Combine domain updates into global update

**Trimmed Mean Algorithm**:
```python
def aggregate(updates):
# 1. Filter by confidence
filtered = [u for u in updates if u.confidence >= min_confidence]

# 2. Union all parameter keys
keys = union_keys(filtered)

# 3. Stack vectors into matrix [n_domains, n_params]
matrix = stack_vectors(filtered, keys)

# 4. Trim top and bottom α fraction
k = floor(α·n_domains)
sorted_vals = sort(matrix, axis=0)
trimmed = sorted_vals[k : n_domains - k]

# 5. Compute mean
aggregate = mean(trimmed, axis=0)

return aggregate, keys, meta
```

**Configuration**:
```python
@dataclass
class CrossMetaConfig:
method: str = "trimmed_mean" # or "median"
trim_alpha: float = 0.1
min_confidence: float = 0.05
```

#### Tier 3: Online Reptile Optimizer

**Purpose**: Apply meta-optimization with adaptive step sizes

**Reptile Update**:
```python
def apply(aggregated_vector, keys, reward, drg_profile):
# 1. Update beta (step size) based on resources
if drg_profile["vram_high"] or drg_profile["latency_high"]:
beta *= 0.8 # Reduce step size under pressure
if drg_profile["bandwidth_free"]:
beta *= 1.1 # Increase when resources available

beta = clip(beta, beta_min, beta_max)

# 2. Record history for rollback
history.append(prior.copy())

# 3. Apply Reptile update: θ ← θ + β·Δθ
prior_vector = array([prior.get(key, 0.0) for key in keys])
updated_vector = prior_vector + beta·aggregated_vector

# 4. Update prior
prior = dict(zip(keys, updated_vector))

# 5. Update reward EMA
reward_ema = (1 - α)·reward_ema + α·reward

return prior
```

**Rollback Mechanism**:
```python
def should_rollback(reward):
delta = reward_ema - reward
return delta > rollback_delta

def rollback():
if history:
prior = history.pop()
return prior
```

**Configuration**:
```python
@dataclass
class MetaOptimizerConfig:
beta_init: float = 0.1
beta_max: float = 0.3
ema_alpha: float = 0.3
rollback_delta: float = 0.1
backup_versions: int = 10
```

#### Tier 4: Meta Scheduling

**Purpose**: Adaptive update intervals based on system load

**Algorithm**:
```python
def should_update(telemetry):
latency = telemetry.get("latency_ms", target_latency)
vram_high = telemetry.get("vram_high", 0.0)
bandwidth_low = telemetry.get("bandwidth_low", 0.0)

# Adapt interval
if latency > target_latency or vram_high:
interval = max(min_interval, floor(interval·0.7)) # Update more frequently
if bandwidth_low:
interval = min(max_interval, interval + 2) # Update less frequently
if latency < target_latency·0.7 and not vram_high:
interval = max(min_interval, round(interval·0.8))

# Add jitter
if jitter > 0:
interval = round(interval·(1 + uniform(-jitter, jitter)))

interval = clip(interval, min_interval, max_interval)

# Check if update is due
if window_counter >= interval:
window_counter = 0
return True

return False
```

**Configuration**:
```python
@dataclass
class MetaSchedulerConfig:
update_interval_windows: int = 10
latency_target_ms: float = 80.0
jitter: float = 0.1
min_interval_windows: int = 3
max_interval_windows: int = 20
```

#### Tier 5: Integrative Governance

**Purpose**: Rule-based meta-governance with safety checks

**Multi-Objective Reward**:
```python
def compute_reward(telemetry):
# Positive terms
accept = telemetry.get('accept_rate', 0.0)
stability = telemetry.get('stability_avg', 0.0)
contrast = telemetry.get('rcl_contrast', 0.0)

# Penalty terms
latency = telemetry.get('latency_ms', 0.0) / 120.0
vram = telemetry.get('vram_util', 0.0)
oom_flag = 1.0 if telemetry.get('oom_flag', False) else 0.0

reward = (
0.35·accept +
0.25·stability +
0.10·contrast -
0.15·latency -
0.10·vram -
0.20·oom_flag
)

return clip(reward, -1.0, 1.0)
```

**Policy Application**:
```python
def apply_policies(telemetry, reward, ema_reward):
changed_knobs = []

# Controller exploration adjustments
if accept < 0.06 and ema_reward < prev_ema_reward:
tau = clamp(tau + 0.1, tau_bounds) # Increase exploration
gamma = clamp(gamma + 0.05, gamma_bounds)
changed_knobs.extend(['tau', 'gamma'])

# Evaluator threshold adjustments
if accept_low_windows >= 5 and accept < 0.03:
g_min = clamp(g_min - 0.002, g_min_bounds) # Lower threshold
changed_knobs.append('g_min')

# Operator tier toggling
if latency > 120.0 or vram > 0.85:
tier3_topk = 0 # Disable expensive operators
tier2_enabled = False
changed_knobs.extend(['tier3_topk', 'tier2_enabled'])

return policy_update, changed_knobs
```

**Safety Checks**:
```python
def safety_checks(reward, ema_reward, previous_snapshot, changed_knobs):
# Check for reward degradation
if history:
prev_entry = history[-1]
delta = prev_entry['reward_avg'] - ema_reward

if delta > rollback_delta:
# Rollback to previous hyperparameters
rollback_previous(prev_entry['knobs_before'])
rollback_count += 1
return True

# Apply cooldown for changed knobs
for knob in changed_knobs:
cooldowns[knob] = cooldown_cycles

return False
```

**Configuration**:
```python
DEFAULT_CONFIG = {
'meta_score': {
'weights': {'accept': 0.35, 'stability': 0.25, 'contrast': 0.1},
'penalties': {'latency': 0.15, 'vram': 0.1, 'oom': 0.2},
'ema_alpha': 0.3,
},
'controller_policy': {
'tau_bounds': [0.5, 1.2],
'gamma_bounds': [0.1, 0.5],
},
'evaluator_policy': {
'g_min_bounds': [0.006, 0.02],
'lambda_ci_bounds': [0.4, 0.6],
},
'drg_policy': {
'sketch_dim_set': [512, 1024, 2048],
'n_paths_max': 128,
},
'safety': {
'rollback_delta': 0.1,
'cooldown_cycles': 5,
},
}
```

#### Complete Meta-Learning Flow

```
Domain 1 → DomainMetaLearner → DomainMetaUpdate (confidence-weighted)
Domain 2 → DomainMetaLearner → DomainMetaUpdate (confidence-weighted)
Domain 3 → DomainMetaLearner → DomainMetaUpdate (confidence-weighted)
↓
CrossDomainMetaAggregator
(trimmed mean aggregation)
↓
OnlineReptileOptimizer
(adaptive step size + rollback)
↓
Global Prior Update
↓
MetaIntegrativeLayer
(policy application + safety)
↓
Controller/Evaluator/Operator Updates
```

#### Usage Example

```python
from scarcity.meta import MetaLearningAgent, MetaLearningConfig
from scarcity.runtime import get_bus

# Create meta-learning agent
config = MetaLearningConfig()
meta_agent = MetaLearningAgent(get_bus(), config)

# Start meta-learning
await meta_agent.start()

# Meta-learning automatically:
# 1. Observes domain performance
# 2. Aggregates cross-domain updates
# 3. Optimizes global prior with Reptile
# 4. Schedules updates adaptively
# 5. Applies governance policies
# 6. Rolls back on reward degradation

# Stop meta-learning
await meta_agent.stop()
```

---


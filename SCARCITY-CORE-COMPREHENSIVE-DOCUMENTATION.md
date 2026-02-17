# SCARCITY Core Library - Comprehensive Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Engine Layer (MPIE)](#engine-layer-mpie)
5. [Federation Layer](#federation-layer)
6. [Meta-Learning Layer](#meta-learning-layer)
7. [Simulation Engine](#simulation-engine)
8. [Dynamic Resource Governor (DRG)](#dynamic-resource-governor-drg)
9. [Stream Processing](#stream-processing)
10. [Runtime System](#runtime-system)
11. [Federated Model Interface (FMI)](#federated-model-interface-fmi)
12. [Mathematical Foundations](#mathematical-foundations)
13. [API Reference](#api-reference)
14. [Implementation Details](#implementation-details)

---

## Overview

SCARCITY is an online-first framework for scarcity-aware deep learning that provides a complete runtime for adaptive, resource-efficient machine learning with real-time performance feedback and dynamic optimization. The core library implements a sophisticated multi-layered architecture designed for federated learning, online inference, and adaptive resource management.

### Key Features

- **Multi-Path Inference Engine (MPIE)**: Online bandit-based path exploration with UCB/Thompson sampling
- **Federated Learning**: Decentralized model aggregation with privacy preservation
- **Meta-Learning**: Cross-domain adaptation using online Reptile optimization
- **Dynamic Resource Governance**: Adaptive resource allocation based on system telemetry
- **Real-time Simulation**: Agent-based modeling with 3D visualization
- **Stream Processing**: Continuous data ingestion with backpressure control
- **Event-Driven Architecture**: Asynchronous pub/sub communication fabric

### Version Information

- **Version**: 1.0.0
- **Author**: Omega Makena
- **License**: See LICENSE file

---

## Architecture

The SCARCITY core library follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  FMI Service  │  Simulation  │  Meta-Learning  │    DRG     │
├─────────────────────────────────────────────────────────────┤
│              Federation Layer (Packets & Coordination)      │
├─────────────────────────────────────────────────────────────┤
│                Engine Layer (MPIE Orchestrator)            │
├─────────────────────────────────────────────────────────────┤
│              Stream Processing & Runtime Bus                │
├─────────────────────────────────────────────────────────────┤
│                    Core Operators & Utils                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Data Ingestion**: Stream sources feed data windows to the engine
2. **Path Exploration**: MPIE proposes and evaluates candidate paths using bandit algorithms
3. **Federation**: Successful paths are packaged and shared across domains
4. **Meta-Learning**: Cross-domain patterns are learned and applied
5. **Resource Management**: DRG monitors system resources and adapts behavior
6. **Simulation**: Agent-based models provide what-if analysis capabilities

---

## Core Components

### Module Structure

```python
scarcity/
├── __init__.py                 # Main package exports
├── engine/                     # Multi-Path Inference Engine
├── federation/                 # Federated learning components
├── meta/                      # Meta-learning algorithms
├── simulation/                # Simulation engine
├── governor/                  # Dynamic Resource Governor
├── stream/                    # Stream processing
├── runtime/                   # Event bus and telemetry
└── fmi/                      # Federated Model Interface
```

### Key Exports

```python
from scarcity.engine import MPIEOrchestrator, BanditRouter, Evaluator
from scarcity.federation import FederationCoordinator, PathPack
from scarcity.meta import MetaLearningAgent, DomainMetaLearner
from scarcity.simulation import SimulationEngine, AgentRegistry
from scarcity.governor import DynamicResourceGovernor
from scarcity.runtime import EventBus, get_bus
```

---

## Engine Layer (MPIE)

The Multi-Path Inference Engine is the core component responsible for online learning and adaptive inference through bandit-based path exploration.

### MPIEOrchestrator

The main orchestrator coordinates the online inference pipeline following the Controller ⇆ Evaluator contract.

```python
class MPIEOrchestrator:
    """
    Multi-Path Inference Engine orchestrator.
    
    Coordinates online inference pipeline under resource constraints.
    Never blocks; maintains bounded state only.
    """
```

#### Key Methods

```python
async def start(self) -> None:
    """Start the orchestrator and subscribe to events."""

async def _handle_data_window(self, topic: str, data: Dict[str, Any]) -> None:
    """Handle incoming data window event with full Controller ⇆ Evaluator contract."""
```

#### Processing Pipeline

1. **Proposal Phase**: Controller proposes candidate paths using UCB/Thompson sampling
2. **Evaluation Phase**: Evaluator scores candidates and computes confidence intervals
3. **Selection Phase**: Diversity-aware selection with cost considerations
4. **Update Phase**: Bandit learning from shaped rewards
5. **Storage Phase**: Accepted paths stored in hypergraph structure
6. **Export Phase**: Insights exported for federation

### BanditRouter (Controller)

Implements online bandit policy for path proposal with exploration/exploitation balance.

```python
class BanditRouter:
    """
    Online bandit router for path proposal.
    
    Balances exploration vs exploitation using UCB/Thompson Sampling.
    Ensures diversity and respects resource caps.
    """
```

#### Bandit Algorithm

The controller uses Upper Confidence Bound (UCB) with diversity bonuses:

```
UCB(arm) = μ(arm) + τ√(2ln(T)/n(arm)) + γ·D(arm) - η·C(arm)
```

Where:
- `μ(arm)`: Mean reward for arm
- `τ`: Temperature parameter (exploration)
- `γ`: Diversity weight
- `η`: Cost weight
- `D(arm)`: Diversity score
- `C(arm)`: Cost estimate

#### Key Features

- **Drift Detection**: Page-Hinkley test for non-stationarity
- **Thompson Sampling**: Fallback for high-drift scenarios
- **Diversity Tracking**: Per-variable coverage counters
- **Resource Awareness**: DRG-driven parameter adaptation

### Evaluator

Scores candidate paths and produces shaped rewards for bandit learning.

```python
class Evaluator:
    """
    Path evaluator with bootstrap confidence intervals.
    
    Computes gain estimates, stability scores, and acceptance decisions.
    """
```

#### Evaluation Metrics

- **Gain**: ΔR² or -ΔNLL improvement
- **Confidence Interval**: Bootstrap-based uncertainty quantification
- **Stability**: Robustness across resamples
- **Cost**: Computational latency penalty

#### Reward Shaping

```python
reward = α·gain + β·diversity - γ·latency_penalty
```

### Data Types

#### Candidate

```python
@dataclass
class Candidate:
    path_id: str                    # Deterministic UUID
    vars: Tuple[int, ...]          # Variable indices
    lags: Tuple[int, ...]          # Lag values
    ops: Tuple[str, ...]           # Operations ("sketch", "attn")
    root: int                      # Root variable (bandit arm)
    depth: int                     # Path length
    domain: int                    # Domain identifier
    gen_reason: str                # Generation method
```

#### EvalResult

```python
@dataclass
class EvalResult:
    path_id: str                   # Reference to candidate
    gain: float                    # ΔR² or -ΔNLL
    ci_lo: float                   # Lower confidence bound
    ci_hi: float                   # Upper confidence bound
    stability: float               # Stability score [0,1]
    cost_ms: float                 # Computation time
    accepted: bool                 # Acceptance decision
    extras: Dict[str, Any]         # Additional metrics
```

#### Reward

```python
@dataclass
class Reward:
    path_id: str                   # Reference to candidate
    arm_key: Tuple[int, int]       # Bandit arm identifier
    value: float                   # Shaped reward [-1, +1]
    latency_penalty: float         # Non-negative penalty
    diversity_bonus: float         # Non-negative bonus
    accepted: bool                 # Acceptance status
```

---

## Federation Layer

The federation layer enables decentralized learning across multiple domains while preserving privacy and ensuring trust.

### FederationCoordinator

Manages peer membership and routing decisions.

```python
class FederationCoordinator:
    """Coordinates membership and routing for federated learning."""
    
    def register_peer(self, peer_id: str, endpoint: str, capabilities: Dict[str, float]) -> PeerInfo
    def update_trust(self, peer_id: str, trust: float) -> None
    def select_peers(self, count: int, min_trust: float = 0.3) -> List[PeerInfo]
```

### Packet Types

The federation layer defines several packet types for different kinds of information exchange:

#### PathPack

Contains discovered paths and their performance metrics.

```python
@dataclass
class PathPack:
    schema_hash: str                           # Schema identifier
    window_range: Tuple[int, int]             # Time window
    domain_id: int                            # Source domain
    revision: int                             # Version number
    edges: List[Tuple[str, str, float, float, float, int]]  # Path edges
    hyperedges: List[Dict[str, Any]]          # Higher-order relationships
    operator_stats: Dict[str, float]          # Performance metrics
    provenance: Provenance                    # Audit trail
```

#### EdgeDelta

Incremental updates to the path graph.

```python
@dataclass
class EdgeDelta:
    schema_hash: str                          # Schema identifier
    domain_id: int                            # Source domain
    revision: int                             # Version number
    upserts: List[Tuple[str, float, float, int, int, int]]  # Edge updates
    prunes: List[str]                         # Edges to remove
```

#### PolicyPack

Meta-learning policy updates.

```python
@dataclass
class PolicyPack:
    controller: Dict[str, float]              # Controller parameters
    evaluator: Dict[str, float]               # Evaluator parameters
    drg: Dict[str, float]                     # Resource governor settings
    evidence: Dict[str, float]                # Supporting evidence
```

#### CausalSemanticPack

Causal relationships and semantic concepts.

```python
@dataclass
class CausalSemanticPack:
    schema_hash: str                          # Schema identifier
    domain_id: int                            # Source domain
    revision: int                             # Version number
    pairs: List[CausalPair]                   # Causal relationships
    concepts: List[ConceptLink]               # Semantic concepts
```

### Privacy and Trust

- **Trust Scoring**: Peer reputation based on contribution quality
- **Privacy Guard**: Differential privacy mechanisms
- **Packet Validation**: Cryptographic integrity checks
- **Selective Sharing**: Trust-based information filtering

---

## Meta-Learning Layer

The meta-learning layer implements cross-domain adaptation using online optimization algorithms.

### MetaLearningAgent

High-level orchestrator for meta-learning processes.

```python
class MetaLearningAgent:
    """
    Coordinates domain meta updates, aggregation, optimisation, storage, and telemetry.
    """
    
    def __init__(self, bus: Optional[EventBus] = None, config: Optional[MetaLearningConfig] = None)
    async def start(self) -> None
    async def stop(self) -> None
```

#### Processing Flow

1. **Domain Updates**: Collect performance metrics from individual domains
2. **Cross-Domain Aggregation**: Combine updates across domains
3. **Online Optimization**: Apply Reptile-style meta-gradient updates
4. **Prior Distribution**: Update global parameter priors
5. **Rollback Mechanism**: Revert poor updates based on performance

### DomainMetaLearner

Handles meta-learning within individual domains.

```python
class DomainMetaLearner:
    """Domain-specific meta-learning with online adaptation."""
    
    def observe(self, domain_id: str, metrics: Dict[str, float], params: Dict[str, float]) -> DomainMetaUpdate
    def adapt_parameters(self, update: DomainMetaUpdate) -> Dict[str, float]
```

### CrossDomainMetaAggregator

Aggregates meta-updates across multiple domains.

```python
class CrossDomainMetaAggregator:
    """Aggregates domain meta-updates for global optimization."""
    
    def aggregate(self, updates: List[DomainMetaUpdate]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]
```

### OnlineReptileOptimizer

Implements online version of the Reptile meta-learning algorithm.

```python
class OnlineReptileOptimizer:
    """Online Reptile optimizer for meta-learning."""
    
    def apply(self, gradient: np.ndarray, keys: List[str], reward: float, drg_profile: Dict[str, Any]) -> Dict[str, float]
    def should_rollback(self, reward: float) -> bool
    def rollback(self) -> Dict[str, float]
```

#### Algorithm

The online Reptile update rule:

```
θ_{t+1} = θ_t + α(φ_i - θ_t)
```

Where:
- `θ_t`: Current meta-parameters
- `φ_i`: Task-specific parameters after adaptation
- `α`: Meta-learning rate (adaptive based on reward)

---

## Simulation Engine

The simulation engine provides agent-based modeling capabilities with real-time visualization.

### SimulationEngine

Main orchestrator for the simulation system.

```python
class SimulationEngine:
    """
    Integrates the simulation loop with SCARCITY runtime.
    """
    
    def __init__(self, registry: AgentRegistry, config: SimulationConfig, bus: Optional[EventBus] = None)
    async def start(self) -> None
    async def stop(self) -> None
    def run_whatif(self, scenario_id: str, node_shocks: Optional[Dict[str, float]] = None, 
                   edge_shocks: Optional[Dict[tuple, float]] = None, horizon: Optional[int] = None) -> Dict[str, Any]
```

### AgentRegistry

Manages simulation agents and their relationships.

```python
class AgentRegistry:
    """Registry for simulation agents with relationship tracking."""
    
    def register_agent(self, agent_id: str, agent_type: str, properties: Dict[str, Any]) -> None
    def update_edges(self, edges: List[Dict[str, Any]]) -> None
    def adjacency_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]
    def node_embeddings(self) -> np.ndarray
```

### SimulationEnvironment

Provides the environment context for agent interactions.

```python
class SimulationEnvironment:
    """Environment for agent-based simulation."""
    
    def step(self) -> Dict[str, Any]
    def state(self) -> EnvironmentState
    def reset(self) -> None
```

### DynamicsEngine

Implements the physics/dynamics of the simulation.

```python
class DynamicsEngine:
    """Simulation dynamics engine."""
    
    def step(self) -> Dict[str, float]
    def apply_shock(self, node_id: str, magnitude: float) -> None
```

### WhatIfManager

Handles scenario analysis and counterfactual simulations.

```python
class WhatIfManager:
    """Manager for what-if scenario analysis."""
    
    def run_scenario(self, scenario_id: str, node_shocks: Optional[Dict[str, float]] = None,
                     edge_shocks: Optional[Dict[tuple, float]] = None, 
                     horizon: Optional[int] = None) -> Dict[str, Any]
```

### VisualizationEngine

Provides 3D visualization capabilities (optional dependency).

```python
class VisualizationEngine:
    """3D visualization engine for simulation."""
    
    def render_frame(self, positions: np.ndarray, values: np.ndarray, 
                     adjacency: np.ndarray, stability: np.ndarray, lod: float = 1.0) -> Dict[str, Any]
```

---

## Dynamic Resource Governor (DRG)

The DRG monitors system resources and dynamically adapts behavior to maintain performance under resource constraints.

### DynamicResourceGovernor

Main coordinator for resource governance.

```python
class DynamicResourceGovernor:
    """
    Coordinates sensors, profiler, policies, and actuators to keep the system stable.
    """
    
    def __init__(self, config: DRGConfig, bus: Optional[EventBus] = None)
    def register_subsystem(self, name: str, handle: SubsystemHandle | object) -> None
    async def start(self) -> None
    async def stop(self) -> None
```

### ResourceSensors

Monitors system resource utilization.

```python
class ResourceSensors:
    """System resource monitoring."""
    
    def sample(self) -> Dict[str, float]
```

Monitored metrics:
- **CPU Usage**: Processor utilization percentage
- **Memory Usage**: RAM consumption
- **GPU Memory**: VRAM utilization (if available)
- **Network Bandwidth**: I/O throughput
- **Disk I/O**: Storage access patterns

### ResourceProfiler

Analyzes resource trends and forecasts future usage.

```python
class ResourceProfiler:
    """Resource trend analysis and forecasting."""
    
    def update(self, metrics: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]
```

Uses exponential moving averages and Kalman filtering for trend analysis.

### Policy System

The DRG uses a rule-based policy system for resource management:

```python
@dataclass
class PolicyRule:
    metric: str                    # Metric to monitor
    threshold: float               # Trigger threshold
    action: str                    # Action to take
    factor: float                  # Action intensity
    
    def triggered(self, value: float) -> bool:
        return value > self.threshold
```

#### Default Policies

- **Memory Pressure**: Reduce batch sizes, enable compression
- **CPU Overload**: Lower sampling rates, reduce parallelism
- **GPU Memory**: Decrease model precision, reduce cache sizes
- **Network Congestion**: Compress packets, reduce federation frequency
- **Disk I/O**: Enable caching, reduce logging verbosity

### ResourceActuators

Executes resource management actions on registered subsystems.

```python
class ResourceActuators:
    """Executes resource management actions."""
    
    def execute(self, subsystem: str, action: str, factor: float) -> bool
```

---

## Stream Processing

The stream processing layer handles continuous data ingestion with adaptive rate control.

### StreamSource

Async data source with PI-controller based rate regulation.

```python
class StreamSource:
    """
    Async data source that ingests data with adaptive rate control.
    
    Features:
    - Async iterator for non-blocking reads
    - PI-controller rate regulation
    - Backpressure detection via bus queue depth
    - Support for CSV, generator, and custom sources
    """
    
    def __init__(self, data_source: Callable | AsyncIterator | str, window_size: int = 1000,
                 name: str = "default", target_latency_ms: float = 100.0)
    async def read_chunk(self) -> Optional[np.ndarray]
    async def stream(self) -> AsyncIterator[np.ndarray]
```

### PIController

Proportional-Integral controller for adaptive rate regulation.

```python
class PIController:
    """
    Proportional-Integral controller for adaptive rate regulation.
    
    Formula: Δt_next = Δt_base + K_p * error + K_i * integral
    """
    
    def update(self, actual_latency_ms: float) -> float
```

The PI controller maintains target latency by adjusting data ingestion rates:

```
error = target_latency - actual_latency
integral += error
dt_next = dt_base + K_p * error + K_i * integral
```

### Data Sources

Supported data source types:
- **CSV Files**: Chunked reading with pandas integration
- **Async Iterators**: Custom async data generators
- **Callables**: Function-based data sources
- **Streaming APIs**: Real-time data feeds

---

## Runtime System

The runtime system provides the foundational event-driven architecture.

### EventBus

Central pub/sub communication fabric for SCARCITY.

```python
class EventBus:
    """
    Asynchronous pub/sub event broker for SCARCITY runtime.
    
    Features:
    - Non-blocking concurrent dispatch
    - Topic-based message routing
    - Automatic error isolation
    - Graceful shutdown support
    """
    
    async def publish(self, topic: str, data: Any) -> None
    def subscribe(self, topic: str, callback: Callable) -> None
    def unsubscribe(self, topic: str, callback: Callable) -> bool
```

#### Key Topics

- `data_window`: Incoming data batches
- `resource_profile`: DRG resource updates
- `processing_metrics`: Engine performance metrics
- `engine.insight`: Discovered insights
- `federation.*`: Federation packets
- `simulation.*`: Simulation events
- `meta_policy_update`: Meta-learning updates

### Global Bus Access

```python
from scarcity.runtime import get_bus

bus = get_bus()  # Get global singleton
```

---

## Federated Model Interface (FMI)

The FMI provides high-level orchestration for federated learning workflows.

### FMIService

Main service orchestrator for FMI operations.

```python
class FMIService:
    """
    Ties together validation, routing, aggregation, and emission.
    """
    
    async def ingest(self, payload: Mapping[str, Any]) -> ProcessOutcome
    def apply_drg_signal(self, signal: str) -> None
    def snapshot(self) -> Dict[str, Any]
```

### Processing Pipeline

1. **Validation**: Packet integrity and trust verification
2. **Routing**: Cohort assignment based on schema/domain
3. **Aggregation**: Cross-domain parameter combination
4. **Emission**: Result distribution to subscribers

### DRG Integration

The FMI adapts to resource constraints:
- **bandwidth_low**: Reduce precision to Q8
- **latency_high**: Suspend POP/CCS packets
- **vram_high**: Defer aggregation
- **util_low**: Restore full precision

---

## Mathematical Foundations

### Bandit Algorithms

The engine uses Upper Confidence Bound (UCB) with diversity bonuses:

```
UCB_i(t) = μ_i(t) + c√(ln(t)/n_i(t)) + γD_i(t) - ηC_i(t)
```

Where:
- `μ_i(t)`: Empirical mean reward for arm i
- `c`: Confidence parameter (temperature)
- `n_i(t)`: Number of times arm i was selected
- `D_i(t)`: Diversity score for arm i
- `C_i(t)`: Cost estimate for arm i

### Diversity Scoring

Diversity is computed as inverse coverage:

```
D_i = 1/√(1 + c_i)
```

Where `c_i` is the coverage count for variable i.

### Reward Shaping

Rewards combine multiple objectives:

```
r = α·gain + β·diversity - γ·latency - δ·cost
```

### Meta-Learning (Online Reptile)

The meta-learning update rule:

```
θ_{t+1} = θ_t + α_t(φ_i - θ_t)
```

With adaptive learning rate:

```
α_t = α_0 · exp(-λ·t) · (1 + β·reward_t)
```

### Sketch Operators

#### Polynomial Sketch

Approximates polynomial features using CountSketch:

```
poly_sketch(x, d) ≈ Σ_{|S|≤d} c_S · Π_{i∈S} x_i
```

#### Tensor Sketch

Approximates Kronecker products:

```
tensor_sketch(x₁, x₂) ≈ x₁ ⊗ x₂
```

Using hash-based convolution.

---

## API Reference

### Engine Layer

```python
# Main orchestrator
from scarcity.engine import MPIEOrchestrator

orchestrator = MPIEOrchestrator()
await orchestrator.start()

# Bandit controller
from scarcity.engine import BanditRouter

controller = BanditRouter(drg=resource_profile)
candidates = controller.propose(window_meta, schema, budget=100)
controller.update(rewards)

# Evaluator
from scarcity.engine import Evaluator

evaluator = Evaluator(drg=resource_profile)
results = evaluator.score(data, candidates)
rewards = evaluator.make_rewards(results, diversity_fn, candidates)
```

### Federation Layer

```python
# Coordinator
from scarcity.federation import FederationCoordinator, CoordinatorConfig

config = CoordinatorConfig(heartbeat_timeout=60.0)
coordinator = FederationCoordinator(config)
coordinator.register_peer("peer1", "http://peer1:8080", {"cpu": 0.8})

# Packets
from scarcity.federation import PathPack, EdgeDelta, PolicyPack

path_pack = PathPack(
    schema_hash="abc123",
    window_range=(0, 1000),
    domain_id=1,
    revision=1,
    edges=[],
    hyperedges=[],
    operator_stats={},
    provenance=provenance
)
```

### Meta-Learning Layer

```python
# Meta-learning agent
from scarcity.meta import MetaLearningAgent, MetaLearningConfig

config = MetaLearningConfig()
agent = MetaLearningAgent(config=config)
await agent.start()

# Domain meta-learner
from scarcity.meta import DomainMetaLearner, DomainMetaConfig

config = DomainMetaConfig(learning_rate=0.01)
learner = DomainMetaLearner(config)
update = learner.observe("domain1", metrics, params)
```

### Simulation Engine

```python
# Simulation setup
from scarcity.simulation import SimulationEngine, AgentRegistry, SimulationConfig

registry = AgentRegistry()
config = SimulationConfig()
engine = SimulationEngine(registry, config)
await engine.start()

# What-if analysis
result = engine.run_whatif(
    scenario_id="shock_test",
    node_shocks={"node1": 0.5},
    horizon=100
)
```

### Dynamic Resource Governor

```python
# DRG setup
from scarcity.governor import DynamicResourceGovernor, DRGConfig

config = DRGConfig(control_interval=0.5)
drg = DynamicResourceGovernor(config)
drg.register_subsystem("engine", engine_handle)
await drg.start()
```

### Stream Processing

```python
# Stream source
from scarcity.stream import StreamSource

source = StreamSource(
    data_source="data.csv",
    window_size=1000,
    target_latency_ms=100.0
)

async for chunk in source.stream():
    # Process chunk
    pass
```

### Runtime System

```python
# Event bus
from scarcity.runtime import get_bus

bus = get_bus()

# Subscribe to events
async def handle_data(topic, data):
    print(f"Received {data} on {topic}")

bus.subscribe("data_window", handle_data)

# Publish events
await bus.publish("data_window", {"data": [1, 2, 3]})
```

---

## Implementation Details

### Performance Characteristics

- **Latency**: Sub-100ms processing for typical workloads
- **Throughput**: 1000+ windows/second on modern hardware
- **Memory**: Bounded state with configurable limits
- **Scalability**: Horizontal scaling via federation

### Resource Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, GPU optional
- **Storage**: Configurable, typically <1GB for metadata

### Dependencies

- **Core**: numpy, asyncio (Python 3.8+)
- **Optional**: pandas (CSV support), matplotlib (visualization)
- **GPU**: CUDA toolkit for GPU acceleration

### Configuration

Most components accept configuration objects:

```python
@dataclass
class ComponentConfig:
    param1: float = 1.0
    param2: int = 100
    param3: str = "default"
```

### Error Handling

- **Graceful Degradation**: System continues with reduced functionality
- **Error Isolation**: Component failures don't cascade
- **Recovery**: Automatic retry with exponential backoff
- **Monitoring**: Comprehensive telemetry and logging

### Testing

The library includes comprehensive test coverage:
- Unit tests for individual components
- Integration tests for component interaction
- Property-based tests for algorithmic correctness
- Performance benchmarks

### Extensibility

The architecture supports extension through:
- **Custom Operators**: Add new sketch operators
- **Custom Policies**: Define DRG policies
- **Custom Sources**: Implement stream sources
- **Custom Aggregators**: Define federation aggregation methods

---

This comprehensive documentation covers all major aspects of the SCARCITY Core library. For specific implementation examples and advanced usage patterns, refer to the individual module documentation and example code in the repository.
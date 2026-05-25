# Simulation Utilities

This page documents the supporting files in the simulation module.

---

## engine.py — Simulation Orchestrator

### `SimulationConfig`

Unified configuration:

```python
@dataclass
class SimulationConfig:
    environment: EnvironmentConfig
    dynamics: DynamicsConfig
    scheduler: SchedulerConfig
    whatif: WhatIfConfig
    visualization: VisualizationConfig
    storage: SimulationStorageConfig
```

### `SimulationEngine`

Main orchestrator that:
- Runs the async simulation loop
- Subscribes to event bus topics
- Coordinates all sub-components

```python
engine = SimulationEngine(
    registry=agent_registry,
    config=SimulationConfig(),
    bus=event_bus
)

engine.start()  # Begin simulation
# ... later ...
engine.stop()   # Graceful shutdown
```

**Event subscriptions**:
- `engine.insight`: New relationships from discovery
- `sim.shock`: External shock requests
- `sim.telemetry`: System performance data

---

## agents.py — Agent Registry

### `NodeAgent`

Represents a variable in the simulation:

```python
@dataclass
class NodeAgent:
    node_id: str
    agent_type: str
    domain: int
    regime: int
    embedding: np.ndarray
    stability: float
    value: float = 0.0
```

### `EdgeLink`

Represents a causal relationship:

```python
@dataclass
class EdgeLink:
    edge_id: str
    source: str
    target: str
    weight: float
    stability: float
    confidence_interval: float
    regime: int
```

### `AgentRegistry`

Converts discovered relationships into simulation structures:

```python
registry = AgentRegistry()

# Load from HypergraphStore snapshot
registry.load_from_store_snapshot(store.snapshot())

# Update with new edges
registry.update_edges(new_edges)

# Get matrices for computation
adj, stability, node_ids = registry.adjacency_matrix()
```

---

## dynamics.py — Dynamic Equations

Applies behavioral equations to propagate state:

```python
class DynamicsEngine:
    def update(self, state, dt):
        # Apply all behavioral equations
        # Returns new state
        pass
```

Integrates with discovered relationships to determine propagation paths.

---

## environment.py — Simulation Environment

Sets up the simulation world:

```python
class SimulationEnvironment:
    def __init__(self, config: EnvironmentConfig):
        self.nodes = {}
        self.edges = {}
        self.history = []
```

Manages:
- Node/edge lifecycles
- Regime tracking
- Environment boundaries

---

## scheduler.py — Time Scheduling

Controls simulation timing:

```python
class SimulationScheduler:
    def __init__(self, config: SchedulerConfig):
        self.dt = config.dt
        self.current_time = 0
        self.tick_count = 0
    
    def advance(self):
        self.current_time += self.dt
        self.tick_count += 1
```

Supports:
- Variable time steps
- Adaptive pacing based on system load
- Synchronization with real time (for dashboards)

---

## monitor.py — State Monitoring

Records and validates simulation state:

```python
class SimulationMonitor:
    def record(self, state):
        self.history.append(state)
        self._check_anomalies(state)
    
    def _check_anomalies(self, state):
        if state["gdp"] < 0:
            logger.warning("Negative GDP detected!")
```

---

## storage.py — Persistence

Saves simulation state:

```python
class SimulationStorage:
    def save_checkpoint(self, state, path):
        # Serialize to disk
        pass
    
    def load_checkpoint(self, path):
        # Restore from disk
        pass
```

Used for:
- Scenario checkpoints
- Recovery after crashes
- What-if branching

---

## whatif.py — Counterfactual Scenarios

### `WhatIfConfig`

```python
@dataclass
class WhatIfConfig:
    default_horizon: int = 12
    max_concurrent_scenarios: int = 5
    fork_on_scenario: bool = True
```

### `WhatIfManager`

Runs counterfactual scenarios:

```python
manager = WhatIfManager(config)

# Run scenario
results = manager.run_scenario(
    base_state=current_state,
    shocks={"interest_rate": 0.02},
    horizon=24
)

# Compare outcomes
print(f"GDP impact: {results['gdp_end'] - current_state['gdp']}")
```

**Process**:
1. Fork current state
2. Apply specified shocks
3. Run simulation for horizon steps
4. Return trajectory and final state

---

## visualization3d.py — 3D Visualization

Renders the relationship graph in 3D:

### `VisualizationConfig`

```python
@dataclass
class VisualizationConfig:
    enabled: bool = False
    update_interval: float = 0.1
    node_scale: float = 1.0
    edge_opacity: float = 0.5
```

### `VisualizationEngine`

```python
viz = VisualizationEngine(config)

# Render frame
frame = viz.render(
    node_positions=registry.node_embeddings(),
    adjacency=adjacency_matrix,
    node_colors=node_values
)

# Get frame data for frontend
data = viz.get_frame_data()
```

Outputs WebGL-compatible data for browser rendering.

---

## Index

| File | Purpose |
|------|---------|
| `sfc.py` | Stock-Flow Consistent economic model |
| `engine.py` | Simulation orchestrator |
| `agents.py` | Node and edge representations |
| `dynamics.py` | Behavioral equation application |
| `environment.py` | Simulation world setup |
| `scheduler.py` | Time step management |
| `monitor.py` | State recording and validation |
| `storage.py` | Checkpoint persistence |
| `whatif.py` | Counterfactual scenarios |
| `visualization3d.py` | 3D graph rendering |

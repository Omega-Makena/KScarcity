# KShield Backend Core — Overview

The **backend/app/core** module provides the FastAPI application's core domain logic, managing the Scarcity framework components.

---

## Purpose

The backend needs to:
- **Coordinate Scarcity components**: Bus, MPIE, DRG, Federation
- **Manage domains**: Multi-domain simulation support
- **Handle configuration**: Settings and dependencies
- **Provide datasets**: Demo data generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ScarcityCoreManager                          │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ EventBus │    │  MPIE    │    │   DRG    │    │ Federation│ │
│   │ (runtime)│    │(discover)│    │(governor)│    │(learning) │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        ▼                                               ▼        │
│   ┌──────────┐                                   ┌──────────┐  │
│   │ Telemetry│                                   │ Simulation│  │
│   │(metrics) │                                   │ (SFC)     │  │
│   └──────────┘                                   └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ScarcityCoreManager (`scarcity_manager.py`)

### Overview

Manages lifecycle of all scarcity core components:

```python
from app.core.scarcity_manager import ScarcityCoreManager

manager = ScarcityCoreManager(settings=get_settings())

# Initialize in dependency order
await manager.initialize()

# Start components
await manager.start()

# ... application runs ...

# Shutdown
await manager.stop()
```

### Initialization Order

1. **Runtime Bus** — Foundation for communication
2. **MPIE Orchestrator** — Causal discovery
3. **DRG** — Resource governor
4. **Federation** — Distributed learning
5. **Meta Learning** — Hyperparameter optimization
6. **Simulation** — SFC economy

### Methods

| Method | Purpose |
|--------|---------|
| `initialize()` | Create all components |
| `start()` | Start all components |
| `stop()` | Graceful shutdown |
| `get_status()` | Component status dict |
| `get_bus_statistics()` | EventBus metrics |
| `get_telemetry_history()` | Historical telemetry |

---

## DomainManager (`domain_manager.py`)

### Overview

Manages multi-domain simulation:

```python
from app.core.domain_manager import DomainManager, DistributionType

manager = DomainManager(persistence_path="domains.json")

# Create domain
domain = manager.create_domain(
    name="Healthcare",
    distribution_type=DistributionType.NORMAL,
    distribution_params={"mean": 0.0, "std": 1.0}
)

# Domain operations
manager.pause_domain(domain.id)
manager.resume_domain(domain.id)
```

### Domain Model

```python
@dataclass
class Domain:
    id: int
    name: str
    distribution_type: DistributionType
    distribution_params: Dict[str, float]
    status: DomainStatus = DomainStatus.ACTIVE
    synthetic_enabled: bool = True
    total_windows: int = 0
    federation_rounds: int = 0
```

### Distribution Types

```python
class DistributionType(Enum):
    NORMAL = "normal"
    SKEWED = "skewed"
    BIMODAL = "bimodal"
```

---

## FederationCoordinator (`federation_coordinator.py`)

Coordinates federated learning across domains:

```python
coordinator = FederationCoordinator(config=config)

# Run federation round
await coordinator.run_round()
```

---

## File Guide

| File | Purpose |
|------|---------|
| `scarcity_manager.py` | ScarcityCoreManager lifecycle |
| `domain_manager.py` | Multi-domain management |
| `domain_data_store.py` | Domain data storage |
| `federation_coordinator.py` | Federation orchestration |
| `multi_domain_generator.py` | Synthetic data generation |
| `config.py` | Application settings |
| `dependencies.py` | FastAPI dependencies |
| `datasets.py` | Dataset utilities |
| `demo_mode.py` | Demo mode handling |
| `error_handlers.py` | Error handling |
| `logging_config.py` | Logging setup |

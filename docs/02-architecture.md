# Architecture & System Design

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Overview │ │  Engine  │ │Federation│ │ Domains  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────┴────────────────────────────────────┐
│                  Backend (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Layer (v2)                           │  │
│  │  /runtime  /mpie  /drg  /federation  /domains        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ScarcityCoreManager (Orchestration)          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ Event Bus
┌────────────────────────┴────────────────────────────────────┐
│              Scarcity Core Components                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Runtime  │ │   MPIE   │ │   DRG    │ │   Meta   │      │
│  │   Bus    │ │Orchestr. │ │ Governor │ │ Learning │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │Federation│ │Simulation│ │  Domain  │                   │
│  │Coordinat.│ │  Engine  │ │ Manager  │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

### Root Directory
```
scace4/
├── backend/                 # Python backend service
├── scarcity/               # Core ML library
├── scarcity-deep-dive/     # React frontend
├── docs/                   # Documentation (this folder)
├── package-lock.json       # Frontend dependencies lock
└── README.md              # Project readme
```

### Backend Structure (`backend/`)
```
backend/
├── app/
│   ├── api/
│   │   ├── v1/            # Deprecated API (mock data)
│   │   └── v2/            # Current API (real components)
│   │       ├── endpoints/
│   │       │   ├── runtime.py      # Runtime Bus API
│   │       │   ├── mpie.py         # MPIE API
│   │       │   ├── drg.py          # DRG API
│   │       │   ├── federation.py   # Federation API
│   │       │   ├── federation_v2.py # Multi-domain federation
│   │       │   ├── meta.py         # Meta-learning API
│   │       │   ├── simulation.py   # Simulation API
│   │       │   ├── domains.py      # Domain management
│   │       │   ├── domain_data.py  # Domain data viz
│   │       │   ├── demo.py         # Demo mode
│   │       │   ├── metrics.py      # System metrics
│   │       │   └── health.py       # Health checks
│   │       └── routes.py           # Router aggregation
│   ├── core/
│   │   ├── config.py              # Configuration
│   │   ├── scarcity_manager.py    # Component orchestration
│   │   ├── domain_manager.py      # Domain lifecycle
│   │   ├── multi_domain_generator.py # Synthetic data
│   │   ├── federation_coordinator.py # Federation logic
│   │   ├── domain_data_store.py   # Data storage
│   │   ├── demo_mode.py           # Demo orchestration
│   │   ├── logging_config.py      # Logging setup
│   │   ├── error_handlers.py      # Error handling
│   │   └── dependencies.py        # DI container
│   ├── engine/
│   │   └── runner.py              # Engine lifecycle
│   ├── schemas/
│   │   ├── domains.py             # Domain schemas
│   │   ├── mpie.py                # MPIE schemas
│   │   ├── metrics.py             # Metrics schemas
│   │   └── ...                    # Other schemas
│   ├── simulation/
│   │   └── manager.py             # Simulation manager
│   └── main.py                    # FastAPI app
├── scripts/
│   ├── run_demo.py                # Demo runner
│   ├── start_data_feed.py         # Data feed starter
│   └── move.py                    # Utility script
├── tests/
│   ├── test_api_endpoints.py      # API tests
│   └── test_v2_endpoints.py       # V2 API tests
├── data/                          # Runtime data
├── logs/                          # Log files
├── artifacts/                     # Generated artifacts
├── requirements.txt               # Python dependencies
├── main.py                        # Entry point
└── README.md                      # Backend docs
```


### Core Library Structure (`scarcity/`)
```
scarcity/
├── runtime/
│   ├── bus.py                 # Event bus implementation
│   └── telemetry.py           # Telemetry collection
├── engine/
│   ├── engine.py              # MPIE Orchestrator
│   ├── controller.py          # Path controller
│   ├── encoder.py             # Feature encoding
│   ├── evaluator.py           # Statistical evaluation
│   ├── store.py               # Hypergraph store
│   ├── resource_profile.py    # Resource profiling
│   └── operators/             # Causal operators
├── governor/
│   ├── drg_core.py            # DRG implementation
│   ├── monitor.py             # Resource monitoring
│   ├── policies.py            # Control policies
│   ├── actuators.py           # Resource actuators
│   └── sensors.py             # System sensors
├── federation/
│   ├── coordinator.py         # Federation coordinator
│   ├── client_agent.py        # Client agent
│   ├── aggregator.py          # Model aggregation
│   ├── privacy_guard.py       # Privacy mechanisms
│   ├── reconciler.py          # State reconciliation
│   └── transport.py           # Network transport
├── meta/
│   ├── meta_learning.py       # Meta-learning agent
│   ├── integrative_meta.py    # Meta supervisor
│   ├── domain_meta.py         # Domain-specific meta
│   ├── cross_meta.py          # Cross-domain meta
│   └── optimizer.py           # Meta optimizer
├── simulation/
│   ├── engine.py              # Simulation engine
│   ├── agents.py              # Agent registry
│   ├── dynamics.py            # System dynamics
│   ├── visualization3d.py     # 3D visualization
│   └── whatif.py              # What-if analysis
├── stream/
│   ├── source.py              # Data sources
│   ├── window.py              # Windowing
│   ├── schema.py              # Schema management
│   └── cache.py               # Data caching
└── dashboard/
    ├── server.py              # Dashboard server
    └── api/                   # Dashboard API
```

### Frontend Structure (`scarcity-deep-dive/src/`)
```
scarcity-deep-dive/src/
├── pages/
│   ├── Overview.tsx           # System overview
│   ├── Runtime.tsx            # Runtime Bus dashboard
│   ├── Engine.tsx             # MPIE dashboard
│   ├── Governor.tsx           # DRG dashboard
│   ├── Federation.tsx         # Federation dashboard
│   ├── FederationDashboard.tsx # Multi-domain federation
│   ├── MetaLearning.tsx       # Meta-learning dashboard
│   ├── Simulation.tsx         # 3D simulation
│   ├── Domains.tsx            # Domain management
│   ├── DomainData.tsx         # Domain data visualization
│   ├── DataIngestion.tsx      # Data upload
│   ├── Debug.tsx              # Debug tools
│   └── Test.tsx               # Test page
├── components/
│   ├── Layout.tsx             # Main layout
│   ├── DemoMode.tsx           # Demo mode UI
│   ├── DemoOrchestrator.tsx   # Demo orchestration
│   ├── DomainUpload.tsx       # File upload
│   ├── simulation/            # Simulation components
│   └── ui/                    # UI components (shadcn)
├── lib/
│   ├── api.ts                 # API client
│   ├── demoData.ts            # Demo data
│   └── utils.ts               # Utilities
├── hooks/
│   ├── use-toast.ts           # Toast notifications
│   └── use-mobile.tsx         # Mobile detection
├── App.tsx                    # App root
└── main.tsx                   # Entry point
```

## Component Interactions

### Data Flow
```
1. Data Ingestion
   User Upload → Domain Manager → Multi-Domain Generator → Runtime Bus

2. Processing
   Runtime Bus → MPIE Orchestrator → Causal Discovery → Hypergraph Store

3. Resource Management
   DRG Monitor → Resource Sensors → Policy Engine → Actuators

4. Federation
   Domain Models → Federation Coordinator → Aggregation → Distribution

5. Visualization
   Hypergraph Store → API → Frontend → 3D Rendering
```

### Event Flow
```
Runtime Bus (Pub/Sub)
├── data_window          # New data arrived
├── processing_metrics   # MPIE metrics
├── resource_profile     # DRG resource state
├── federation_update    # Federation events
├── meta_update          # Meta-learning events
└── simulation_state     # Simulation updates
```

## Design Patterns

### 1. Event-Driven Architecture
All components communicate via the Runtime Bus using publish-subscribe pattern.

**Benefits**:
- Loose coupling
- Async processing
- Easy to add new components

### 2. Dependency Injection
Components receive dependencies through constructors.

**Example**:
```python
class MPIEOrchestrator:
    def __init__(self, bus: EventBus):
        self.bus = bus
```

### 3. Manager Pattern
ScarcityCoreManager orchestrates component lifecycle.

**Responsibilities**:
- Initialize components in correct order
- Start/stop components
- Handle errors and cleanup

### 4. Repository Pattern
Hypergraph store abstracts data persistence.

**Benefits**:
- Testable
- Swappable backends
- Clear interface

### 5. Strategy Pattern
Federation uses different aggregation strategies.

**Strategies**:
- FedAvg (Federated Averaging)
- Weighted (by domain size)
- Adaptive (by performance)

# Backend Implementation Guide

## Overview

The backend is a FastAPI application that orchestrates the SCARCITY framework components and exposes them via REST API.

**Technology Stack**:
- Python 3.11+
- FastAPI 0.115.0
- Uvicorn (ASGI server)
- NumPy (numerical computing)
- Pydantic (data validation)
- asyncio (async/await)

## Application Lifecycle

### 1. Startup Sequence

**File**: `backend/app/main.py`

```python
@app.on_event("startup")
async def startup_event():
    """
    Startup sequence:
    1. Initialize dataset registry
    2. Create simulation manager
    3. Start engine runner
    4. Initialize scarcity core components
    5. Start scarcity core components
    """
    # 1. Dataset registry
    dataset_registry = DatasetRegistry(...)
    app.state.dataset_registry = dataset_registry
    
    # 2. Simulation manager
    simulation = SimulationManager(...)
    app.state.simulation = simulation
    
    # 3. Engine runner
    engine_runner = EngineRunner()
    await engine_runner.start()
    app.state.engine_runner = engine_runner
    
    # 4. Scarcity core
    if settings.scarcity_enabled:
        scarcity_manager = ScarcityCoreManager()
        await scarcity_manager.initialize()
        await scarcity_manager.start()
        app.state.scarcity_manager = scarcity_manager
```

### 2. Component Initialization Order

**Critical**: Components must be initialized in dependency order

```
1. Runtime Bus (foundation)
2. Domain Manager (data sources)
3. Multi-Domain Generator (data generation)
4. Domain Data Store (data storage)
5. MPIE Orchestrator (causal discovery)
6. DRG (resource monitoring)
7. Federation Coordinator (model sharing)
8. Meta-Learning Agent (optimization)
9. Simulation Engine (visualization)
```

**Implementation**: `backend/app/core/scarcity_manager.py`

```python
async def initialize(self):
    # 1. Runtime Bus
    self.bus = get_bus()
    
    # 2. Domain Manager
    self.domain_manager = DomainManager()
    
    # 3. Multi-Domain Generator
    self.multi_domain_generator = MultiDomainDataGenerator(
        domain_manager=self.domain_manager,
        bus=self.bus
    )
    
    # 4. Domain Data Store
    self.domain_data_store = DomainDataStore(bus=self.bus)
    
    # 5. MPIE
    self.mpie = MPIEOrchestrator(bus=self.bus)
    
    # 6. DRG
    self.drg = DynamicResourceGovernor(bus=self.bus)
    self.drg.register_subsystem("mpie", self.mpie)
    
    # 7. Federation
    self.federation_coordinator_v2 = FederationCoordinator()
    
    # 8. Meta-Learning
    self.meta = MetaLearningAgent(bus=self.bus)
    
    # 9. Simulation
    self.simulation = SimulationEngine(bus=self.bus)
```

### 3. Shutdown Sequence

**Reverse order** of startup:

```python
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown in reverse order:
    1. Stop scarcity core
    2. Stop engine runner
    3. Clean up simulation
    """
    # 1. Scarcity core
    if hasattr(app.state, "scarcity_manager"):
        await app.state.scarcity_manager.stop()
    
    # 2. Engine runner
    if hasattr(app.state, "engine_runner"):
        await app.state.engine_runner.stop()
    
    # 3. Simulation
    if hasattr(app.state, "simulation"):
        del app.state.simulation
```

## API Structure

### API Versioning

**v1 API** (Deprecated):
- Path: `/api/v1`
- Status: Mock data only
- Purpose: Legacy compatibility

**v2 API** (Current):
- Path: `/api/v2`
- Status: Production
- Purpose: Real scarcity components

### Endpoint Organization

**File**: `backend/app/api/v2/routes.py`

```python
api_v2_router = APIRouter()

# Health
api_v2_router.include_router(health.router, tags=["health-v2"])

# Metrics
api_v2_router.include_router(metrics.router, prefix="/metrics")

# Domains
api_v2_router.include_router(domains.router, prefix="/domains")

# Domain Data
api_v2_router.include_router(domain_data.router, prefix="/domains")

# Demo
api_v2_router.include_router(demo.router, prefix="/demo")

# Runtime Bus
api_v2_router.include_router(runtime.router, prefix="/runtime")

# MPIE
api_v2_router.include_router(mpie.router, prefix="/mpie")

# DRG
api_v2_router.include_router(drg.router, prefix="/drg")

# Federation
api_v2_router.include_router(federation.router, prefix="/federation")
api_v2_router.include_router(federation_v2.router, prefix="/federation-v2")

# Meta-Learning
api_v2_router.include_router(meta.router, prefix="/meta")

# Simulation
api_v2_router.include_router(simulation.router, prefix="/simulation")
```

## Key Modules

### 1. Domain Manager

**File**: `backend/app/core/domain_manager.py`

**Purpose**: Manage domain lifecycle and configuration

**Key Classes**:
```python
class Domain:
    id: int
    name: str
    distribution_type: DistributionType
    distribution_params: Dict[str, float]
    status: DomainStatus
    synthetic_enabled: bool
    total_windows: int
    manual_uploads: int
    federation_rounds: int

class DomainManager:
    domains: Dict[int, Domain]
    name_registry: Dict[str, int]
    
    def create_domain(...) -> Domain
    def get_domain(domain_id: int) -> Domain
    def pause_domain(domain_id: int)
    def resume_domain(domain_id: int)
    def remove_domain(domain_id: int)
    def list_domains() -> List[Domain]
```

**Persistence**:
- Saves to: `data/domains.json`
- Auto-saves on changes
- Loads on startup

### 2. Multi-Domain Generator

**File**: `backend/app/core/multi_domain_generator.py`

**Purpose**: Generate synthetic data for domains

**Key Classes**:
```python
class DataGenerator:
    distribution_type: DistributionType
    params: Dict[str, float]
    
    def generate(window_size: int) -> np.ndarray

class DomainScheduler:
    tasks: Dict[int, asyncio.Task]
    
    def start_domain(domain_id, interval, offset, callback)
    def stop_domain(domain_id)

class MultiDomainDataGenerator:
    domain_manager: DomainManager
    bus: EventBus
    generators: Dict[int, DataGenerator]
    scheduler: DomainScheduler
    
    async def start()
    async def stop()
    async def generate_for_domain(domain_id) -> np.ndarray
```

**Data Flow**:
```
1. Scheduler triggers generation for domain
2. DataGenerator creates synthetic data
3. Data published to Runtime Bus
4. MPIE processes data
5. Domain statistics updated
```

### 3. Federation Coordinator

**File**: `backend/app/core/federation_coordinator.py`

**Purpose**: Coordinate federated learning

**Key Classes**:
```python
class ModelUpdate:
    domain_id: int
    timestamp: datetime
    weights: np.ndarray
    num_samples: int
    loss: float

class Connection:
    from_domain: int
    to_domain: int
    established_at: datetime
    updates_shared: int

class ModelAggregator:
    strategy: AggregationStrategy
    
    def aggregate(updates: List[ModelUpdate]) -> np.ndarray

class FederationCoordinator:
    active: bool
    connections: Dict[Tuple[int, int], Connection]
    aggregator: ModelAggregator
    pending_updates: Dict[int, List[ModelUpdate]]
    
    def enable_federation()
    def create_connection(from_domain, to_domain)
    async def share_update(from_domain, to_domain, update)
    def aggregate_updates(domain_id) -> np.ndarray
    def create_full_mesh(domain_ids)
```

**Topologies**:
- Full Mesh: All-to-all connections
- Ring: Each domain connects to next
- Star: Central coordinator

### 4. Domain Data Store

**File**: `backend/app/core/domain_data_store.py`

**Purpose**: Store and retrieve domain data windows

**Key Classes**:
```python
class DomainDataStore:
    data: Dict[int, deque]  # domain_id → windows
    max_windows_per_domain: int
    
    async def start()
    async def stop()
    def get_windows(domain_id, limit, offset, source) -> List[DataWindow]
    def get_statistics(domain_id) -> DomainStatistics
    def get_latest_window(domain_id) -> DataWindow
```

**Storage**:
- In-memory circular buffer
- Max 1000 windows per domain
- Automatic eviction of old data

### 5. Scarcity Manager

**File**: `backend/app/core/scarcity_manager.py`

**Purpose**: Orchestrate all scarcity components

**Key Methods**:
```python
class ScarcityCoreManager:
    bus: EventBus
    mpie: MPIEOrchestrator
    drg: DynamicResourceGovernor
    federation_coordinator: FederationCoordinator
    meta: MetaLearningAgent
    simulation: SimulationEngine
    domain_manager: DomainManager
    
    async def initialize()
    async def start()
    async def stop()
    def get_status() -> dict
    def get_telemetry_history() -> List[dict]
```

**Telemetry Collection**:
- Subscribes to core topics
- Stores last 1000 events
- Provides query interface

## Configuration

**File**: `backend/app/core/config.py`

**Environment Variables**:
```python
class Settings(BaseSettings):
    # API
    project_name: str = "Scarce Demo Backend"
    api_v1_prefix: str = "/api/v1"
    api_v2_prefix: str = "/api/v2"
    
    # CORS
    allow_origins: list[str] = ["http://localhost:3000", ...]
    
    # Simulation
    simulation_seed: int = 42
    simulation_tick_seconds: float = 1.0
    
    # Scarcity Core
    scarcity_enabled: bool = True
    scarcity_mpie_enabled: bool = True
    scarcity_drg_enabled: bool = True
    scarcity_federation_enabled: bool = False
    scarcity_meta_enabled: bool = True
    
    # Resource Limits
    scarcity_mpie_max_candidates: int = 200
    scarcity_mpie_resamples: int = 1000
    scarcity_drg_control_interval: float = 0.5
    scarcity_drg_cpu_threshold: float = 90.0
    scarcity_drg_memory_threshold: float = 85.0
```

**Usage**:
```python
from app.core.config import get_settings

settings = get_settings()
if settings.scarcity_enabled:
    # Initialize scarcity components
    pass
```

## Error Handling

**File**: `backend/app/core/error_handlers.py`

**Global Exception Handlers**:
```python
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

## Logging

**File**: `backend/app/core/logging_config.py`

**Configuration**:
```python
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
```

**Usage**:
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Component started")
logger.error("Error occurred", exc_info=True)
```

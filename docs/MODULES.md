# Module Reference — K-Scarcity / K-SHIELD

---

## kshiked/core/

### `scarcity_bridge.py` — ScarcityBridge
Universal adapter connecting K-SHIELD to the Scarcity Engine.

```python
bridge = ScarcityBridge()
bridge.train("data/kenya_world_bank.csv")        # learns 306+ causal hypotheses
economy  = bridge.create_learned_economy()        # SFC with discovered relationships
top      = bridge.get_top_relationships(10)        # ranked causal chains
conf_map = bridge.get_confidence_map()             # per-variable confidence 0–1
score    = bridge.validate()                       # historical accuracy replay
```

**DRG Assurance Levels returned:**
- `HIGH` — confidence ≥ 0.85, recent data
- `MEDIUM` — confidence 0.65–0.85
- `LOW` — confidence < 0.65 or stale data
- `FALLBACK` — discovery failed, uses hardcoded baselines

---

## scarcity/simulation/

### Wave 2 Typed Multi-Sector SFC Stack

The simulation package now includes a typed, equation-first multi-sector engine that complements the legacy `sfc.py` and `research_sfc.py` paths.

| Module | Role |
|--------|------|
| `types.py` | Canonical typed contracts: `EconomyState`, `PolicyState`, `ShockVector`, `StepResult`, `SectorFeedback` |
| `parameters.py` | Kenya-calibrated parameter container `AllParams` with sector-level production, IO, household, fiscal, monetary, external, and banking blocks |
| `coupling_interface.py` | Cross-model coupling contracts and aggregation rules (`aggregate_feedback`, `MacroExposure`) |
| `accounting.py` | Residual-based stock-flow consistency checks and warning generation |
| `production.py` | CES production system with Cobb-Douglas limit handling near unit elasticity |
| `labor_market.py` | Employment closure, labor supply shock handling, and wage dynamics |
| `price_system.py` | Inflation/CPI/relative-price dynamics with long-run CPI anchoring |
| `households.py` | Disposable income, consumption, savings, deposits, and household loan updates |
| `government.py` | Fiscal balance, debt accumulation, and government bond-holder decomposition |
| `monetary.py` | Policy-rate setting with pass-through to loan/deposit/government rates |
| `foreign.py` | Trade, remittances, aid, BoP closure, FX pressure, exchange rate, and reserves |
| `banking.py` | Credit evolution, NPL stress, equity, reserves, and CAR updates |
| `sfc_engine.py` | `MultiSectorSFCEngine` orchestration, quarterly stepping, simulation loop, and steady-state search |

Minimal usage:

```python
from scarcity.simulation import MultiSectorSFCEngine
from scarcity.simulation.parameters import AllParams

engine = MultiSectorSFCEngine(params=AllParams.default_kenya())
results = engine.simulate(quarters=40)
state, iterations = engine.find_steady_state()
```

### Simulation Tests

| Test | Scope |
|------|-------|
| `scarcity/simulation/tests/test_coupling.py` | Coupling aggregation and macro exposure consistency |
| `scarcity/simulation/tests/test_accounting.py` | Accounting residual checks and inconsistency detection |
| `scarcity/simulation/tests/test_production.py` | CES/CD behavior and monotonic production responses |
| `scarcity/simulation/tests/test_steady_state.py` | Convergence and 200-quarter stability checks |

---

### `governance.py` — EconomicGovernor / EventActuator / SimSensor

| Class | Role |
|-------|------|
| `EconomicGovernor` | Enforces resource stability constraints, applies policy transmission |
| `EventActuator` | Executes governance signals to SFC economy |
| `SimSensor` | Extracts economic state vectors for analysis |

---

### `shocks.py` — Stochastic Shock Types

| Class | Model |
|-------|-------|
| `ImpulseShock` | Classic impulse with exponential decay |
| `OUProcessShock` | Ornstein-Uhlenbeck mean-reverting process |
| `BrownianShock` | Geometric Brownian Motion |
| `MarkovSwitchingShock` | Hamilton (1989) regime-switching |
| `JumpDiffusionShock` | Poisson jump process |
| `StudentTShock` | Fat-tailed shocks for stress testing |

---

### `policies.py` — Policy Registry
Default policy library: inflation targeting, counter-cyclical fiscal, exchange rate management.  
Configurable monetary, fiscal, and sectoral instrument parameters.

### `tensor_policies.py` — Policy Tensor Engine
Multi-dimensional policy space. Policies represented as tensors for composition and optimization.

---

## kshiked/pulse/

### `sensor.py` — PulseSensor
Main orchestrator. Maintains registry of 15+ signal detectors. Maps social media text → signal detections → PulseState updates.

### `primitives.py` — PulseState
Core state model:
```
PulseState
├── ScarcityVector[domain]     → resource scarcity per sector
├── ActorStress[actor_type]    → stress levels by actor category
├── BondStrength               → social cohesion metrics
└── instability_index          → aggregate instability 0–1
```

### `indices.py` — 8 Threat Indices

| Index | Key Inputs |
|-------|-----------|
| `PI` — Polarization | Language extremity, identity framing, bond fracture |
| `LEI` — Legitimacy Erosion | Authority rejection signals, institutional dismissal |
| `MRS` — Mobilization Readiness | Anger + scarcity + coordination signals |
| `ECI` — Elite Cohesion | Leadership disagreement, elite defection signals |
| `IWI` — Information Warfare | Rumor velocity, conspiracy propagation |
| `SFI` — Security Friction | Force use signals, stability erosion |
| `ECR` — Economic Cascade Risk | Multi-sector scarcity co-occurrence |
| `ETM` — Ethnic Tension Matrix | 12 Kenya ethnic group tension tracking |

**Severity levels:** CRITICAL (≥0.8) / HIGH (≥0.6) / ELEVATED (≥0.4) / NORMAL (<0.4)

### `simulation_connector.py` — SimulationShockGenerator
Maps threat indices to economic shocks:

| Index Threshold | Shock Generated |
|----------------|-----------------|
| Polarization HIGH | Confidence shock |
| LEI HIGH | Confidence + GDP shock |
| MRS HIGH | GDP + Inflation shock |
| ECI HIGH | GDP + Trade shock |
| IWI HIGH | Inflation + Confidence shock |
| SFI HIGH | GDP + Trade shock |
| ECR HIGH | GDP + Inflation + Currency shocks |
| ETM HIGH | Confidence + GDP shocks |

### `scrapers/` — Social Media Ingestion

| Module | Source |
|--------|--------|
| `x_client.py` | Twitter / X |
| `facebook_scraper.py` | Facebook |
| `instagram_scraper.py` | Instagram |
| `telegram_scraper.py` | Telegram |
| `reddit_scraper.py` | Reddit |
| `ecommerce/` | Jumia · Jiji · Kilimall price data |

---

## kshiked/simulation/

### Architecture Snapshot

The simulation layer combines scenario authoring, shock compilation, policy control,
post-processing sector projection, and data-driven validation.

**Core files:**
- `scenario_templates.py` — registries, preset scenarios, policy presets, merge helpers
- `compiler.py` — compiles stochastic shocks into channel vectors (+ metadata)
- `controller.py` — policy feedback loop over `SFCEconomy`
- `sector_engine.py` — transforms macro trajectory into 6-sector state projections
- `validation.py` — historical validation, moment matching, out-of-sample and retrodiction runners

### Execution Pipeline

1. Build shocks/policies from registries and presets (`scenario_templates.py`).
2. Compile shocks into SFC-compatible vectors (`ShockCompiler`).
3. Run macro trajectory in Scarcity SFC engine (`scarcity.simulation.sfc`).
4. Project macro outputs into multi-sector outcomes (`SectorSimulator.project`).
5. Score realism and stability with validation stack (`ValidationRunner`, `RetrodictionRunner`, etc.).

### Scenario and Policy Registries (`scenario_templates.py`)

Registry-first design (extensible, data-driven):
- `SHOCK_REGISTRY` — shock definitions and ranges (includes sectoral shocks + SFC mappings)
- `POLICY_INSTRUMENT_REGISTRY` — instrument metadata by policy domain
- `POLICY_TEMPLATES` — 15 preset policy responses
- `SCENARIO_LIBRARY` — 16 named scenario templates

Composition helpers:
- `merge_shock_vectors(...)` — additive superposition of multiple scenarios + optional custom shocks
- `merge_policy_instruments(...)` — layered policy preset merge with custom override precedence

Template model:
- `ScenarioTemplate` supports timed shock generation with `shock_onset`, `shock_duration`, and
  `shock_shape` in `{step, pulse, ramp, decay}`.

### Simulation Modes and Ripple Models (`sector_engine.py`)

**SimulationMode:**

| Mode | Description |
|------|-------------|
| `SINGLE_SECTOR` | Deep simulation for one sector with spillover hints for others |
| `MULTI_SECTOR` | Selected sectors with cross-sector ripple application |
| `FULL_SIMULATION` | Full six-sector projection with stacked shocks and weights |

**RippleModel:**
- `SIMULTANEOUS` — direct impacts applied immediately
- `CASCADING` — staged propagation across orders with decay
- `WEIGHTED_INTERDEPENDENCY` — influence-matrix-adjusted propagation

### Policy Control Loop (`controller.py`)

`PolicyController` runs a closed-loop process each step:
1. Extract state vector from economy outcomes/channels.
2. Evaluate actions with `PolicyTensorEngine`.
3. Map actions into concrete policy-rate/fiscal overrides.
4. Inject overrides through `economy.config.policy_schedule` for next-step execution.
5. Step the economy and append trajectory frame.

### Shock Compiler Surface (`compiler.py`)

`ShockCompiler.compile_with_metadata(...)` returns:
- channel vectors (`demand_shock`, `supply_shock`, `fiscal_shock`, `fx_shock`)
- metadata bundle including `regime_paths`, `jump_times`, and `confidence_bands`

### Validation and Retrodiction Stack (`validation.py`)

- `ValidationRunner` — episode detection + historical replay scoring
- `MomentMatcher` — distribution and autocorrelation matching
- `OutOfSampleValidator` — rolling-window holdout RMSE evaluation
- `ConvergenceDiagnostics` — Monte Carlo convergence checks
- `RetrodictionRunner` — named episode replay with direction/range scoring

### Export Surface and Boundaries

`kshiked/simulation/__init__.py` currently re-exports scenario/policy accessors plus
optional `FallbackBlender`, `ValidationRunner`, and `EpisodeDetector`.

Operational notes:
- `sector_engine.py` is a post-processing layer (does not mutate SFC internals).
- `controller.py` currently applies policy overrides via schedule mutation strategy.
- `compiler.py` uses `Any` in type annotations in `compile_with_metadata`.

---

## kshiked/federation/

### `node.py` — AegisNode
Extends `FederationClientAgent` (from Scarcity library).

Key behaviours:
- Security lattice clearance enforcement
- Per-packet trust scoring
- Knowledge graph merging from external nodes
- Ed25519 message authentication (CryptoSigner)

### `gossip.py` — Defense Gossip Protocol
Signal propagation between institution nodes.  
Exponential time-decay weighting of stale updates.  
EMA-based backoff under high latency.

### `security.py` — Cryptographic Primitives
- Pairwise HKDF-SHA256 masking
- Ed25519 / X25519 key management
- Byzantine detection utilities

---

## kshiked/causal/

### `economic_causal_discovery.py`
Uses Scarcity's `OnlineDiscoveryEngine` trained on World Bank Kenya dataset.  
Outputs: JSON causal graph with 40+ indicator nodes and edge strengths.

**Key indicators:** GDP, Inflation, Unemployment, Trade Balance, FX Rate, M2, Public Debt, Agri Output, Health Expenditure, Education Spend, Poverty Rate, Gini, Social Cohesion, Conflict Events, Rainfall, Food Prices.

---

## kshiked/causal_adapter/

| Module | Role |
|--------|------|
| `runner.py` | Orchestrates discovery pipeline end-to-end |
| `artifacts.py` | Caches discovered graphs, versioned |
| `config.py` | Training configuration (confidence thresholds, coverage) |
| `dataset.py` | Data pipeline from CSV to discovery input |
| `integration.py` | Adapter patterns for external callers |
| `spec_builder.py` | Generates discovery specifications |

---

## kshiked/hub.py — KShieldHub

Singleton. Central access to all subsystems.

```python
hub = KShieldHub.get_instance()
hub.pulse          # PulseSensor
hub.bridge         # ScarcityBridge
hub.simulate(...)  # run projections
hub.get_indices()  # current threat indices
```

---

## kshiked/ui/institution/backend/

| Module | Role |
|--------|------|
| `analytics_engine.py` | Cost of delay, inaction projections, county convergence |
| `report_narrator.py` | Threat indices → plain-English narratives |
| `executive_bridge.py` | Aggregates data for executive views |
| `federation_bridge.py` | Links institution nodes to Aegis Protocol |
| `learning_engine.py` | Federated learning round coordination |
| `auto_pipeline.py` | Automated CSV → discovery → projection pipeline |
| `research_engine.py` | Research-grade analysis tools for analysts |
| `sector_reports.py` | Per-sector status grids and summaries |
| `ontology.py` | Shared economic concept taxonomy |
| `schema_manager.py` | Institution data schema validation |
| `history_middleware.py` | Audit trail and decision history |
| `messaging.py` | Cross-institution secure messaging |
| `data_sharing.py` | Secure data sharing policies |
| `scarcity_bridge.py` | Institution-level ScarcityBridge adapter |
| `delta_sync.py` | Incremental sync for federated updates |
| `project_signals.py` | Institution project tracking signals |
| `models.py` | SQLAlchemy / SQLite models |

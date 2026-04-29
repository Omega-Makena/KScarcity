# Module Reference — K-Scarcity / K-SHIELD

See [BENCHMARK.md](BENCHMARK.md) for the master benchmark that verifies every component listed here.

---

## scarcity/engine/

### Online Discovery Engine (`engine_v2.py`) — `OnlineDiscoveryEngine`

Entry point for streaming causal discovery. Manages the full hypothesis lifecycle across
any variable set and any data frequency.

```python
engine = OnlineDiscoveryEngine(
    explore_interval=10,   # how often exploration hypotheses are proposed
    mode="balanced",       # "balanced" | "conservative" | "aggressive"
    buffer_size=150,       # observation window for all hypothesis buffers
)
engine.initialize_v2(var_names)          # seeds all pair/triplet hypotheses
engine.update(data_dict)                 # advance one observation
engine.get_candidate_paths()             # ranked edges (confidence ≥ 0.25)
```

`buffer_size` threads through every hypothesis constructor (both in `initialize_v2` and
in exploration steps) so high-frequency tick data (`buffer_size=50`) and monthly macro
data (`buffer_size=300`) are each handled with an appropriate observation window.

### Hypothesis Types — `relationships.py` / `relationships_extended.py`

15 relational hypothesis types across two files:

| Type | Algorithm | Implementation |
|------|-----------|---------------|
| `CausalHypothesis` | Granger causality (RLS regression, dual-direction); signed directional confidence `\|conf_fwd−conf_bwd\|`; F-ratio asymmetry guard; live-direction override | Window-batch |
| `CorrelationalHypothesis` | Welford online Pearson correlation | Truly online |
| `TemporalHypothesis` | AR/VAR (RLS forgetting factor) | Truly online |
| `FunctionalHypothesis` | Polynomial regression (RLS) | Truly online |
| `EquilibriumHypothesis` | Kalman filter mean-reversion | Truly online |
| `CompositionalHypothesis` | Sum constraint error | Window-batch |
| `CompetitiveHypothesis` | Negative correlation / zero-sum | Window-batch |
| `SynergisticHypothesis` | Interaction regression | Window-batch |
| `ProbabilisticHypothesis` | Distribution shift (Cohen's d) | Window-batch |
| `StructuralHypothesis` | Per-group Welford ANOVA/ICC | Truly online |
| `MediatingHypothesis` | Baron-Kenny mediation (Sobel test); lowered to p<0.20, min_n=20 for short series | Window-batch |
| `ModeratingHypothesis` | Interaction moderation (F-test) | Window-batch |
| `GraphHypothesis` | Shared-variance network structure | Window-batch |
| `SimilarityHypothesis` | k-means clustering | Window-batch |
| `LogicalHypothesis` | Boolean rule induction | Window-batch |

#### Cold-start Guard — `_not_ready()`

All 15 types use the `_not_ready()` sentinel for early returns before their observation
buffer reaches the required minimum:

```python
# returned until buffer is full
{'fit_score': 0.0, 'confidence': 0.0, 'evidence': n, 'stability': 0.0, 'ready': False}

# returned once ready
{'fit_score': ..., 'confidence': ..., ..., 'ready': True}
```

`confidence: 0.0` is blocked by the existing `confidence < 0.25` filter in
`get_candidate_paths()`, so cold-start noise never enters the proposal pool.
The `ready` flag allows downstream consumers to distinguish cold-start from a genuine
zero-confidence finding.

### Configuration — `relationship_config.py`

Centralised config dataclasses for all 15 types (see `HypothesisConfig`). All thresholds
and forgetting factors are overridable at construction time; defaults are documented in
`CausalConfig`, `CorrelationalConfig`, `TemporalConfig`, etc.

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

## scarcity/federation/

### Transport Layer

### `transport.py` — BaseTransport / LoopbackTransport / SimulatedNetworkTransport

| Symbol | Role |
|-------|------|
| `TransportConfig` | Protocol and reconnect configuration envelope |
| `BaseTransport` | Shared lifecycle (`start`, `stop`) and handler dispatch contract |
| `LoopbackTransport` | In-process transport for tests and single-node development |
| `SimulatedNetworkTransport` | Delay-injected transport for latency-aware simulation |
| `build_transport` | Protocol router (`loopback`, `sim`, `ws`) |

### `ws_transport.py` — WebSocketTransport / WSTransportConfig

Production distributed federation transport using websocket links.

**Server-side behavior**
- `start()` binds websocket server on `host:port`.
- `_handle_connection()` parses JSON, enforces optional `auth_token`, then dispatches to registered handlers.
- `connected_clients` tracks active inbound websocket clients.

**Client-side behavior**
- `send(topic, payload)` broadcasts to configured peers, or to inbound clients when no peers are configured.
- `send_to(endpoint, topic, payload)` sends a targeted packet.
- `_ensure_connection()` opens or reuses peer links with lock protection and reconnect timeout.
- `_listen_peer()` keeps outbound peer links bidirectional by consuming inbound frames.
- `connected_peers` tracks healthy outbound links.

**Key configuration (`WSTransportConfig`)**

| Param | Default | Purpose |
|------|---------|---------|
| `host` | `0.0.0.0` | Server bind host |
| `port` | `8765` | Server bind port |
| `peer_endpoints` | `None` | Outbound peer websocket URLs |
| `ping_interval` | `20.0` | Keepalive cadence |
| `ping_timeout` | `10.0` | Dead-peer timeout |
| `max_message_size` | `10 MB` | Payload size guardrail |
| `auth_token` | `None` | Optional shared-secret inbound gate |

### Federation Transport Test Coverage

- `scarcity/tests/test_audit_transport.py`: protocol selection assertions for `sim`, `loopback`, `ws`, and `websocket`.
- `scarcity/tests/test_ws_transport.py`: websocket transport lifecycle, auth enforcement, dispatch path, retry path, and introspection properties.

### Meta-Learning Layer (Phases 2–5)

#### `domain_server.py` — DomainServer / DomainServerRegistry

Per-domain logical meta agents. Each basket owns one `DomainServer` that holds a domain-specific base model and episodic memory.

| Symbol | Role |
|--------|------|
| `DomainServerConfig` | `memory_capacity`, `reptile_lr`, `hit_decay`, `min_episodes_for_prior` |
| `DomainServer` | `.update_base_params()`, `.adapt()`, `.record()`, `.suggest_prior()` |
| `DomainServerRegistry` | Basket-keyed dict; `.get_or_create(domain_id, basket_id)` |

Key attributes: `.base_params`, `.hit_rate`, `.memory_size`, `.round_id`

#### `global_meta_memory.py` — GlobalMetaMemory

Cross-domain episodic prior store. Aggregates domain base params after each round and answers prior queries.

| Symbol | Role |
|--------|------|
| `GlobalMetaMemoryConfig` | `memory_capacity`, `min_domains_for_aggregate`, `retrieval_top_k`, `blend_alpha` |
| `GlobalMetaMemory` | `.aggregate(registry, performance_map)`, `.suggest_prior(domain_id, context)`, `.memory_size` |

#### `packets.py` — Phase 4 Protocol Bridge Additions

Three new packet types for adaptation signalling (alongside existing `PathPack`, `EdgeDelta`, `PolicyPack`, `CausalSemanticPack`):

| Packet | Topic | Purpose |
|--------|-------|---------|
| `AdaptationRequest` | `federation.adaptation_request` | Client → DomainServer warm-start query |
| `AdaptationResponse` | `federation.adaptation_response` | DomainServer → Client prior reply |
| `DomainSyncPacket` | `federation.domain_sync` | DomainServer → Global state snapshot |

All have `.to_dict()` / `.from_dict()` round-trip. `serialise_packet()` routes to topic. `normalise_packets()` groups by type.

Test coverage: `scarcity/tests/test_adaptation_packets.py`

---

## scarcity/meta/ — Phase 5 Additions

#### `domain_server_meta.py` — DomainServerMeta

Federation-to-meta bridge. Observes `DomainServer` instances via duck typing and converts state to `DomainMetaUpdate` objects.

| Symbol | Role |
|--------|------|
| `DomainServerMetaConfig` | `hit_rate_weight`, `memory_weight`, `min_confidence`, `meta_lr_min/max`, `performance_gain_boost` |
| `DomainServerMeta` | `.observe(server, performance)`, `.observe_registry(registry, performance_map)`, `.status()` |

Confidence formula: `hit_rate_w × hit_rate + mem_w × log1p(mem) / log1p(ref) + gain_boost × gain`
Delta formula: `meta_lr(confidence) × (curr_params − prev_params)`

Test coverage: `scarcity/tests/test_domain_server_meta.py`

#### `cross_meta.py` — CrossDomainMetaLearner (Phase 5b addition)

Memory-backed cross-domain aggregation. Wraps `CrossDomainMetaAggregator` as fallback and blends with `GlobalMetaMemory` prior.

| Symbol | Role |
|--------|------|
| `CrossDomainMetaLearnerConfig` | `memory_reference_capacity`, `max_memory_quality`, `fallback_method` |
| `CrossDomainMetaLearner` | `.aggregate(updates, context)` → `(vec, keys, meta)` |

Blend: `result = (1−quality) × fallback + quality × prior`. Quality grows 0 → 0.8 as episodes fill.

Test coverage: `scarcity/tests/test_cross_domain_meta_learner.py`

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

---

## scripts/

### Comprehensive Benchmark Harness (`benchmark_harness.py`)

Orchestrates all 26 benchmark stages covering the full K-Scarcity architecture. Single entry
point for claim validation, regression testing, and claim integrity reporting.

```bash
python scripts/benchmark_harness.py              # all 26 stages
python scripts/benchmark_harness.py --fast       # reduced trial counts
python scripts/benchmark_harness.py --skip-slow  # skip stages > 5 min
python scripts/benchmark_harness.py --stage 9 10 11.1 11.2   # specific stages
python scripts/benchmark_harness.py --list       # list all stages and exit
```

**Stage groups:**

| Group | Stages | Coverage |
|-------|--------|----------|
| Foundation | 0, 1.1–1.4 | Engine identity, non-IID, null FPR, temporal ordering |
| Discovery | 2.1–2.3 | Four-condition matrix, baselines, cross-method comparison |
| Federation | 3.1–3.4 | Evidence-sharing ablation, DP, Byzantine robustness |
| Simulation | 4.1–4.3 | SFC identity, directional validation, null shock |
| Meta-learning | 5.1–5.3 | Pretrain inversion, pioneer sweep, MetaIntegrativeLayer |
| DRG | 6.1–6.2 | Assurance levels, self-regulation loop |
| Causal | 7 | DoWhy ATE pipeline (SKIP if dowhy not installed) |
| Integration | 8.1 | EventBus static + live wiring audit |
| Prediction | 9 | Rolling leave-one-year-out MAE (6 methods) |
| Regime transfer | 10 | Post-2008 adaptation: AR1-fixed vs rolling vs Scarcity |
| Sparsity/buffer | 11.1, 11.2 | Data-drop degradation curves; buffer size sweep |

**Outputs:** `artifacts/harness/harness_results.json` and `artifacts/harness/claim_integrity_matrix.json`.

**Stage modules** (`scripts/stages/`):

| Module | Stages |
|--------|--------|
| `stage0_identity.py` | 0 — AST-based engine identity audit |
| `stage1_foundation.py` | 1.1–1.4 — non-IID, null FPR, temporal ordering, Pearson baseline |
| `stage2_discovery.py` | 2.1–2.3 — four-condition matrix, Granger/VAR baselines |
| `stage3_federation.py` | 3.1–3.4 — evidence-sharing, DP sweep, Byzantine robustness |
| `stage4_simulation.py` | 4.1–4.3 — SFC accounting, 12-shock directional validation |
| `stage5_meta.py` | 5.1–5.3 — pretrain inversion, pioneer sweep, MetaIntegrativeLayer |
| `stage6_drg.py` | 6.1–6.2 — DRG assurance levels, self-regulation loop |
| `stage7_causal.py` | 7 — DoWhy ATE sign accuracy |
| `stage8_integration.py` | 8.1 — EventBus wiring audit |
| `stage9_prediction_mae.py` | 9 — rolling MAE: Mean/LocalAR1/FedAvgAR1/OracleAR1/ScarcityLocal/ScarcityFed |
| `stage10_regime_transfer.py` | 10 — regime transfer with synthetic structural break at 2008 |
| `stage11_sparsity_buffer.py` | 11.1–11.2 — sparsity sweep and buffer size sweep |
| `utils.py` | Shared helpers: `make_result`, `load_ground_truth`, `build_hub`, `stream_rows`, ... |

---

### Relationship Discovery Benchmark (`benchmark_discovery.py`)

Evaluates how well the online discovery engine recovers 25 theory-grounded macro/financial
relationships from real time-series data.

```bash
python scripts/benchmark_discovery.py \
  --fred --fred-key <KEY> \
  --country USA --peers CAN,GBR \
  --start 1980 --end 2023 \
  --pretrain-live --pretrain-start 1980 --pretrain-end 2005 \
  --output artifacts/meta/discovery_benchmark.txt
```

**Modes:**
- `--fred` — fetch FRED quarterly data (176 obs for USA 1980–2023); omit for dry-run with
  synthetic data
- `--pretrain-live` — pretrain on World Bank annual data before streaming FRED observations
- `--peers` — comma-separated ISO-3 list of federated peer nodes

**Evaluation:** dual-path — (1) low-threshold (conf ≥ 0.10) ensemble perturbation test;
(2) direct hypothesis scan at p < 0.10 as fallback for hypotheses that fire but do not
respond to perturbation (e.g. `CausalHypothesis` with `direction=-1`).

**Key results — FRED USA (best condition D, pretrained + fed):** 68% discovery, 36% overall
recall, 53% recall on the 17/25 testable relationships, 75% conf-weighted sign accuracy.
Infrastructure basket is untestable with FRED (no `electricity_access`/`internet_users`).

**Key results — World Bank Kenya 1980–2023 (44 obs, best condition D: pretrained + fed):**
92% discovery, 44% recall, **47% structural recall**, 48% sign accuracy.
All 4 baskets testable. Structural recall improved +5 pp over the prior run (42% → 47%)
driven by ECM cointegration reset at the pretrain/live boundary and majority-sign voting.
See `BENCHMARK_FINDINGS.md §31.2` for full methodology and per-condition breakdown.

See `documentation/scarcity-docs/BENCHMARK_FINDINGS.md §31` for full methodology and
`artifacts/meta/discovery_benchmark.txt` for per-relationship detail.

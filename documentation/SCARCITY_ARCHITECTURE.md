# SCARCITY Architecture Overview

> System architecture for the SENTINEL platform — Strategic National Economic & Threat Intelligence Layer

---

## System Layers

The platform is organised into four layers, each depending only on the layers below it:

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                     PRESENTATION LAYER                             │
 │                                                                    │
 │  K-SHIELD (8505)      Institution Portal (8506)   SENTINEL (8507) │
 │  kshield/page.py      institution/page.py          sentinel_dash   │
 │  ├─ Causal            ├─ Executive Dashboard       ├─ Live Map     │
 │  ├─ Terrain           ├─ Admin Governance          ├─ Federation   │
 │  ├─ Simulation        ├─ Developer Dashboard       ├─ Policy Chat  │
 │  └─ Impact            ├─ Spoke (Local)             ├─ Signals      │
 │                       ├─ Collaboration Room        ├─ Escalation   │
 │                       └─ FL Dashboard              └─ Causal Sim   │
 ├────────────────────────────────────────────────────────────────────┤
 │                       API LAYER                                    │
 │                                                                    │
 │  backend/app/api/v2/  (FastAPI — current)                          │
 │  backend/app/api/v1/  (FastAPI — deprecated)                       │
 ├────────────────────────────────────────────────────────────────────┤
 │                    INTELLIGENCE LAYER                              │
 │                                                                    │
 │  kshiked.pulse          15 social signals + NLP pipeline          │
 │  kshiked.core           EconomicGovernor, shocks, ScarcityBridge  │
 │  kshiked.hub            KShieldHub — singleton orchestrator       │
 │  kshiked.causal_adapter Causal pipeline bridge                    │
 │  kshiked.federation     Aegis Protocol — defense sector FL        │
 │  kshiked.simulation     Kenya calibration + scenario templates    │
 │  federated_databases    Federation data plane + audit             │
 ├────────────────────────────────────────────────────────────────────┤
 │                    FOUNDATION LAYER                                │
 │                                                                    │
 │  scarcity.engine        OnlineDiscoveryEngine (15 hypotheses)     │
 │  scarcity.simulation    Multi-sector SFC engine + IO structure    │
 │  scarcity.federation    Federated learning + secure aggregation   │
 │  scarcity.causal        DoWhy causal inference                    │
 │  scarcity.meta          Meta-learning agent (Reptile/MAML)        │
 │  scarcity.governor      Dynamic Resource Governor (DRG)           │
 │  scarcity.fmi           Federated Metadata Interchange            │
 │  scarcity.stream        Windowed data ingestion                   │
 │  scarcity.runtime       EventBus, telemetry                       │
 │  scarcity.synthetic     Synthetic data pipeline                   │
 │  scarcity.analytics     Policy terrain + aggregation utilities    │
 └────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

| Component | Package | Key Entrypoint | Purpose |
|-----------|---------|----------------|---------|
| Discovery Engine | `scarcity.engine` | `OnlineDiscoveryEngine` | Real-time hypothesis discovery from streaming data |
| Simulation — SFC | `scarcity.simulation` | `SFCEconomy` | Legacy 4-sector Stock-Flow Consistent model |
| Simulation — Multi-Sector | `scarcity.simulation` | `MultiSectorSFCEngine` | Typed 4-sector engine with 8 behavioral blocks |
| IO Structure | `scarcity.simulation` | `IOConfig`, `LeontiefModel` | 9-sector KNBS IO matrix + SFC aggregation bridge |
| Causal Inference | `scarcity.causal` | `CausalRunner` | DoWhy-based causal identification and estimation |
| Federation | `scarcity.federation` | `FederationClientAgent` | Multi-agency federated learning without data sharing |
| Meta Learning | `scarcity.meta` | `MetaLearner` | Learning-to-learn for hypothesis tuning |
| Governor | `scarcity.governor` | `DynamicResourceGovernor` | CPU/memory throttling; DRG assurance levels |
| FMI | `scarcity.fmi` | `FederatedMetadataInterchange` | Schema exchange for federated deployments |
| Stream | `scarcity.stream` | `StreamIngester` | Windowed data ingestion |
| Runtime | `scarcity.runtime` | `EventBus` | Pub/sub event bus and telemetry |
| Synthetic | `scarcity.synthetic` | `SyntheticPipeline` | Test data generation (accounts, content, behavior) |
| Analytics | `scarcity.analytics` | `AnalyticsModule` | Aggregation and policy terrain utilities |
| Pulse Engine | `kshiked.pulse` | `PulseSensor` | 15 SIGINT signal detectors + NLP + LLM pipeline |
| Governance | `kshiked.core` | `EconomicGovernor` | Policy simulation and shock modelling |
| Scarcity Bridge | `kshiked.core` | `ScarcityBridge` | Connects KShield to Scarcity discovery engine |
| Hub | `kshiked.hub` | `KShieldHub` | Singleton orchestrator unifying Pulse + Scarcity |
| Causal Adapter | `kshiked.causal_adapter` | `AdapterConfig` | Bridge between Scarcity causal engine and KShield |
| Threat Indices | `kshiked.pulse.indices` | `compute_threat_report` | 8 composite threat indices (PI, LEI, MRS, ECI, IWI, SFI, ECR, ETM) |
| Kenya Calibration | `kshiked.simulation` | `calibrate_from_data` | World Bank data → SFC parameters |
| Scenario Templates | `kshiked.simulation` | `get_scenario_by_id` | 9 Kenya shock scenarios + 8 policy presets |
| Dashboard | `kshiked.ui` | `render_sentinel_dashboard` | Routed Streamlit Command Center |
| Cost of Delay | `kshiked.ui.institution.backend` | `analytics_engine` | KES billions delay cost quantification |
| Report Export | `kshiked.ui.institution` | `unified_report_export` | PDF + ZIP cross-dashboard exports |
| KShield Federation | `kshiked.federation` | — | Aegis Protocol — defense sector federation |
| Federated Databases | `federated_databases` | `ScarcityFederationManager` | Node DBs + control plane + ML sync + audit |
| Shock Compiler | `kshiked.simulation` | `ShockCompiler` | Transforms stochastic shocks into SFC vectors |

---

## Primary Data Flow — Pulse Pipeline

```
                   RAW TEXT (Social Media / News)
                            │
                   ┌────────▼────────┐
                   │   PulseSensor   │  kshiked.pulse
                   │  15 detectors   │  Keywords + NLP + LLM
                   │  process_text() │
                   └────────┬────────┘
                            │ SignalDetections
                   ┌────────▼────────┐
                   │  SignalMapper   │  15 detectors → PulseState
                   │  update_state() │
                   └────────┬────────┘
                            │ PulseState
                   ┌────────▼────────────┐
                   │  ThreatIndexReport  │  8 indices computed
                   │  PI·LEI·MRS·ECI     │  (Polarization, Legitimacy,
                   │  IWI·SFI·ECR·ETM    │   Mobilization, Cohesion,
                   └────────┬────────────┘   IW, Security, Economic, Ethnic)
                            │
                   ┌────────▼────────┐
                   │   KShieldHub    │  kshiked.hub
                   │                 │
                   │  ┌─── Pulse ◄───┘
                   │  │
                   │  ├─── get_shock_vector()
                   │  │         │
                   │  │   ┌─────▼───────────┐
                   │  │   │  ShockCompiler  │  kshiked.simulation
                   │  │   │  compile()      │  Impulse/OU/Brownian
                   │  │   └─────┬───────────┘
                   │  │         │
                   │  ├─────────▼───────────┐
                   │  │  EconomicGovernor   │  kshiked.core
                   │  │  step()             │  SFC policy simulation
                   │  └─────┬───────────────┘
                   │        │
                   │  ┌─────▼───────────┐
                   │  │  SFCEconomy /   │  scarcity.simulation
                   │  │  MultiSector    │  Macro state update
                   │  │  SFCEngine      │
                   │  └─────┬───────────┘
                   │        │
                   └────────▼──────────────────────┐
                            │                      │
                   ┌────────▼────────┐    ┌────────▼──────────┐
                   │   Dashboard     │    │  REST API (v2)    │
                   │   (Streamlit)   │    │  (FastAPI)        │
                   └─────────────────┘    └───────────────────┘
```

---

## Parallel Pipeline — OnlineDiscoveryEngine

```
     STREAMING ROWS (CSV / API / World Bank)
                  │
         ┌────────▼──────────┐
         │  StreamIngester   │  scarcity.stream
         │  windowed input   │
         └────────┬──────────┘
                  │
         ┌────────▼──────────────────────────────────────────┐
         │  OnlineDiscoveryEngine                            │  scarcity.engine
         │                                                   │
         │  Hypothesis Pool (15 types):                      │
         │  Causal · Correlational · Temporal · Functional   │
         │  Equilibrium · Compositional · Competitive        │
         │  Synergistic · Probabilistic · Structural         │
         │  Mediating · Moderating · Graph · Similarity      │
         │  Logical                                          │
         │                                                   │
         │  Algorithms: Vectorized Batch RLS (einsum)        │
         │  CountSketch · TensorSketch · Page-Hinkley drift  │
         │  Thompson sampling · Counterfactual Jacobian      │
         └────────┬──────────────────────────────────────────┘
                  │ Confidence-weighted hypotheses
         ┌────────▼──────────┐
         │  KnowledgeGraph   │  Hypergraph store with temporal decay
         │  get_knowledge_   │
         │  graph()          │
         └────────┬──────────┘
                  │
         ┌────────▼──────────┐    ┌────────────────────────┐
         │  CausalRunner     │    │   MetaLearner          │
         │  DoWhy pipeline   │    │   Reptile optimizer    │
         │  (DoWhy + IV)     │    │   hypothesis tuning    │
         └────────┬──────────┘    └────────────────────────┘
                  │
         ┌────────▼──────────┐
         │  Causal Network   │  → K-SHIELD Causal Tab
         │  Visualization    │  → ScarcityBridge.create_learned_economy()
         └───────────────────┘
```

---

## Simulation Engine — Internal Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                    scarcity/simulation/                              │
 │                                                                      │
 │  IO LAYER (new — Item 15)                                            │
 │  ┌─────────────────────────────────────────────────────────────┐    │
 │  │  io_structure.py                                            │    │
 │  │  ├─ SubSectorType (9 KNBS sectors)                          │    │
 │  │  ├─ IOConfig (9×9 matrix + sector_shares + shock_sensitivity│    │
 │  │  ├─ default_kenya_io_config()                               │    │
 │  │  ├─ KNBS_TO_SFC_SECTOR (concordance map)                    │    │
 │  │  ├─ aggregate_io_to_sfc_sectors() → 3×3 block + imports     │    │
 │  │  └─ LeontiefModel (Leontief inverse solver)                 │    │
 │  │                                                             │    │
 │  │  parameters.py                                              │    │
 │  │  └─ AllParams → InputOutputParams (4×4 reconciled matrix)   │    │
 │  │     ← 3×3 block derived from KNBS aggregation               │    │
 │  │     ← INFORMAL row/col from field estimates                 │    │
 │  └─────────────────────────────────────────────────────────────┘    │
 │                                                                      │
 │  ENGINE PATHS                                                        │
 │  ┌──────────────────────────┐  ┌─────────────────────────────────┐  │
 │  │  sfc.py  (legacy path)   │  │  sfc_engine.py  (typed path)    │  │
 │  │  SFCEconomy              │  │  MultiSectorSFCEngine            │  │
 │  │  SFCConfig (40+ params)  │  │                                 │  │
 │  │  4 balance-sheet sectors │  │  Behavioral Blocks:             │  │
 │  │  Phillips Curve          │  │  ├─ production.py   CES output  │  │
 │  │  Taylor Rule             │  │  ├─ labor_market.py wages/unemp │  │
 │  │  Okun's Law              │  │  ├─ price_system.py CPI/imports │  │
 │  │  Step cycle (5 stages)   │  │  ├─ households.py  income/cons  │  │
 │  └──────────────────────────┘  │  ├─ government.py  taxes/debt  │  │
 │                                 │  ├─ monetary.py    Taylor+pass │  │
 │                                 │  ├─ foreign.py     CA/KA/FX   │  │
 │                                 │  └─ banking.py     CAR/NPL    │  │
 │                                 │                               │  │
 │                                 │  Support:                     │  │
 │                                 │  ├─ coupling_interface.py     │  │
 │                                 │  ├─ accounting.py (residuals) │  │
 │                                 │  └─ types.py (contracts)      │  │
 │                                 └─────────────────────────────────┘  │
 │                                                                      │
 │  EXTENDED MODULES                                                    │
 │  ┌─────────────────────────────────────────────────────────────┐    │
 │  │  bayesian.py          Bayesian VARX forecasting             │    │
 │  │  financial_accelerator.py  BGG financial accelerator        │    │
 │  │  heterogeneous.py     Heterogeneous agent extension         │    │
 │  │  agents.py            Agent-based model overlay             │    │
 │  │  open_economy.py      Open economy extensions               │    │
 │  │  learned_sfc.py       ScarcityBridge-trained SFC wrapper    │    │
 │  │  research_sfc.py      Research configuration variants       │    │
 │  │  whatif.py            Counterfactual what-if engine         │    │
 │  │  dynamics.py          Economic dynamics utilities           │    │
 │  │  scenario.py          Scenario management                   │    │
 │  │  scheduler.py         Simulation scheduler                  │    │
 │  │  monitor.py           Runtime monitoring                    │    │
 │  │  storage.py           Run artifact persistence              │    │
 │  │  visualization3d.py   3D visualisation helpers              │    │
 │  └─────────────────────────────────────────────────────────────┘    │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## IO Structure — Sector Reconciliation (Item 15)

Three separate sector frameworks exist in the codebase, now bridged:

```
 ┌─────────────────────────┐     ┌──────────────────────────────────────┐
 │  KNBS 9-Sector IO       │     │  SFC 4-Sector Production Model       │
 │  (io_structure.py)      │     │  (sfc_engine.py / parameters.py)     │
 │                         │     │                                      │
 │  agriculture            │     │  AGRICULTURE                         │
 │  manufacturing    ──────┼─────│  MANUFACTURING  (mfg+mine+const+     │
 │  mining          │      │     │                  water aggregated)   │
 │  construction    │      │     │  SERVICES       (srv+hlth+trans+     │
 │  water           │      │     │                  secur aggregated)   │
 │  services        │      │     │  INFORMAL       (field estimates)    │
 │  health          │      │     └──────────────────────────────────────┘
 │  transport       │      │
 │  security        │      │     ┌──────────────────────────────────────┐
 │                  │      │     │  Policy 6-Sector Model               │
 │  LeontiefModel   │      │     │  (sector_engine / sector_registry)   │
 │  Hawkins-Simon   │      │     │  Economics/Finance · Healthcare      │
 │  check           │      │     │  Environment/Water · Social          │
 └─────────────────────────┘     │  Education/Labor · Security          │
           │                     └──────────────────────────────────────┘
           │ aggregate_io_to_sfc_sectors()
           │ Standard IO aggregation formula:
           │ A_agg[I,J] = Σ_{i∈I} Σ_{j∈J} A[i,j]·x_j/X_J
           ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Reconciled 4×4 Matrix in InputOutputParams             │
 │  (3×3 block KNBS-derived; INFORMAL from field estimates)│
 │                                                         │
 │  Column sums: AGR=0.42  MFG=0.46  SRV=0.49  (all <1)  │
 │  Hawkins-Simon: SATISFIED                               │
 └─────────────────────────────────────────────────────────┘
```

---

## Federation Architecture — Aegis Protocol

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                  SENTINEL FEDERATION HUB                        │
 │                                                                 │
 │   Receives:  Encrypted model updates (gradients, not raw data)  │
 │   Applies:   Trimmed-Mean / Element-wise Median aggregation     │
 │   Defends:   Krum + Multi-Krum + Bulyan (Byzantine resistance)  │
 │   Outputs:   Updated global prior → broadcast to all nodes      │
 └────────────────────────┬────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────────┐
          │               │               │       │
          ▼               ▼               ▼       ▼
     ┌─────────┐     ┌─────────┐    ┌─────────┐  ...
     │  NIS    │     │  DCI    │    │  CBK    │
     │ Local   │     │ Local   │    │ Local   │
     │ Training│     │ Training│    │ Training│
     │ Q8 quant│     │ Q8 quant│    │ Q8 quant│
     │ update  │     │ update  │    │ update  │
     └─────────┘     └─────────┘    └─────────┘

  Security Layers:
  ├─ HKDF-SHA256 pairwise masking (Bonawitz-style)
  ├─ Differential Privacy (ε-δ Gaussian noise)
  ├─ Trust scoring: Agreement 60% / Compliance 30% / Impact 10%
  └─ Security lattice: UNCLASSIFIED → RESTRICTED → SECRET → TOP_SECRET
```

---

## DRG Assurance Model

The Dynamic Resource Governor assigns assurance levels to all projections:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  DynamicResourceGovernor                                     │
  │                                                              │
  │  Inputs: confidence score + data freshness + CPU/mem load   │
  │                                                              │
  │  HIGH     ≥ 0.85, recent data  → Policy-grade projection    │
  │  MEDIUM   0.65–0.85            → Directionally reliable     │
  │  LOW      < 0.65 or stale      → Indicative only            │
  │  FALLBACK Discovery failed     → Hardcoded SFC baselines    │
  │                                                              │
  │  Actuators: throttle engine, shed workloads, alert telemetry │
  └──────────────────────────────────────────────────────────────┘
```

---

## Configuration

All backend settings are loaded from environment variables with the `SCARCE_` prefix. See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for the full reference of 60+ settings.

Key toggles:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCARCE_SCARCITY_ENABLED` | `True` | Master switch for scarcity components |
| `SCARCE_SCARCITY_FEDERATION_ENABLED` | `False` | Federation layer |
| `SCARCE_SCARCITY_SIMULATION_ENABLED` | `False` | Simulation engine |
| `SCARCE_SCARCITY_MPIE_ENABLED` | `True` | MPIE hypothesis orchestrator |
| `SCARCE_SCARCITY_DRG_ENABLED` | `True` | Dynamic Resource Governor |

---

## Directory Structure

```
scace4/
├── scarcity/                         # Foundation Layer (pip-installable)
│   ├── engine/                       # Hypothesis discovery
│   │   ├── engine_v2.py              # OnlineDiscoveryEngine (current)
│   │   ├── engine.py                 # Legacy engine
│   │   ├── relationships.py          # 15 hypothesis types
│   │   ├── relationships_extended.py # Extended relationship operators
│   │   ├── vectorized_core.py        # Batch RLS via numpy.einsum
│   │   ├── anomaly.py                # RRCF anomaly detection
│   │   ├── forecasting.py            # Bayesian VARX forecasting
│   │   ├── arbitration.py            # Hypothesis arbitration
│   │   ├── bandit_router.py          # Thompson sampling router
│   │   └── operators/                # Composable operator modules
│   │       ├── attention_ops.py
│   │       ├── causal_semantic_ops.py
│   │       ├── evaluation_ops.py
│   │       ├── integrative_ops.py
│   │       ├── relational_ops.py
│   │       ├── sketch_ops.py
│   │       ├── stability_ops.py
│   │       └── structural_ops.py
│   ├── simulation/                   # SFC economic model
│   │   ├── sfc.py                    # SFCEconomy (legacy, 4-sector)
│   │   ├── sfc_engine.py             # MultiSectorSFCEngine (typed)
│   │   ├── io_structure.py           # 9-sector KNBS IO + aggregation
│   │   ├── parameters.py             # AllParams (KNBS-reconciled)
│   │   ├── types.py                  # Typed contracts
│   │   ├── production.py             # CES production block
│   │   ├── labor_market.py           # Labor + wage dynamics
│   │   ├── price_system.py           # CPI + import prices
│   │   ├── households.py             # Income + consumption
│   │   ├── government.py             # Fiscal block
│   │   ├── monetary.py               # Taylor Rule + pass-through
│   │   ├── foreign.py                # External sector + FX
│   │   ├── banking.py                # Credit + CAR + NPL
│   │   ├── coupling_interface.py     # Cross-sector feedback
│   │   ├── accounting.py             # SFC identity checks
│   │   ├── bayesian.py               # Bayesian VARX
│   │   ├── financial_accelerator.py  # BGG accelerator
│   │   ├── heterogeneous.py          # Heterogeneous agents
│   │   ├── agents.py                 # Agent-based overlay
│   │   ├── open_economy.py           # Open economy extensions
│   │   ├── learned_sfc.py            # ScarcityBridge-trained SFC
│   │   ├── research_sfc.py           # Research variants
│   │   ├── whatif.py                 # Counterfactual engine
│   │   ├── dynamics.py               # Dynamics utilities
│   │   ├── scenario.py               # Scenario management
│   │   ├── scheduler.py              # Run scheduler
│   │   ├── monitor.py                # Runtime monitoring
│   │   ├── storage.py                # Artifact persistence
│   │   ├── visualization3d.py        # 3D visualisation
│   │   └── tests/                    # Simulation test suite
│   ├── federation/                   # Federated learning
│   │   ├── client_agent.py           # FederationClientAgent
│   │   ├── aggregator.py             # Trimmed-Mean / Median
│   │   ├── secure_aggregation.py     # Shamir secret sharing
│   │   ├── privacy_guard.py          # Differential privacy
│   │   ├── gossip.py                 # Gossip protocol
│   │   ├── hierarchical.py           # Two-layer aggregation
│   │   ├── trust_scorer.py           # Node trust scoring
│   │   └── ws_transport.py           # WebSocket transport
│   ├── causal/                       # DoWhy causal inference
│   │   ├── engine.py                 # CausalRunner
│   │   ├── identification.py         # Backdoor / IV
│   │   ├── estimation.py             # Linear / Forest estimators
│   │   ├── validation.py             # Refutation tests
│   │   └── time_series.py            # Granger causality
│   ├── meta/                         # Meta-learning
│   │   ├── meta_learning.py          # MetaLearner (Reptile)
│   │   ├── integrative_meta.py       # Cross-domain meta
│   │   └── optimizer.py              # Reptile / MAML optimizer
│   ├── governor/                     # Resource governance
│   │   ├── drg_core.py               # DynamicResourceGovernor
│   │   ├── monitor.py                # CPU/memory sensors
│   │   ├── actuators.py              # Workload shedding
│   │   └── policies.py               # Throttle policies
│   ├── fmi/                          # Federated Metadata Interchange
│   ├── stream/                       # Data ingestion / windowing
│   ├── runtime/                      # EventBus, telemetry
│   ├── synthetic/                    # Test data generators
│   └── analytics/                    # Policy terrain utilities
│
├── kshiked/                          # Intelligence Layer (Kenya-specific)
│   ├── core/                         # Governance + ScarcityBridge
│   ├── pulse/                        # 15 SIGINT signals
│   │   ├── sensor.py                 # PulseSensor
│   │   ├── detectors.py              # Signal detectors
│   │   ├── indices.py                # 8 threat indices
│   │   ├── llm/                      # LLM analysis pipeline
│   │   │   ├── analyzer.py           # Main LLM analyzer
│   │   │   ├── policy_chatbot.py     # Policy chat
│   │   │   └── policy_extractor.py   # Policy extraction
│   │   └── ingestion/                # Data ingestion pipeline
│   ├── simulation/                   # Kenya calibration + scenarios
│   ├── causal_adapter/               # Causal bridge to KShield
│   ├── federation/                   # Aegis Protocol
│   ├── hub.py                        # KShieldHub orchestrator
│   └── ui/                           # Streamlit dashboards
│       ├── sentinel_dashboard.py     # SENTINEL (port 8507)
│       ├── sentinel/                 # SENTINEL views
│       │   ├── router.py             # Routed navigation
│       │   ├── live_map.py           # Kenya county map
│       │   ├── executive.py          # Executive summary
│       │   ├── operations.py         # Operational alerts
│       │   ├── signals.py            # Signal deep-dive
│       │   ├── federation.py         # FL round management
│       │   ├── policy_chat.py        # LLM policy chat
│       │   ├── causal_sim.py         # Interactive causal sim
│       │   └── escalation.py         # Escalation management
│       ├── kshield/                  # K-SHIELD (port 8505)
│       │   ├── page.py               # Auth gate + 4 sub-cards
│       │   ├── causal/               # Causal relationships tab
│       │   ├── terrain/              # 3D policy terrain tab
│       │   ├── simulation/           # Simulation workspace
│       │   │   ├── view.py           # Main simulation view
│       │   │   ├── sector_dashboard.py # Sector-level dashboard
│       │   │   └── workbench/        # Advanced workbench
│       │   └── impact/               # Policy impact tab
│       ├── institution/              # Institution Portal (port 8506)
│       │   ├── page.py               # Multi-role router
│       │   ├── executive_dashboard.py
│       │   ├── admin_governance.py
│       │   ├── developer_dashboard.py
│       │   ├── local_dashboard.py    # Spoke (county-level)
│       │   ├── collab_room.py        # Collaboration room
│       │   ├── fl_dashboard.py       # FL round management
│       │   ├── unified_report_export.py # PDF + ZIP export
│       │   └── backend/
│       │       ├── analytics_engine.py  # Cost of Delay engine
│       │       └── executive_bridge.py  # DRG-backed bridge
│       └── home/                     # Landing page
│
├── federated_databases/              # Federation data plane
│   ├── scarcity_federation.py        # Node registration + sync rounds
│   ├── storage.py                    # Node/control SQLite backends
│   └── pipeline.py                   # Single-node / federated wrapper
│
├── backend/                          # API Layer
│   └── app/
│       ├── api/v2/                   # Current REST API (FastAPI)
│       └── api/v1/                   # Deprecated
│
├── documentation/                    # Architecture documentation
├── scarcity-deep-dive/               # Vite + React frontend
└── artifacts/                        # Run artifacts (effects, graphs)
```

---

*Last updated: 2026-04-18*

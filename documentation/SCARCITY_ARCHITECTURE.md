# SCARCITY Architecture Overview

> System architecture for the SENTINEL platform — Strategic National Economic & Threat Intelligence Layer

---

## System Layers

The platform is organised into four layers, each depending only on the layers below it:

```
 ┌────────────────────────────────────────────────┐
 │            PRESENTATION LAYER                   │
 │  sentinel_dashboard.py · scarcity-deep-dive     │
 │  (Streamlit 9-tab + Vite frontend)              │
 ├────────────────────────────────────────────────┤
 │               API LAYER                         │
 │  backend/app/api/v2 (FastAPI — current)         │
 │  backend/app/api/v1 (FastAPI — deprecated)      │
 ├────────────────────────────────────────────────┤
 │           INTELLIGENCE LAYER                    │
 │  kshiked.pulse    — 15 social signals           │
 │  kshiked.core     — EconomicGovernor, shocks    │
 │  kshiked.hub      — KShieldHub orchestrator     │
 │  kshiked.causal_adapter — causal pipeline       │
 │  kshiked.federation — Aegis Protocol            │
 ├────────────────────────────────────────────────┤
 │           FOUNDATION LAYER                      │
 │  scarcity.engine   — OnlineDiscoveryEngine      │
 │  scarcity.federation — Federated Learning       │
 │  scarcity.simulation — SFCEconomy               │
 │  scarcity.causal   — DoWhy causal inference     │
 │  scarcity.meta     — Meta-learning agent        │
 │  scarcity.runtime  — EventBus, telemetry        │
 └────────────────────────────────────────────────┘
```

---

## Component Map

| Component | Package | Key Entrypoint | Purpose |
|-----------|---------|----------------|---------|
| Discovery Engine | `scarcity.engine` | `OnlineDiscoveryEngine` | Real-time hypothesis discovery from streaming data |
| Federation | `scarcity.federation` | `FederationClientAgent` | Multi-agency federated learning without data sharing |
| Simulation | `scarcity.simulation` | `SFCEconomy` | Stock-Flow Consistent macroeconomic model |
| Causal Inference | `scarcity.causal` | `CausalRunner` | DoWhy-based causal identification and estimation |
| Meta Learning | `scarcity.meta` | `MetaLearner` | Learning-to-learn for hypothesis tuning |
| Governor | `scarcity.governor` | `DynamicResourceGovernor` | CPU/memory throttling for engine workloads |
| FMI | `scarcity.fmi` | `FederatedMetadataInterchange` | Schema exchange for federated deployments |
| Stream | `scarcity.stream` | `StreamIngester` | Windowed data ingestion |
| Runtime | `scarcity.runtime` | `EventBus` | Pub/sub event bus and telemetry |
| Synthetic | `scarcity.synthetic` | `SyntheticPipeline` | Test data generation (accounts, content, behavior) |
| Analytics | `scarcity.analytics` | `AnalyticsModule` | Aggregation and reporting utilities |
| Pulse Engine | `kshiked.pulse` | `PulseSensor` | 15 SIGINT signal detectors + NLP pipeline |
| Governance | `kshiked.core` | `EconomicGovernor` | Policy simulation and shock modelling |
| Hub | `kshiked.hub` | `KShieldHub` | Singleton orchestrator unifying Pulse + Scarcity |
| Causal Adapter | `kshiked.causal_adapter` | `AdapterConfig` | Bridge between Scarcity causal engine and KShield |
| Threat Indices | `kshiked.pulse.indices` | `compute_threat_report` | 5 composite threat indices |
| Dashboard | `kshiked.ui` | `render_sentinel_dashboard` | 9-tab Streamlit Command Center |
| KShield Federation | `kshiked.federation` | — | Aegis Protocol — defense sector federation |
| Shock Compiler | `kshiked.simulation` | `ShockCompiler` | Transforms stochastic shocks into SFC vectors |

---

## Primary Data Flow

```
                   RAW TEXT
                     │
            ┌────────▼────────┐
            │  PulseSensor    │  (kshiked.pulse)
            │  process_text() │
            └────────┬────────┘
                     │ SignalDetections
            ┌────────▼────────┐
            │  SignalMapper   │  15 detectors → PulseState
            │  update_state() │
            └────────┬────────┘
                     │ PulseState + ThreatIndexReport
            ┌────────▼────────┐
            │  KShieldHub     │  (kshiked.hub)
            │                 │
            │  ┌─── Pulse ◄───┘
            │  │
            │  ├─── get_shock_vector()
            │  │        │
            │  │   ┌────▼────────────┐
            │  │   │ ShockCompiler   │  (kshiked.simulation)
            │  │   │ compile()       │  Impulse/OU/Brownian → vectors
            │  │   └────┬────────────┘
            │  │        │
            │  ├────────▼────────────┐
            │  │  EconomicGovernor   │  (kshiked.core)
            │  │  step()             │  SFC policy simulation
            │  └────┬────────────────┘
            │       │
            │  ┌────▼──────────┐
            │  │  SFCEconomy   │  (scarcity.simulation)
            │  │  step()       │  Macro state update
            │  └────┬──────────┘
            │       │
            └───────▼──────────────────────┐
                    │                      │
           ┌────────▼────────┐    ┌────────▼──────────┐
           │  Dashboard      │    │  REST API (v2)     │
           │  (Streamlit)    │    │  (FastAPI)         │
           └─────────────────┘    └───────────────────┘
```

### Parallel Pipeline: OnlineDiscoveryEngine

```
     STREAMING ROWS (CSV / API)
              │
     ┌────────▼──────────┐
     │  OnlineDiscovery  │  (scarcity.engine)
     │  Engine           │
     │  process_row()    │
     └────────┬──────────┘
              │ Hypothesis pool (15 types)
     ┌────────▼──────────┐
     │  KnowledgeGraph   │  (scarcity.engine)
     │  get_knowledge_   │
     │  graph()          │
     └────────┬──────────┘
              │
     ┌────────▼──────────┐
     │  Causal Network   │  Dashboard Tab 4
     │  Visualization    │
     └───────────────────┘
```

---

## Configuration

All backend settings are loaded from environment variables with the `SCARCE_` prefix. See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for the full reference of 60+ settings.

Key toggles:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCARCE_SCARCITY_ENABLED` | `True` | Master switch for scarcity components |
| `SCARCE_SCARCITY_FEDERATION_ENABLED` | `False` | Federation layer (stub) |
| `SCARCE_SCARCITY_SIMULATION_ENABLED` | `False` | Simulation engine (stub) |
| `SCARCE_SCARCITY_MPIE_ENABLED` | `True` | MPIE hypothesis orchestrator |
| `SCARCE_SCARCITY_DRG_ENABLED` | `True` | Dynamic Resource Governor |

---

## Directory Structure

```
scace4/
├── scarcity/                       # Foundation Layer
│   ├── engine/                     # Hypothesis discovery (engine_v2.py)
│   ├── federation/                 # Federated learning
│   ├── simulation/                 # SFC economic model
│   ├── causal/                     # DoWhy causal inference
│   ├── meta/                       # Meta-learning
│   ├── governor/                   # Resource control
│   ├── fmi/                        # Federated metadata interchange
│   ├── stream/                     # Data ingestion / windowing
│   ├── runtime/                    # EventBus, telemetry
│   ├── synthetic/                  # Test data generation
│   └── analytics/                  # Aggregation utilities
├── kshiked/                        # Intelligence Layer
│   ├── core/                       # EconomicGovernor, shocks
│   ├── pulse/                      # 15 SIGINT signals + NLP
│   ├── sim/                        # Backtesting & Monte Carlo
│   ├── ui/                         # Streamlit dashboard (9 tabs)
│   ├── causal_adapter/             # Causal pipeline bridge
│   ├── federation/                 # Aegis Protocol
│   ├── simulation/                 # ShockCompiler
│   ├── data/                       # GeoJSON, news DB
│   ├── analysis/                   # Data quality tools
│   └── hub.py                      # KShieldHub orchestrator
├── backend/                        # API Layer
│   └── app/
│       ├── api/v1/                 # v1 endpoints (deprecated)
│       ├── api/v2/                 # v2 endpoints (current)
│       └── core/                   # Config, managers
├── scarcity-deep-dive/             # Presentation (Vite frontend)
├── documentation/                  # This documentation tree
└── tests/                          # Test suite
```

---

*Last updated: 2026-02-11*

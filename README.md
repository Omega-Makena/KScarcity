# SENTINEL

**Strategic National Economic & Threat Intelligence Layer**

An AI-powered early warning system that detects emerging threats to national stability by fusing economic indicators, social signals, and critical infrastructure data — enabling multi-agency collaboration without sharing sensitive raw data.

---

##  What It Does

SENTINEL transforms scattered data streams into actionable intelligence by:

- **Discovering** causal relationships between economic and social instability
- **Detecting** signal silence and coordinated "going dark" patterns
- **Monitoring** critical infrastructure stress beyond social media
- **Forecasting** escalation pathways with time-to-event estimates
- **Enabling** multi-agency collaboration through federated learning
- **Supporting** human decision-making with competing hypothesis frameworks

---

## Scarcity vs KShield vs SENTINEL

These three names appear everywhere in the codebase — here is the relationship:

```
┌──────────────────────────────────────────────────────────────────┐
│  SENTINEL  — the user-facing product                             │
│  Streamlit Command Center on port 8501                           │
│  12 sidebar views + K-SHIELD module (4 cards, 11 sim tabs)       │
├──────────────────────────────────────────────────────────────────┤
│  KShield (kshiked/)  — Kenya-specific intelligence layer         │
│  Pulse (15 signals), calibration (World Bank → SFC),             │
│  9 scenario templates, 8 policy templates, federation bridge     │
├──────────────────────────────────────────────────────────────────┤
│  Scarcity (scarcity/)  — domain-agnostic Python library          │
│  OnlineDiscoveryEngine, SFCEconomy, FederationClientAgent,       │
│  DoWhy causal, Meta-learning, EventBus runtime                   │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency rule**: Scarcity never imports KShield. KShield imports Scarcity. SENTINEL imports both.

See the full guide: [Scarcity vs SENTINEL](documentation/SCARCITY_VS_SENTINEL.md)

---

##  Architecture

```
                              ANALYST LAYER
                    Human oversight, feedback, explainability
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              META-INTEL      DECISION         INSTITUTIONAL
            Unknown-unknown   Latency model    Strain indicators
            Competing hypo    Escalation path  Leadership signals
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              SIGNAL INTEL    THREAT ASSESS    INFRASTRUCTURE
            Silence detect    Actor cap/intent   Power/telecom
            Going dark        Stability buffer   Cascade paths
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              CAUSAL RIGOR    ADAPTIVE INTEL    PERSISTENCE
            Granger tests     Baseline learn    Decay model
            Counterfactual    Regime shift      Reinforcement
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                              FOUNDATION
                    ┌───────────────┼───────────────┐
                    │               │               │
                 SCARCITY       KSHIELD        FEDERATION
               Discovery      Pulse Engine    Multi-agency
               15 hypothesis   15 signals     Secure + private
```

---

##  Core Modules

### Scarcity Engine
Real-time discovery of causal relationships between economic variables.

```python
from scarcity.engine import OnlineDiscoveryEngine

engine = OnlineDiscoveryEngine(explore_interval=10)
engine.initialize(schema)          # or engine.initialize_v2(schema)
status = engine.process_row(row)   # main tick
graph  = engine.get_knowledge_graph()
```

- 15 hypothesis types (causal, temporal, structural, advanced)
- Confidence-weighted hypergraph with temporal decay
- Thompson sampling for exploration-exploitation

### KShield Pulse
Social signal detection and threat index computation.

```python
from kshiked.pulse import PulseSensor

sensor = PulseSensor(use_nlp=True)
detections = sensor.process_text(text, metadata)
shock_vector = sensor.get_shock_vector(variables)
```

- 15 signal types from survival stress to mobilization readiness
- 5 threat indices: Polarization, Legitimacy Erosion, Mobilization Readiness, Elite Cohesion, Information Warfare
- Kenya-specific: 47-county mapping, ethnic tension tracking

### Federated Learning
Multi-agency collaboration without raw data sharing.

```python
from scarcity.federation import FederationClientAgent, SecureAggregator

client = FederationClientAgent(node_id="agency_1", reconciler=reconciler)
aggregator = SecureAggregator(min_participants=3)
```

- Hierarchical two-layer aggregation
- Secure aggregation with Shamir secret sharing
- Differential privacy (local + central)
- Push-pull gossip with materiality thresholds

### Economic Governance
Stock-Flow Consistent macroeconomic simulation.

```python
from kshiked.core import EconomicGovernor

governor = EconomicGovernor(config, env)
await governor.step(current_tick)
```

- SFC model with households, firms, banks, government
- PID control for policy execution
- Shock modeling (impulse, stochastic, mean-reverting)

### K-SHIELD Simulation Pipeline
Kenya-calibrated SFC simulation with 9 scenarios and 11 analysis tabs.

```python
from scarcity.simulation.sfc import SFCEconomy, SFCConfig
from kshiked.simulation.kenya_calibration import calibrate_from_data
from kshiked.simulation.scenario_templates import get_scenario_by_id

calib = calibrate_from_data(steps=50, policy_mode="custom")
scenario = get_scenario_by_id("oil_crisis")
calib.config.shock_vectors = scenario.build_shock_vectors(50)

econ = SFCEconomy(calib.config)
econ.initialize()
trajectory = econ.run(50)
```

- 9 named scenarios (Oil Crisis, Drought, Debt Crisis, Perfect Storm, etc.)
- 8 policy templates (CBK Tightening, Austerity, Rate Cap 2016, etc.)
- NK Phillips Curve with inflation anchoring
- 11 dynamic 3D-capable analysis tabs in the dashboard

---

##  Multi-Agency Federation

SENTINEL enables agencies to collaborate on threat models **without ever sharing raw data**.

```
                    ┌─────────────────────────────────────┐
                    │         SENTINEL FEDERATION HUB     │
                    │   Receives: Model updates (numbers) │
                    │   Never sees: Raw data              │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────┬───────────┼───────────┬───────────┐
           ▼           ▼           ▼           ▼           ▼
       ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
       │  NIS  │   │  DCI  │   │  CBK  │   │  KDF  │   │  CA   │
       │ Data  │   │ Data  │   │ Data  │   │ Data  │   │ Data  │
       │ stays │   │ stays │   │ stays │   │ stays │   │ stays │
       │ HERE  │   │ HERE  │   │ HERE  │   │ HERE  │   │ HERE  │
       └───────┘   └───────┘   └───────┘   └───────┘   └───────┘
```

**Privacy Guarantees:**
- Local training — raw data never leaves agency
- Gradient encryption — only encrypted updates transmitted
- Differential privacy — noise prevents reverse-engineering
- Secure aggregation — hub cannot see individual contributions

---

##  Capabilities

### Currently Implemented 

| Category | Capabilities |
|----------|--------------|
| **Discovery** | 15 hypothesis types, hypergraph store, temporal decay |
| **Signals** | 15 signal detectors, 5 threat indices, NLP pipeline |
| **Simulation** | SFC 4-sector model, NK Phillips Curve, 9 scenarios, 8 policies, 11 dynamic 3D tabs |
| **Federation** | Secure aggregation, differential privacy, gossip protocol |
| **Governance** | PID control, crisis modes, multi-policy coordination |
| **Geo-Mapping** | 47 Kenya counties, ethnic tension tracking |
| **K-SHIELD** | Auth-gated module with Causal, Terrain, Simulation, Impact cards |

### Under Development 

| Category | Capabilities |
|----------|--------------|
| **Causal Rigor** | Granger tests, counterfactual validation, confidence bounds |
| **Signal Intelligence** | Silence detection, going-dark indicators, false calm |
| **Infrastructure** | Power/telecom stress, cascade path modeling |
| **Decision Support** | Time-to-escalation, escalation pathways, fragility index |
| **Meta-Intelligence** | Unknown-unknown detection, competing hypotheses |
| **Analyst Layer** | Feedback learning, explainability, hypothesis override |

---

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Omega-Makena/scarcity.git
cd scarcity

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install the scarcity library (editable mode)
pip install -e pypi/

# For backend development, also install backend deps
pip install -r backend/requirements.txt

# For the Streamlit dashboard
pip install streamlit plotly pandas numpy
```

### Run the Dashboard

```bash
# From the project root
streamlit run kshiked/ui/sentinel_dashboard.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

##  Project Structure

```
scace4/
├── scarcity/                       # Core discovery engine (domain-agnostic)
│   ├── engine/                     # Hypothesis discovery (engine_v2.py)
│   ├── federation/                 # Federated learning
│   ├── simulation/                 # SFC economic model (SFCEconomy, SFCConfig)
│   ├── causal/                     # Causal inference (DoWhy)
│   ├── meta/                       # Meta-learning
│   ├── governor/                   # Resource control
│   ├── fmi/                        # Federated metadata interchange
│   ├── stream/                     # Data ingestion / windowing
│   └── runtime/                    # EventBus, telemetry
├── kshiked/                        # KShield intelligence (Kenya-specific)
│   ├── core/                       # Governance & shocks
│   ├── pulse/                      # Signal detection (15 signals)
│   ├── sim/                        # Backtesting & Monte Carlo
│   ├── simulation/                 # Kenya calibration, 9 scenarios, 8 policies
│   ├── analysis/                   # Data quality & crash analysis
│   ├── ui/                         # SENTINEL dashboard (Streamlit)
│   │   ├── sentinel_dashboard.py   # Entry point — 12 sidebar views
│   │   └── kshield/               # K-SHIELD sub-module
│   │       ├── page.py            # Auth + landing + routing
│   │       ├── simulation.py      # 11 analysis tabs (2040 lines)
│   │       ├── causal.py          # Causal Relationships card
│   │       ├── terrain.py         # Policy Terrain card
│   │       └── impact.py          # Policy Impact card
│   ├── causal_adapter/             # Causal adapter layer
│   ├── federation/                 # KShield federation bridge
│   └── hub.py                      # KShieldHub orchestrator
├── backend/                        # REST API (FastAPI)
│   └── app/
│       ├── api/v1/                 # v1 endpoints (deprecated)
│       ├── api/v2/                 # v2 endpoints (current)
│       └── core/                   # Config, managers
├── scarcity-deep-dive/             # Interactive frontend (Vite + React)
├── documentation/                  # Technical docs
│   ├── scarcity-docs/              # Engine documentation (11 modules)
│   ├── kshield-docs/               # KShield documentation (12 modules)
│   ├── SCARCITY_VS_SENTINEL.md     # Relationship guide (3-layer architecture)
│   ├── SIMULATION_ENGINE.md        # Full SFC + calibration + scenario reference
│   ├── DASHBOARD_ROUTING.md        # Navigation architecture
│   ├── SCARCITY_ARCHITECTURE.md    # System architecture overview
│   ├── CONFIG_REFERENCE.md         # All 60+ environment variables
│   └── SENTINEL_ROADMAP.md         # Full roadmap
└── tests/                          # Test suite
```
---

##  Documentation

| Document | Description |
|----------|-------------|
| [Scarcity vs SENTINEL](documentation/SCARCITY_VS_SENTINEL.md) | **Start here** — 3-layer architecture relationship guide |
| [Simulation Engine](documentation/SIMULATION_ENGINE.md) | Full SFC + calibration + scenario pipeline reference |
| [Dashboard Routing](documentation/DASHBOARD_ROUTING.md) | Navigation architecture (HOME → K-SHIELD → tabs) |
| [SENTINEL Roadmap](documentation/SENTINEL_ROADMAP.md) | Full technical roadmap with 50 capabilities |
| [Scarcity Architecture](documentation/SCARCITY_ARCHITECTURE.md) | System architecture overview with data flow diagrams |
| [Configuration Reference](documentation/CONFIG_REFERENCE.md) | All 60+ environment variables |
| [Scarcity Docs](documentation/scarcity-docs/INDEX.md) | Detailed engine documentation (11 modules) |
| [KShield Docs](documentation/kshield-docs/INDEX.md) | Intelligence layer documentation (12 modules) |

---

##  Use Cases

### National Security
- Early warning for civil unrest
- Election violence prevention
- Cross-border threat monitoring
- Critical infrastructure protection

### Economic Stability
- Currency attack detection
- Supply chain disruption alerts
- Food security intelligence
- Inflation pressure forecasting

### Multi-Agency Coordination
- Privacy-preserving intelligence fusion
- Cross-domain threat correlation
- Unified alerting without data exposure

---

##  Security Features

| Feature | Implementation |
|---------|----------------|
| Authentication | SSO/MFA ready |
| Authorization | Role-based access control |
| Encryption | End-to-end (in-transit + at-rest) |
| Privacy | Differential privacy + secure aggregation |
| Audit | Immutable action logging |
| Deployment | Air-gap option available |

---

##  Key Metrics

| Metric | Target |
|--------|--------|
| Lead time for warnings | 48-72 hours |
| False alarm rate | < 15% |
| Reliability score | > 80% |
| Decision-latency accuracy | ±6 hours |

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

##  Team

**Omega Makena** — Lead Developer

---

##  Roadmap

**Target Completion: March 20, 2026**

See [SENTINEL_ROADMAP.md](documentation/SENTINEL_ROADMAP.md) for the complete 50-capability roadmap.

---

*Built for Kenya. Designed for national security.*

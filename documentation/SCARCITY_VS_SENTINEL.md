# Scarcity vs SENTINEL — Relationship Guide

> Understanding the layered architecture: how the **Scarcity** library powers the **SENTINEL** platform, and how **KShield** bridges them.

---

## The Three Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SENTINEL                                        │
│              (The Product — what the end-user sees)                       │
│                                                                          │
│   Streamlit Command Center · 13 routed views · 4 K-modules              │
│   Analyst-facing dashboards with threat maps, simulation UI,             │
│   federation status, and decision support.                               │
├─────────────────────────────────────────────────────────────────────────┤
│                          KSHIELD                                         │
│        (The Intelligence Layer — domain-specific adapters)               │
│                                                                          │
│   Kenya calibration · 9 scenario templates · 8 policy presets            │
│   15 social signal detectors · Pulse threat indices · County mapping     │
│   Causal adapter · Federation bridge (Aegis Protocol)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                          SCARCITY                                        │
│          (The Foundation — domain-agnostic engine library)               │
│                                                                          │
│   OnlineDiscoveryEngine · SFCEconomy · FederationClientAgent             │
│   DoWhy causal inference · Meta-learning · EventBus runtime              │
│   No country-specific code — works on any economic dataset               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Is Scarcity?

**Scarcity** (`scarcity/`) is a **domain-agnostic Python library** for discovering, simulating, and federating economic relationships from streaming data.

It knows nothing about Kenya, K-SHIELD, dashboards, or threat intelligence. It provides **pure algorithms**:

| Package | Responsibility |
|---------|---------------|
| `scarcity.engine` | Online hypothesis discovery — 15 relationship types, Thompson sampling, confidence-weighted hypergraph |
| `scarcity.simulation` | Stock-Flow Consistent (SFC) macroeconomic model with 4 sectors, Phillips Curve, Taylor Rule, Okun's Law |
| `scarcity.federation` | Federated learning — secure aggregation (Shamir), differential privacy, gossip protocol |
| `scarcity.causal` | Rigorous causal inference via DoWhy — identification, estimation, refutation |
| `scarcity.meta` | Meta-learning (Reptile/MAML) for hypothesis tuning across data regimes |
| `scarcity.governor` | Dynamic resource governance — CPU/memory throttling for engine workloads |
| `scarcity.fmi` | Federated Metadata Interchange — schema exchange between deployment nodes |
| `scarcity.stream` | Data ingestion — windowing, time-alignment, format adapters |
| `scarcity.runtime` | EventBus architecture, telemetry, diagnostics |
| `scarcity.analytics` | Policy terrain analysis utilities |
| `scarcity.synthetic` | Synthetic data generators for testing (accounts, behaviors, content) |

### Design Principle
> Scarcity should be **installable via pip** and usable by any developer in any country on any dataset. No hardcoded indicator names, no country-specific assumptions, no UI dependencies.

---

## What Is KShield?

**KShield** (`kshiked/`) is the **Kenya-specific intelligence layer** — it adapts the generic Scarcity algorithms for the specific problem domain of national economic and threat intelligence.

KShield is where **domain knowledge** lives:

| Package | What It Adds Over Scarcity |
|---------|---------------------------|
| `kshiked.core` | `EconomicGovernor` — orchestrates multi-policy execution with PID control, crisis modes, and Kenya-specific sector definitions |
| `kshiked.pulse` | `PulseSensor` — 15 social signal detectors trained on Kenya-relevant patterns (ethnic tension, diaspora flows, M-Pesa velocity, BBI-era language) |
| `kshiked.simulation` | `kenya_calibration.py` — derives SFC parameters from World Bank Kenya data. `scenario_templates.py` — 9 Kenya-specific shock scenarios (KES depreciation, Rift Valley drought, SGR debt corridor) |
| `kshiked.causal_adapter` | Bridges Scarcity's `OnlineDiscoveryEngine` → KShield UI. Translates discovered hypotheses into analyst-readable causal chains |
| `kshiked.federation` | Aegis Protocol — defense-sector federation bridge. Maps to Kenya's institutional structure (NIS, DCI, CBK, KDF, CA) |
| `kshiked.hub` | `KShieldHub` — central orchestrator that wires Pulse + Causal + Simulation + Federation into a single real-time event loop |
| `kshiked.ui` | SENTINEL Command Center — routed Streamlit UI with K-SHIELD sub-module (Causal, Terrain, Simulation, Impact) |
| `kshiked.analysis` | Offline diagnostics — data quality checks and historical crash identification |

### Design Principle
> KShield should **never modify Scarcity source code**. It extends via composition — importing Scarcity classes, wrapping them with calibration/configuration, and exposing them through Kenya-tuned APIs.

---

## What Is SENTINEL?

**SENTINEL** is the **user-facing product** — the complete analyst workstation. It is not a separate codebase; it is the top-level composition of Scarcity + KShield into a deployable application.

SENTINEL comprises:

| Component | Files | Purpose |
|-----------|-------|---------|
| Command Center | `kshiked/ui/sentinel_dashboard.py`, `kshiked/ui/sentinel/router.py` | Streamlit entrypoint + routed navigation with deep links (`?view=...`) |
| K-SHIELD Module | `kshiked/ui/kshield/page.py` | Auth gate → landing page → 4 sub-cards (Causal, Terrain, Simulation, Impact) |
| K-PULSE Module | via `sentinel_dashboard.py` Signal Intelligence tab | Real-time social signal monitoring |
| K-COLLAB Module | `kshiked/ui/sentinel/federation.py` + `federated_databases/` | Federated DB node control, sync rounds, metrics, audit trail |
| K-EDUCATION Module | via `sentinel_dashboard.py` Document Intelligence tab | Explainable analytics and public knowledge |
| Policy Intelligence | `kshiked/ui/sentinel/policy_chat.py` | Bill analysis chat with URL-linked evidence traces |
| Backend API | `backend/app/` | FastAPI REST API (v2) for programmatic access |
| Frontend | `scarcity-deep-dive/` | Vite + React interactive visualisation (secondary UI) |

---

## How They Interact — Data Flow

```
  World Bank CSV                    User-uploaded CSV
       │                                  │
       └──────────┬───────────────────────┘
                  │
                  ▼
      ┌─────────────────────┐
      │  kenya_calibration   │ ← KShield: derives SFC params from data
      │  .calibrate_from_data│   (tax_rate, spending_ratio, inflation_target...)
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  scenario_templates  │ ← KShield: builds shock vector time-series
      │  .build_shock_vectors│   (oil crisis, drought, KES depreciation...)
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  scarcity.simulation │ ← Scarcity: runs the generic SFC engine
      │  .SFCEconomy.run()   │   (4 sectors, Phillips Curve, Taylor Rule)
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  simulation.py (UI)  │ ← SENTINEL: renders 11 analysis tabs
      │  render_simulation() │   (3D state cube, IRFs, Monte Carlo, etc.)
      └─────────────────────┘
```

### The SFC Simulation Pipeline in Detail

```
 calibrate_from_data()   →   SFCConfig (40+ parameters)
       +                           +
 build_shock_vectors()   →   shock_vectors[t][demand|supply|fiscal|fx]
       +                           +
 SFCEconomy(config)      →   4 sector balance sheets initialised
       │
       ├── step() × N quarters
       │   ├── 1. Apply shock vector
       │   ├── 2. Policy response (Taylor Rule / custom)
       │   ├── 3. Economic dynamics (Phillips, Okun, fiscal)
       │   ├── 4. Outcome scoring (14 dimensions)
       │   └── 5. Record 4D frame
       │
       └── trajectory: List[Dict]
             ├── t (int)
             ├── shock_vector  {demand, supply, fiscal, fx}
             ├── policy_vector {policy_rate, fiscal_impulse, ...}
             ├── channels      {output_gap, inflation_gap, credit_spread}
             ├── outcomes      {gdp_growth, inflation, unemployment, ...}
             ├── flows         {consumption, investment, govt_spending, ...}
             └── sector_balances {households, firms, government, banks}
```

---

## The Separation in Practice

| Question | Answer |
|----------|--------|
| Can Scarcity run without KShield? | **Yes.** Install via `pip install -e pypi/` and use `SFCEconomy`, `OnlineDiscoveryEngine`, etc. directly. |
| Can KShield run without Scarcity? | **No.** KShield imports `scarcity.simulation.sfc`, `scarcity.engine`, `scarcity.federation` etc. |
| Can SENTINEL run without both? | **No.** The dashboard imports KShield modules which import Scarcity modules. |
| Where do I add a new country? | Create a new calibration module (like `kenya_calibration.py`) and scenario templates. Scarcity's SFC engine works unchanged. |
| Where do I add a new hypothesis type? | In `scarcity/engine/`. KShield and SENTINEL pick it up automatically. |
| Where do I add a new dashboard view? | In `kshiked/ui/sentinel/router.py` (`NAV_OPTIONS`) and corresponding `kshiked/ui/sentinel/*.py` renderer. |
| Where is the SFC model? | `scarcity/simulation/sfc.py` — the core engine. `kshiked/simulation/kenya_calibration.py` — the Kenya parameter wrapper. |

---

## Dependency Direction

```
  SENTINEL (product)
      │
      ├── imports → KShield (intelligence layer)
      │                 │
      │                 └── imports → Scarcity (foundation library)
      │
      └── imports → Scarcity (directly, for federation/meta)
```

> Dependencies flow **downward only**. Scarcity never imports KShield. KShield never imports sentinel_dashboard.py. This ensures the foundation library remains reusable.

---

## Package Distribution

| Package | Install Path | Audience |
|---------|-------------|----------|
| Scarcity | `pip install -e pypi/` | Python developers, data scientists — any country, any dataset |
| KShield | `import kshiked` (local) | Kenya security analysts, CBK, NIS, intelligence agencies |
| SENTINEL | `streamlit run kshiked/ui/sentinel_dashboard.py` | Analyst workstation operators — the complete product |

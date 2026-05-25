# KShield — Documentation Index

Complete documentation for the **KShield** economic monitoring and early warning system.

---

## Quick Navigation

| Module | Description |
|--------|-------------|
| [core](./core/INDEX.md) | Economic governance, policies, shocks |
| [pulse](./pulse/INDEX.md) | Social sensing engine — 15 signals |
| [sim](./sim/INDEX.md) | Backtesting and Monte Carlo |
| [causal](./causal/INDEX.md) | Causal inference adapter |
| [hub](./hub/INDEX.md) | KShieldHub central orchestrator |
| [ui](./ui/INDEX.md) | SENTINEL Command Center dashboard (routed multi-view UI) |
| [federation-kshield](./federation-kshield/INDEX.md) | Aegis Protocol — defense sector federation |
| [causal-adapter](./causal-adapter/INDEX.md) | Causal pipeline bridge (Scarcity ↔ KShield) |
| [simulation-kshield](./simulation-kshield/INDEX.md) | ShockCompiler — stochastic → SFC vectors |
| [data](./data/INDEX.md) | GeoJSON boundaries, news DB, analysis tools |
| [backend-core](./backend-core/INDEX.md) | FastAPI core logic |
| [backend-api](./backend-api/INDEX.md) | REST API endpoints |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KSHIELD FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Tier 4: Backend API                             │   │
│   │              FastAPI · REST Endpoints · WebSocket                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Tier 3: Core Manager                            │   │
│   │     ScarcityCoreManager · DomainManager · FederationCoordinator      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Tier 2: Pulse Engine                            │   │
│   │       PulseSensor · Detectors · Indices · Co-occurrence              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Tier 1: Core Logic                              │   │
│   │   EconomicGovernor · PolicyTensorEngine · ShockManager               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reading Order

### For New Users

1. **[Core Overview](./core/00_overview.md)** — Governance basics
2. **[Pulse Overview](./pulse/00_overview.md)** — Signal detection
3. **[Backend API](./backend-api/00_overview.md)** — REST endpoints

### For Data Scientists

1. **[Pulse Signals](./pulse/00_overview.md)** — 15 signal types
2. **[Sim Backtesting](./sim/00_overview.md)** — Monte Carlo validation
3. **[Core Shocks](./core/00_overview.md)** — OU, Brownian processes

### For Developers

1. **[Backend Core](./backend-core/00_overview.md)** — ScarcityCoreManager
2. **[Backend API](./backend-api/00_overview.md)** — API design
3. **[Scarcity Integration](../scarcity-docs/INDEX.md)** — Related framework

---

## Module Summaries

### Core (`kshiked.core`)

Economic governance system:
- **EconomicGovernor**: PID control loop
- **PolicyTensorEngine**: Vectorized evaluation
- **ShockManager**: Impulse, OU, Brownian shocks
- **Policies**: Monetary/fiscal with crisis modes

### Pulse (`kshiked.pulse`)

Social sensing engine:
- **15 signals**: Survival stress to mobilization
- **Detectors**: NLP-enhanced (sentiment, emotion)
- **Indices**: PI, LEI, MRS, ECI, IWI
- **Co-occurrence**: Time-weighted analysis
- **Kenya focus**: Ethnic tension tracking

### Sim (`kshiked.sim`)

Backtesting and validation:
- **BacktestEngine**: Calibrate γ parameter
- **MonteCarloSimulator**: Parallel runs
- **CountryProfile**: Pre-configured shocks
- **SystemicShock**: Historical events

### Backend Core

Application lifecycle:
- **ScarcityCoreManager**: Component coordination
- **DomainManager**: Multi-domain simulation
- **FederationCoordinator**: Federated learning

### Backend API

REST endpoints:
- **/api/v1/** (deprecated, mock data): status, metrics, domains, data, onboarding, mpie, risk
- **/api/v2/** (current): health, metrics, domains, demo, runtime, mpie, drg, federation, meta, simulation

---

## Related Documentation

| Documentation | Description |
|---------------|-------------|
| [Scarcity](../scarcity-docs/INDEX.md) | Core discovery framework |

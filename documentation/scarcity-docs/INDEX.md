# Scarcity Library — Documentation Index

Complete documentation for the **Scarcity** library — an online relationship discovery and simulation framework for economic data.

---

## Quick Navigation

| Module | Description |
|--------|-------------|
| [engine](./engine/INDEX.md) | Core discovery engine — hypotheses, evaluation, relationships |
| [federation](./federation/INDEX.md) | Federated learning with privacy guarantees |
| [simulation](./simulation/INDEX.md) | SFC economic simulation |
| [causal](./causal/INDEX.md) | Rigorous causal inference (DoWhy) |
| [meta](./meta/INDEX.md) | Meta-learning and governance |
| [governor](./governor/INDEX.md) | Dynamic resource control |
| [fmi](./fmi/INDEX.md) | Federated metadata interchange |
| [stream](./stream/INDEX.md) | Data ingestion and windowing |
| [runtime](./runtime/INDEX.md) | EventBus and telemetry |
| [analytics](./analytics/INDEX.md) | Policy terrain analysis |\r\n| [synthetic](./synthetic/INDEX.md) | Test data generation (accounts, content, behavior) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SCARCITY FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Tier 5: Meta Layer                          │   │
│   │         MetaIntegrativeLayer · MetaLearningAgent · Reptile          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Tier 4: Federation Layer                        │   │
│   │   HierarchicalFederation · Gossip · SecureAggregation · FMI          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Tier 3: Discovery Engine                       │   │
│   │     OnlineDiscoveryEngine · Hypothesis · Evaluator · BanditRouter    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Tier 2: Simulation                             │   │
│   │             SFCEconomy · SimulationEngine · WhatIfManager            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Tier 1: Infrastructure                         │   │
│   │   EventBus · Telemetry · DynamicResourceGovernor · Stream            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reading Order

### For New Users

1. **[Engine Overview](./engine/00_overview.md)** — Understand relationship discovery
2. **[Simulation Overview](./simulation/00_overview.md)** — SFC economics basics
3. **[Runtime Overview](./runtime/00_overview.md)** — EventBus fundamentals

### For Federated Deployments

1. **[Federation Overview](./federation/00_overview.md)** — Hierarchical federation
2. **[FMI Overview](./fmi/00_overview.md)** — Metadata interchange
3. **[Meta Overview](./meta/00_overview.md)** — Cross-domain learning

### For Production Operations

1. **[Governor Overview](./governor/00_overview.md)** — Resource management
2. **[Stream Overview](./stream/00_overview.md)** — Data pipeline
3. **[Runtime Overview](./runtime/00_overview.md)** — Monitoring

---

## Module Summaries

### Engine

The core discovery system:
- **Hypotheses**: 15 relationship types (causal, temporal, structural, etc.)
- **Evaluation**: Bootstrap confidence intervals, predictive gain
- **Routing**: Multi-armed bandit for exploration-exploitation
- **Storage**: Hypergraph with temporal decay

### Federation

Privacy-preserving distributed learning:
- **Hierarchical**: Two-layer aggregation with domain baskets
- **Privacy**: Local DP, secure aggregation, central DP
- **Robustness**: Byzantine-resistant aggregation (Krum, Bulyan)
- **Communication**: Gossip protocol with materiality detection

### Simulation

Stock-Flow Consistent economic model:
- **Sectors**: Households, Firms, Banks, Government, Foreign
- **Identities**: Accounting constraints always hold
- **Dynamics**: Behavioral equations, Phillips curve, Taylor rule
- **What-If**: Counterfactual scenario testing

### Causal

Rigorous causal inference:
- **Identification**: Backdoor, IV, mediation
- **Estimation**: Linear, propensity score, causal forests
- **Validation**: Refutation tests
- **Integration**: DoWhy and EconML

### Meta

Adaptive hyperparameter learning:
- **Governance**: Rule-based policy engine
- **Learning**: Domain and cross-domain priors
- **Optimization**: Online Reptile algorithm
- **Safety**: Rollback on performance drops

### Governor

Dynamic resource control:
- **Sensors**: CPU, GPU, memory, I/O monitoring
- **Profiling**: EMA and Kalman filtering
- **Policies**: Rule-based threshold triggers
- **Actuators**: Batch reduction, throttling, load shedding

### FMI

Federated metadata interchange:
- **Packets**: MSP, POP, CCS types
- **Validation**: Schema registry
- **Aggregation**: Trimmed mean, weighted confidence
- **Privacy**: Optional DP noise

### Stream

Data ingestion pipeline:
- **Sources**: CSV, API, generators
- **Windowing**: Overlapping with normalization
- **Rate Control**: PI controller for backpressure
- **Statistics**: Online mean/variance (Welford)

### Runtime

Core infrastructure:
- **EventBus**: Async pub/sub messaging
- **Telemetry**: Latency, throughput, drift monitoring
- **Probe**: System resource collection

### Analytics

Policy analysis:
- **Terrain**: Response surfaces over policy space
- **Overlays**: Stability and risk visualization

---

## Statistics

| Module | Files | Docs |
|--------|-------|------|
| engine | 31 | 17 |
| federation | 18 | 10 |
| simulation | 11 | 4 |
| causal | 7 | 3 |
| meta | 11 | 3 |
| governor | 9 | 2 |
| fmi | 9 | 2 |
| stream | 8 | 2 |
| runtime | 3 | 2 |
| analytics | 1 | 2 |
| **Total** | **108** | **47** |

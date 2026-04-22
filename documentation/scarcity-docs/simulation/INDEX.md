# Simulation Module — Documentation Index

Complete documentation for the `scarcity.simulation` module — economic simulation with SFC dynamics.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and concepts |
| [01_sfc.md](./01_sfc.md) | SFCEconomy — Stock-Flow Consistent model |
| [02_utilities.md](./02_utilities.md) | Engine, agents, what-if, visualization |
| [03_agents_sectors.md](./03_agents_sectors.md) | AgentRegistry, NodeAgent, EdgeLink; households, government, banking, production, labor, monetary, foreign, financial accelerator, heterogeneous agents, IO structure |
| [04_scenario_dynamics.md](./04_scenario_dynamics.md) | ShockProcess, DynamicsEngine, SimulationEnvironment, WhatIfManager, Bayesian estimation |
| [05_infrastructure.md](./05_infrastructure.md) | EconomyState, parameters, accounting checks, coupling interface, LearnedSFCEconomy, open economy, price system, SFC engine, storage, scheduler, monitor |

---

## Key Concepts

### Stock-Flow Consistency

- Every flow has source and destination
- Balance sheets must balance
- Sectoral balances sum to zero

### Sectors

Five economic sectors:
- Households, Firms, Banks, Government, Foreign

### Integration

Discovered relationships become simulation dynamics:
- Engine discovers "A → B"
- Simulation uses edge weight for propagation

---

## Quick Start

```python
from scarcity.simulation.sfc import SFCEconomy

economy = SFCEconomy()
economy.initialize(gdp=100.0)
economy.run(100)

print(economy.get_state())
```

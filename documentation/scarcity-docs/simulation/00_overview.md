# Scarcity Simulation Module — Overview

The **simulation module** transforms the discovered relationships from the engine into a **runnable economic model**. It implements Stock-Flow Consistent (SFC) dynamics with proper accounting identities.

---

## Purpose

Once the engine discovers relationships like "interest_rate → inflation", the simulation module can:

1. **Project forward**: Given current state, what happens next?
2. **Test scenarios**: What if we raise interest rates by 1%?
3. **Validate discoveries**: Do the discovered relationships produce realistic dynamics?
4. **Power dashboards**: Live visualization of economic state

---

## Core Concepts

### Stock-Flow Consistency (SFC)

The simulation follows SFC principles from post-Keynesian economics:

- **Stocks**: Balance sheet items (assets, liabilities, net worth)
- **Flows**: Income, expenses, transfers between sectors
- **Accounting identities**: Must always hold (e.g., Assets = Liabilities + Net Worth)

This ensures the simulation is **internally consistent** — money doesn't appear or disappear.

### Sectors

The economy is divided into sectors:

| Sector | Assets | Liabilities |
|--------|--------|-------------|
| **Households** | Deposits, bonds | — |
| **Firms** | Capital, deposits | Loans, equity |
| **Banks** | Loans, reserves | Deposits |
| **Government** | — | Bonds |
| **Foreign** | Forex reserves | — |

### Key Identities

1. **Sectoral balance**: Sum of all sectors' net lending = 0
2. **Balance sheet**: Assets = Liabilities + Net Worth (per sector)
3. **Flow of funds**: Every flow has a source and destination

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SimulationEngine                             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  SFCEconomy │  │ AgentRegistry│  │ WhatIfManager│              │
│  │  (dynamics) │  │   (nodes)    │  │  (scenarios) │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Event Bus                                 ││
│  │  Topics: sim.state, sim.shock, engine.insight               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────┐
│  Dashboard  │
│ (Streamlit) │
└─────────────┘
```

---

## Key Components

### SFCEconomy (`sfc.py`)

The core economic model:

```python
from scarcity.simulation.sfc import SFCEconomy, SFCConfig

economy = SFCEconomy(config=SFCConfig())
economy.initialize(gdp=100.0)

# Simulate
for _ in range(100):
    economy.step()
    state = economy.get_state()
    print(f"GDP: {state['gdp']:.2f}")
```

### SimulationEngine (`engine.py`)

Orchestrates the simulation loop:

```python
from scarcity.simulation import SimulationEngine, SimulationConfig

engine = SimulationEngine(
    registry=agent_registry,
    config=SimulationConfig()
)
engine.start()
```

Integrates with event bus for real-time updates.

### AgentRegistry (`agents.py`)

Converts discovered relationships into simulation entities:
- `NodeAgent`: Individual economic variable
- `EdgeLink`: Causal relationship between variables

### WhatIfManager (`whatif.py`)

Runs counterfactual scenarios:

```python
results = engine.run_whatif(
    scenario_id="rate_hike",
    node_shocks={"interest_rate": 0.01},  # +1%
    horizon=12  # 12 periods
)
```

---

## Data Flow

### From Engine to Simulation

```
1. OnlineDiscoveryEngine discovers relationship
2. Publishes to "engine.insight" topic
3. SimulationEngine receives via event bus
4. AgentRegistry updates its edge weights
5. Next simulation step uses new relationships
```

### Simulation Tick Cycle

```
1. Dynamics.update()
   - Apply behavioral equations
   - Update sector balance sheets
   - Check accounting identities

2. Monitor.record()
   - Record state to history
   - Check for anomalies

3. Publish state
   - Send to "sim.state" topic
   - Dashboard receives and renders

4. Scheduler.advance()
   - Move to next time step
```

---

## File Guide

| File | Purpose |
|------|---------|
| `sfc.py` | Stock-Flow Consistent economic model |
| `engine.py` | Simulation orchestrator |
| `agents.py` | Node and edge representations |
| `dynamics.py` | Dynamic equation handlers |
| `environment.py` | Simulation environment setup |
| `whatif.py` | Counterfactual scenario manager |
| `scheduler.py` | Time step scheduling |
| `monitor.py` | State monitoring and recording |
| `storage.py` | Simulation state persistence |
| `visualization3d.py` | 3D network visualization |

---

## Integration Points

### With Engine Module

```python
# Engine discovers relationships
knowledge = engine.get_knowledge_graph()

# Convert to simulation structure
registry = AgentRegistry()
registry.load_from_store_snapshot(store.snapshot())

# Run simulation with discovered dynamics
sim_engine = SimulationEngine(registry, config)
```

### With Governor Module

The `EconomicGovernor` can:
- Apply policy actions to the simulation
- Observe simulation state for policy decisions
- Run what-if scenarios before committing policies

### With Dashboard

Real-time state updates via event bus:
- `sim.state`: Current economic indicators
- `sim.shock`: Applied shocks
- `sim.whatif`: Scenario results

---

## Shock System

Apply exogenous shocks to the economy:

```python
economy.apply_shock("demand", magnitude=0.1)   # +10% demand
economy.apply_shock("monetary", magnitude=0.02)  # +2% rate
economy.apply_shock("fiscal", magnitude=0.05)   # +5% spending
economy.apply_shock("supply", magnitude=-0.1)   # -10% supply
```

Shocks propagate through the behavioral equations over time.

---

## Behavioral Equations

The SFC model uses these core equations:

### Consumption
```
C = c1 * Y_d + c2 * W[-1]
```
(Consumption depends on disposable income + lagged wealth)

### Investment
```
I = i1 * (Y - Y[-1]) + i2 * r[-1]
```
(Investment depends on growth + interest rates)

### Money Demand
```
M = λ0 + λ1 * Y - λ2 * r
```
(Money demand increases with income, decreases with rates)

### Phillips Curve
```
π = π[-1] + β * (u* - u)
```
(Inflation responds to unemployment gap)

---

## Validation

The simulation continuously validates:

1. **Accounting identities**: All sectors' net lending sums to zero
2. **Balance sheet consistency**: A = L + NW for each sector
3. **No infinite values**: Bounded dynamics
4. **Reasonable ranges**: GDP > 0, unemployment ∈ [0, 1], etc.

Violations trigger warnings and optional corrections.

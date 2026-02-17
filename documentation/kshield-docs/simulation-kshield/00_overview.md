# KShield Simulation — Overview

> `kshiked.simulation` — Transforms stochastic shocks into SFC-compatible vectors.

---

## Purpose

This module bridges the gap between KShield's rich stochastic shock definitions (Impulse, Ornstein-Uhlenbeck, Brownian Motion) and the Scarcity SFC engine which requires deterministic time-series vectors. The **ShockCompiler** generates these vectors at initialization time, keeping the core SFC engine fast and vectorised.

---

## File Guide

| File | Size | Purpose |
|------|------|---------|
| `compiler.py` | 4.9 KB | `ShockCompiler` — converts `Shock` objects to `np.ndarray` vectors |
| `controller.py` | 5.4 KB | `SimulationController` — orchestrates scenario runs and parameter sweeps |

---

## ShockCompiler

### Constructor

```python
ShockCompiler(steps=50, dt=1.0, seed=42)
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `steps` | `int` | `50` | Number of time steps to generate |
| `dt` | `float` | `1.0` | Time step size |
| `seed` | `int` | `42` | Random seed for reproducibility |

### `compile(shocks) → Dict[str, np.ndarray]`

Generates deterministic vectors for all active shocks, mapped to the 4 canonical SFC channels:

| Channel Key | Triggered By |
|-------------|-------------|
| `demand_shock` | consumption, investment, demand keywords |
| `supply_shock` | productivity, supply keywords |
| `fiscal_shock` | spending, tax, fiscal keywords |
| `fx_shock` | currency, rate, FX keywords |

### Shock Types Supported

| Type | Class | Parameters |
|------|-------|------------|
| Impulse | `ImpulseShock` | `magnitude`, `decay` |
| OU Process | `OUProcessShock` | `theta`, `mu`, `sigma`, `dt` |
| Brownian | `BrownianShock` | `drift`, `volatility`, `dt` |

### Example

```python
from kshiked.simulation.compiler import ShockCompiler
from kshiked.core.shocks import ImpulseShock

compiler = ShockCompiler(steps=100, seed=42)
shocks = [ImpulseShock(target_metric="demand", magnitude=-0.05, decay=0.9)]
vectors = compiler.compile(shocks)
# vectors["demand_shock"] → np.array of 100 values, decaying from -0.05
```

---

## Integration with SFC

```
Shock definitions (kshiked.core.shocks)
    │
    ▼
ShockCompiler.compile()
    │
    ▼
Dict[str, np.ndarray]  (4 canonical channels)
    │
    ▼
SFCConfig.shock_vectors  (scarcity.simulation.sfc)
    │
    ▼
SFCEconomy.step()
```

---

*Source: `kshiked/simulation/` · Last updated: 2026-02-11*

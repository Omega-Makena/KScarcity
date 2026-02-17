# Sim Module — Documentation Index

Complete documentation for the `kshiked.sim` module — backtesting and simulation.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Backtest, Monte Carlo |

---

## Key Concepts

### Backtesting

- Load historical data
- Calibrate γ (graph weight)
- Validate against known events

### Monte Carlo

- Multiple simulation runs
- Confidence intervals
- Shock response analysis

### Country Profiles

- Pre-configured country data
- Historical shock events

---

## Quick Start

```python
from kshiked.sim import BacktestEngine, SimulationConfig

config = SimulationConfig(start_year=2010, end_year=2022)
engine = BacktestEngine("data.csv", config)
engine.load_data()
gamma = engine.calibrate()
results = engine.run_simulation(gamma)
```

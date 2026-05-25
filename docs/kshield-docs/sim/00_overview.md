# KShield Sim Module — Overview

The **sim module** provides backtesting and simulation capabilities for KShield economic predictions.

---

## Purpose

Before deploying predictions, you need to:
- **Validate against history**: Does the model predict known crises?
- **Calibrate parameters**: Optimize graph propagation weights
- **Stress test**: Monte Carlo with known shocks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BacktestEngine                            │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ LoadData │───►│ BuildGrph│───►│ RunSim   │───►│ Report   │ │
│   │ (CSV)    │    │(discover)│    │ (shocks) │    │ (MD)     │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                  │
│                     ┌───────────────┐                           │
│                     │ MonteCarlo    │                           │
│                     │ (parallel)    │                           │
│                     └───────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## BacktestEngine (`backtest_prediction.py`)

### Configuration

```python
@dataclass
class SimulationConfig:
    start_year: int = 2010
    end_year: int = 2022
    calibration_end: int = 2019
    monte_carlo_runs: int = 50
    noise_sigma: float = 0.02
    damping: float = 0.95
    growth_nodes: List[str] = None
```

### Usage

```python
from kshiked.sim.backtest_prediction import BacktestEngine, SimulationConfig

config = SimulationConfig(
    start_year=2010,
    end_year=2022,
    monte_carlo_runs=50
)

engine = BacktestEngine(csv_path="kenya_data.csv", config=config)
engine.load_data()

# Calibrate gamma (graph propagation weight)
gamma = engine.calibrate()

# Run with shocks
results = engine.run_simulation(gamma, shocks=kenya_shocks)

# Generate report
engine.generate_report(results, "backtest_report.md")
```

---

## Systemic Shocks

### SystemicShock

```python
@dataclass
class SystemicShock:
    year: int
    impacts: Dict[str, float]
    
    def apply(self, current_year, state):
        if current_year == self.year and not self.triggered:
            for variable, impact in self.impacts.items():
                state[variable] *= (1 + impact)
            self.triggered = True
```

### Example: Kenya Shocks

```python
KENYA_PROFILE = CountryProfile(
    name="Kenya",
    csv_path="API_KEN_DS2_en_csv_v2.csv",
    historical_shocks=[
        # 2011: Drought & Inflation
        SystemicShock(2011, {
            "GDP (current US$)": -0.05,
            "Inflation, consumer prices (annual %)": 0.50
        }),
        # 2020: COVID-19
        SystemicShock(2020, {
            "GDP (current US$)": -0.08,
            "Trade in services (% of GDP)": -0.30
        })
    ]
)
```

---

## Monte Carlo Simulation

```python
class MonteCarloSimulator:
    def run_parallel(self) -> pd.DataFrame:
        # Runs config.monte_carlo_runs iterations
        # Returns aggregated results with confidence intervals
```

Provides:
- Mean trajectory
- Standard deviation
- Confidence bands
- Shock response analysis

---

## CountryProfile

```python
@dataclass
class CountryProfile:
    name: str
    csv_path: str
    currency_symbol: str = "US$"
    historical_shocks: List[SystemicShock] = None
```

Use profiles to quickly backtest different countries:

```python
await run_country_backtest(KENYA_PROFILE)
```

---

## Calibration

The engine finds optimal γ (gamma) for graph propagation:

```python
gamma = engine.calibrate()
# Tests: [0.001, 0.005, 0.01, 0.05, 0.1]
# Minimizes: MAE on calibration period
```

---

## Demo Simulation (`demo_economic_simulation.py`)

Interactive demonstration:

```python
python -m kshiked.sim.demo_economic_simulation
```

Shows:
- Graph construction from data
- Relationship discovery
- Policy simulation
- Governor response

---

## File Guide

| File | Purpose |
|------|---------|
| `backtest_prediction.py` | BacktestEngine, MonteCarloSimulator, CountryProfile |
| `demo_economic_simulation.py` | Interactive demo |
| `run_economic_simulation.py` | Standalone simulation runner |
| `run_governance.py` | Governance loop demo |

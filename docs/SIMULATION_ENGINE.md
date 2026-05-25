# Simulation Engine — Complete Reference

> The full SFC simulation pipeline: from data calibration through scenario execution to multi-tab visualisation.

---

## Architecture Overview

The simulation pipeline has three layers:

```
┌────────────────────────────────────────────────────────────────────┐
│  PRESENTATION   kshiked/ui/kshield/simulation/                     │
│                 Simulation workspace + sector dashboard + workbench │
├────────────────────────────────────────────────────────────────────┤
│  CALIBRATION    kshiked/simulation/kenya_calibration.py             │
│  + SCENARIOS    kshiked/simulation/scenario_templates.py            │
│                 Data-driven parameters + 9 shock templates          │
├────────────────────────────────────────────────────────────────────┤
│  ENGINE (two paths — choose by use case)                           │
│                                                                    │
│  scarcity/simulation/sfc.py          (legacy — SFCEconomy)         │
│  scarcity/simulation/sfc_engine.py   (typed  — MultiSectorSFCEngine│
│                                                                    │
│  IO Foundation:                                                    │
│  scarcity/simulation/io_structure.py (9-sector KNBS IO model)      │
│  scarcity/simulation/parameters.py   (AllParams — KNBS-reconciled) │
└────────────────────────────────────────────────────────────────────┘
```

---

## Engine Paths at a Glance

```
 ┌──────────────────────────────┐  ┌──────────────────────────────────┐
 │  sfc.py — Legacy Path        │  │  sfc_engine.py — Typed Path      │
 │                              │  │                                  │
 │  SFCEconomy                  │  │  MultiSectorSFCEngine            │
 │  SFCConfig (~40 params)      │  │  AllParams (KNBS-reconciled)     │
 │  4 balance-sheet sectors:    │  │  4 production sectors:           │
 │   Households · Firms         │  │   Agriculture · Manufacturing    │
 │   Banks · Government         │  │   Services · Informal            │
 │                              │  │                                  │
 │  Aggregate-level dynamics:   │  │  8 dedicated behavioral blocks:  │
 │   Phillips Curve             │  │   production.py                  │
 │   Taylor Rule                │  │   labor_market.py                │
 │   Okun's Law                 │  │   price_system.py                │
 │   14-dimension outcomes      │  │   households.py                  │
 │                              │  │   government.py                  │
 │  Step cycle (5 stages):      │  │   monetary.py                    │
 │   Shocks → Policy →          │  │   foreign.py                     │
 │   Dynamics → Outcomes →      │  │   banking.py                     │
 │   Logging                    │  │                                  │
 │                              │  │  Support modules:                │
 │  Best for:                   │  │   coupling_interface.py          │
 │  - Kenya calibration UI      │  │   accounting.py (SFC checks)     │
 │  - Scenario comparisons      │  │   types.py (contracts)           │
 │  - Monte Carlo               │  │                                  │
 └──────────────────────────────┘  │  Best for:                       │
                                    │  - Sector-level analysis         │
                                    │  - IO multiplier propagation     │
                                    │  - Research configurations       │
                                    └──────────────────────────────────┘
```

---

## 1. IO Foundation — KNBS Reconciliation

### Problem Solved

`parameters.py` defines a 4-sector production model while `io_structure.py` holds a 9-sector KNBS IO matrix. Before Item 15, these were inconsistent with no link between them.

### Current Architecture

```
 io_structure.py                          parameters.py
 ┌───────────────────────────────┐        ┌──────────────────────────────┐
 │  9-Sector KNBS IO Matrix      │        │  AllParams                   │
 │  (Kenya 2017 SUT)             │        │  └─ InputOutputParams        │
 │                               │        │     ├─ io_matrix (4×4)       │
 │  SubSectorType enum:          │        │     │  ← KNBS-derived 3×3    │
 │   agriculture                 │        │     │  ← field estimates INFM │
 │   manufacturing               │        │     └─ import_content (4×1)  │
 │   mining                      │        └──────────────────────────────┘
 │   construction        ────────┼──►  aggregate_io_to_sfc_sectors()
 │   water                       │        │
 │   services                    │        │  Standard IO aggregation:
 │   health                      │        │  A_agg[I,J] = Σ A[i,j]·x_j/X_J
 │   transport                   │
 │   security                    │    KNBS concordance:
 │                               │     AGRICULTURE ← agriculture (22.8%)
 │  IOConfig:                    │     MANUFACTURING ← mfg(7.6%) + mine(0.5%)
 │   io_matrix (9×9 ndarray)     │                   + const(7.1%) + water(0.9%)
 │   sector_shares (GDP weights) │     SERVICES ← srv(49.0%) + hlth(2.4%)
 │   shock_sensitivity           │                + trans(5.8%) + secur(3.7%)
 │                               │     INFORMAL ← not in KNBS SUT (field est.)
 │  LeontiefModel:               │
 │   Leontief inverse solver     │    Validation: Column sums < 1.0 (Hawkins-Simon)
 │   Hawkins-Simon check         │    AGR=0.42  MFG=0.46  SRV=0.49  ✓
 └───────────────────────────────┘
```

### AllParams Structure

```python
@dataclass
class AllParams:
    national_accounts: NationalAccountsParams  # KNBS/World Bank 2019-2023 averages
    production: ProductionParams               # CES: TFP, alpha, sigma, h, delta, K/Y
    io: InputOutputParams                      # 4×4 reconciled IO matrix + imports
    households: HouseholdParams                # MPC, quintiles, food shares, poverty line
    government: GovernmentParams               # Tax rates, expenditure shares, debt params
    monetary: MonetaryParams                   # Taylor Rule, spread, FX reserves
    external: ExternalParams                   # Trade elasticities, world growth
    banking: BankingParams                     # LTD, CAR, NPL, credit rationing

    @staticmethod
    def default_kenya() -> AllParams: ...
```

---

## 2. SFC Engine — Legacy Path (`scarcity/simulation/sfc.py`)

### What It Models

A Stock-Flow Consistent macroeconomic model where every financial flow has a source and destination, and every sector's balance sheet must add up.

| Sector | Assets | Liabilities |
|--------|--------|-------------|
| Households | Deposits | Loans |
| Firms | Capital | Loans, Equity |
| Banks | Loans (to HH + Firms) | Deposits |
| Government | — | Bonds |

### Key Classes

#### `SFCConfig` (~40 fields)

| Category | Parameters |
|----------|-----------|
| **Consumption** | `consumption_propensity` (0.8), `wealth_effect` (0.02) |
| **Investment** | `investment_sensitivity` (0.5), `base_investment_ratio` (0.2), `depreciation_rate` (0.05) |
| **Monetary** | `target_inflation` (0.02), `taylor_rule_phi` (1.5), `taylor_rule_psi` (0.5), `neutral_rate` (0.03) |
| **Fiscal** | `tax_rate` (0.25), `spending_ratio` (0.20), `fiscal_impulse_baseline` (0.0) |
| **Phillips Curve** | `phillips_coefficient` (0.15), `inflation_anchor_weight` (0.7) |
| **Okun's Law** | `okun_coefficient` (0.02), `nairu` (0.05) |
| **Policy Mode** | `"on"` (Taylor Rule), `"off"` (frozen), `"custom"` (user instruments) |
| **Shocks** | `shock_vectors` (list per step), `shock_schedule` (named events) |
| **Constraints** | `constraints` (dict of variable → (min, max) bounds) |

#### `SFCEconomy` — Methods

| Method | Description |
|--------|-------------|
| `initialize(gdp=100)` | Sets up consistent balance sheets for all 4 sectors |
| `step()` | Advances one period: shocks → policy → dynamics → outcomes → logging |
| `run(steps)` | Calls `step()` N times, returns `trajectory: List[Dict]` |
| `run_scenario(config, seed)` | Static convenience — creates economy, initializes, runs |

### The Step Cycle

```
1. SHOCKS       — Apply shock_vector[t]: demand, supply, fiscal, FX
2. POLICY       — Taylor Rule (auto) or custom instruments
3. DYNAMICS     — GDP · consumption · investment · govt spending · savings
                  Phillips Curve (inflation with NK anchoring)
                  Okun's Law (unemployment from output gap)
                  Financial stability (credit spread, leverage)
4. OUTCOMES     — Score 14 dimensions dynamically
5. LOGGING      — Record 4D frame + legacy history
```

### Frame Structure

```python
{
    "t": int,
    "shock_vector":    {"demand_shock": float, "supply_shock": float,
                        "fiscal_shock": float, "fx_shock": float},
    "policy_vector":   {"policy_rate": float, "fiscal_impulse": float, ...},
    "channels":        {"output_gap": float, "inflation_gap": float,
                        "credit_spread": float},
    "outcomes":        {"gdp_growth": float, "inflation": float,
                        "unemployment": float, "household_welfare": float,
                        "real_consumption": float, "savings_rate": float,
                        "household_net_worth": float, "debt_to_gdp": float,
                        "fiscal_deficit_gdp": float, "fiscal_space": float,
                        "investment_ratio": float, "capital_stock": float,
                        "financial_stability": float,   # 0=crisis, 1=stable
                        "cost_of_living_index": float},
    "flows":           {"consumption": float, "investment": float,
                        "govt_spending": float, "tax_revenue": float,
                        "savings": float, "fiscal_deficit": float,
                        "subsidy": float},
    "sector_balances": {"households": float, "firms": float,
                        "government": float, "banks": float},
}
```

### Key Economic Equations

**New Keynesian Phillips Curve** (inflation with anchoring):
```
π(t) = α·π(t-1) + (1-α)·π* + κ·output_gap
```
Where `α = inflation_anchor_weight` (0.7), `κ = phillips_coefficient` (0.15), `π* = target_inflation`.

**Taylor Rule** (monetary policy):
```
i(t) = r* + π* + φ·(π - π*) + ψ·output_gap
```

**Okun's Law** (labor market):
```
u(t) = u(t-1) + β·Δoutput_gap
```

---

## 3. Multi-Sector Engine — Typed Path (`scarcity/simulation/sfc_engine.py`)

### Architecture

```
 MultiSectorSFCEngine
 │
 ├── AllParams (KNBS-calibrated parameters)
 │    ├── NationalAccountsParams  — GDP shares, employment, growth rates
 │    ├── ProductionParams        — CES: TFP, capital share, elast. of subst.
 │    ├── InputOutputParams       — 4×4 IO matrix (KNBS-derived 3×3 + INFORMAL)
 │    ├── HouseholdParams         — MPC, quintile shares, food burden, poverty line
 │    ├── GovernmentParams        — Tax rates, expenditure composition, debt
 │    ├── MonetaryParams          — Taylor Rule + spreads + FX reserves
 │    ├── ExternalParams          — Trade elasticities, export composition
 │    └── BankingParams           — LTD, CAR, NPL, credit rationing thresholds
 │
 ├── Behavioral Blocks (one module per domain)
 │    ├── production.py      → sectoral CES output, TFP shocks, capital accumulation
 │    ├── labor_market.py    → employment by sector, unemployment, real wages
 │    ├── price_system.py    → CPI, import prices, relative prices, NK anchoring
 │    ├── households.py      → disposable income, consumption, savings, deposits/loans
 │    ├── government.py      → VAT/income/corp taxes, spending, transfers, debt/bonds
 │    ├── monetary.py        → CBK rate rule, market-rate pass-through (loan/deposit)
 │    ├── foreign.py         → exports/imports/remittances/aid + CA + KA + reserves + FX
 │    └── banking.py         → credit supply, deposits, equity, CAR, NPL, reserve closure
 │
 └── Accounting Support
      ├── coupling_interface.py  → aggregate_feedback(), macro state exposure
      └── accounting.py          → SFC identity residual checks (warns on drift)
```

### Type Contracts (`types.py`)

```python
@dataclass
class EconomyState:
    gdp: float; capital: dict[Sector, float]
    labor: dict[Sector, float]; wages: dict[Sector, float]
    prices: dict[Sector, float]; fx_rate: float
    ...

@dataclass
class PolicyState:
    interest_rate: float; fiscal_stance: float; ...

@dataclass
class ShockVector:
    tfp: dict[Sector, float]; demand: float
    fx: float; fiscal: float; ...

@dataclass
class StepResult:
    state: EconomyState; policy: PolicyState
    flows: dict; diagnostics: dict
```

### Engine Methods

| Method | Description |
|--------|-------------|
| `step(shock, feedback)` | Advance one quarter, return `StepResult` |
| `simulate(quarters, shocks, feedbacks)` | Run repeated quarterly steps |
| `find_steady_state(max_iter, tol)` | Neutral-shock convergence search |

---

## 4. Extended Simulation Modules

| Module | Purpose |
|--------|---------|
| `bayesian.py` | Bayesian VARX forecasting — probabilistic uncertainty quantification |
| `financial_accelerator.py` | Bernanke-Gertler-Gilchrist financial accelerator for credit amplification |
| `heterogeneous.py` | Heterogeneous household agent extension — distributional dynamics |
| `agents.py` | Agent-based model overlay — individual firm/household decisions |
| `open_economy.py` | Open economy extensions — Mundell-Fleming, UIP condition |
| `learned_sfc.py` | ScarcityBridge-trained SFC — discovery engine informs calibration |
| `research_sfc.py` | Research configuration variants for academic use |
| `whatif.py` | Counterfactual what-if engine — path-divergence from baseline |
| `dynamics.py` | Shared dynamics utilities — growth accounting, convergence checks |
| `scenario.py` | Scenario management — build, store, compare named scenarios |
| `scheduler.py` | Simulation scheduler — time-step ordering and parallelism |
| `monitor.py` | Runtime monitoring — convergence, stability alerts |
| `storage.py` | Run artifact persistence — effects.jsonl, summary.json |
| `visualization3d.py` | 3D visualisation helpers — surface + trajectory renderers |

---

## 5. Kenya Calibration (`kshiked/simulation/kenya_calibration.py`)

### Purpose

Bridges the generic SFC engine to Kenya-specific data. Reads the World Bank CSV (or any user-uploaded data) and derives SFC parameters with per-parameter confidence tracking.

### Key Exports

| Export | Type | Description |
|--------|------|-------------|
| `calibrate_from_data(loader, steps, policy_mode, overrides)` | Function → `CalibrationResult` | Main entry point. Returns `.config`, `.params`, `.overall_confidence` |
| `OUTCOME_DIMENSIONS` | Dict (11 entries) | Metadata for each outcome dimension — label, unit, format, higher_is, category |
| `DEFAULT_DIMENSIONS` | List (5 entries) | `["gdp_growth", "inflation", "unemployment", "household_welfare", "debt_to_gdp"]` |

### CalibrationResult

```python
@dataclass
class CalibrationResult:
    config: SFCConfig
    params: Dict[str, ParamInfo]
    overall_confidence: float   # 0.0 – 1.0

@dataclass
class ParamInfo:
    value: float
    source: str     # "data" or "default"
    confidence: float
    note: str
```

### Calibration Fallback

When the World Bank CSV lacks an indicator, the calibrator falls back to middle-income-country defaults and marks the parameter's source as `"default"` with lower confidence.

---

## 6. Scenario Templates (`kshiked/simulation/scenario_templates.py`)

### ScenarioTemplate Dataclass

```python
@dataclass
class ScenarioTemplate:
    id: str; name: str; category: str; context: str
    shocks: Dict[str, float]       # {"supply_shock": -0.15, "fx_shock": 0.05}
    shock_onset: int               # Quarter when shock hits
    shock_duration: int            # 0 = permanent step
    shock_shape: str               # "step" | "pulse" | "ramp" | "decay"
    suggested_policy: dict
    suggested_dimensions: list
```

### 9 Scenario Library

| ID | Name | Category | Key Shocks |
|----|------|----------|------------|
| `oil_crisis` | Oil Price Spike (+30%) | Supply | supply + FX |
| `drought` | Severe Drought (-20% Agri) | Supply | supply + demand |
| `food_price_surge` | Food Price Surge (+25%) | Supply | supply (ramped) |
| `kes_depreciation` | Shilling Depreciation (-15%) | External | FX |
| `global_recession` | Global Recession | External | demand + FX |
| `aid_reduction` | Foreign Aid Cut (-30%) | External | fiscal |
| `debt_crisis` | Sovereign Debt Crisis | Fiscal | fiscal + FX |
| `perfect_storm` | Perfect Storm | Combined | supply + demand + FX |
| `stimulus_boom` | Government Stimulus Boom | Fiscal | fiscal (positive) |

### 8 Policy Templates

| Key | Name | Instruments |
|-----|------|-------------|
| `do_nothing` | Do Nothing | No intervention |
| `cbk_tightening` | CBK Tightening | +2pp policy rate |
| `aggressive_tightening` | Aggressive Tightening | Major rate hike + CRR |
| `fiscal_stimulus` | Fiscal Stimulus | More spending + subsidies |
| `austerity` | Austerity / IMF Package | Spending cuts + tax hikes |
| `rate_cap_2016` | Kenya 2016 Rate Cap | Interest rate cap at 11% |
| `expansionary_mix` | Expansionary Mix | Lower rates + targeted subsidies |
| `price_controls` | Price Controls | Cap fuel + food prices |

---

## 7. Simulation UI (`kshiked/ui/kshield/simulation/`)

### Layout

```
 kshiked/ui/kshield/simulation/
 ├── view.py            Main simulation view — scenario config + run
 ├── sector_dashboard.py Sector-level output breakdown
 ├── core_analysis.py   Core analysis tabs (IRF, Monte Carlo, etc.)
 ├── advanced.py        Advanced analysis (parameter surface, stress matrix)
 ├── run.py             Run orchestration + trajectory storage
 ├── scenario_config.py Scenario + policy configuration panels
 ├── _shared.py         Shared utilities + dimension discovery
 ├── guide.py           User guide tab
 ├── research.py        Research-mode configuration
 ├── param_surface.py   Parameter sweep → 3D response surface
 ├── backtest.py        Historical backtest engine
 └── workbench/
     └── view.py        Advanced research workbench
```

### Analysis Tabs

All tabs are **fully dynamic** — dimension labels, axis options, and metrics are auto-discovered from the trajectory data at runtime.

| # | Tab | Visualisation | 3D |
|---|-----|---------------|----|
| 1 | **Scenario Runner** | Impact delta cards + time-series with shock onset marker | — |
| 2 | **Sensitivity Matrix** | Policy-outcome correlation heatmap | — |
| 3 | **3D State Cube** | User picks X/Y/Z/Color from any discovered dimension | Yes |
| 4 | **Compare Runs** | Overlay up to 5 trajectories on a single chart | — |
| 5 | **Phase Explorer** | 2D or 3D trajectory through any state space | Yes |
| 6 | **Impulse Response** | IRFs as % deviation from pre-shock baseline | Yes |
| 7 | **Flow Dynamics** | Waterfall bar + Sankey of economic flows | Yes |
| 8 | **Monte Carlo** | Parameter jitter → N simulations → fan charts | Yes |
| 9 | **Stress Matrix** | All 9 scenarios × all outcomes → heatmap | Yes |
| 10 | **Parameter Surface** | Sweep parameter → 3D response surface | Yes |
| 11 | **Sector Dashboard** | Sector-level output, employment, prices | — |
| 12 | **Diagnostics** | Calibration table, SFC balance check, channel dynamics | — |

### Dynamic Dimension Discovery

```python
_discover_dimensions(trajectory) → {
    "outcomes":        ["gdp_growth", "inflation", "unemployment", ...],
    "channels":        ["output_gap", "inflation_gap", "credit_spread"],
    "flows":           ["consumption", "investment", "govt_spending", ...],
    "sector_balances": ["households", "firms", "government", "banks"],
    "policy_vector":   ["policy_rate", "fiscal_impulse", ...],
    "shock_vector":    ["demand_shock", "supply_shock", "fx_shock", ...],
}
```

---

## 8. Quick Start

```python
# Legacy path — Kenya calibration UI flow
from scarcity.simulation.sfc import SFCEconomy, SFCConfig

cfg = SFCConfig(steps=50)
econ = SFCEconomy(cfg)
econ.initialize()
trajectory = econ.run(50)

# With Kenya calibration + scenario
from kshiked.simulation.kenya_calibration import calibrate_from_data
from kshiked.simulation.scenario_templates import get_scenario_by_id

calib = calibrate_from_data(steps=50, policy_mode="custom")
scenario = get_scenario_by_id("oil_crisis")
calib.config.shock_vectors = scenario.build_shock_vectors(50)

econ = SFCEconomy(calib.config)
econ.initialize()
trajectory = econ.run(50)

final = trajectory[-1]
print(f"GDP Growth: {final['outcomes']['gdp_growth']:.2%}")
print(f"Inflation:  {final['outcomes']['inflation']:.2%}")
print(f"Debt/GDP:   {final['outcomes']['debt_to_gdp']:.2%}")

# Typed multi-sector path — sector-level analysis
from scarcity.simulation.sfc_engine import MultiSectorSFCEngine
from scarcity.simulation.parameters import AllParams

params = AllParams.default_kenya()
engine = MultiSectorSFCEngine(params)
result = engine.simulate(quarters=40)

# IO aggregation bridge — derive 4-sector matrix from KNBS 9-sector data
from scarcity.simulation.io_structure import aggregate_io_to_sfc_sectors

agg = aggregate_io_to_sfc_sectors()
print(agg["io_block"])       # 3×3 block (AGR/MFG/SRV)
print(agg["import_content"]) # import fractions by SFC sector
```

---

## 9. Benchmark Results

The legacy SFC engine passes 18/19 benchmark tests (97.3% score):

| Test | Status |
|------|--------|
| Baseline GDP stability | PASS |
| Positive GDP growth | PASS |
| Demand shock GDP reduction | PASS |
| Supply shock inflation increase | PASS |
| Inflation bounded [-10%, 49%] | PASS |
| Unemployment bounded [2%, 30%] | PASS |
| SFC accounting identity holds | PASS |
| Phillips Curve (inflation ↔ output gap) | PASS |
| Taylor Rule (rate responds to inflation) | PASS |
| NK Phillips Curve anchoring | PASS |
| Inflation spiral prevention | PASS |
| Hypothesis promotion fallback | PASS |
| Multi-scenario comparison | PASS |
| Custom policy override | PASS |
| Shock timing (onset/shape) | PASS |
| Long-run convergence (200q) | PASS |
| Consumption > 0 always | PASS |
| Sector balance consistency | PASS |
| Okun's Law sensitivity (direction) | NEEDS REVIEW |

The typed multi-sector engine is validated by:

| Test | File |
|------|------|
| CES production accuracy | `scarcity/simulation/tests/test_production.py` |
| 200-quarter steady state | `scarcity/simulation/tests/test_steady_state.py` |
| Cross-sector coupling | `scarcity/simulation/tests/test_coupling.py` |
| SFC accounting residuals | `scarcity/simulation/tests/test_accounting.py` |
| IO/SFC plugin interface | `scarcity/simulation/tests/test_sfc_plugins.py` |

---

*Last updated: 2026-04-18*

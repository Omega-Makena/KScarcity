# Simulation — Infrastructure

---

## types.py — Core Types

### `Sector` enum

```python
class Sector(Enum):
    AGRICULTURE   = "agri"
    MANUFACTURING = "mfg"
    SERVICES      = "svc"
    INFORMAL      = "inf"
```

`SECTORS` tuple contains all four in the canonical order used for dict iteration.

### `EconomyState`

Frozen dataclass — the complete snapshot of the economy at one quarter:

| Group | Fields |
|-------|--------|
| Time | `quarter` |
| Output (real) | `Y`, `Y_pot` — sectoral output and potential |
| Capital | `K` (sectoral), `K_pub` (public) |
| Labor | `N` (sectoral), `N_s` (supply), `U` (unemployment) |
| Prices | `P` (sectoral), `P_cpi`, `P_imp`, `E_nom` |
| Government | `B_gov`, `B_bank`, `B_cb`, `B_foreign`, `G_exp`, `G_inv`, `DEFICIT` |
| Households | `C`, `S_h`, `D_h`, `L_h` |
| Banking | `L_f`, `D_f`, `BANK_EQUITY`, `BANK_CAR`, `NPL_ratio` |
| External | `EX`, `IM`, `CA`, `KA`, `RES_fx`, `remittances`, `aid` |
| Rates | `i_cb`, `i_loan`, `i_dep`, `i_gov`, `i_taylor`, `pi_cpi` |

### `SectorFeedback`

Carries external signals into the macro computation loop:

```python
@dataclass(frozen=True)
class SectorFeedback:
    labor_supply_shock: float
    labor_productivity_factor: Dict[Sector, float]
    capital_destruction: Dict[Sector, float]
    demand_shift: Dict[Sector, float]
    trade_disruption: Dict[Sector, float]
    fx_pressure: float
    fiscal_pressure: float
    yield_factor: float   # Agriculture-specific
```

### `ShockVector`

Point-in-time shocks applied each period: `supply_shock`, `demand_shock`, `rainfall_shock`, `world_demand_shock`, `remittance_shock`, `aid_shock`, `risk_premium_shock`.

---

## parameters.py — Calibrated Parameters

All parameter dataclasses are calibrated to Kenyan national accounts (KNBS/World Bank 2019–2023 averages).

### `NationalAccountsParams`

| Parameter | Default | Source |
|-----------|---------|--------|
| `gdp_share[AGRICULTURE]` | 0.218 | KNBS |
| `gdp_share[MANUFACTURING]` | 0.164 | KNBS |
| `gdp_share[SERVICES]` | 0.473 | KNBS |
| `gdp_share[INFORMAL]` | 0.145 | KNBS |
| `gdp_real_2023` | 10 980.0 | World Bank |
| `labor_force_2023` | 23.5M | KNBS |
| `unemployment_rate_2023` | 5.4% | KNBS |
| `population_growth_rate` | 2.2% | KNBS |
| `tfp_growth_trend` | 0.8% | Estimated |

Validated: `gdp_share` and `employment_share` must each sum to 1.0 (checked in `__post_init__`).

### Other parameter dataclasses

| Dataclass | Purpose |
|-----------|---------|
| `ProductionParams` | CES α (capital share) and σ (substitution elasticity) by sector |
| `HouseholdParams` | Consumption propensities `c_1` (income), `c_2` (wealth); consumption shares by sector |
| `BankingParams` | Leverage ratio, credit rationing threshold, min capital adequacy, risk weights |
| `GovernmentParams` | Spending ratios, debt management rules, bond maturity |
| `MonetaryParams` | Taylor rule coefficients (φ_π, φ_y), i_neutral, i_floor, i_ceiling, spread parameters |
| `ExternalParams` | Export/import GDP ratios, FX managed-float parameters, remittance/aid shares |
| `InputOutputParams` | Leontief IO matrix coefficients between sectors |

---

## accounting.py — Stock-Flow Consistency Checks

`run_accounting_checks(prev_state, state, flows)` — verifies the model satisfies all accounting identities each period.

**Seven residual checks**:

| Residual | Identity |
|----------|----------|
| 1 | National income identity: Y = C + I + G + NX |
| 2 | Household budget: ΔD_h = S_h + ΔL_h |
| 3 | Government budget: ΔB_gov = DEFICIT |
| 4 | Bank balance sheet: assets = liabilities + equity |
| 5 | Bond market clearing: B_gov = B_bank + B_cb + B_foreign |
| 6 | BOP identity: CA + KA − ΔRES_fx × E_nom = 0 |
| 7 | Walras law: Σ excess_demand_s = 0 |

Returns `dict[str, float]` of residuals. Non-zero residuals indicate a coding error in the sector blocks.

---

## coupling_interface.py — Sector Feedback Aggregation

### `AggregatedFeedback`

Frozen dataclass that consolidates feedback signals from all sector extension models:

```python
@dataclass(frozen=True)
class AggregatedFeedback:
    labor_supply_shock: float
    productivity_shock: Dict[Sector, float]
    capital_destruction: Dict[Sector, float]
    demand_shift: Dict[Sector, float]
    trade_disruption: Dict[Sector, float]
    fx_pressure: float
    fiscal_pressure: float
    yield_factor: float
```

`AggregatedFeedback.neutral()` — returns identity values (1.0 for multipliers, 0.0 for additive shocks).

### `aggregate_feedback(feedbacks: List[SectorFeedback])`

Combines multiple sector feedback objects:
- **Multiplicative factors** (productivity, demand_shift, trade_disruption): multiplied across sources
- **Additive factors** (fx_pressure, fiscal_pressure): summed
- **Capital destruction**: combined as `1 − Π(1 − d_i)` (survival probability)

---

## learned_sfc.py — Hypothesis-Driven Economy

### `LearnedSFCEconomy`

SFC model where economic relationships come from the discovery engine's learned hypotheses (306+ relationships) rather than hardcoded equations.

```python
from scarcity.simulation.learned_sfc import LearnedSFCEconomy, LearnedSFCConfig

economy = LearnedSFCEconomy(
    bridge=scarcity_bridge,       # Trained ScarcityBridge
    sfc_config=SFCConfig(),       # Parametric fallback
    learned_config=LearnedSFCConfig(steps=20, enable_fallback=True),
)
economy.initialize()
trajectory = economy.run(steps=20)
```

**`LearnedSFCConfig`**:

| Field | Default | Description |
|-------|---------|-------------|
| `steps` | 20 | Simulation horizon |
| `enable_fallback` | True | Mix learned + parametric |
| `fallback_weight_override` | None | Fixed blend weight (None = use per-variable confidence) |
| `initial_state_source` | "data" | "data" or "manual" |

**FallbackBlender**: blends learned predictions with parametric `SFCEconomy` output using per-variable confidence scores from the hypothesis pool. High-confidence learned relationships dominate; low-confidence falls back to parametric equations.

---

## open_economy.py — Open Economy Extension

Activates the FOREIGN sector stub from `sfc.py` with full open-economy dynamics.

**`OpenEconomyConfig`** — calibrated to Kenya's external sector (CBK, World Bank, KNBS):

| Parameter | Description |
|-----------|-------------|
| Exchange rate model | UIP–PPP hybrid with managed float |
| Trade | REER-elastic exports, income-elastic imports |
| Capital account | FDI + portfolio flows + hot money |
| Reserves | Import-cover adequacy management |
| Remittances | ~3.5% of GDP (World Bank) |

`OpenEconomyExtension.step(state, shocks)` → extends `ForeignComputation` with capital account dynamics, reserve adequacy monitoring, and exchange rate pressure from capital flows.

---

## price_system.py — Sticky Price Dynamics

### `PriceComputation`

```python
@dataclass(frozen=True)
class PriceComputation:
    P_new: Dict[Sector, float]
    P_cpi: float
    profits: Dict[Sector, float]
    unit_cost: Dict[Sector, float]
    pi_cpi: float                   # CPI inflation rate
```

### `compute_prices_and_profits()`

Cost-plus pricing with partial adjustment:

```
unit_cost_s = (wage_bill_s + int_input_cost_s + capital_cost_s + import_cost_s) / Y_gross_s
P_target_s  = unit_cost_s × (1 + markup_s)
P_new_s     = P_prev_s + price_adjustment_speed × (P_target_s − P_prev_s)
```

Sector markups: Agriculture 10%, Manufacturing 15%, Services 20%, Informal 5%.

CPI is the consumption-share weighted average of sector prices.

---

## research_sfc.py — Research Workbench

High-level wrapper for exploratory simulation work. Provides:
- Parameter sweep utilities
- Multi-scenario comparison
- Sensitivity analysis across policy dimensions
- Results serialization for dashboard consumption

---

## sfc_engine.py — SFC Computation Orchestrator

Orchestrates the full quarterly computation loop by calling each sector block in the correct order:

```
1. Labor market (wages, employment, unemployment)
2. Production (output by sector, using CES/CD functions)
3. Price system (sector prices, CPI, profits)
4. Monetary block (policy rates, FX intervention)
5. Government block (taxes, spending, deficit, bonds)
6. Household block (income, consumption, saving, deposits)
7. Banking block (loans, deposits, capital adequacy)
8. Foreign block (trade, BOP, exchange rate, reserves)
9. Accounting checks (residual verification)
```

`SFCEngine.step(state, shocks, feedback)` → `EconomyState` — advances the economy by one quarter.

---

## storage.py — Trajectory Persistence

`SimulationStorage`: serializes and loads `EconomyState` trajectories to/from JSON or Parquet for dashboard and analysis consumption.

---

## scheduler.py — Simulation Scheduler

`SimulationScheduler`: manages multi-scenario job queues and parallel execution when running grid sweeps or Monte Carlo batches.

---

## monitor.py — Runtime Monitoring

`SimulationMonitor`: tracks per-step statistics, detects numerical instability (NaN/Inf in state variables), and publishes progress to the EventBus.

---

## visualization3d.py — 3D Surface Rendering

Provides Plotly-compatible surface data structures for terrain visualization. Closely related to `analytics/terrain.py` — converts simulation trajectory matrices into `go.Surface`-ready dicts.

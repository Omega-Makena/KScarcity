# Simulation — Agents and Sector Blocks

This page documents the agent registry and the sector-level computation modules that implement the multi-sector Stock-Flow Consistent macro model.

---

## agents.py — Agent Registry

### `NodeAgent`

Represents a simulation entity derived from a discovered variable in the hypergraph store.

```python
@dataclass
class NodeAgent:
    node_id: str
    agent_type: str
    domain: int
    regime: int          # Maps to income quintile in heterogeneous extension
    embedding: np.ndarray
    stability: float
    value: float = 0.0
```

### `EdgeLink`

Represents a causal link between two agents, derived from discovered edges.

```python
@dataclass
class EdgeLink:
    edge_id: str
    source: str
    target: str
    weight: float
    stability: float
    confidence_interval: float
    regime: int
```

### `AgentRegistry`

Source of truth for the `SimulationEnvironment`. Translates hypergraph store snapshots into unified `NodeAgent` and `EdgeLink` collections.

```python
registry = AgentRegistry()
registry.load_from_snapshot(snapshot)    # Populates nodes and edges
registry.nodes()                         # → Dict[str, NodeAgent]
registry.adjacency_matrix()              # → (adjacency, stability, node_ids)
```

---

## households.py — Household Sector

### `HouseholdComputation`

Frozen dataclass holding all household-sector outputs for a single period:

| Field | Description |
|-------|-------------|
| `Y_disp` | Disposable income |
| `C` | Aggregate consumption |
| `C_by_sector` | Sectoral consumption allocation |
| `S_h` | Household saving |
| `D_h_new` | New deposit stock |
| `L_h_new` | New loan stock |
| `delta_D_h`, `delta_L_h` | Flow changes in deposits/loans |
| `GINI` | Gini coefficient |
| `POVERTY` | Poverty rate |

### Key functions

**`compute_disposable_income(w, N, dividends, rem_h, transfers_gov, tax_rate_income, tax_rate_vat, C_guess)`**

```
Y_disp = wage_income + DIV + REM_h + TRANS - TAX_income - VAT(C)
wage_income = Σ_s w[s] × N[s]
tax_vat = tax_rate_vat × C / (1 + tax_rate_vat)
```

**`compute_consumption(Y_disp, D_h_prev, L_h_prev, params)`**

```
C = c_1 × Y_disp + c_2 × W_h_prev
W_h_prev = D_h_prev − L_h_prev
```

**`allocate_consumption_by_sector(C, demand_shift, params)`**

```
C_s = consumption_shares[s] × C × demand_shift[s]
```

---

## government.py — Fiscal Block

### `GovernmentComputation`

Frozen dataclass with all fiscal outputs:

| Field | Description |
|-------|-------------|
| `T_rev` | Total tax revenue |
| `T_income`, `T_corporate`, `T_vat`, `T_trade` | Tax revenue components |
| `G_exp`, `G_inv`, `G_total` | Government expenditure and investment |
| `DEFICIT` | Fiscal deficit |
| `delta_B_gov` | Change in government bond stock |
| `B_gov_new`, `B_bank_new`, `B_cb_new`, `B_foreign_new` | Debt holdings by sector |
| `K_pub_new` | Public capital stock |

### `compute_government_block()`

Implements the full fiscal block:

```
T_income   = tax_rate_income × Σ_s w[s] × N[s]
T_corporate = tax_rate_corporate × Σ_s max(profits[s], 0)
T_vat      = tax_rate_vat × C / (1 + tax_rate_vat)
T_trade    = trade_tax_rate × Σ_s max(IM[s], 0)
T_rev      = T_income + T_corporate + T_vat + T_trade
```

Government spending is a fixed share of NGDP adjusted by fiscal_pressure. Deficit = G_total − T_rev + interest payments. Bond issuance split: 25% absorbed by central bank, rest by banks and foreign sector.

---

## banking.py — Banking Sector

### `BankingComputation`

Frozen dataclass with banking-sector outputs including loans, deposits, capital adequacy, and NPL ratio.

### `compute_banking_block()`

Key logic:

- **Credit ceiling**: `max_loans = BANK_EQUITY × max_leverage_ratio`
- **Credit multiplier**: linear ramp from 0 to 1 as CAR rises from `min_capital_adequacy` to `credit_rationing_threshold`
- **Loan allocation**: available credit distributed across sectors (capped per sector at `available_credit / 4`)
- **NPL dynamics**: NPL ratio rises with output gap and unemployment
- **Capital adequacy**: `BANK_CAR = BANK_EQUITY / RWA`; RWA computed using Basel-style risk weights

---

## production.py — Production Functions

Implements sector-level output with CES (Constant Elasticity of Substitution) production functions, falling back to Cobb-Douglas when σ → 1.

### Key functions

**`_ces_or_cd_output(A_eff, alpha, sigma, K_eff, hN)`**

```
σ → 1 (|σ-1| ≤ 0.01):  Y = A_eff × K_eff^α × hN^(1-α)   [Cobb-Douglas]
otherwise:              Y = A_eff × (α × K_eff^ρ + (1-α) × hN^ρ)^(1/ρ)
                        ρ = (σ-1)/σ
```

**`_productivity_multiplier(sector, shocks, feedback)`**

Combines supply shock, labor productivity factor, and sector-specific multipliers. Agriculture additionally incorporates `rainfall_shock` and `yield_factor`.

**`_destruction_by_sector(feedback)`** — clips capital destruction rates to [0, 1].

---

## labor_market.py — Labor Market

### `LaborMarketComputation`

```python
@dataclass(frozen=True)
class LaborMarketComputation:
    N: Dict[Sector, float]      # Employment by sector
    N_s_total: float             # Total labor supply
    U: float                     # Unemployment rate
    w: Dict[Sector, float]       # Nominal wages by sector
```

### `compute_labor_market()`

Implements Phillips curve wage dynamics and Okun-adjusted labor demand.

**Capital-labour substitution**:
```
wage_term = (1 + pi_cpi_prev)^sigma
```

When inflation is positive, real wages fall, firms demand more labour.

**Phillips curve**:
```
Δw/w = phillips_slope × (U_nairu − U) + π_CPI
```

**Okun adjustment** on employment: output growth translates to employment via `okun_elasticity`.

Labor supply grows at `labor_force_growth_rate` with `labor_supply_shock` multiplier.

---

## monetary.py — Monetary Policy Block

### `MonetaryComputation`

| Field | Description |
|-------|-------------|
| `i_cb` | Central bank policy rate |
| `i_loan`, `i_dep`, `i_gov` | Lending, deposit, and government bond rates |
| `i_taylor` | Taylor rule rate |
| `pi_cpi` | CPI inflation |
| `delta_res_fx_intervention` | FX reserve change from intervention |
| `import_cover_months` | Reserves in months of imports |

### `compute_monetary_block()`

**Taylor Rule**:
```
i_taylor = i_neutral + φ_π × (π_CPI − π_target) + φ_y × output_gap
i_cb = smoothing × i_cb_prev + (1 − smoothing) × i_taylor
i_cb = clamp(i_cb, i_floor, i_ceiling)
```

Override: if `i_target_override` is set, `i_cb` is replaced directly.

Spreads: `i_loan = i_cb + spread_loan`, `i_dep = max(0, i_cb + spread_deposit)`, `i_gov = i_cb + spread_govt + risk_premium_shock`.

FX intervention: reduces reserves when import cover falls below target; adds to reserves when above target.

---

## foreign.py — External Sector

### `ForeignComputation`

Covers exports, imports (final + intermediate), remittances, aid, current account (CA), capital account (KA), balance of payments (BOP), FX reserve change, and exchange rate dynamics.

### `compute_foreign_block()`

- Exports: base share of nominal GDP, adjusted by real exchange rate and world demand
- Imports: income-elastic with IO intermediate inputs; adjusted by trade disruption shocks
- CA = EX − IM + remittances + aid
- BOP = CA + KA − ΔRES_fx
- Exchange rate: managed float with UIP + PPP hybrid; FX intervention from `delta_RES_fx_intervention`

---

## financial_accelerator.py — BGG Financial Accelerator

Implements Bernanke-Gertler-Gilchrist dynamics within the SFC framework.

**`FinancialAcceleratorConfig`** — calibrated to Kenya's banking sector (CBK Financial Stability Reports):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_npl_rate` | 0.12 | Baseline NPL ratio (~Kenya average) |
| `npl_gdp_sensitivity` | 0.8 | NPL rise per unit GDP contraction |

**Mechanisms implemented**:
1. NPL dynamics with endogenous default rates
2. Collateral / LTV constraints on credit
3. Capital adequacy (CAR) with Basel-style risk weights
4. Credit cycle amplification
5. Endogenous credit spread driven by bank health and NPL ratio

---

## heterogeneous.py — Heterogeneous Agents

Disaggregates the household sector into income quintiles and labor market segments.

### `IncomeQuintile` enum

`Q1` (bottom 20%) through `Q5` (top 20%) — maps to `NodeAgent.regime` field.

### `LaborType` enum

`FORMAL` / `INFORMAL` — Kenyan informal sector modeled separately.

**Mechanisms**:
1. Income quintile distribution with differentiated consumption propensities (lower quintiles consume more of income)
2. Formal/Informal segmentation with separate wage dynamics
3. Gini coefficient tracking
4. Inequality dynamics under policy shocks

---

## io_structure.py — Input-Output Structure

Implements Leontief Input-Output matrix for inter-sector flows.

### `SubSectorType` enum

Production sub-sectors: AGRICULTURE, MANUFACTURING, SERVICES, MINING, CONSTRUCTION.  
Crisis/public sectors: HEALTH, WATER, TRANSPORT, SECURITY.

**Mechanisms**:
1. Leontief IO matrix for intermediate demand flows between sectors
2. Value-added chain tracking
3. Sector-specific TFP and productivity shocks
4. Structural change analysis via IO coefficient evolution

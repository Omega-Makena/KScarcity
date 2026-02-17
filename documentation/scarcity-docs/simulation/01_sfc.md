# sfc.py — Stock-Flow Consistent Economy

The `SFCEconomy` class implements a **Stock-Flow Consistent** economic model with proper balance sheets, accounting identities, and behavioral equations.

---

## What is SFC?

Stock-Flow Consistent modeling is an approach from post-Keynesian economics that:
- Tracks every stock (balance sheet item) and flow (transaction)
- Ensures accounting identities always hold
- Models all sectors of the economy consistently

**Key principle**: Every financial flow has a source and destination. Money doesn't appear or disappear.

---

## Sectors

### `SectorType` (Enum)

```python
class SectorType(Enum):
    HOUSEHOLDS = "households"
    FIRMS = "firms"
    BANKS = "banks"
    GOVERNMENT = "government"
    FOREIGN = "foreign"
```

### `Sector` (Dataclass)

Each sector has:

```python
@dataclass
class Sector:
    name: str
    sector_type: SectorType
    assets: Dict[str, float]        # What it owns
    liabilities: Dict[str, float]   # What it owes
    income: float = 0.0
    expenses: float = 0.0
```

**Properties**:
- `total_assets()`: Sum of all asset values
- `total_liabilities()`: Sum of all liabilities
- `net_worth()`: Assets - Liabilities
- `net_lending()`: Income - Expenses
- `balance_sheet_identity()`: Check A = L + NW

---

## Configuration

### `SFCConfig`

```python
@dataclass
class SFCConfig:
    # Behavioral parameters
    consumption_propensity_income: float = 0.8  # c1
    consumption_propensity_wealth: float = 0.2  # c2
    investment_accelerator: float = 0.1         # i1
    investment_sensitivity: float = -0.5        # i2
    money_demand_income: float = 0.5            # λ1
    money_demand_rate: float = -0.1             # λ2
    inflation_persistence: float = 0.9
    phillips_slope: float = 0.5                 # β
    
    # Structural parameters
    gov_spending_ratio: float = 0.2
    tax_rate: float = 0.25
    depreciation_rate: float = 0.05
    
    # Simulation parameters
    dt: float = 1.0
    interest_rate_min: float = 0.0
    interest_rate_max: float = 0.20
```

---

## Class: `SFCEconomy`

### Initialization

```python
economy = SFCEconomy(config=SFCConfig())
```

Creates empty sectors. Call `initialize()` to set initial conditions.

### `initialize(gdp=100.0)`

Sets up consistent starting positions:

```python
economy.initialize(gdp=100.0)
```

**Initial state**:
- GDP = 100
- Unemployment = 5%
- Inflation = 2%
- Interest rate = 3%
- Debt/GDP = 60%

Balance sheets are initialized to satisfy accounting identities.

---

## Simulation Step

### `step()`

Advance economy by one time period:

```python
economy.step()
```

**Step sequence**:

1. **Compute disposable income**
   ```
   Y_d = Y * (1 - tax_rate)
   ```

2. **Household decisions**
   ```
   C = c1 * Y_d + c2 * Wealth
   ```

3. **Firm decisions**
   ```
   I = i1 * ΔY + i2 * r
   ```

4. **Government spending**
   ```
   G = g_ratio * Y
   ```

5. **Aggregate demand**
   ```
   Y_new = C + I + G
   ```

6. **Labor market**
   ```
   u = max(0, u * (1 - gdp_adjustment * ΔY))
   ```

7. **Inflation dynamics (Phillips curve)**
   ```
   π = π_persistence * π + β * (u* - u)
   ```

8. **Monetary policy (Taylor rule)**
   ```
   r = r* + φ_π * (π - π*) + φ_u * (u - u*)
   ```

9. **Update balance sheets**
   - Households: Deposits increase by savings
   - Firms: Capital depreciates, loans adjust
   - Banks: Deposits match loans
   - Government: Debt increases by deficit

10. **Check accounting identities**
    ```
    Σ net_lending = 0 (within tolerance)
    ```

### `run(steps)`

Run multiple steps:

```python
economy.run(100)
```

---

## State Access

### `get_state() -> Dict`

Current macroeconomic variables:

```python
state = economy.get_state()
# Returns:
{
    "gdp": 105.3,
    "inflation": 0.023,
    "unemployment": 0.048,
    "interest_rate": 0.035,
    "government_debt_gdp": 0.62,
    "consumption": 85.2,
    "investment": 12.1,
    "government_spending": 21.0
}
```

### History

All states are recorded:

```python
# Access full history
for state in economy.history:
    print(state["gdp"])
```

---

## Shocks

### `apply_shock(shock_type, magnitude)`

Apply exogenous disturbance:

```python
economy.apply_shock("demand", 0.1)  # +10% demand shock
```

**Shock types**:

| Type | Effect |
|------|--------|
| `demand` | Temporarily boost aggregate demand |
| `supply` | Productivity/capacity change |
| `monetary` | Interest rate adjustment |
| `fiscal` | Government spending change |

Shocks are one-time disturbances that propagate through dynamics.

---

## Accounting Validation

### `accounting_identity() -> bool`

Checks that net lending sums to zero:

```python
if not economy.accounting_identity():
    logger.warning("Accounting identity violation!")
```

The tolerance is 1e-6 of GDP to account for floating-point errors.

### Balance Sheet Identity

For each sector:
```
Assets = Liabilities + Net Worth
```

Checked after each step.

---

## Calibration

To calibrate for a specific economy:

```python
config = SFCConfig(
    consumption_propensity_income=0.75,  # Kenya: higher propensity
    tax_rate=0.20,                        # Lower tax base
    gov_spending_ratio=0.15,              # Smaller government
    phillips_slope=0.7                    # Steeper Phillips curve
)

economy = SFCEconomy(config)
economy.initialize(gdp=10000)  # Start at Kenya's GDP
```

---

## Validation Function

### `validate_sfc_economy()`

Tests that the model maintains consistency:

```python
from scarcity.simulation.sfc import validate_sfc_economy

is_valid = validate_sfc_economy()
# Returns True if all identities hold over 100 steps
```

---

## Edge Cases

### Negative Values

GDP and employment are floored at reasonable minimums:
- GDP: 1e-6 (prevent division by zero)
- Unemployment: 0.001 (near-full employment limit)
- Interest rates: 0 to 20% (ZLB to hyperinflation)

### Extreme Shocks

Very large shocks (>50%) may break dynamics:
- Clip to reasonable ranges
- Log warnings
- System recovers over time

### Long Simulations

After many steps:
- Check for drift in accounting identities
- Monitor for explosive dynamics
- Use shorter dt if instability occurs

---

## Example: Full Simulation

```python
from scarcity.simulation.sfc import SFCEconomy, SFCConfig
import matplotlib.pyplot as plt

# Setup
config = SFCConfig()
economy = SFCEconomy(config)
economy.initialize(gdp=100.0)

# Baseline simulation
economy.run(50)

# Apply shock
economy.apply_shock("monetary", 0.02)  # Raise rates 2%

# Continue simulation
economy.run(50)

# Plot results
gdp = [s["gdp"] for s in economy.history]
inflation = [s["inflation"] for s in economy.history]

plt.subplot(2, 1, 1)
plt.plot(gdp)
plt.title("GDP")

plt.subplot(2, 1, 2)
plt.plot(inflation)
plt.title("Inflation")

plt.show()
```

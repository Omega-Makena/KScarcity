# KShield Core Module — Overview

The **core module** provides the governance and shock management system for KShield economic simulations.

---

## Purpose

KShield needs to:
- **Apply economic policies**: Monetary and fiscal rules with PID control
- **Handle shocks**: Stochastic processes for stress testing
- **Integrate with SFC**: Real economic dynamics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EconomicGovernor                          │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ SimSensor│───►│ PolicyEng│───►│ Actuator │───►│ EventBus │ │
│   │ (read)   │    │ (PID)    │    │(commands)│    │(publish) │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        ▼                                               ▼        │
│   ┌──────────┐                                   ┌──────────┐  │
│   │ SFCEcon  │◄─────────────────────────────────│ ShockMgr │  │
│   │ (model)  │                                   │(stoch)   │  │
│   └──────────┘                                   └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## EconomicGovernor (`governance.py`)

### Overview

Orchestrates economic policy and control:

```python
from kshiked.core import EconomicGovernor, EconomicGovernorConfig

config = EconomicGovernorConfig(
    control_interval=1,
    policies=default_economic_policies()
)
governor = EconomicGovernor(config, env=simulation_environment)

# Each tick
await governor.step(current_tick)
```

### Control Loop

```python
async def step(self, current_tick: int):
    # 1. Run SFC economic dynamics
    self.sfc.step()
    
    # 2. Sync state to environment
    self.actuator.sync_to_environment()
    
    # 3. Check governance interval
    if current_tick % self.config.control_interval != 0:
        return
    
    # 4. Read current state
    vals, names = self.sensors.get_vector()
    
    # 5. Evaluate policies (PID)
    action_signals = self.engine.evaluate(vals, dt=control_interval)
    
    # 6. Execute control actions
    await self.actuator.execute_signals(action_signals, metrics)
```

### SimSensor

Reads current economic state:

```python
sensor = SimSensor(env)
values, names = sensor.get_vector()
# Returns numpy array and list of metric names
```

### EventActuator

Executes governance decisions via SFC:

```python
await actuator.execute_signals({
    "tighten_policy": 0.5,
    "stimulus_package": 0.0
}, metrics)
# Publishes monetary_policy_update, fiscal_policy_update to EventBus
```

---

## Shock System (`shocks.py`)

### ShockType

```python
class ShockType(Enum):
    IMPULSE = "impulse"      # One-time hit with decay
    OU_PROCESS = "ou_process" # Mean-reverting volatility
    BROWNIAN = "brownian"     # Random walk
```

### ImpulseShock

Classic one-time hit:

```python
shock = ImpulseShock(
    name="oil_crisis",
    target_metric="GDP (current US$)",
    magnitude=-5.0,  # -5% immediate hit
    decay=0.9        # 10% decay per step
)
```

### OUProcessShock

Ornstein-Uhlenbeck mean-reverting process:

```python
shock = OUProcessShock(
    name="forex_volatility",
    target_metric="Exchange Rate",
    theta=0.15,  # Mean reversion speed
    mu=0.0,      # Long-run mean
    sigma=0.2    # Volatility
)
```

Uses Euler-Maruyama discretization: `dx = θ(μ - x)dt + σdW`

### BrownianShock

Geometric Brownian Motion:

```python
shock = BrownianShock(
    name="asset_prices",
    target_metric="Stock Index",
    drift=0.01,
    volatility=0.1
)
```

### ShockManager

Coordinates multiple shocks:

```python
manager = ShockManager()
manager.add_shock(ImpulseShock(...))
manager.add_shock(OUProcessShock(...))

# Each step
manager.apply_shocks(env_state)
```

---

## Economic Policies (`policies.py`)

### EconomicPolicy

Enhanced policy with PID control:

```python
@dataclass
class EconomicPolicy(PolicyRule):
    authority: str = "Central Bank"
    cooldown: int = 5
    temporal_lag: int = 0
    uncertainty_tolerance: float = 0.2
    
    # PID Control
    kp: float = 0.0  # Proportional
    ki: float = 0.0  # Integral
    kd: float = 0.0  # Derivative
    
    # Crisis Mode
    crisis_threshold: float = 999.0
    crisis_multiplier: float = 5.0
```

### Default Policies

```python
policies = default_economic_policies()
# Returns:
# monetary: [inflation_control, gdp_stimulus]
# fiscal: [debt_management]
```

**Monetary Policy** (Inflation Target):
- Metric: Inflation (annual %)
- Threshold: 5.0%
- Action: tighten_policy
- PID: Kp=0.1, Ki=0.01, Kd=0.05
- Crisis: 15% (hyperinflation)

**Fiscal Policy** (Debt Management):
- Metric: External debt (% of GDP)
- Threshold: 70%
- Action: austerity
- PID: Kp=0.05, Ki=0.001, Kd=0.0

---

## File Guide

| File | Purpose |
|------|---------|
| `governance.py` | EconomicGovernor, EventActuator, SimSensor |
| `shocks.py` | ShockManager, ImpulseShock, OUProcessShock, BrownianShock |
| `policies.py` | EconomicPolicy, default_economic_policies |
| `tensor_policies.py` | PolicyTensorEngine (vectorized evaluation) |

---

## Integration

### With Scarcity

Uses `scarcity.simulation.sfc.SFCEconomy` for real dynamics:

```python
from scarcity.simulation.sfc import SFCEconomy

sfc = SFCEconomy()
sfc.initialize(gdp=100.0)
sfc.apply_shock("monetary", 0.01)  # Raise rates
```

### With EventBus

Publishes policy updates:

```python
await bus.publish("monetary_policy_update", {
    "instrument": "interest_rate",
    "delta": 0.01,
    "new_rate": 0.05
})
```

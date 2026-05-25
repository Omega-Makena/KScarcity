# Simulation — Scenarios and Dynamics

---

## scenario.py — Scenario Management

### `ShockShape` enum

```
STEP      — jump and stay at magnitude
PULSE     — jump then return to zero immediately
RAMP      — linear increase over duration
DECAY     — jump then exponential decay at decay_rate
CYCLICAL  — sine wave at frequency
```

### `ShockProcess`

Defines a shock as a time-series process rather than a point value.

```python
@dataclass
class ShockProcess:
    target: str          # e.g., "demand_shock", "supply_shock"
    magnitude: float
    start_time: int      # Step number when shock begins
    duration: int
    shape: ShockShape = ShockShape.STEP
    decay_rate: float = 0.5
    frequency: float = 0.1
    # Stochastic parameters
    distribution: Optional[str] = None   # "normal" or "uniform"
    std_dev: float = 0.0
```

**`generate_vector(total_steps)`** → `np.ndarray` — generates the full time-series for the shock including stochastic sampling when `distribution` is set.

### Scenario structure

Scenarios are UUID-identified, persistable objects stored under `scenarios/` directory.

```python
from scarcity.simulation.scenario import ShockProcess, ShockShape

shock = ShockProcess(
    target="demand_shock",
    magnitude=-0.15,     # -15% demand contraction
    start_time=4,
    duration=8,
    shape=ShockShape.DECAY,
    decay_rate=0.3,
)
vector = shock.generate_vector(total_steps=20)
```

**Policies as instruments**: alongside shocks, scenarios define policy instruments with rules, constraints, and implementation lags.

---

## dynamics.py — Causal Propagation

### `DynamicsConfig`

```python
@dataclass
class DynamicsConfig:
    global_damping: float = 0.9    # Dampens all propagation weights
    delta_t: float = 1.0           # Integration time step
    stability_floor: float = 0.05  # Prevents zero-weight edges
```

### `DynamicsEngine`

Propagates values through the causal graph derived from `SimulationEnvironment`.

```python
engine = DynamicsEngine(environment, DynamicsConfig())
state_delta = engine.step()  # Returns {node_id: new_value}
```

**Propagation rule** (one step):

```
weights   = adjacency × stability × global_damping
incoming  = weights.T @ old_values
outgoing  = weights.sum(axis=0) × old_values
delta     = Δt × (incoming − outgoing)
new_values = old_values + delta
```

After propagation: Gaussian noise is applied (`apply_noise`), then an energy cap is enforced (`enforce_energy_cap`) to prevent unbounded growth.

---

## environment.py — Simulation State Container

### `EnvironmentConfig`

```python
@dataclass
class EnvironmentConfig:
    damping: float = 0.9
    noise_sigma: float = 0.01
    energy_cap: float = 5.0
    seed: int = 42
```

### `EnvironmentState`

Holds the complete state of the simulation at one timestep:

```python
@dataclass
class EnvironmentState:
    values: np.ndarray       # Current node values
    node_ids: List[str]      # Ordered node identifiers
    adjacency: np.ndarray    # Weighted causal adjacency matrix
    stability: np.ndarray    # Per-edge stability weights
    timestamp: int = 0
```

### `SimulationEnvironment`

```python
env = SimulationEnvironment(registry=agent_registry, config=EnvironmentConfig())

state = env.state()                        # Current EnvironmentState
env.update_values(new_values)             # Write new node values
env.apply_noise(values)                   # Add Gaussian noise
env.enforce_energy_cap(old, new)          # Clip magnitudes
```

State is built from `AgentRegistry.adjacency_matrix()` on initialization. Empty registries are handled gracefully (empty arrays).

---

## whatif.py — Counterfactual Scenarios

### `WhatIfConfig`

```python
@dataclass
class WhatIfConfig:
    horizon_steps: int = 12      # Forward-simulation steps
    bootstrap_runs: int = 8      # Monte Carlo runs for CIs
    noise_sigma: float = 0.02    # Noise magnitude in bootstrap
```

### `WhatIfManager`

Forks the simulation state to test "what if" hypotheses. Computes delta trajectories against a no-intervention baseline.

```python
manager = WhatIfManager(
    environment=env,
    dynamics_config=DynamicsConfig(),
    config=WhatIfConfig(horizon_steps=12, bootstrap_runs=8),
)

result = manager.run_scenario(
    scenario_id="inflation_shock",
    interventions={"inflation": 0.08},   # Override specific node values
)
# result: baseline trajectory, scenario trajectory, delta, confidence intervals
```

**Workflow**:
1. Deep-copy current environment state (fork)
2. Apply interventions to forked state
3. Run `horizon_steps` of `DynamicsEngine` on forked state
4. Run same steps on unperturbed baseline
5. Repeat `bootstrap_runs` times with noise injected → compute CI bands
6. Return `{baseline, scenario, delta, ci_lower, ci_upper}`

---

## bayesian.py — Bayesian Parameter Estimation

Implements Bayesian inference for SFCConfig parameters without PyMC/Stan — pure NumPy for portability.

### `ParameterPrior`

```python
@dataclass
class ParameterPrior:
    name: str
    distribution: str = "normal"   # "normal", "uniform", "lognormal", "beta"
    mean: float = 0.0
    std: float = 1.0
    lower: float = -np.inf
    upper: float = np.inf
    alpha: float = 2.0             # Beta distribution shape
    beta_param: float = 2.0
```

`log_prior(value)` — evaluates log-prior density.

Priors are derived from Kenya calibration ranges and meta-learning global_prior concept.

### Metropolis-Hastings MCMC

Samples the joint posterior of `SFCConfig` parameters:

```python
from scarcity.simulation.bayesian import BayesianSFCEstimator

estimator = BayesianSFCEstimator(base_config=SFCConfig(), priors=prior_list)
posterior_samples = estimator.run_mcmc(data=observed_trajectories, n_samples=1000)
ci = estimator.credible_interval(posterior_samples, level=0.9)
```

### Online Bayesian Updating

Extended Kalman filter (`KalmanFilter1D` from `engine/algorithms_online.py`) applied to parameter tracking — parameters update as new data arrives, avoiding full MCMC each period.

### Model Comparison

Bayesian Information Criterion (BIC) for comparing SFCConfig variants:

```
BIC = k × ln(n) − 2 × ln(L̂)
```

where k = number of parameters, n = data points, L̂ = maximised likelihood.

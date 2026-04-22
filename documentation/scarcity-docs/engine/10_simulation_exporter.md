# Engine — Policy Simulation, Export, and Economic Engine

---

## simulation.py — PolicySimulator

Sandboxed simulation environment for "what-if" analysis. Propagates shocks through the discovered knowledge graph (hypothesis pool) without modifying the live engine state.

### `PolicySimulator`

```python
from scarcity.engine.simulation import PolicySimulator

simulator = PolicySimulator(engine_hypotheses=engine.hypotheses)
```

**Hypothesis selection**:
1. Uses all `ACTIVE` hypotheses from the pool (deep-copied)
2. Fallback: if no ACTIVE hypotheses exist (small-data scenario), uses `TENTATIVE` hypotheses with `confidence ≥ 0.35`

**State machine**:

```python
simulator.set_initial_state({"gdp": 100.0, "inflation": 0.03})
simulator.perturb("oil_price", 120.0)    # Inject a shock
simulator.set_policy("tax_rate", 0.25)   # Persistent policy override
simulator.run(time_horizon=20)           # Simulate N steps
history = simulator.history              # List of state dicts per step
```

**Propagation** — each step:
1. Apply persistent `policies` overrides to state
2. For each active hypothesis H with source variable in state: `predicted = H.predict(state[source])`
3. Update target: `state[target] = momentum × state[target] + (1 − momentum) × predicted`
4. Record state to history

`momentum = 0.5` — inertia factor (0 = instant adoption, 1 = no change).

**Integration with `EconomicDiscoveryEngine`**:

```python
sim = engine.get_simulation_handle()
sim.set_initial_state(initial_state)
sim.set_policy("interest_rate", 0.12)
sim.run(20)
df = pd.DataFrame(sim.history)
```

---

## exporter.py — Insight Emitter

`Exporter` acts as the outbound gateway for the inference engine — broadcasts discovered relationships to the EventBus and accumulates batched path packs.

```python
exporter = Exporter()
exporter.emit_insights(
    accepted_edges=edge_list,
    resource_profile=resource_profile,
)
```

**`emit_insights(accepted_edges, resource_profile)`**:

- Emits an immediate insight payload every window: `{"edges": [...], "count": N, "timestamp": t}`
- Accumulates edges for batched `PathPack` emission at interval `resource_profile["export_interval"]` (default: every 10 windows)
- `export_count` tracks total emissions; `last_pack_time` controls pack scheduling

Path packs are emitted to the EventBus on topic `inference.path_pack` for downstream consumers (dashboard, FMI pipeline).

---

## resource_profile.py — Resource Profile

`DEFAULT_RESOURCE_PROFILE` — the canonical set of engine tuning parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_paths` | 200 | Active hypothesis paths to track |
| `precision` | "fp16" | Numeric precision for weights |
| `sketch_dim` | 512 | Sketch operator dimension |
| `window_size` | 256 | Rolling window size |
| `resamples` | 8 | Bootstrap resamples for CIs |
| `export_interval` | 10 | Windows between path pack exports |
| `branch_width` | 1 | Hypothesis branching factor |
| `tier2_enabled` | True | Enable Tier-2 aggregation layer |
| `tier3_topk` | 5 | Top-K paths explored at Tier 3 |

**`clone_default_profile()`** — returns a deep copy of `DEFAULT_RESOURCE_PROFILE`. Always use this when creating per-domain or per-engine instances to avoid aliasing the global defaults.

The DRG and MetaIntegrativeLayer both read and write resource profiles through the EventBus — changes propagate from the governor to the engine without direct coupling.

---

## economic_engine.py — Economic Discovery Engine

Specialized wrapper around `OnlineDiscoveryEngine` for macroeconomic datasets. Handles the mapping between user-friendly variable names and World Bank indicator codes.

```python
from scarcity.engine.economic_engine import EconomicDiscoveryEngine

engine = EconomicDiscoveryEngine()
engine.process_row({"GDP_Growth": 5.1, "Inflation": 4.2, ...})
```

### `EconomicDiscoveryEngine`

Wraps `OnlineDiscoveryEngine` (engine_v2.py) and extends it with:

1. **Variable whitelist**: only tracks variables in `economic_config.ECONOMIC_VARIABLES` (World Bank codes)
2. **Name mapping**: `CODE_TO_NAME` maps World Bank codes → friendly names (e.g., `"NY.GDP.MKTP.KD.ZG"` → `"GDP_Growth"`)
3. **Pre-populated hypothesis pool**: 306+ hypotheses created upfront

**Hypothesis pre-population** (`_populate_initial_hypotheses`):

For N=18 economic variables:
- `18 × 17 = 306` pairwise `VectorizedFunctionalHypothesis` (A→B and B→A for each pair)
- 18 autoregressive `TemporalLagHypothesis` (A(t−1) → A(t)) for momentum/inertia

Total: 324 hypotheses initialized before any data is seen.

**Simulation handle**: `engine.get_simulation_handle()` returns a `PolicySimulator` bound to the current hypothesis pool — used by `TerrainGenerator` to run policy sweeps.

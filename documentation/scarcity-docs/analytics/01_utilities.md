# Analytics Module — Utilities

---

## terrain.py — Policy Terrain Generator

### Purpose

Generates response surfaces ("terrains") by sweeping over pairs of policy parameters and running a grid of economic simulations. The terrain metaphor maps directly to the economic interpretation:

| Terrain Concept | Economic Meaning |
|-----------------|-----------------|
| Surface height (Z) | System response — performance or welfare |
| Axes (X, Y) | Policy stance — fiscal and monetary position |
| Walking the surface | Time / simulation evolution |

### `TerrainGenerator`

```python
from scarcity.analytics.terrain import TerrainGenerator

generator = TerrainGenerator(engine=economic_engine)
```

Requires an initialized `EconomicDiscoveryEngine` instance with `get_simulation_handle()`.

#### `generate_surface()`

```python
result = generator.generate_surface(
    initial_state={"gdp": 100.0, "inflation": 0.03},
    x_policy="fiscal_stance",
    y_policy="monetary_rate",
    z_response="gdp",
    x_range=(-0.5, 0.5),
    y_range=(0.02, 0.15),
    steps=10,
    time_horizon=20,
    max_points=400,   # cap to avoid combinatorial blow-up
)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_state` | dict | Starting values for all simulation variables |
| `x_policy` | str | Policy variable swept on the X axis |
| `y_policy` | str | Policy variable swept on the Y axis |
| `z_response` | str | Response variable measured as surface height |
| `x_range` | (float, float) | Min/max for X axis |
| `y_range` | (float, float) | Min/max for Y axis |
| `steps` | int | Grid resolution — runs `steps × steps` simulations |
| `time_horizon` | int | Years each simulation runs |
| `max_points` | int | Cap on total simulations; auto-reduces `steps` if exceeded |

**Return value**

```python
{
    "x": np.ndarray,          # 1D array of X coordinate values (length = steps)
    "y": np.ndarray,          # 1D array of Y coordinate values (length = steps)
    "z": np.ndarray,          # 2D (steps × steps) surface height matrix
    "overlays": {
        "stability": np.ndarray,  # Std-dev of z_response over time horizon
        "risk": np.ndarray,       # Max system_stress from sim.meta_history
    }
}
```

**Surface computation**

- `z`: mean of `z_response` over the time horizon (sustained performance, not final value)
- `stability`: standard deviation of `z_response` — lower = smoother trajectory
- `risk`: maximum `system_stress` metric recorded in `sim.meta_history`; points with risk > 0.8 are set to `NaN` (rendered as holes in 3D plots)

**Grid budget control**

If `steps × steps > max_points`, steps is reduced to `floor(sqrt(max_points))` before the grid is built. A warning is logged.

### Integration with Dashboard

The returned dict feeds directly into the Plotly surface renderer in `dashboard/server.py`. `x` and `y` are 1D coordinate arrays; `z` is a 2D matrix — matching Plotly's `go.Surface` expected layout.

```python
import plotly.graph_objects as go

result = generator.generate_surface(...)
fig = go.Figure(go.Surface(x=result["x"], y=result["y"], z=result["z"]))
fig.show()
```

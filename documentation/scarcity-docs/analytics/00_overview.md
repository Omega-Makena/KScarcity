# Scarcity Analytics Module — Overview

The **analytics module** provides advanced analysis and visualization capabilities for the discovered economic relationships.

---

## Purpose

Once relationships are discovered and simulated, users want to:
- **Visualize policy landscapes**: How does GDP respond to fiscal/monetary policy?
- **Find optimal policies**: What policy mix maximizes welfare?
- **Identify risk zones**: Where do policies lead to instability?

---

## TerrainGenerator (`terrain.py`)

### Overview

Generates response surfaces by sweeping over policy parameters:

```python
from scarcity.analytics import TerrainGenerator

terrain = TerrainGenerator(engine=discovery_engine)

result = terrain.generate_surface(
    initial_state={"gdp": 100, "inflation": 0.02},
    x_policy="fiscal_stance",
    y_policy="monetary_stance",
    z_response="gdp",
    x_range=(-0.1, 0.1),
    y_range=(-0.05, 0.05),
    steps=10,
    time_horizon=20
)
```

### Terrain Concept

The terrain represents:
- **Surface height (Z)**: System response (GDP, welfare, employment)
- **X/Y axes**: Policy positions (fiscal stance, interest rate)
- **Walking**: How the system evolves over time under a policy

### Output Structure

```python
{
    "x": np.array([...]),    # X-axis values (fiscal)
    "y": np.array([...]),    # Y-axis values (monetary)
    "z": np.array([[...]]),  # 2D response surface
    "overlays": {
        "stability": np.array([[...]]),  # Volatility at each point
        "risk": np.array([[...]])        # Crash risk at each point
    }
}
```

### Overlays

#### Stability

Lower volatility = higher stability:
```python
stability = hist[z_response].std()
```

#### Risk

Maximum system stress experienced:
```python
risk = max([m['system_stress'] for m in sim.meta_history])
```

Unreachable zones (risk > 0.8) are marked with NaN.

### Usage with Visualization

```python
import plotly.graph_objects as go

terrain_data = terrain.generate_surface(...)

fig = go.Figure(data=[
    go.Surface(
        x=terrain_data["x"],
        y=terrain_data["y"],
        z=terrain_data["z"],
        colorscale="Viridis"
    )
])
fig.show()
```

---

## Performance Notes

- Grid complexity: O(steps²)
- Each point runs a full simulation
- Use `max_points` to cap computation:

```python
result = terrain.generate_surface(..., steps=20, max_points=400)
# Automatically reduces steps if 20² > 400
```

---

## Index

| File | Purpose |
|------|---------|
| `terrain.py` | TerrainGenerator for policy response surfaces |

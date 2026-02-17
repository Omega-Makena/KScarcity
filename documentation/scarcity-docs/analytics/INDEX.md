# Analytics Module — Documentation Index

Complete documentation for the `scarcity.analytics` module — policy terrain analysis.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — TerrainGenerator |

---

## Key Concepts

### Policy Terrain

- Z-height = system response (GDP, welfare)
- X/Y axes = policy positions
- Overlays = stability and risk

---

## Quick Start

```python
from scarcity.analytics import TerrainGenerator

terrain = TerrainGenerator(engine=my_engine)
result = terrain.generate_surface(
    initial_state={...},
    x_policy="fiscal",
    y_policy="monetary",
    z_response="gdp",
    x_range=(-0.1, 0.1),
    y_range=(-0.05, 0.05)
)
```

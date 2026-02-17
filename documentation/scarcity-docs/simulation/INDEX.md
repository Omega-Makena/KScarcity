# Simulation Module — Documentation Index

Complete documentation for the `scarcity.simulation` module — economic simulation with SFC dynamics.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and concepts |
| [01_sfc.md](./01_sfc.md) | SFCEconomy — Stock-Flow Consistent model |
| [02_utilities.md](./02_utilities.md) | Engine, agents, what-if, visualization |

---

## Key Concepts

### Stock-Flow Consistency

- Every flow has source and destination
- Balance sheets must balance
- Sectoral balances sum to zero

### Sectors

Five economic sectors:
- Households, Firms, Banks, Government, Foreign

### Integration

Discovered relationships become simulation dynamics:
- Engine discovers "A → B"
- Simulation uses edge weight for propagation

---

## Quick Start

```python
from scarcity.simulation.sfc import SFCEconomy

economy = SFCEconomy()
economy.initialize(gdp=100.0)
economy.run(100)

print(economy.get_state())
```

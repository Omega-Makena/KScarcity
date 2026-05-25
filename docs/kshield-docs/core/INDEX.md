# Core Module — Documentation Index

Complete documentation for the `kshiked.core` module — economic governance and shocks.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Governance, policies, shocks |

---

## Key Concepts

### Economic Governor

- PID control for policy rules
- SFC integration for real dynamics
- EventBus for observation

### Shock Types

- **Impulse**: One-time hit with decay
- **OU Process**: Mean-reverting volatility
- **Brownian**: Random walk

---

## Quick Start

```python
from kshiked.core import EconomicGovernor, EconomicGovernorConfig

governor = EconomicGovernor(EconomicGovernorConfig(), env)
await governor.step(tick)
```

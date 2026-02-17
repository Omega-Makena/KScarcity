# Backend Core — Documentation Index

Complete documentation for the `backend/app/core` module — application core logic.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — ScarcityCoreManager, domains |

---

## Key Concepts

### ScarcityCoreManager

- Lifecycle: initialize → start → stop
- Manages: Bus, MPIE, DRG, Federation, Simulation
- Telemetry collection

### Domain Management

- Multi-domain simulation
- Distribution types: normal, skewed, bimodal
- Persistence to disk

---

## Quick Start

```python
from app.core.scarcity_manager import ScarcityCoreManager

manager = ScarcityCoreManager()
await manager.initialize()
await manager.start()
```

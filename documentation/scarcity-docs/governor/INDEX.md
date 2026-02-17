# Governor Module — Documentation Index

Complete documentation for the `scarcity.governor` module — Dynamic Resource Governor.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and control loop |

---

## Key Concepts

### Control Loop

1. Sample metrics (sensors)
2. Forecast pressure (profiler)
3. Evaluate rules (policies)
4. Execute actions (actuators)

### Collected Metrics

- CPU, GPU utilization
- Memory, VRAM usage
- I/O throughput
- Latency

---

## Quick Start

```python
from scarcity.governor import DynamicResourceGovernor, DRGConfig

governor = DynamicResourceGovernor(DRGConfig())
governor.register_subsystem("engine", my_engine)
await governor.start()
```

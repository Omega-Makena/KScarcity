# Runtime Module — Documentation Index

Complete documentation for the `scarcity.runtime` module — EventBus and Telemetry.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — EventBus and monitoring |

---

## Key Concepts

### EventBus

- Async pub/sub messaging
- Topic-based routing
- Error isolation

### Telemetry

- Latency tracking (EMA)
- Throughput counting
- Drift detection (Page-Hinkley)
- System resource probing

---

## Quick Start

```python
from scarcity.runtime import get_bus, Telemetry

bus = get_bus()
telemetry = Telemetry(bus=bus)

bus.subscribe("my_topic", my_handler)
await telemetry.start()
```

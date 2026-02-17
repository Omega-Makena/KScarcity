# FMI Module — Documentation Index

Complete documentation for the `scarcity.fmi` module — Federated Metadata Interchange.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Packet types and aggregation |

---

## Key Concepts

### Packet Types

- **MSP**: Meta Signal Pack — current state
- **POP**: Policy Outcome Pack — policy results
- **CCS**: Concept Causal Summary — discovered causality

### Aggregation

- Trimmed mean for robustness
- Weighted by confidence
- Optional DP noise

---

## Quick Start

```python
from scarcity.fmi import FMIEmitter, FMIAggregator

emitter = FMIEmitter(domain_id="site_001")
packet = emitter.emit_msp(metrics={...})

aggregator = FMIAggregator()
result = aggregator.aggregate("cohort", [packet1, packet2])
```

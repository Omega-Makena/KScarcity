# FMI Module — Documentation Index

Complete documentation for the `scarcity.fmi` module — Federated Metadata Interchange.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Packet types and aggregation |
| [01_contracts.md](./01_contracts.md) | PacketType, PacketBase, MSP/POP/CCS, SchemaDefinition, FMIContractRegistry |
| [02_pipeline.md](./02_pipeline.md) | FMIRouter, FMIAggregator, FMIEncoder, FMIEmitter, FMIValidator, FMIService, FMITelemetry |

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

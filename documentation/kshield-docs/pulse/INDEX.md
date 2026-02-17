# Pulse Module — Documentation Index

Complete documentation for the `kshiked.pulse` module — social sensing engine.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Signal detection, indices |

---

## Key Concepts

### Signal Detection

- 15 signal types for economic/political stress
- NLP-enhanced detection (sentiment, emotion, entities)
- Keyword fallback for speed

### Threat Indices

- **PI**: Polarization Index
- **LEI**: Legitimacy Erosion Index
- **MRS**: Mobilization Readiness Score
- **ECI**: Elite Cohesion Index
- **IWI**: Information Warfare Index

### Time Weighting

- Exponential, linear, step decay
- Rolling windows with configurable half-life

---

## Quick Start

```python
from kshiked.pulse import PulseSensor

sensor = PulseSensor(use_nlp=True)
detections = sensor.process_text("Prices are killing us")
state = sensor.get_state()
```

# Pulse Module — Documentation Index

Complete documentation for the `kshiked.pulse` module — social sensing engine.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Signal detection, indices |
| [llm/01_llm_architecture.md](./llm/01_llm_architecture.md) | Ollama LLM integration — models, pipeline, config |
| [llm/02_policy_chatbot_architecture.md](./llm/02_policy_chatbot_architecture.md) | Policy Impact Chatbot — design, data flow, UI, evidence tracing |

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

### News Traceability

- News ingestion persists URL-linked records with extraction metadata.
- Full-content extraction is stored with `extracted_text`, `content_hash`, and extraction status.
- Policy chat outputs include URL + evidence excerpt + trace pointer (record/file reference).

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

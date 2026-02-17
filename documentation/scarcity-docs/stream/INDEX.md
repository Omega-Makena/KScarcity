# Stream Module — Documentation Index

Complete documentation for the `scarcity.stream` module — data ingestion and windowing.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Windowing and rate control |

---

## Key Concepts

### Windowing

- Overlapping windows with configurable stride
- Online normalization (Welford's algorithm)
- EMA smoothing

### Rate Control

- PI controller for adaptive pacing
- Backpressure detection
- Target latency maintenance

---

## Quick Start

```python
from scarcity.stream import StreamSource, WindowBuilder

source = StreamSource("data.csv")
builder = WindowBuilder()

async for chunk in source.stream():
    windows = builder.process_chunk(chunk)
    for window in windows:
        engine.predict(window)
```

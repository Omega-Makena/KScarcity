# Scarcity Stream Module — Overview

The **stream module** handles continuous data ingestion, windowing, and preprocessing for online learning. It provides rate-controlled streaming with adaptive feedback.

---

## Purpose

The discovery engine needs continuous data input:
- **Windowed data**: Fixed-size overlapping windows
- **Normalized data**: Z-score or min-max normalized
- **Rate-controlled**: Adaptive to system load
- **Missing data handled**: LOCF or interpolation

The stream module:
1. **Ingests data** from various sources (CSV, API, generators)
2. **Windows data** with overlap and normalization
3. **Regulates rate** using PI controller
4. **Handles backpressure** from downstream

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Stream Pipeline                        │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Source  │───►│ Sharder  │───►│ Window   │───►│  Cache   │  │
│  │ (ingest) │    │ (split)  │    │ Builder  │    │ (buffer) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       ▼                                               ▼         │
│  ┌──────────┐                                   ┌──────────┐   │
│  │   PI     │                                   │  Schema  │   │
│  │Controller│                                   │ (types)  │   │
│  └──────────┘                                   └──────────┘   │
│                                                                   │
│                        ┌───────────────┐                         │
│                        │   Federator   │                         │
│                        │   (merge)     │                         │
│                        └───────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### StreamSource (`source.py`)

Async data ingestion with rate control:

```python
source = StreamSource(
    data_source="data.csv",
    window_size=1000,
    target_latency_ms=100.0
)

async for chunk in source.stream():
    # Process chunk
    pass
```

**Features**:
- CSV, async iterator, or callable sources
- PI controller for rate regulation
- Backpressure detection

### PIController

Adaptive rate control:

```python
controller = PIController(
    target_latency=100.0,  # ms
    k_p=0.1,               # Proportional gain
    k_i=0.01               # Integral gain
)

# After processing
delay = controller.update(actual_latency_ms)
await asyncio.sleep(delay)
```

### WindowBuilder (`window.py`)

Creates normalized overlapping windows:

```python
builder = WindowBuilder(
    window_size=2048,
    stride=1024,
    normalization="z-score",
    ema_alpha=0.3
)

windows = builder.process_chunk(raw_data)
# Returns list of normalized windows
```

**Features**:
- Welford's algorithm for online statistics
- Z-score or min-max normalization
- EMA smoothing
- Missing data handling (LOCF, interpolation)

### WelfordStats

Online mean/variance computation:

```python
stats = WelfordStats(n_features=10)
stats.update(new_sample)

mean = stats.mean
std = stats.get_std()
```

No need to store all samples — O(1) per update.

### EMASmoother

Noise reduction:

```python
smoother = EMASmoother(alpha=0.3, n_features=10)
smoothed = smoother.smooth(raw_values)
```

---

## File Guide

| File | Purpose |
|------|---------|
| `source.py` | StreamSource and PIController |
| `window.py` | WindowBuilder, WelfordStats, EMASmoother |
| `cache.py` | Data caching for replay |
| `replay.py` | Historical data replay |
| `schema.py` | Data schema definitions |
| `sharder.py` | Data sharding for parallel processing |
| `federator.py` | Stream federation from multiple sources |

---

## Configuration

### StreamSource

```python
StreamSource(
    data_source: str | Callable | AsyncIterator,
    window_size: int = 1000,
    name: str = "default",
    target_latency_ms: float = 100.0
)
```

### WindowBuilder

```python
WindowBuilder(
    window_size: int = 2048,
    stride: int = 1024,
    normalization: str = "z-score",  # or "min-max", "none"
    ema_alpha: float = 0.3,
    fill_method: str = "locf"  # or "interpolation", "zero"
)
```

---

## Integration

### With Engine

```python
# Stream feeds the engine
source = StreamSource("data.csv")
builder = WindowBuilder()

async for chunk in source.stream():
    windows = builder.process_chunk(chunk)
    for window in windows:
        engine.process_window(window)
```

### With Governor

Governor can adjust stream parameters:

```python
# Governor detects high latency
builder.set_window_size(1024)  # Reduce window
builder.set_stride(512)        # Less overlap
```

---

## Usage Example

```python
from scarcity.stream import StreamSource, WindowBuilder

# Setup pipeline
source = StreamSource("economic_data.csv", window_size=1000)
builder = WindowBuilder(window_size=2048, normalization="z-score")

# Process stream
async for chunk in source.stream():
    windows = builder.process_chunk(chunk)
    
    for window in windows:
        # Window is normalized, smoothed, ready for engine
        predictions = engine.predict(window)
        
# Get statistics
print(source.get_stats())
print(builder.get_stats())
```

---

## Edge Cases

### Missing Data

Handled automatically:
- **LOCF**: Last Observation Carried Forward (default)
- **Interpolation**: Linear interpolation
- **Zero**: Fill with zeros

### Concept Drift

Reset statistics when distribution changes:

```python
if drift_detected:
    builder.reset_stats()
```

### Backpressure

PI controller slows down when downstream is overwhelmed:
- Increases delay between chunks
- Prevents buffer overflow
- Maintains target latency

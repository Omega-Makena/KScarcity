# Scarcity Runtime Module — Overview

The **runtime module** provides the core infrastructure for all Scarcity components: the **EventBus** for decoupled communication and **Telemetry** for real-time monitoring.

---

## Purpose

All Scarcity components need:
- **Communication**: Publish/subscribe messaging between modules
- **Monitoring**: Track latency, throughput, drift, resources
- **Coordination**: Graceful startup and shutdown

The runtime module provides:
1. **EventBus**: Async pub/sub message broker
2. **Telemetry**: Real-time metrics collection and publishing
3. **System Probe**: CPU/GPU/memory monitoring

---

## EventBus (`bus.py`)

### Overview

Asynchronous event-driven message broker for intra-process communication:

```python
from scarcity.runtime import get_bus

# Get global singleton
bus = get_bus()

# Subscribe to topic
async def handler(topic, data):
    print(f"Received: {data}")

bus.subscribe("engine.insight", handler)

# Publish message
await bus.publish("engine.insight", {"relationship": "A→B"})
```

### Features

- **Non-blocking dispatch**: All subscribers called concurrently
- **Error isolation**: One subscriber failure doesn't affect others
- **Graceful shutdown**: Wait for pending tasks before stopping

### Key Methods

| Method | Purpose |
|--------|---------|
| `subscribe(topic, callback)` | Register handler for topic |
| `unsubscribe(topic, callback)` | Remove handler |
| `publish(topic, data)` | Send message to all subscribers |
| `topics()` | List active topics |
| `get_stats()` | Get message counts |
| `shutdown(timeout)` | Graceful shutdown |

### Global Singleton

```python
from scarcity.runtime import get_bus, reset_bus

bus = get_bus()  # Get or create global instance
reset_bus()      # Reset (for testing)
```

---

## Telemetry (`telemetry.py`)

### Overview

Real-time monitoring and performance feedback:

```python
from scarcity.runtime import Telemetry, get_bus

telemetry = Telemetry(bus=get_bus(), publish_interval=3.0)
await telemetry.start()

# Record latency
telemetry.record_latency(45.0)

# Get snapshot
snapshot = telemetry.get_snapshot()
```

### Components

#### `LatencyTracker`

EMA-based latency tracking:

```python
tracker = LatencyTracker(alpha=0.3)
tracker.record(45.0)  # ms
latency = tracker.get_latency()
```

#### `ThroughputCounter`

Sliding window event counting:

```python
counter = ThroughputCounter(window_seconds=1.0)
counter.record_event()
rate = counter.get_rate()  # events/second
```

#### `DriftMonitor`

Page-Hinkley test for drift detection:

```python
monitor = DriftMonitor(threshold=3.0)
drift = monitor.update(new_value)
if drift is not None:
    print(f"Drift detected: {drift}")
```

#### `SystemProbe`

Resource monitoring:

```python
probe = SystemProbe()
metrics = probe.probe()
# Returns: {"cpu_util": 0.45, "memory_mb": 4096, "gpu_util": 0.75, ...}
```

### Telemetry Class

Main orchestrator:

```python
telemetry = Telemetry(publish_interval=3.0)

# Start background collection
await telemetry.start()

# Record metrics
telemetry.record_latency(45.0)
telemetry.record_throughput()

# Get snapshot
snapshot = telemetry.get_snapshot()
# {"latency_ms": 42.5, "throughput_eps": 1000, "cpu_util": 0.4, ...}

# Check drift
drift = telemetry.check_drift(value)

# Stop
await telemetry.stop()
```

---

## Common Topics

The EventBus uses these standard topics:

| Topic | Publisher | Data |
|-------|-----------|------|
| `data_window` | StreamSource | Raw data windows |
| `engine.insight` | Engine | Discovered relationships |
| `telemetry` | Telemetry | System metrics |
| `processing_metrics` | Engine | Processing stats |
| `meta.update` | MetaLayer | Hyperparameter changes |
| `drg.signal.*` | Governor | Control signals |
| `sim.state` | Simulation | Economic state |

---

## Integration

### With All Modules

Every module uses the global bus:

```python
from scarcity.runtime import get_bus

bus = get_bus()
bus.subscribe("my_topic", my_handler)
```

### With Dashboard

Telemetry publishes to `telemetry` topic:

```python
bus.subscribe("telemetry", update_dashboard)
```

---

## Usage Example

```python
from scarcity.runtime import EventBus, Telemetry, get_bus

# Get bus
bus = get_bus()

# Setup telemetry
telemetry = Telemetry(bus=bus)

# Define handlers
async def on_insight(topic, data):
    print(f"New relationship: {data}")
    telemetry.record_latency(data.get('latency_ms', 0))

# Subscribe
bus.subscribe("engine.insight", on_insight)

# Start telemetry
await telemetry.start()

# ... application runs ...

# Shutdown
await telemetry.stop()
await bus.shutdown()
```

---

## Index

| File | Purpose |
|------|---------|
| `bus.py` | EventBus pub/sub broker |
| `telemetry.py` | Telemetry, LatencyTracker, ThroughputCounter, DriftMonitor, SystemProbe |

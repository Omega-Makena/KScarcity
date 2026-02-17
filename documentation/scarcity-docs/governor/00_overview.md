# Scarcity Governor Module — Overview

The **governor module** implements the **Dynamic Resource Governor (DRG)** — a control system that monitors system resources and applies backpressure or scaling when needed.

---

## Purpose

Machine learning systems can exhaust resources:
- **VRAM**: GPU memory fills with inference batches
- **CPU**: Processing threads saturate cores
- **Memory**: Buffer queues grow unbounded
- **Latency**: Response times spike under load

The governor:
1. **Monitors resources** via telemetry sensors
2. **Predicts pressure** using EMA and Kalman filtering
3. **Applies policies** when thresholds are exceeded
4. **Executes actions** like batch size reduction or rate limiting

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              DynamicResourceGovernor (Core Loop)                │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Sensors  │───►│ Profiler │───►│ Policies │───►│ Actuators│ │
│   │ (sample) │    │ (EMA +   │    │ (rules)  │    │ (execute)│ │
│   │          │    │  Kalman) │    │          │    │          │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        ▼                                               ▼        │
│   ┌──────────┐                                   ┌──────────┐  │
│   │ Monitor  │                                   │ Registry │  │
│   │ (record) │                                   │(subsys)  │  │
│   └──────────┘                                   └──────────┘  │
│                                                                  │
│                        ┌──────────┐                             │
│                        │  Hooks   │                             │
│                        │(EventBus)│                             │
│                        └──────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Control Loop

Every `control_interval` seconds (default: 0.5s):

```python
while running:
    # 1. Sample current metrics
    metrics = sensors.sample()  # CPU, GPU, memory, I/O
    
    # 2. Update profiles and forecasts
    ema, forecast = profiler.update(metrics)
    
    # 3. Evaluate policies
    decisions = evaluate_policies(metrics, ema, forecast)
    
    # 4. Execute actions
    for subsystem, rule in decisions:
        actuators.execute(subsystem, rule.action, rule.factor)
    
    # 5. Publish telemetry
    hooks.publish_telemetry(metrics)
```

---

## Key Components

### DynamicResourceGovernor (`drg_core.py`)

The main controller:

```python
from scarcity.governor import DynamicResourceGovernor, DRGConfig

governor = DynamicResourceGovernor(config=DRGConfig())

# Register subsystems to control
governor.register_subsystem("inference_engine", engine)
governor.register_subsystem("discovery", discovery_engine)

# Start control loop
await governor.start()
```

### ResourceSensors (`sensors.py`)

Collects system telemetry:

```python
sensors = ResourceSensors(config=SensorConfig())
metrics = sensors.sample()
# Returns: {"cpu_util": 0.45, "mem_util": 0.60, "gpu_util": 0.75, ...}
```

**Metrics collected**:
- `cpu_util`: CPU utilization (0-1)
- `mem_util`: RAM utilization (0-1)
- `gpu_util`: GPU utilization (0-1)
- `vram_util`: GPU memory utilization (0-1)
- `disk_read_mb`, `disk_write_mb`: I/O rates
- `net_sent_mb`, `net_recv_mb`: Network rates

### ResourceProfiler (`profiler.py`)

Smooths and forecasts metrics:

```python
profiler = ResourceProfiler(config=ProfilerConfig())
ema, forecast = profiler.update(metrics)
# ema: Exponential moving average
# forecast: Kalman filter prediction
```

### Policies (`policies.py`)

Rule-based decisions:

```python
@dataclass
class PolicyRule:
    metric: str        # Which metric to check
    threshold: float   # Trigger threshold
    action: str        # What action to take
    factor: float      # Action parameter
    
    def triggered(self, value: float) -> bool:
        return value > self.threshold
```

**Default policies**:
- `vram_util > 0.85` → reduce batch size
- `cpu_util > 0.90` → throttle processing
- `latency_ms > 500` → shed load

### ResourceActuators (`actuators.py`)

Executes control actions:

```python
actuators.execute("inference_engine", "reduce_batch", 0.5)
# Reduces batch size by 50%
```

**Actions**:
- `reduce_batch`: Decrease batch size
- `throttle`: Slow down processing rate
- `shed_load`: Drop low-priority work
- `scale_up`: Request more resources

---

## Configuration

### `DRGConfig`

```python
@dataclass
class DRGConfig:
    sensor: SensorConfig
    profiler: ProfilerConfig
    control_interval: float = 0.5  # Seconds between loops
    policies: Dict[str, List[PolicyRule]]
    monitor: MonitorConfig
```

### `SensorConfig`

```python
@dataclass
class SensorConfig:
    interval_ms: int = 250  # Min time between samples
```

---

## File Guide

| File | Purpose |
|------|---------|
| `drg_core.py` | DynamicResourceGovernor main class |
| `sensors.py` | ResourceSensors telemetry collection |
| `profiler.py` | ResourceProfiler EMA/Kalman |
| `policies.py` | PolicyRule definitions |
| `actuators.py` | ResourceActuators execution |
| `registry.py` | SubsystemRegistry for managed systems |
| `monitor.py` | DRGMonitor for recording |
| `hooks.py` | DRGHooks for EventBus |

---

## Integration

### With Engine

```python
# Engine registers itself
governor.register_subsystem("engine", engine)

# Governor can now:
# - Reduce hypothesis pool size
# - Throttle path evaluation
# - Adjust batch sizes
```

### With EventBus

```python
# Governor publishes telemetry
bus.publish("drg.telemetry", {"metrics": {...}})

# Governor publishes control signals
bus.publish("drg.signal.throttle", {"subsystem": "engine", ...})
```

### With Meta Layer

The meta layer can observe DRG signals:
```python
# Meta layer sees resource pressure
if telemetry["vram_high"]:
    # Adjust hyperparameters to reduce memory usage
```

---

## Usage Example

```python
from scarcity.governor import DynamicResourceGovernor, DRGConfig
from scarcity.governor.policies import PolicyRule

# Custom policy
custom_policies = {
    "inference": [
        PolicyRule("vram_util", 0.8, "reduce_batch", 0.5),
        PolicyRule("latency_ms", 1000, "shed_load", 0.2)
    ]
}

config = DRGConfig(policies=custom_policies)
governor = DynamicResourceGovernor(config)

await governor.start()
# ... runs control loop ...
await governor.stop()
```

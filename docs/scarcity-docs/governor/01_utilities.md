# Governor Module — Component Reference

Detailed API documentation for all components of `scarcity.governor`.

---

## drg_core.py — DynamicResourceGovernor

### `DRGConfig`

```python
@dataclass
class DRGConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    control_interval: float = 0.5
    policies: Dict[str, List[PolicyRule]] = None  # defaults to default_policies()
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensor` | `SensorConfig` | `SensorConfig()` | Telemetry sensor configuration |
| `profiler` | `ProfilerConfig` | `ProfilerConfig()` | EMA/Kalman profiler configuration |
| `control_interval` | `float` | `0.5` | Seconds between control loop iterations |
| `policies` | `Dict[str, List[PolicyRule]]` | `default_policies()` | Subsystem → rule mapping |
| `monitor` | `MonitorConfig` | `MonitorConfig()` | Monitor logging configuration |

### `DynamicResourceGovernor`

Central controller that coordinates all governor subsystems.

```python
class DynamicResourceGovernor:
    def __init__(self, config: DRGConfig, bus: Optional[EventBus] = None)
    async def start() -> None
    async def stop() -> None
    def register_subsystem(name: str, handle: SubsystemHandle | object) -> None
```

**Internal control loop (`_loop`)** — executes every `control_interval` seconds:

1. **Sample** — `sensors.sample()` collects current system metrics
2. **Profile** — `profiler.update(metrics)` returns `(ema, forecast)`
3. **Evaluate** — `_evaluate_policies(metrics, ema, forecast)` checks rules against Kalman forecast (falling back to raw metrics)
4. **Dispatch** — executes actuations and publishes signals via `hooks`
5. **Record** — `monitor.record({...metrics, ...ema})`

**`_evaluate_policies`** returns `List[Tuple[str, PolicyRule]]` — list of `(subsystem_name, triggered_rule)` pairs for which actuator execution succeeded.

---

## sensors.py — ResourceSensors

### `SensorConfig`

```python
@dataclass
class SensorConfig:
    interval_ms: int = 250
```

Minimum milliseconds between samples; enforced via a blocking sleep if called too early.

### `ResourceSensors`

Collects CPU, GPU, memory, and I/O telemetry as a unified dict. Optional dependencies (`psutil`, `torch`, `pynvml`) are handled gracefully — missing libraries return zero values for that metric group.

**`sample() -> Dict[str, float]`** — returns:

| Key | Description |
|-----|-------------|
| `cpu_util` | CPU utilization fraction (0–1), via psutil |
| `cpu_freq` | CPU frequency in MHz |
| `mem_util` | RAM utilization fraction (0–1) |
| `mem_available_gb` | Available RAM in GB |
| `swap_util` | Swap utilization fraction |
| `gpu_util` | GPU compute utilization (0–1); prefers pynvml, falls back to torch |
| `vram_util` | GPU memory utilization fraction (0–1) |
| `disk_read_mb` | Cumulative disk read bytes converted to MB |
| `disk_write_mb` | Cumulative disk write bytes converted to MB |
| `net_sent_mb` | Cumulative network bytes sent in MB |
| `net_recv_mb` | Cumulative network bytes received in MB |

---

## profiler.py — ResourceProfiler

### `ProfilerConfig`

```python
@dataclass
class ProfilerConfig:
    ema_alpha: float = 0.3
    kalman_Q: float = 0.01
    kalman_R: float = 0.1
```

| Field | Description |
|-------|-------------|
| `ema_alpha` | Smoothing factor α for EMA (higher = faster response) |
| `kalman_Q` | Process noise covariance (higher = more responsive filter) |
| `kalman_R` | Measurement noise covariance (higher = smoother filter) |

### `KalmanState`

```python
@dataclass
class KalmanState:
    estimate: float = 0.0
    error_cov: float = 1.0
```

Per-metric Kalman state: current estimate and error covariance.

### `ResourceProfiler`

**`update(metrics) -> Tuple[ema_dict, forecast_dict]`**

For each metric key:

- **EMA**: `ema = (1 - α) × prev + α × new`
- **Kalman predict**: `P_pred = P + Q`
- **Kalman gain**: `K = P_pred / (P_pred + R)`
- **Kalman update**: `estimate = estimate + K × (measurement - estimate)`, `P = (1 - K) × P_pred`

The forecast dict contains the Kalman updated estimate for each metric. Policies use `forecast.get(metric, metrics.get(metric, 0.0))`.

---

## policies.py — PolicyRule

### `PolicyRule`

```python
@dataclass
class PolicyRule:
    metric: str
    threshold: float
    action: str
    direction: str = ">"   # ">" or "<"
    factor: float = 0.5
    priority: int = 1
```

**`triggered(value: float) -> bool`** — returns `value >= threshold` when `direction=">"`, else `value <= threshold`.

### `default_policies()`

Returns a `Dict[str, List[PolicyRule]]` with five pre-configured subsystems:

| Subsystem | Metric | Threshold | Action | Direction |
|-----------|--------|-----------|--------|-----------|
| `simulation` | `vram_util` | 0.90 | `scale_down` | `>` |
| `simulation` | `fps` | 25.0 | `increase_lod` | `<` |
| `mpie` | `cpu_util` | 0.85 | `reduce_batch` | `>` |
| `meta` | `vram_util` | 0.85 | `drop_low_priority` | `>` |
| `federation` | `latency_ms` | 150.0 | `delay_sync` | `>` |
| `memory` | `mem_util` | 0.90 | `flush_cache` | `>` |

---

## actuators.py — ResourceActuators

### `ResourceActuators`

```python
class ResourceActuators:
    def __init__(self, registry: SubsystemRegistry)
    def execute(subsystem: str, action: str, factor: float) -> bool
```

`execute` looks up the subsystem handle via `registry.get(subsystem)`, resolves the action name to a method name via an internal map, and calls `handle.call(method, factor=factor)`. Returns `False` if the subsystem is not registered or the action is unsupported.

Supported actions (action string → method called on handle):

| Action | Method |
|--------|--------|
| `scale_down` | `scale_down` |
| `scale_up` | `scale_up` |
| `reduce_batch` | `reduce_batch` |
| `drop_low_priority` | `drop_low_priority` |
| `delay_sync` | `delay_sync` |
| `flush_cache` | `flush_cache` |
| `increase_lod` | `increase_lod` |

---

## registry.py — SubsystemRegistry

### `SubsystemHandle`

```python
@dataclass
class SubsystemHandle:
    name: str
    handle: Any

    def call(self, method: str, *args, **kwargs) -> bool
```

`call` uses `getattr(handle, method)` and invokes if callable. Returns `True` on success, `False` if method not found.

### `SubsystemRegistry`

```python
class SubsystemRegistry:
    def register(name: str, handle: Any) -> None
    def get(name: str) -> Optional[SubsystemHandle]
    def all() -> Dict[str, SubsystemHandle]
```

Wraps raw handle objects in `SubsystemHandle` on registration. `get` returns `None` if subsystem is not registered.

---

## monitor.py — DRGMonitor

### `MonitorConfig`

```python
@dataclass
class MonitorConfig:
    log_dir: Path = Path("logs/drg")
    level: str = "INFO"
```

Directory is created on init if it does not exist.

### `DRGMonitor`

**`record(metrics: Dict[str, float]) -> None`** — appends `{timestamp: time.time(), **metrics}` to internal history. Keeps the last 1000 entries (older entries are dropped).

**`dump() -> Path`** — writes the full history list to `log_dir/drg_metrics.json` as pretty-printed JSON. Returns the file path.

---

## hooks.py — DRGHooks

### `DRGHooks`

```python
class DRGHooks:
    def __init__(self, bus: EventBus | None = None)
    async def publish_signal(signal: str, payload: Dict[str, Any]) -> None
    async def publish_telemetry(payload: Dict[str, Any]) -> None
```

**`publish_signal`** publishes to topic `drg.signal.{signal}` with the action payload (subsystem, metric, threshold, factor).

**`publish_telemetry`** publishes to topic `drg.telemetry` with metrics, EMA, and Kalman forecast dicts.

---

## Usage Example

```python
from scarcity.governor import DynamicResourceGovernor, DRGConfig

drg = DynamicResourceGovernor(DRGConfig())
drg.register_subsystem("inference_engine", engine_handle)
await drg.start()
# DRG now runs every 0.5s, adjusting registered subsystems
await drg.stop()
```

Custom policies:

```python
from scarcity.governor.policies import PolicyRule
from scarcity.governor import DRGConfig, DynamicResourceGovernor

config = DRGConfig(
    control_interval=1.0,
    policies={
        "inference_engine": [
            PolicyRule(metric="vram_util", threshold=0.80, action="reduce_batch", factor=0.5),
        ]
    }
)
drg = DynamicResourceGovernor(config)
```

Subscribing to DRG telemetry on the event bus:

```python
async def on_drg_telemetry(topic, data):
    print(data["metrics"]["cpu_util"])

bus.subscribe("drg.telemetry", on_drg_telemetry)
bus.subscribe("drg.signal.scale_down", on_scale_down_signal)
```

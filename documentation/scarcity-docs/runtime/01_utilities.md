# Runtime Module — EventBus and Telemetry Reference

API documentation for `scarcity.runtime.bus` and `scarcity.runtime.telemetry`.

---

## EventBus (bus.py)

Async pub/sub message broker. All SCARCITY components communicate through the global singleton.

### Singleton access

```python
from scarcity.runtime import get_bus, reset_bus

bus = get_bus()      # Returns or creates the global EventBus instance
reset_bus()          # Destroys the global instance (testing only)
```

### `EventBus`

```python
class EventBus:
    async def publish(topic: str, data: Any) -> None
    def subscribe(topic: str, callback: Callable) -> None
    def unsubscribe(topic: str, callback: Callable) -> bool
    def topics() -> List[str]
    def get_stats() -> Dict[str, int]
    async def wait_for_idle() -> None
    async def shutdown(timeout: float = 5.0) -> None
```

**`subscribe(topic, callback)`** — registers a callable to receive `(topic, data)` on every publish to that topic. Both sync and async callbacks are supported.

**`publish(topic, data)`** — dispatches to all subscribers concurrently via `asyncio.create_task`. Subscriber errors are isolated: one failing callback does not prevent others from receiving the message. Raises `RuntimeError` if the bus has been shut down.

**`unsubscribe(topic, callback)`** — removes the callback. Returns `True` if found and removed, `False` if not found. Cleans up empty topic entries.

**`topics()`** — returns the list of topic strings that currently have at least one subscriber.

**`get_stats()`** — returns:

```python
{
    "messages_published": int,
    "messages_delivered": int,
    "delivery_errors": int,
    "topics_active": int,        # current count of topics with subscribers
    "total_subscribers": int,    # sum of all subscriber callbacks across all topics
}
```

**`wait_for_idle()`** — awaits all in-flight dispatch tasks. Useful before asserting state in tests.

**`shutdown(timeout=5.0)`** — sets `_running = False`, waits up to `timeout` seconds for pending tasks, cancels any remaining tasks, then clears all subscribers and tasks. After shutdown, `publish` raises `RuntimeError`.

### Usage

```python
from scarcity.runtime import get_bus

bus = get_bus()

async def my_handler(topic: str, data: dict) -> None:
    print(f"[{topic}] {data}")

bus.subscribe("data_window", my_handler)
await bus.publish("data_window", {"x": 42})

# Inspect
print(bus.get_stats())

# Cleanup
await bus.shutdown()
```

---

## Telemetry (telemetry.py)

Runtime performance monitoring and metrics publishing.

### `LatencyTracker`

EMA-based latency tracker.

```python
class LatencyTracker:
    def __init__(self, alpha: float = 0.3)
    def record(duration_ms: float) -> None
    def get_latency() -> float
    def reset() -> None
```

EMA update: `L_t = α × duration_ms + (1 − α) × L_{t-1}`. First observation initializes without smoothing.

### `ThroughputCounter`

Sliding-window event rate counter.

```python
class ThroughputCounter:
    def __init__(self, window_seconds: float = 1.0)
    def record_event() -> None
    def get_rate() -> float   # events per second
```

`record_event` timestamps each event and purges events older than `window_seconds`. `get_rate` returns `count / window_seconds`.

### `DriftMonitor`

Welford online mean/variance with z-score alert threshold.

```python
class DriftMonitor:
    def __init__(self, threshold: float = 3.0)
    def update(value: float) -> Optional[float]
```

**`update`** — updates Welford running mean and variance. After at least 10 observations with positive variance, computes `z = |value − mean| / std`. Returns the z-score if it exceeds `threshold`, otherwise returns `None`. Logs a warning when drift is detected.

### `SystemProbe`

System resource snapshot.

```python
class SystemProbe:
    def probe() -> Dict[str, float]
    def get_last_metrics() -> Dict[str, float]
```

**`probe`** returns a dict that may include (depending on available libraries):

| Key | Source | Description |
|-----|--------|-------------|
| `cpu_percent` | psutil | CPU utilization % |
| `memory_mb` | psutil | RAM used in MB |
| `vram_total_gb` | torch.cuda | Total VRAM in GB |
| `vram_used_gb` | torch.cuda | Used VRAM in GB |
| `gpu_util` | pynvml | GPU utilization % |
| `gpu_memory_util` | pynvml | GPU memory utilization fraction |
| `gpu_count` | torch.cuda | Number of GPUs |
| `gpu_name` | torch.cuda | GPU model name |

Missing libraries produce zero values for their respective fields.

### `Telemetry`

Main orchestrator. Runs a background loop that publishes a snapshot every `publish_interval` seconds.

```python
class Telemetry:
    def __init__(self, bus: Optional[EventBus] = None, publish_interval: float = 3.0)
    async def start() -> None
    async def stop() -> None
    def record_latency(duration_ms: float) -> None
    def record_message() -> None
    def record_error() -> None
    def record_metric(name: str, value: float) -> None
    def check_drift(value: float) -> Optional[float]
```

**Internal loop** — collects a snapshot every `publish_interval` seconds and publishes to the `"telemetry"` topic. The snapshot dict includes all system metrics, bus latency/throughput, error count (rolling 1-minute window), drift score, and any custom metrics.

**`meta_metrics` subscription** — `Telemetry` subscribes to the `"meta_metrics"` topic on init and forwards all numeric fields as custom metrics via `record_metric`. Unsubscribes cleanly on `stop()`.

**Snapshot keys** (representative):

| Key | Description |
|-----|-------------|
| `timestamp` | Unix timestamp |
| `cpu_percent` | CPU utilization |
| `memory_mb` | RAM used in MB |
| `bus_latency_ms` / `latency_ms` | EMA bus latency |
| `bus_throughput` | Messages per second |
| `fps` | Derived from latency: `1000 / latency_ms` |
| `errors_last_minute` | Error events in past 60 s |
| `drift_score` | Current Welford mean (not z-score) |

Custom metrics are merged at the top level.

### Usage

```python
from scarcity.runtime import get_bus
from scarcity.runtime.telemetry import Telemetry

bus = get_bus()
telemetry = Telemetry(bus=bus, publish_interval=3.0)
await telemetry.start()

# From any component:
telemetry.record_latency(45.2)
telemetry.record_message()
telemetry.record_metric("inference_batch_size", 32)

drift = telemetry.check_drift(some_value)

# Consume telemetry snapshots
async def on_telemetry(topic, snapshot):
    print(snapshot["cpu_percent"])

bus.subscribe("telemetry", on_telemetry)

await telemetry.stop()
```

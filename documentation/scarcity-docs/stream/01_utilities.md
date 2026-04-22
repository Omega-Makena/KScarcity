# Stream Module — Component Reference

API documentation for all components of `scarcity.stream`.

---

## source.py — StreamSource

Async data ingestion with PI-controller rate regulation.

### `PIController`

```python
class PIController:
    def __init__(self, target_latency: float = 100.0, k_p: float = 0.1, k_i: float = 0.01)
    def update(actual_latency_ms: float) -> float
```

Adaptive delay controller. Formula:

```
Δt_next = Δt_base + K_p × error + K_i × integral
error = target_latency - actual_latency_ms
```

Integral is clamped to `±1000` (windup protection). Output is clipped to `[0.01, 10.0]` seconds.

### `StreamSource`

```python
class StreamSource:
    def __init__(
        self,
        data_source: Callable | AsyncIterator | str,
        window_size: int = 1000,
        name: str = "default",
        target_latency_ms: float = 100.0,
    )
    async def read_chunk() -> Optional[np.ndarray]
    async def stream() -> AsyncIterator[np.ndarray]
    def stop() -> None
    def get_stats() -> Dict[str, Any]
```

`data_source` accepts:
- `str` — CSV file path; read in `window_size` row chunks via pandas (requires pandas)
- `AsyncIterator` — iterated `window_size` items per chunk
- `Callable` — called each chunk; if the result is a coroutine it is awaited

**`read_chunk`** — reads one chunk, records latency, updates PI controller. Returns `None` at end-of-data.

**`stream`** — async generator that yields chunks and sleeps `current_delay` seconds between reads (delay set by PI controller).

**`get_stats`** — returns `{name, chunks_read, rows_read, errors, last_latency_ms, current_delay, is_running}`.

---

## window.py — WindowBuilder

Online windowing and normalization.

### `WelfordStats`

```python
class WelfordStats:
    def __init__(self, n_features: int)
    def update(x: np.ndarray) -> None    # accepts (n_samples, n_features) or (n_features,)
    def get_std() -> np.ndarray          # returns std per feature; floor 1e-8
```

Incremental mean and variance using Welford's algorithm. Safe for online use — no storage of samples.

### `EMASmoother`

```python
class EMASmoother:
    def __init__(self, alpha: float = 0.3, n_features: int = 1)
    def smooth(x: np.ndarray) -> np.ndarray
```

EMA per feature: `value = α × x + (1 − α) × value`. Initializes to the first sample on first call.

### `WindowBuilder`

```python
class WindowBuilder:
    def __init__(
        self,
        window_size: int = 2048,
        stride: int = 1024,
        normalization: str = "z-score",  # "z-score" | "min-max" | "none"
        ema_alpha: float = 0.3,
        fill_method: str = "locf",       # "locf" | "linear"
    )
    def process_chunk(chunk: np.ndarray) -> List[np.ndarray]
    def reset_stats() -> None
    def set_window_size(new_size: int) -> None
    def set_stride(new_stride: int) -> None
    def get_stats() -> Dict
```

**`process_chunk`** — appends chunk to a rolling deque buffer, extracts overlapping windows of size `window_size` with step `stride`, applies missing data handling then normalization to each window. Returns a list of normalized `np.ndarray` windows.

**Missing data** — `locf`: last-observed-carry-forward (forward fill then backward fill for leading NaNs); `linear`: per-feature linear interpolation with zero fill when all values are NaN.

**Normalization** — `z-score`: `(window − mean) / std` using Welford running statistics; `min-max`: `(window − min) / (max − min + 1e-8)` when bounds are set; `none`: pass-through.

**Governor integration** — `set_window_size` and `set_stride` allow the DRG to adapt the window parameters at runtime. `reset_stats` clears Welford statistics for drift recovery.

---

## cache.py — CacheManager

LRU cache with temporal decay weighting.

```python
class CacheManager:
    def __init__(self, max_size: int = 1000, decay_factor: float = 0.01)
    def get(key: str) -> Optional[np.ndarray]
    def put(key: str, data: np.ndarray, metadata: Optional[Dict] = None) -> None
    def get_weight(key: str) -> float
    def clear() -> None
    def get_stats() -> Dict
    def size() -> int
```

Backed by `collections.OrderedDict` (insertion-order = LRU order). When `max_size` is reached, the oldest (least recently used) entry is evicted.

**`get`** — moves the item to the end (most recently used) before returning. Records a hit.

**`put`** — does nothing if key already exists (moves to end). Otherwise inserts and evicts if at capacity.

**`get_weight(key)`** — temporal decay: `exp(−λ × age_seconds)` where `λ = decay_factor`. Returns `0.0` if key not in cache.

**`get_stats`** — returns `{size, max_size, hits, misses, hit_ratio, evictions, insertions}`.

---

## schema.py — SchemaManager

UUID-based schema versioning and lineage tracking.

### `FieldMetadata`

```python
@dataclass
class FieldMetadata:
    name: str
    dtype: str
    unit: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
```

### `SchemaVersion`

```python
@dataclass
class SchemaVersion:
    version_uuid: str   # MD5 hash of schema fields (first 16 chars)
    fields: List[FieldMetadata]
    created_at: str     # ISO format UTC
    hash: str           # Full MD5 of sorted field JSON
```

### `SchemaManager`

```python
class SchemaManager:
    def infer_schema(data: np.ndarray, field_names: Optional[List[str]] = None) -> SchemaVersion
    def validate_data(data: np.ndarray) -> Tuple[bool, Optional[str]]
    def get_schema_diff(old: SchemaVersion, new: SchemaVersion) -> Dict[str, Any]
    def get_current_schema() -> Optional[SchemaVersion]
    def get_field_index(field_name: str) -> Optional[int]
    def load_history() -> None
```

**`infer_schema`** — computes an MD5 hash of the field definitions. If the hash matches the current schema, returns it unchanged (no new version). Otherwise creates a new `SchemaVersion` with a fresh UUID and appends to history. Optionally persists history to `schema_history_file`.

**`validate_data`** — checks feature count matches current schema. Returns `(True, None)` if no schema exists yet.

**`get_schema_diff`** — returns `{added_fields, removed_fields, type_changes, breaking}`. `breaking=True` when fields are removed or types changed.

---

## sharder.py — StreamSharder

Domain-based adaptive stream partitioning.

```python
class StreamSharder:
    def __init__(self, n_shards: int = 3, rebalance_threshold: float = 2.0)
    def assign_shard(data: np.ndarray, metadata: Optional[Dict] = None) -> int
    def record_latency(shard_id: int, latency_ms: float) -> None
    def rebalance() -> None
    def get_shard_stats() -> Dict
    def get_stats() -> Dict
```

Uses MiniBatch K-Means clustering (scikit-learn) to assign data windows to shards. Falls back to round-robin if scikit-learn is not installed.

**`assign_shard`** — extracts a feature vector (mean of window if 2D), predicts cluster assignment, updates the clusterer with `partial_fit`.

**`rebalance`** — triggered when any shard's mean latency exceeds `rebalance_threshold × overall_mean`. Reinitializes the clusterer to redistribute assignments.

---

## replay.py — ReplayManager

Fault-tolerant event log with checkpoint recovery.

```python
class ReplayManager:
    def __init__(self, log_file: str = "logs/stream/replay.log", checkpoint_dir: str = "logs/stream/checkpoints")
    async def log_event(event: Dict) -> int          # returns byte offset
    async def replay_events(start_offset: int = 0, end_offset: Optional[int] = None) -> List[Dict]
    async def save_checkpoint(offset: int, metadata: Optional[Dict] = None) -> str
    async def load_latest_checkpoint() -> Optional[Dict]
    async def recover() -> Tuple[int, List[Dict]]
    def check_heartbeat() -> bool
    def get_stats() -> Dict
```

Uses an append-only JSONL log file. Each entry includes `offset`, `timestamp`, `checksum` (MD5 of event dict), and `event`. Checksums are verified on replay; mismatches are logged.

**`recover`** — loads the most recent checkpoint (by filename sort) and replays events from that offset. Returns `(last_offset, events)`.

**`check_heartbeat`** — returns `True` if the last logged event was within `heartbeat_timeout` (30 s) of now.

Uses `aiofiles` for async I/O when available; falls back to synchronous file operations.

---

## federator.py — StreamFederator

Multi-node stream sharing via WebSocket.

```python
class StreamFederator:
    def __init__(self, node_id: str, listen_port: int = 8765)
    async def start_server() -> None
    async def broadcast_data(data_window: Dict, domain_id: Optional[int] = None) -> None
    def get_stats() -> Dict
```

WebSocket-based peer communication. Nodes exchange `hello` handshakes on connect and `heartbeat` messages to track liveness. `broadcast_data` sends to all connected peers. Conflict resolution uses latest-timestamp-wins.

Requires `websockets` library. Falls back gracefully when unavailable.

**`get_stats`** — returns `{node_id, connected, peer_count, messages_sent, messages_received, conflicts_resolved}`.

---

## Usage

```python
from scarcity.stream.source import StreamSource
from scarcity.stream.window import WindowBuilder
from scarcity.stream.cache import CacheManager

source = StreamSource("data/stream.csv", window_size=1024, name="main")
builder = WindowBuilder(window_size=2048, stride=1024, normalization="z-score")
cache = CacheManager(max_size=500)

async for chunk in source.stream():
    windows = builder.process_chunk(chunk)
    for i, window in enumerate(windows):
        key = f"window_{source._stats['chunks_read']}_{i}"
        cache.put(key, window)
```

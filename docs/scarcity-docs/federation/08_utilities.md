# Federation Utilities

This page documents the smaller utility files in the federation module.

---

## packets.py — Wire Formats

Defines the data structures exchanged between clients and coordinator.

### `PathPack`

A compressed representation of discovered paths:

```python
@dataclass
class PathPack:
    source_idx: int
    target_idx: int
    lag: int
    effect: float
    confidence: float
    stability: float
```

### `EdgeDelta`

Incremental update to an edge:

```python
@dataclass
class EdgeDelta:
    source_id: int
    target_id: int
    weight_delta: float    # Change in weight
    ci_delta: Tuple[float, float]  # Change in CI
    hits_delta: int
```

### `PolicyPack`

Policy configuration bundle:

```python
@dataclass
class PolicyPack:
    g_min: float           # Minimum gain
    stability_min: float   # Minimum stability
    n_proposals: int       # Paths per window
    # ...
```

### `CausalSemanticPack`

Higher-level causal structure:

```python
@dataclass
class CausalSemanticPack:
    dag_edges: List[Tuple[int, int, float]]
    confounders: List[int]
    interventions: List[Dict]
```

---

## codec.py — Serialization

### `PayloadCodec`

Efficient serialization for federation messages:

```python
codec = PayloadCodec()

# Serialize
bytes_data = codec.encode(path_pack)

# Deserialize  
path_pack = codec.decode(bytes_data, PathPack)
```

**Features**:
- Compact binary format (msgpack-based)
- Schema versioning for compatibility
- Compression optional

---

## validator.py — Packet Validation

### `PacketValidator`

Validates incoming packets:

```python
validator = PacketValidator()

is_valid, error = validator.validate(packet)
if not is_valid:
    reject(error)
```

**Checks**:
- Schema conformance
- Value bounds (e.g., confidence in [0,1])
- Signature verification (if signed)
- Replay detection (via nonces)

---

## privacy_guard.py — Privacy Enforcement

### `PrivacyGuard`

Enforces privacy policies:

```python
guard = PrivacyGuard(
    max_epsilon=10.0,
    max_delta=1e-4
)

# Check before DP operation
if guard.can_proceed(epsilon=1.0, delta=1e-5):
    perform_operation()
    guard.record(epsilon=1.0, delta=1e-5)
else:
    raise PrivacyBudgetExhausted()
```

**Integrates with**:
- GossipProtocol (local DP)
- Layer2Aggregator (central DP)
- PrivacyAccountant (tracking)

---

## trust_scorer.py — Client Trust

### `TrustScorer`

Computes trust scores for clients:

```python
scorer = TrustScorer()

# Update trust based on behavior
scorer.update(client_id="hospital_1", 
              metric="outlier_count", 
              value=0)

trust = scorer.get_trust("hospital_1")
# Returns: 0.95
```

**Factors**:
- Frequency of outlier updates
- Consistency with aggregate
- Uptime/availability
- Historical accuracy

**Use cases**:
- Weight updates by trust
- Exclude low-trust clients from aggregation
- Flag suspicious behavior

---

## scheduler.py — Federation Scheduling

### `FederationScheduler`

Coordinates federation rounds:

```python
scheduler = FederationScheduler(
    round_duration=60.0,      # Seconds per round
    gossip_frequency=5,       # Gossip rounds per federation round
    aggregation_delay=10.0    # Wait for stragglers
)

# Main loop
while True:
    scheduler.tick()
    
    if scheduler.should_gossip():
        run_gossip()
    
    if scheduler.should_aggregate():
        run_aggregation()
        scheduler.advance_round()
```

---

## client_agent.py — Client-Side Agent

### `FederationClientAgent`

Wraps a local engine for federation:

```python
agent = FederationClientAgent(
    engine=my_discovery_engine,
    client_id="hospital_1"
)

# Extract shareable update from engine
update = agent.get_shareable_update()

# Apply received global model
agent.apply_global_model(global_model)
```

**Handles**:
- Converting engine state to federation format
- Integrating received updates
- Managing local vs. global model tension

---

## coordinator.py — Server-Side Coordinator

### `FederationCoordinator`

High-level coordinator logic:

```python
coordinator = FederationCoordinator()

# Process incoming update
coordinator.handle_update(client_id, update)

# Trigger aggregation
if coordinator.should_aggregate():
    global_model = coordinator.aggregate()
    coordinator.broadcast(global_model)
```

---

## reconciler.py — Store Reconciliation

### `StoreReconciler`

Merges distributed HypergraphStores:

```python
reconciler = StoreReconciler(strategy="union")

merged_store = reconciler.reconcile([store_1, store_2, store_3])
```

**Strategies**:
- `union`: Include edge if in any store
- `intersection`: Include edge if in all stores
- `weighted`: Weight by source count/confidence

### `build_reconciler(strategy) -> StoreReconciler`

Factory function:

```python
reconciler = build_reconciler("weighted")
```

---

## transport.py — Network Abstraction

### Transport Layer

Abstract interface for network communication:

```python
class Transport(ABC):
    @abstractmethod
    def send(self, to: str, message: bytes) -> None: ...
    
    @abstractmethod
    def receive(self, timeout: float) -> Optional[Tuple[str, bytes]]: ...
```

**Implementations**:
- `InMemoryTransport`: For testing
- `HTTPTransport`: REST-based
- `GRPCTransport`: High-performance RPC (future)

---

## Index

### Core Files

| File | Purpose |
|------|---------|
| `hierarchical.py` | Main orchestrator |
| `aggregator.py` | Byzantine-robust aggregation |
| `gossip.py` | Push-pull gossip with local DP |
| `basket.py` | Domain basket management |
| `buffer.py` | Staleness-aware update buffer |
| `layers.py` | Layer 1 & 2 aggregation |
| `secure_aggregation.py` | Cryptographic secure sum |

### Utility Files

| File | Purpose |
|------|---------|
| `packets.py` | Wire formats |
| `codec.py` | Serialization |
| `validator.py` | Packet validation |
| `privacy_guard.py` | Privacy enforcement |
| `trust_scorer.py` | Client trust |
| `scheduler.py` | Round scheduling |
| `client_agent.py` | Client wrapper |
| `coordinator.py` | Server logic |
| `reconciler.py` | Store merging |
| `transport.py` | Network abstraction |

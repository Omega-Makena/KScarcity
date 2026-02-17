# gossip.py — Gossip Protocol with Local DP

The gossip module implements **peer-to-peer communication** within domain baskets. Every message is clipped and noised with local differential privacy before transmission.

---

## Purpose

Rather than sending all updates to a central server, clients within a basket **gossip** with each other:
- Reduces load on central infrastructure
- Faster convergence within similar domains
- Privacy preserved via local DP (server never sees raw updates)

---

## Push-Pull Hybrid Protocol

### Push

When local state changes significantly ("material" change):
1. Clip update to L2 norm
2. Add calibrated Gaussian noise (local DP)
3. Send to K randomly selected peers

**Trigger**: MaterialityDetector checks drift from last pushed state.

### Pull

Periodically:
1. Request current summaries from K random peers
2. Average received summaries into local state
3. Update local reference

**Interval**: Configurable (default: every 2 seconds of simulated time).

---

## Core Classes

### `GossipConfig`

```python
@dataclass
class GossipConfig:
    peers_per_round: int = 5         # K peers per gossip round
    pull_interval: float = 2.0       # Seconds between pulls
    push_drift_threshold: float = 0.1  # Relative change to trigger push
    clip_norm: float = 1.0           # L2 clipping bound
    local_dp_epsilon: float = 1.0    # Privacy parameter
    local_dp_delta: float = 1e-5     # Privacy failure probability
    max_messages_per_day: int = 24   # Rate limit
    message_ttl: float = 300.0       # Message expiry (seconds)
```

### `GossipMessage`

A single gossip message:

```python
@dataclass
class GossipMessage:
    sender_id: str              # Who sent it
    basket_id: str              # Which basket
    summary_vector: np.ndarray  # The (clipped+noised) update
    sequence_number: int        # For ordering
    timestamp: float            # When created
    round_id: int               # Training round
```

Methods:
- `age()`: Seconds since creation
- `is_expired(ttl)`: True if older than TTL

---

## Local Differential Privacy

### `LocalDPMechanism`

Calibrates Gaussian noise for (ε, δ)-DP:

```python
dp = LocalDPMechanism(
    epsilon=1.0,
    delta=1e-5,
    sensitivity=1.0  # L2 clipping norm
)

noisy_vector = dp.clip_and_noise(raw_vector)
```

**Noise calibration**:
```
σ = sensitivity × √(2 × ln(1.25/δ)) / ε
```

This is the **Gaussian mechanism** for (ε, δ)-DP.

### Methods

- `add_noise(vector)`: Add calibrated Gaussian noise
- `clip_and_noise(vector)`: Clip to L2 norm, then add noise

**Why local DP?** Even if the gossip network is compromised, each message reveals at most (ε, δ) information about the sender's data.

---

## MaterialityDetector

Decides when to push:

```python
detector = MaterialityDetector(drift_threshold=0.1)

should_push = detector.should_push("client_1", new_state)
if should_push:
    send_gossip(new_state)
    detector.update_state("client_1", new_state)
```

### Drift Computation

```python
drift = ||new_state - old_state|| / (||old_state|| + ε)
```

High drift → State changed significantly → Worth pushing.

### Purposes

1. **Bandwidth conservation**: Don't push if nothing changed
2. **Privacy conservation**: Fewer pushes = less budget consumed
3. **Convergence optimization**: Focus communication on active learning phases

---

## PeerSampler

Samples K peers with rotation for privacy:

```python
sampler = PeerSampler(rotation_window=10)

peers = sampler.sample(
    basket_peers=["a", "b", "c", "d", "e"],
    k=2,
    exclude="a"  # Don't sample self
)
# Returns: ["c", "d"]
```

### Rotation Window

A peer cannot be resampled within `rotation_window` rounds:
- Spreads privacy budget across more peers
- Prevents over-querying the same clients
- Improves coverage

### Methods

- `sample(basket_peers, k, exclude)`: Sample k peers
- `advance_round()`: Move to next round, update rotation
- `rotate()`: Alias for advance_round

---

## GossipProtocol

Main protocol class:

### Initialization

```python
protocol = GossipProtocol(
    config=GossipConfig(),
    basket_manager=basket_manager
)
```

### `run_round(basket_id) -> int`

Execute one gossip round for a basket:

```python
n_messages = protocol.run_round("healthcare")
```

**Steps**:
1. Get all clients in basket
2. For each client:
   a. Check if push needed (MaterialityDetector)
   b. If push: clip+noise state, send to K peers
   c. Process incoming messages (pull)
   d. Average into local state

Returns: Number of messages exchanged.

### `receive_messages(client_id) -> List[GossipMessage]`

Get pending messages for a client:
- Filters expired messages
- Consumes from inbox

### `send_message(msg: GossipMessage)`

Queue a message for delivery:
- Validates rate limits
- Stores in recipient's inbox

---

## Message Flow

```
Client A                     Client B
    │                           │
    │  1. Detect material       │
    │     change                │
    │                           │
    │  2. Clip to L2=1.0        │
    │     [0.3, -0.5, 0.2] →    │
    │     [0.27, -0.45, 0.18]   │
    │                           │
    │  3. Add Gaussian noise    │
    │     [0.32, -0.41, 0.21]   │
    │                           │
    │  ────────────────────►    │
    │     GossipMessage         │
    │                           │
    │                           │  4. Receive and validate
    │                           │
    │                           │  5. Average into local
    │                           │     state
```

---

## Rate Limiting

To prevent budget exhaustion:
- `max_messages_per_day`: Hard limit on outgoing messages
- Messages exceeding limit are dropped
- Counter resets at round boundaries

---

## Edge Cases

### No Peers Available

If basket has only one client:
- Gossip is a no-op
- Client waits for Layer 1 aggregation

### Message Expiration

Old messages (age > TTL) are discarded:
- Prevents processing stale information
- Avoids "ghost" updates from long-dead clients

### Network Partitions

If some peers are unreachable:
- Protocol continues with available peers
- Convergence is slower but still happens

---

## Privacy Budget per Round

Each gossip message consumes privacy budget:
- 1 message at ε=1.0 consumes ~1.0ε
- With K=5 peers per round: ~5.0ε per round
- With 10 clients: total ~50ε per round (if everyone pushes)

**Budget management**:
- Only push when material
- Increase ε (less noise, less privacy) to stretch budget
- Reduce K to reduce message count

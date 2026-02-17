# buffer.py — Staleness-Aware Update Buffer

The `UpdateBuffer` stores pending client updates, tracks their age, and triggers aggregation when conditions are met.

---

## Purpose

Updates don't arrive simultaneously:
- Some clients are faster than others
- Network latency varies
- Clients may skip rounds

The buffer:
1. Stores updates with metadata (timestamp, round_id)
2. Tracks staleness for age-weighted aggregation
3. Triggers aggregation when thresholds are met
4. Prevents replay attacks and over-participation

---

## Core Classes

### `BufferConfig`

```python
@dataclass
class BufferConfig:
    max_staleness: int = 5           # Max rounds old before discarding
    min_updates_trigger: int = 3     # Minimum updates to trigger
    time_trigger_seconds: float = 60.0  # Trigger if this time elapsed
    participation_cap: int = 3       # Max updates per client per round
```

### `BufferedUpdate`

```python
@dataclass
class BufferedUpdate:
    client_id: str
    basket_id: str
    update: np.ndarray
    round_id: int
    timestamp: float
    staleness: int = 0  # How many rounds old
```

---

## Class: `UpdateBuffer`

### Initialization

```python
buffer = UpdateBuffer(config=BufferConfig(
    max_staleness=5,
    min_updates_trigger=3
))
```

### `add(update: BufferedUpdate) -> bool`

Add an update to the buffer:

```python
success = buffer.add(BufferedUpdate(
    client_id="hospital_1",
    basket_id="healthcare_basket_0",
    update=np.array([...]),
    round_id=42,
    timestamp=time.time()
))
```

Returns `False` if:
- Participation cap exceeded
- Replay detected
- Update is too stale

### `get_basket_updates(basket_id) -> List[BufferedUpdate]`

Retrieve updates for a basket:

```python
updates = buffer.get_basket_updates("healthcare_basket_0")
```

Returns updates sorted by timestamp (oldest first).

### `pop_basket_updates(basket_id) -> List[BufferedUpdate]`

Get and **remove** updates for aggregation:

```python
updates = buffer.pop_basket_updates("healthcare_basket_0")
# Buffer is now empty for this basket
```

### `advance_round()`

Move to next round:

```python
buffer.advance_round()
```

Increments staleness counters and prunes old updates.

---

## Trigger Engine

### `TriggerEngine`

Decides when to trigger aggregation:

```python
trigger = TriggerEngine(config=BufferConfig())

should_aggregate = trigger.should_trigger(
    basket_id="healthcare_basket_0",
    n_updates=5,
    time_since_last=45.0
)
```

**Trigger conditions** (any one triggers):
1. `n_updates >= min_updates_trigger`
2. `time_since_last >= time_trigger_seconds`
3. External force flag

### `check_and_clear(basket_id) -> bool`

Check and reset trigger state:

```python
if trigger.check_and_clear("healthcare_basket_0"):
    run_aggregation()
```

Clears the trigger after checking (prevents double-firing).

---

## Replay Guard

### `ReplayGuard`

Prevents replay attacks:

```python
guard = ReplayGuard()

is_valid = guard.check((client_id, round_id, nonce))
if not is_valid:
    reject_update("Replay detected")
```

**Tracks**:
- (client_id, round_id) pairs already seen
- Nonces for cryptographic freshness

### Methods

- `check(key) -> bool`: True if not seen before
- `mark(key)`: Record as seen
- `clear_round(round_id)`: Clear old entries

---

## Privacy Accountant

### `PrivacyAccountant`

Tracks cumulative privacy loss:

```python
accountant = PrivacyAccountant(
    total_epsilon=10.0,
    total_delta=1e-4
)

can_proceed = accountant.check(epsilon=1.0, delta=1e-5)
if can_proceed:
    # Do DP operation
    accountant.spend(epsilon=1.0, delta=1e-5)
```

### Budget Composition

Uses **simple composition** by default:
- ε_total = Σ ε_i
- δ_total = Σ δ_i

Advanced: Moments accountant for tighter bounds (future).

### Methods

- `check(epsilon, delta) -> bool`: Would spending exceed budget?
- `spend(epsilon, delta)`: Consume budget
- `remaining() -> Tuple[float, float]`: Get remaining (ε, δ)
- `reset()`: Refill budget (new epoch)

---

## Staleness-Aware Aggregation

When aggregating:

```python
def aggregate_with_staleness(updates):
    weighted_sum = np.zeros(dim)
    weight_sum = 0
    
    for u in updates:
        # Exponential decay by staleness
        w = 0.9 ** u.staleness
        weighted_sum += w * u.update
        weight_sum += w
    
    return weighted_sum / weight_sum
```

**Effect**: Recent updates have more influence than stale ones.

---

## Participation Cap

Prevents over-contribution:

```python
# BufferConfig.participation_cap = 3

# Client submits 4 updates in one round
buffer.add(update_1)  # Accepted
buffer.add(update_2)  # Accepted
buffer.add(update_3)  # Accepted
buffer.add(update_4)  # REJECTED
```

**Purpose**: 
- Prevents single client from dominating
- Limits privacy leakage per client

---

## Internal State

### Buffer State

```python
{
    "healthcare_basket_0": [
        BufferedUpdate(client_id="h1", ...),
        BufferedUpdate(client_id="h2", ...),
    ],
    "finance_basket_0": [
        BufferedUpdate(client_id="f1", ...),
    ]
}
```

### Participation Counts

```python
{
    ("hospital_1", 42): 2,  # Client, round → count
    ("hospital_2", 42): 1,
}
```

### Replay History

```python
{
    ("hospital_1", 42, "nonce_abc"),
    ("hospital_2", 42, "nonce_def"),
}
```

---

## Edge Cases

### Empty Buffer

Aggregation produces None or default model.

### All Updates Stale

After `max_staleness` rounds, updates are pruned:
- Basket may have no aggregatable updates
- Aggregation skipped

### Client Submits Old Round

```python
# Current round: 50
buffer.add(BufferedUpdate(..., round_id=45))
# staleness = 50 - 45 = 5
# If max_staleness = 5: just barely accepted
# If max_staleness = 4: rejected
```

---

## Statistics

### `get_stats() -> Dict`

```python
{
    "n_buffered": 42,
    "n_baskets_with_updates": 5,
    "avg_staleness": 0.8,
    "oldest_update_age": 3,
    "participation_violations": 2,
    "replay_attempts_blocked": 0
}
```

# hierarchical.py — Hierarchical Federation Orchestrator

The `HierarchicalFederation` class is the **main entry point** for the federation system. It integrates all components — baskets, gossip, buffers, aggregation layers, and secure aggregation.

---

## Purpose

This is the class you'll interact with most when setting up federated learning. It:

- Manages client registration and domain assignment
- Coordinates gossip rounds within baskets
- Triggers Layer 1 and Layer 2 aggregation based on configurable conditions
- Tracks privacy budget consumption
- Produces the global model for distribution

---

## Configuration

### `HierarchicalFederationConfig`

```python
@dataclass
class HierarchicalFederationConfig:
    basket: BasketConfig           # Basket settings
    gossip: GossipConfig           # Gossip protocol settings
    buffer: BufferConfig           # Update buffer settings
    layer1: Layer1Config           # Intra-basket aggregation
    layer2: Layer2Config           # Cross-basket aggregation
    total_epsilon: float = 10.0    # Total privacy budget (ε)
    total_delta: float = 1e-4      # Privacy failure probability (δ)
    auto_aggregate: bool = True    # Auto-run aggregation on triggers
    vector_dim: int = 64           # Dimension of update vectors
```

---

## Class: `HierarchicalFederation`

### Initialization

```python
fed = HierarchicalFederation(config=HierarchicalFederationConfig(
    total_epsilon=10.0,
    auto_aggregate=True,
    vector_dim=128
))
```

Creates internal components:
- `BasketManager`: Manages domain groupings
- `GossipProtocol`: Handles intra-basket communication
- `UpdateBuffer`: Stores pending updates with staleness tracking
- `Layer1Aggregator`: Aggregates within baskets
- `Layer2Aggregator`: Aggregates across baskets
- `PrivacyAccountant`: Tracks budget consumption

---

## Lifecycle Methods

### `register_client(client_id, domain_id, features=None, identity_keypair=None)`

Adds a client to the federation:

```python
fed.register_client(
    client_id="hospital_1",
    domain_id="healthcare",
    identity_keypair=IdentityKeyPair.generate()  # For secure aggregation
)
```

The client is automatically assigned to the appropriate basket based on `domain_id`.

### `unregister_client(client_id) -> bool`

Removes a client. Returns `True` if found.

### `advance_round()`

Moves to the next training round. Clears round-specific state.

---

## Update Submission

### `submit_update(client_id, update, round_id=None) -> bool`

Submit a client's update vector:

```python
success = fed.submit_update(
    client_id="hospital_1",
    update=np.array([0.1, -0.2, 0.05, ...]),
    round_id=42
)
```

**What happens**:
1. Validate the update (not replayed, within participation cap)
2. Store in the buffer with timestamp and round_id
3. If `auto_aggregate=True`, check if triggers should fire
4. Return success status

---

## Gossip

### `run_gossip_round() -> Dict[str, int]`

Execute one gossip round across all baskets:

```python
stats = fed.run_gossip_round()
# Returns: {"healthcare": 15, "finance": 8}  # Messages exchanged
```

For each basket:
1. Sample K peers per client
2. Check materiality (has local state changed significantly?)
3. Clip and noise the update (local DP)
4. Exchange messages
5. Average received updates into local state

---

## Aggregation

### `maybe_aggregate() -> Optional[np.ndarray]`

Check triggers and run aggregation if conditions met:

```python
global_model = fed.maybe_aggregate()
if global_model is not None:
    # New global model available
    distribute_to_clients(global_model)
```

**Layer 1 triggers** (per basket):
- Minimum clients have submitted updates
- Time since last aggregation exceeds threshold
- Buffer staleness exceeds tolerance

**Layer 2 triggers**:
- Minimum baskets have Layer 1 aggregates
- Global update interval exceeded

### `force_aggregate() -> Optional[np.ndarray]`

Force aggregation regardless of triggers. Useful for:
- End of training
- Debugging
- When you know enough data is ready

---

## Model Access

### `get_global_model() -> Optional[np.ndarray]`

Get the current global model (from last Layer 2 aggregation):

```python
model = fed.get_global_model()
```

Returns `None` if no aggregation has happened yet.

### `get_basket_model(basket_id) -> Optional[np.ndarray]`

Get basket-specific model:

```python
healthcare_model = fed.get_basket_model("healthcare")
```

### `get_meta_params() -> Dict`

Get shared meta-parameters (hyperparameters learned across rounds).

---

## Privacy Tracking

### `get_privacy_budget() -> Tuple[float, float]`

Check remaining privacy budget:

```python
remaining_eps, remaining_delta = fed.get_privacy_budget()
print(f"Budget remaining: ε={remaining_eps:.2f}, δ={remaining_delta:.2e}")
```

### Budget Consumption

Operations that consume budget:
- Gossip message (local DP): ~0.5ε per message
- Layer 2 aggregation (central DP): ~1.0ε per round
- Secure aggregation itself: cryptographic, no DP cost

---

## Statistics

### `get_stats() -> Dict`

Federation health metrics:

```python
{
    "n_clients": 25,
    "n_baskets": 3,
    "n_updates_buffered": 42,
    "n_layer1_aggregations": 10,
    "n_layer2_aggregations": 2,
    "privacy_consumed_epsilon": 3.5,
    "current_round": 15,
    "basket_stats": {
        "healthcare": {"clients": 12, "updates": 20},
        "finance": {"clients": 8, "updates": 15},
        ...
    }
}
```

---

## Internal Workflow

### When `submit_update` is called:

```
1. Validate update (ReplayGuard, participation cap)
2. Store in UpdateBuffer with metadata
3. If auto_aggregate:
   a. Check basket trigger (TriggerEngine)
   b. If triggered: run Layer 1 for that basket
   c. Check global trigger
   d. If triggered: run Layer 2
```

### Layer 1 Aggregation (per basket):

```
1. Collect updates from buffer for this basket
2. Apply Byzantine-robust aggregation (Krum/Bulyan)
3. Produce basket aggregate
4. Store as BasketModel
5. Clear processed updates from buffer
```

### Layer 2 Aggregation (global):

```
1. Collect basket aggregates
2. Weight by basket size or trust score
3. Run secure aggregation (cryptographic)
4. Apply central DP noise
5. Produce GlobalMetaModel
6. Distribute to baskets (optional)
```

---

## Edge Cases

### Not Enough Clients

If a basket has fewer than the minimum required clients:
- Gossip still runs (even 2 clients can gossip)
- Layer 1 waits until threshold met
- Updates stay in buffer

### Privacy Budget Exhausted

When `get_privacy_budget()` returns (0, 0):
- Gossip messages blocked
- Aggregation blocked
- System is read-only until budget renewed

### Client Dropout

If clients go offline during secure aggregation:
- Protocol is dropout-tolerant up to threshold
- If too many drop, aggregation aborts and retries next round

---

## Example: Full Workflow

```python
from scarcity.federation import (
    HierarchicalFederation, 
    HierarchicalFederationConfig
)
import numpy as np

# Initialize
fed = HierarchicalFederation(HierarchicalFederationConfig(
    total_epsilon=10.0,
    vector_dim=64
))

# Register clients
for i in range(10):
    domain = "healthcare" if i < 5 else "finance"
    fed.register_client(f"client_{i}", domain_id=domain)

# Training loop
for round_id in range(100):
    # Each client computes local update
    for i in range(10):
        update = compute_local_update(f"client_{i}")
        fed.submit_update(f"client_{i}", update, round_id=round_id)
    
    # Run gossip within baskets
    fed.run_gossip_round()
    
    # Check for aggregation
    global_model = fed.maybe_aggregate()
    if global_model is not None:
        print(f"Round {round_id}: New global model!")
    
    fed.advance_round()

# Final state
print(fed.get_stats())
```

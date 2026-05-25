# secure_aggregation.py — Cryptographic Secure Aggregation

The secure aggregation module implements **cryptographic protocols** that allow the coordinator to compute the sum of client updates without seeing individual contributions.

---

## Purpose

Even with local DP, the central coordinator sees:
- Basket-level aggregates (potentially identifying)
- Who contributed what weight
- Timing patterns

Secure aggregation hides individual basket contributions:
- Coordinator only sees the **sum**
- No single entity learns individual values
- Dropout-tolerant (works with client failures)

---

## Cryptographic Background

### Secret Sharing

Instead of sending `x` directly:
1. Client generates random mask `r`
2. Sends `x + r` to coordinator
3. Masks cancel out when summed (if designed correctly)

### Diffie-Hellman Masks

Pairs of clients agree on shared secrets:
```
client_A and client_B compute:
  mask_AB = hash(DH_shared_secret_AB)

A sends: x_A + mask_AB
B sends: x_B - mask_AB

Sum: (x_A + mask_AB) + (x_B - mask_AB) = x_A + x_B
```

---

## Key Management

### `IdentityKeyPair`

Long-term identity keys:

```python
identity = IdentityKeyPair.generate()
public_key = identity.public
private_key = identity.private
```

Used to:
- Authenticate clients
- Derive ephemeral keys
- Sign messages

### `EphemeralKeyPair`

Per-round temporary keys:

```python
ephemeral = EphemeralKeyPair.generate()
```

Used to:
- Generate pairwise masks for this round
- Discarded after aggregation
- Limits exposure from key compromise

### `EphemeralKeyRecord`

Links ephemeral keys to clients:

```python
@dataclass
class EphemeralKeyRecord:
    client_id: str
    round_id: int
    ephemeral_public: bytes
    signature: bytes  # Signed by identity key
```

---

## Client Side

### `SecureAggClient`

Client-side protocol implementation:

```python
client = SecureAggClient(
    client_id="hospital_1",
    identity_keypair=identity
)

# Round 1: Key exchange
ephemeral = client.generate_ephemeral(round_id=42)
coordinator.submit_ephemeral(ephemeral)

# Round 2: Receive other clients' keys
peer_keys = coordinator.get_peer_keys(round_id=42)
client.set_peer_keys(peer_keys)

# Round 3: Submit masked update
masked = client.mask_update(my_update, round_id=42)
coordinator.submit_masked(client_id, masked, round_id=42)
```

### Masking Process

```python
def mask_update(self, update, round_id):
    masked = update.copy()
    
    for peer_id, peer_public in self.peer_keys.items():
        # Compute shared mask with this peer
        shared_secret = diffie_hellman(self.ephemeral.private, peer_public)
        mask = derive_mask(shared_secret, round_id, len(update))
        
        # Add or subtract based on ID ordering
        if self.client_id < peer_id:
            masked += mask
        else:
            masked -= mask
    
    return masked
```

---

## Coordinator Side

### `SecureAggCoordinator`

Server-side protocol implementation:

```python
coordinator = SecureAggCoordinator(
    min_clients=5,
    vector_dim=64
)

# Phase 1: Collect ephemeral keys
for client_id, ephemeral in ephemeral_submissions:
    coordinator.submit_ephemeral(client_id, ephemeral)

# Phase 2: Distribute keys to clients
peer_keys = coordinator.get_peer_keys(round_id=42)
# Send to each client

# Phase 3: Collect masked updates
for client_id, masked in update_submissions:
    coordinator.submit_masked(client_id, masked, round_id=42)

# Phase 4: Compute aggregate
aggregate = coordinator.finalize(round_id=42)
```

### `finalize(round_id) -> np.ndarray`

Unmask and sum:

```python
def finalize(self, round_id):
    # All masks cancel out!
    # If client A added mask_AB and client B subtracted mask_AB,
    # the sum is just x_A + x_B
    
    return sum(self.masked_updates.values())
```

---

## Dropout Handling

What if some clients go offline?

### Threshold Scheme

Protocol works if at least t out of n clients complete all phases:

```python
coordinator = SecureAggCoordinator(
    min_clients=5,     # Minimum to proceed
    threshold=0.7      # At least 70% must complete
)
```

### Recovery Keys

Optional: Secret-share recovery keys so survivors can unmask on behalf of dropouts.

---

## Protocol Phases

```
Phase 1: Advertise (key exchange)
├── Each client generates ephemeral keypair
├── Signs with identity key
└── Submits to coordinator

Phase 2: Share (key distribution)
├── Coordinator collects all ephemeral publics
└── Distributes to all clients

Phase 3: Masked Update
├── Each client computes pairwise masks
├── Adds update + sum of masks
└── Submits masked vector

Phase 4: Aggregate
├── Coordinator sums all masked vectors
├── Masks cancel out (by construction)
└── Returns aggregate
```

---

## Security Properties

### What Coordinator Learns

- Sum of all updates ✓
- Number of participants ✓
- Who participated ✓

### What Coordinator DOESN'T Learn

- Individual client updates ✗
- Pairwise differences ✗
- Update ordering (beyond timing) ✗

### Assumptions

- Honest-but-curious coordinator (follows protocol)
- No collusion between coordinator and clients
- At least threshold clients are honest

---

## Integration with Federation

```python
# In Layer2Aggregator.aggregate():

if config.use_secure_aggregation:
    # Clients submit encrypted basket aggregates
    for basket_id, model in basket_models:
        masked = client.mask_update(model.aggregate)
        coordinator.submit_masked(basket_id, masked, round_id)
    
    # Compute secure sum
    aggregate = coordinator.finalize(round_id)
else:
    # Plain weighted average
    aggregate = weighted_average(basket_models)
```

---

## Performance Considerations

### Computation

- Key generation: ~1ms per client
- Masking: O(n × d) where n=clients, d=dimension
- Aggregation: O(n × d)

### Communication

- Phase 1: n × |ephemeral_key| bytes
- Phase 2: n² × |ephemeral_key| bytes (each client gets all keys)
- Phase 3: n × d × sizeof(float) bytes

### Optimization: Sparse Aggregation

For high-dimensional updates with sparsity:
- Only mask non-zero coordinates
- Reduces communication significantly

---

## Edge Cases

### Single Client

Secure aggregation with one client:
- No masks to add/cancel
- Updates trivially revealed
- Falls back to direct submission (or waits for more clients)

### Malicious Clients

Protocol doesn't prevent:
- Clients submitting garbage
- Byzantine attacks on the sum

**Solution**: Combine with Byzantine-robust aggregation at Layer 1.

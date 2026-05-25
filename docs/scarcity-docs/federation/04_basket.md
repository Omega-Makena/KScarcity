# basket.py — Domain Basket Management

The **BasketManager** organizes clients into **domain baskets** — groups of clients with similar data distributions that can share information more freely.

---

## Purpose

Not all clients are alike:
- Hospitals have patient data
- Banks have financial transactions
- Retailers have purchase histories

Grouping by domain:
- **Improves convergence**: Similar data → similar models → faster agreement
- **Reduces noise**: Cross-domain updates might confuse models
- **Enables specialization**: Domain-specific relationships can emerge

---

## Core Concepts

### Domain vs. Basket

- **Domain**: Semantic category (e.g., "healthcare", "finance")
- **Basket**: Operational grouping of clients within a domain

Usually 1:1, but multiple baskets per domain are possible (for very large domains).

### Basket Lifecycle

```
FORMING → ACTIVE → STALE → DISBANDED
```

- **FORMING**: Not enough clients yet
- **ACTIVE**: Normal operation
- **STALE**: No recent activity
- **DISBANDED**: Closed, clients reassigned

---

## Data Structures

### `BasketConfig`

```python
@dataclass
class BasketConfig:
    min_clients: int = 3        # Minimum to start gossip
    max_clients: int = 100      # Split if exceeded
    stale_threshold: float = 600.0  # Seconds of inactivity
    allow_cross_domain: bool = False  # Allow multi-domain baskets
```

### `ClientInfo`

```python
@dataclass
class ClientInfo:
    client_id: str
    domain_id: str
    basket_id: Optional[str]
    features: Optional[np.ndarray]  # For fingerprinting
    joined_at: float
    last_active: float
    n_updates: int
    trust_score: float
```

### `BasketInfo`

```python
@dataclass
class BasketInfo:
    basket_id: str
    domain_id: str
    status: BasketStatus
    clients: Set[str]
    created_at: float
    last_activity: float
```

### `BasketStatus` (Enum)

```python
class BasketStatus(Enum):
    FORMING = "forming"
    ACTIVE = "active"
    STALE = "stale"
    DISBANDED = "disbanded"
```

---

## Class: `BasketManager`

### Initialization

```python
manager = BasketManager(config=BasketConfig(
    min_clients=3,
    max_clients=50
))
```

### `add_client(client_id, domain_id, features=None)`

Register a new client:

```python
basket_id = manager.add_client(
    client_id="hospital_1",
    domain_id="healthcare",
    features=np.array([...])  # Optional fingerprint
)
```

**Assignment logic**:
1. Look for existing basket in domain
2. If found and not full: assign
3. If full: create new basket
4. If no basket: create first basket for domain

Returns: Assigned basket_id

### `remove_client(client_id) -> bool`

Unregister a client:

```python
manager.remove_client("hospital_1")
```

If basket becomes empty, it's marked for cleanup.

### `get_basket(basket_id) -> Optional[BasketInfo]`

Get basket details:

```python
info = manager.get_basket("healthcare_basket_0")
print(f"Clients: {info.clients}, Status: {info.status}")
```

### `get_client_basket(client_id) -> Optional[str]`

Look up which basket a client belongs to:

```python
basket_id = manager.get_client_basket("hospital_1")
```

### `get_basket_clients(basket_id) -> Set[str]`

Get all clients in a basket:

```python
clients = manager.get_basket_clients("healthcare_basket_0")
```

---

## Fingerprinting (Optional)

Clients can provide a **feature vector** for grouping:

```python
manager.add_client(
    "hospital_1",
    domain_id="healthcare",
    features=np.array([1.0, 0.5, 0.3])  # E.g., [size, specialty, region]
)
```

Future enhancement: Use features to cluster clients into optimal baskets (beyond just domain).

---

## Activity Tracking

### `touch_client(client_id)`

Mark client as active:

```python
manager.touch_client("hospital_1")
```

Called when client submits an update.

### `touch_basket(basket_id)`

Mark basket as active:

```python
manager.touch_basket("healthcare_basket_0")
```

Called during gossip or aggregation.

### Staleness Detection

```python
stale_baskets = manager.get_stale_baskets(threshold=600.0)
```

Returns baskets with no activity in the last 600 seconds.

---

## Basket Operations

### `create_basket(domain_id) -> str`

Manually create a basket:

```python
basket_id = manager.create_basket("new_domain")
```

### `disband_basket(basket_id)`

Close a basket:

```python
manager.disband_basket("old_basket")
```

Clients are reassigned to other baskets in the same domain.

### `split_basket(basket_id) -> Tuple[str, str]`

Split an oversized basket:

```python
new_basket_1, new_basket_2 = manager.split_basket("huge_basket")
```

Clients are divided (randomly or by features) between the two new baskets.

---

## Statistics

### `get_stats() -> Dict`

```python
{
    "n_clients": 50,
    "n_baskets": 5,
    "baskets_by_domain": {
        "healthcare": 2,
        "finance": 2,
        "retail": 1
    },
    "avg_clients_per_basket": 10,
    "n_stale_baskets": 0
}
```

---

## Edge Cases

### Domain with No Clients

Empty domain has no baskets. First client creates the first basket.

### Single-Client Basket

Valid but limited:
- Gossip is a no-op
- Layer 1 aggregation waits for more clients
- Client still participates in Layer 2 (via its basket)

### Client Joins/Leaves During Round

- **Join**: Starts participating in next gossip round
- **Leave**: Pending messages are dropped, basket continues

---

## Integration Points

- **`HierarchicalFederation`**: Uses BasketManager to route updates
- **`GossipProtocol`**: Queries basket membership for peer sampling
- **`Layer1Aggregator`**: Aggregates per-basket

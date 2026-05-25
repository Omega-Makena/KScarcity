# DomainServer — Per-Domain Logical Meta Agent

`scarcity/federation/domain_server.py`

---

## Purpose

`DomainServer` is the per-domain stateful agent in the federated meta-learning pipeline. Each domain (identified by a `domain_id` and `basket_id` pair) owns one `DomainServer` that:

- Maintains a **REPTILE-style base parameter vector** for that domain
- Stores an **episodic memory** of past adaptation episodes
- Supplies **prior parameter suggestions** to new contexts based on retrieved episodes
- Tracks a **hit rate** (how often retrieved priors prove useful) and decays it each round

`DomainServerRegistry` manages a collection of `DomainServer` instances keyed by basket, providing a `get_or_create` factory so callers do not need to manage lifecycle directly.

---

## Configuration

### `DomainServerConfig`

| Param | Default | Purpose |
|-------|---------|---------|
| `memory_capacity` | `64` | Maximum number of episodes stored in episodic memory |
| `min_episodes_for_prior` | `4` | Minimum episodes required before `suggest_prior` returns a result |
| `reptile_lr` | `0.1` | REPTILE learning rate for `update_base_params` |
| `hit_decay` | `0.9` | Multiplicative decay applied to `hit_rate` each round |
| `max_base_params` | `128` | Maximum number of keys tracked in `base_params` |

---

## API

### `DomainServer(domain_id, basket_id, config=None)`

Creates a new domain server for the given domain and basket.

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain_id` | `str` | Unique identifier for the domain |
| `basket_id` | `str` | Basket this domain belongs to |
| `config` | `DomainServerConfig` | Optional config; defaults to `DomainServerConfig()` |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `base_params` | `Dict[str, float]` | Current REPTILE base parameter vector |
| `hit_rate` | `float` | Proportion of rounds where retrieved prior was useful |
| `memory_size` | `int` | Number of episodes currently stored |
| `round_id` | `int` | Monotonically increasing round counter |

---

### `.update_base_params(vec, keys, reward)`

Applies a REPTILE-style update to `base_params` using the supplied delta vector.

```python
server.update_base_params(
    vec=np.array([0.05, -0.02, 0.01]),
    keys=["tau", "g_min", "gamma_diversity"],
    reward=0.75
)
```

The update rule is:

```
base_params[key] += reptile_lr * vec[i]
```

Reward is recorded alongside the update for downstream aggregation.

---

### `.adapt(context, prior_params) → Dict`

Retrieves adapted parameters for a given context, starting from `prior_params`. Returns the adapted parameter dictionary. If no prior is available for this context, `prior_params` is returned unchanged.

```python
adapted = server.adapt(
    context={"sector": "healthcare", "round": 3},
    prior_params={"tau": 0.9, "g_min": 0.01}
)
```

---

### `.record(context, prior_params, adapted_params, delta)`

Stores an adaptation episode in episodic memory. Episodes record the context, the before/after parameter states, and the delta for future retrieval.

```python
server.record(
    context={"sector": "healthcare", "round": 3},
    prior_params={"tau": 0.9},
    adapted_params={"tau": 0.92},
    delta=0.02
)
```

When `memory_capacity` is reached, the oldest episode is evicted.

---

### `.suggest_prior(context) → Optional[Dict]`

Queries episodic memory for the best-matching prior for `context`. Returns `None` if fewer than `min_episodes_for_prior` episodes are stored. Otherwise returns the parameter dict from the closest matching episode.

```python
prior = server.suggest_prior({"sector": "healthcare", "round": 5})
if prior:
    adapted = server.adapt(context, prior)
```

---

## DomainServerRegistry

A basket-keyed dictionary of `DomainServer` instances.

### `DomainServerRegistry.get_or_create(domain_id, basket_id, config=None) → DomainServer`

Returns an existing server for `(domain_id, basket_id)` or creates one if none exists.

```python
registry = DomainServerRegistry()
server = registry.get_or_create("healthcare", "basket_africa")
```

Iterating over a registry yields all `DomainServer` instances:

```python
for server in registry:
    print(server.domain_id, server.hit_rate)
```

---

## Usage Example

```python
from scarcity.federation.domain_server import (
    DomainServer, DomainServerConfig, DomainServerRegistry
)
import numpy as np

# Create registry and servers
registry = DomainServerRegistry()
server_a = registry.get_or_create("healthcare", "basket_east_africa")
server_b = registry.get_or_create("finance", "basket_east_africa")

# Update base params after a federation round
server_a.update_base_params(
    vec=np.array([0.03, -0.01]),
    keys=["tau", "g_min"],
    reward=0.8
)

# Store an episode
ctx = {"sector": "healthcare", "n_clients": 5}
server_a.record(
    context=ctx,
    prior_params={"tau": 0.9, "g_min": 0.01},
    adapted_params={"tau": 0.93, "g_min": 0.009},
    delta=0.05
)

# Retrieve a prior for a new context
prior = server_a.suggest_prior({"sector": "healthcare", "n_clients": 6})
print(prior)  # {"tau": 0.93, "g_min": 0.009}
```

---

## Notes

- `hit_rate` decays by `hit_decay` each round regardless of usage. A high sustained `hit_rate` indicates that the server's episodic memory is producing useful priors.
- `base_params` is bounded by `max_base_params`. Keys beyond the limit are ignored.
- `DomainServer` is imported by `DomainServerMeta` via **duck typing** — no reverse import is required. Any object exposing `base_params`, `hit_rate`, `memory_size`, and `domain_id` can be observed by `DomainServerMeta`.

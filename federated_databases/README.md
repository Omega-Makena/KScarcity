# Federated Databases (Scarcity)

`federated_databases/` is a first-class federation module that pairs:

- per-node database isolation (`org_a`, `org_b`, ...), and
- ML synchronization rounds (single-node and federated modes)

for the Sentinel + Scarcity stack.

## What it provides

- Node registration and storage backends (`SQLite` today, backend field is extensible).
- Per-node local sample stores, model updates, and shared signal tables.
- Control-plane persistence for:
  - sync rounds,
  - model/signal exchange audit logs,
  - global model state.
- Scarcity ML pipeline modes:
  - `single_node`: local training on one node,
  - `federated`: node-local training + weighted aggregation + broadcast.

## Layout

- `models.py`: federation dataclasses.
- `storage.py`: `NodeStorage` and `ControlPlaneStorage` (SQLite schema + IO).
- `scarcity_federation.py`: `ScarcityFederationManager` orchestration.
- `pipeline.py`: `ScarcityMLPipeline` mode wrapper.

Runtime artifacts are written to:

- `federated_databases/runtime/federation_control.sqlite`
- `federated_databases/runtime/nodes/*.sqlite`
- `federated_databases/runtime/audit_log.jsonl`

## Quick usage

```python
from federated_databases import get_scarcity_federation, ScarcityMLPipeline

manager = get_scarcity_federation()
manager.register_node("org_a", county_filter="Nairobi")
manager.register_node("org_b", county_filter="Mombasa")

pipeline = ScarcityMLPipeline(manager)
pipeline.run(mode="single_node", node_id="org_a")
pipeline.run(mode="federated")

print(manager.get_status())
```

## Data flow

1. Load latest live synthetic policy rows from `data/synthetic_kenya_policy/tweets.csv`.
2. Partition rows to node-local databases.
3. Train local updates (`criticality -> binary label`) on each node.
4. Aggregate updates (`weighted_fedavg`) and broadcast global weights.
5. Log full exchange and round metadata for auditability.

## Notes

- The module is designed for realistic orchestration boundaries (node isolation + control plane) while remaining lightweight.
- UI integration for registration, sync triggering, metrics, and audit visualization is in `kshiked/ui/sentinel/federation.py`.

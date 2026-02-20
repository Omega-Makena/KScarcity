# Federated Data Access Layer (package: `federated_databases`)

`federated_databases/` is a first-class federation module that pairs:

- per-node database isolation (`org_a`, `org_b`, ...), and
- ML synchronization rounds (single-node and federated modes)

for the Sentinel + Scarcity stack.

It should be treated as a **Federated Data Access / Data Virtualization / Federated Execution Layer**:
- It sits above institutional systems (Oracle/Postgres/SQL Server/files/APIs).
- It does not replace those systems.
- It orchestrates validated federated analytics and policy-governed outputs.

## Non-goals

- Building a new database engine
- Replacing institutional warehouses/lakes
- Central ETL replication of raw institutional data

## What it provides

- Node registration and storage backends (`SQLite` today, backend field is extensible).
- Adapter-based connectors (`SQLite`, `Postgres`, and stubs for Oracle/SQL Server/Azure/API).
- Project-scoped governance (participants, allowed datasets/computations, purpose gating).
- Canonical schema mappings + data quality metadata for virtualized query fields.
- Compatibility scoring + automatic basket formation for heterogeneous participation.
- Per-node local sample stores, model updates, and shared signal tables.
- Control-plane persistence for:
  - sync rounds,
  - model/signal exchange audit logs,
  - global model state.
- Federated query control plane:
  - parse -> policy+contract validation -> plan -> pushdown -> execute -> suppression -> audit.
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

## Compatibility and baskets

K-Collab assumes heterogeneity by default and computes a per-node compatibility score `[0,1]`:

- `schema`: 0.30
- `temporal`: 0.15
- `statistical`: 0.20
- `quality`: 0.15
- `policy`: 0.10
- `operational`: 0.10

Basket tiers:

- `>= 0.75`: full compatibility basket
- `0.50 - 0.75`: partial participation basket
- `< 0.50`: excluded (with explicit reason)

Query execution is basket-aware: plan per basket -> local pushdown -> merge -> k-suppression -> provenance.

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

## Coordination workflow (registration -> analytics -> FL readiness)

1. Register/connect external sources and validate connector health.
2. Publish datasets with contracts + canonical mappings + quality metadata.
3. Create collaboration project (participants, datasets, governance).
4. Run compatibility analysis and form baskets.
5. Execute federated analytics query with basket-aware planner.
6. Validate FL readiness from basket outputs (no central raw data movement).

## Hard-problem split (DB vs FL)

The platform now reports two separate hard-problem tracks:

- DB/data-access side (federated query execution)
- FL side (federated training orchestration)

Problem #7 is explicitly non-IID on both sides:

- DB side: institution/group contribution skew
- FL side: client update heterogeneity/drift

## Architecture (text diagram)

```text
Oracle / Postgres / SQLServer / BigQuery / Lake / API
                |         |           |       |
                +---------+-----------+-------+
                              |
                   K-Collab Federated Data Access Layer
           (contracts + policy + compatibility + planner + executor + audit)
                              |
                +-------------+----------------+
                |                              |
        Federated Analytics Outputs      Federated ML Readiness Input
       (aggregated, suppressed,          (baskets + governance + privacy
        provenance-rich)                  budget constraints)
```

## Notes

- The module is designed for realistic orchestration boundaries (node isolation + control plane) while remaining lightweight.
- UI integration for registration, sync triggering, metrics, and audit visualization is in `kshiked/ui/sentinel/federation.py`.

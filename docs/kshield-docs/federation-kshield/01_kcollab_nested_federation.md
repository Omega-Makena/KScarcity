# K-Collab Nested Federation (Federated Data Access + Federated ML)

## Summary
K-Collab now provides a nested federation collaboration layer across both:
1. Federated Data Access / Virtualization (virtual query control plane)
2. Federated ML (nested orchestration over existing scarcity hierarchical federation)

Implementation is reuse-first:
- Reused as-is: `federated_databases/scarcity_federation.py`, `scarcity/federation/*`, `kshiked/ui/sentinel/federation.py` route
- Added adapters/glue only for topology versioning, policy/contracts/planner/executor, and ML job orchestration

## New Modules

### `k_collab/`
- `topology/`: schema validation, text graph preview, version history, topology diff
- `audit/`: append-only JSONL audit log with chain hash
- `common/`: versioned JSONL store
- `ui/`: service bootstrap shared by Streamlit

### `federated_databases/`
- `catalog/`: connector + dataset placement registry
- `contracts/`: dataset contracts (schema/classification/PII/allowed ops)
- `policy/`: ABAC/RBAC hybrid policy evaluator
- `planner/`: query parser + routing plan
- `executor/`: node execution + k-threshold suppression
- `connectors/`: SQLite node connector
- `control_plane.py`: planner -> policy -> contract -> execute -> suppress -> audit

This package name is retained for compatibility, but architecturally it functions as a federated data-access layer above external systems.

### `federated_ml/`
- `topology_adapter/`: topology -> participant/domain mapping
- `orchestration/`: nested job start/round/complete wrapper over `HierarchicalFederation`
- `registry/`: versioned model artifact registry
- `audit_hooks/`: ML lifecycle + compliance log hooks

## Sentinel UI Integration
Updated `kshiked/ui/sentinel/federation.py` now has 4 tabs:
1. Topology Builder
2. Federated Data Access (connectors/contracts/policy/query/audit)
3. Federated ML (job builder/run console/model registry)
4. Legacy Control (existing federation manager controls)

## Governance Checks Enforced
- Clearance gating: user clearance vs dataset classification
- Purpose gating: purpose tag must be role-authorized
- Deny operations: forbidden operations rejected
- Cross-agency guardrail: aggregate-only behavior
- Suppression: k-threshold group suppression in executor
- Audit logging: append-only event log with plan metadata + suppression count

## How To Run
1. Launch Streamlit dashboard and navigate to `FEDERATION`.
2. In `Topology Builder`, save/update topology JSON.
3. In `Federated Data Access`, sync connectors, then run safe SQL/JSON-DSL query.
4. In `Federated ML`, start job, run rounds, complete to register model.

## Architectural Guardrail
- K-Collab does not build new databases, warehouses, lakes, or centralized ETL copies.
- Data stays at institutions; computation is federated to local connectors.

## Tests
Added `tests/test_kcollab_federated_db_control_plane.py` covering:
- contract enforcement
- policy evaluation decisions
- planner routing correctness
- suppression logic

Run:

```bash
python3 -m pytest tests/test_federated_databases_smoke.py tests/test_kcollab_federated_db_control_plane.py -q
```

## Next Hard Problems
- Key management / KMS-backed secret distribution
- Remote attestation for connectors and training clients
- mTLS + channel-level policy enforcement
- Real secure aggregation protocol (not in-process simulation)
- Differential privacy accountant with budget exhaustion policy
- TEE-backed query execution for high-classification domains
- Non-IID data handling across both planes: FL client drift/skew and federated DB statistical heterogeneity/biased aggregates

# K-Collab Hard-Problem Foundations (Phase 1)

This phase starts implementation of the seven hard problems while keeping K-Collab as a **Federated Execution Layer** (not a database/lake/warehouse).

## Architecture Positioning

K-Collab sits above existing systems:

- Oracle / Postgres / SQL Server / BigQuery / data lakes / files / APIs
- Provides federated query planning + policy/contract enforcement + suppression + audit
- Optionally orchestrates federated ML using existing FL logic

## Implemented in This Phase

### 1) Collaboration Project Construct
- Added versioned project registry:
  - `k_collab/projects/registry.py`
- Project defines:
  - participants
  - objective
  - allowed datasets/domains
  - allowed computations (`analytics`, `federated_ml`)
  - governance rules (purpose allowlist, thresholds)
- DB query execution now enforces project allowlists.

### 2) Heterogeneous Connector Adapters
- Added adapter factory + connectors:
  - Real: `federated_databases/connectors/postgres_node.py` (uses `psycopg` if installed)
  - Existing: SQLite
  - Stubs: Oracle, SQL Server, Azure, HTTP API
- Control plane remains source-agnostic via adapter factory.

### 3) Canonical Schema + Data Quality Metadata
- Added canonical mapping registry:
  - `federated_databases/contracts/mapping.py`
- Supports canonical-to-local field mapping and quality metadata.
- Query execution now applies canonical mapping before pushdown.

### 4) Query Studio + Plan Viewer Enhancements
- Executor now returns execution trace per node:
  - pushed-down group/filter/metric details
  - returned row counts
- Sentinel DB pane now displays plan viewer trace.

### 5) Audit/Lineage Expansion
- Query audit now includes:
  - topology/policy/contract/canonical/project versions
  - suppression counts
  - non-IID diagnostics
  - data quality summary

### 6) Reuse Existing FL Logic (No Reimplementation)
- Kept FL core in `scarcity.federation.*`.
- Extended orchestrator wrapper only:
  - project gating
  - minimum remaining epsilon guard
  - non-IID update diagnostics

### 7) Non-IID Foundations (Both Data-Access + FL)
- Federated data-access side:
  - per-group node contribution skew diagnostics
  - `federated_databases/executor/non_iid.py`
- FL side:
  - update heterogeneity metrics (norm CV + cosine distance)
  - `federated_ml/orchestration/non_iid.py`

## Decentralization Mapping

- Federated DB: decentralized data access
- Federated ML: decentralized learning
- Collaboration layer: decentralized coordination
- Audit/policy: decentralized trust

## Not Built (By Design)

K-Collab does **not** replace institutional infrastructure.
It orchestrates it.

## Validation

Run:

```bash
python3 -m pytest \
  tests/test_federated_databases_smoke.py \
  tests/test_kcollab_federated_db_control_plane.py \
  tests/test_kcollab_non_iid.py -q
```

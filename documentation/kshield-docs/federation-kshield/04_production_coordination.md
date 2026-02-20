# K-Collab Production Coordination (Federated Data Access)

## Scope

K-Collab is a federated data access/execution layer above existing institutional systems.
It does not replace databases, lakes, or warehouses.

## Heterogeneity-first execution model

1. Connector registration validates source trust and channel metadata.
2. Dataset contracts define schema/classification/allowed operations.
3. Canonical mappings normalize virtual field names and quality metadata.
4. Compatibility scoring runs per node and forms baskets:
   - `>=0.75`: full basket
   - `0.50-0.75`: partial basket
   - `<0.50`: excluded with reasoning
5. Planner executes per basket with local pushdown.
6. Basket outputs are merged and privacy-suppressed (`k-threshold`).
7. Provenance and coverage metadata are returned and audit-logged.

## Compatibility score components

- Schema compatibility: 0.30
- Temporal compatibility: 0.15
- Statistical compatibility: 0.20
- Data quality compatibility: 0.15
- Policy compatibility: 0.10
- Operational capability: 0.10

## Walkthrough (UI/CLI equivalent)

1. Organization registration and connector health check.
2. Dataset publishing check (contract + mapping + quality metadata).
3. Project creation/update with governance constraints.
4. Compatibility analysis and basket map generation.
5. Federated analytics run with plan viewer and provenance.
6. Federated ML readiness validation from basket outputs (no training).

## Readiness quality gates

- Heterogeneous sources connect.
- Compatibility scoring runs.
- Baskets form automatically.
- Federated analytics executes successfully.
- Provenance metadata is visible.
- Excluded nodes/institutions include explicit reasons.
- Walkthrough completes without manual fixes.

For DB-vs-FL hard-problem separation (with non-IID as problem #7 on both sides), see `05_hard_problems_split.md`.

## Architecture diagram (text)

```text
Institutional Systems (Oracle/Postgres/SQLServer/BigQuery/Lake/API)
                               |
                               v
                K-Collab Federated Data Access Layer
   (contracts + policy + compatibility + planner + executor + audit)
                               |
                  +------------+-------------+
                  |                          |
        Federated Analytics Output    FL Readiness Input
        (aggregate-only, suppressed,  (baskets + governance +
         provenance + coverage)        privacy constraints)
```

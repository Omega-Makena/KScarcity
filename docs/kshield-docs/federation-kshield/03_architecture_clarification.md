# Architecture Clarification: K-Collab as Federated Data Access Layer

## What K-Collab Is
K-Collab is a **Federated Data Access / Data Virtualization / Federated Query Execution** layer.

It sits above heterogeneous institutional systems and orchestrates policy-governed computation.

## What K-Collab Is Not
K-Collab is **not**:
- a new database engine
- a data warehouse replacement
- a data lake replacement
- a centralized ETL pipeline for raw institutional data
- a centralized schema authority that replaces institutional ownership

## Core Principle
- Data stays where it lives.
- Computation travels to the data.
- Only approved, aggregated/derived outputs are returned.

## Scope of the Layer
1. Register external connectors/adapters.
2. Enforce data contracts and policy constraints.
3. Build a logical virtual view for planning.
4. Plan and execute federated analytics/feature extraction.
5. Optionally orchestrate federated ML over existing FL logic.
6. Enforce suppression, purpose-of-use, clearance, and auditing.

## Terminology
Use:
- Federated Data Access
- Data Federation
- Data Virtualization Layer
- Federated Query Execution

Avoid:
- Distributed database
- New database system
- Centralized data platform

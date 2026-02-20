# Hard Problems Split: DB Side vs FL Side

## Why split

K-Collab is one collaboration layer, but production risks are different across:

- federated DB/data-access execution, and
- federated ML orchestration.

The platform now reports both tracks separately while keeping one governance and audit plane.

## DB-side hard problems (federated data access)

1. Heterogeneous source connectivity
2. Schema/contract/canonical alignment
3. Compatibility basket formation and partial participation
4. Planner routing and pushdown fidelity
5. Privacy suppression (k-threshold and safe output)
6. Provenance/coverage/exclusion explainability
7. Non-IID representational skew visibility across institutions

## FL-side hard problems (federated learning)

1. Topology-participant alignment
2. Basket participation viability
3. Update/gradient drift handling
4. Privacy budget compatibility and guards
5. Secure aggregation posture
6. Model registry traceability
7. Non-IID update heterogeneity visibility and mitigation readiness

## Important alignment

Hard problem **#7 is non-IID on both sides**:

- DB side: skew in contribution/coverage across institutions and groups.
- FL side: skew/divergence in client updates during co-learning.

Both are surfaced in reports and should be handled by governance before relying on outputs.

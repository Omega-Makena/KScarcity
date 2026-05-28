"""Phase 1 — Synthetic ground truth data generator.

Known causal graph with 10 variables and 6 relationship types.
Generating equations are fixed; do not modify them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ground_truth(N: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data from a known ground truth graph.

    Ground truth relationships:
      1. V1 → V2          Linear causal:          V2 = 0.7·V1 + ε(0, 0.3)
      2. V1 → V3          Linear causal chain:     V3 = 0.5·V1 + ε(0, 0.3)
      3. V3 → V4          Causal (chain):          V4 = 0.6·V3 + 0.3·V1·V5 + ε(0, 0.3)
      4. V5 ← L1 → V6    Confounded correlation:  V5 = L1 + ε; V6 = 0.8·L1 + ε
      5. V7               Equilibrium (OU):        dV7 = 0.5·(2.0 − V7)dt + 0.3·dW
      6. V8 + V9 ≈ const  Competitive (zero-sum):  V9 = 5.0 − V8 + ε(0, 0.2)
      7. V5+V6+V10 ≈ 1    Compositional:           V10 = 1.0 − V5 − V6 + ε(0, 0.05)
      8. V1·V5 → V4       Synergistic interaction:  (already in V4 equation above)

    Null relationships (must NOT be discovered):
      V8 ⊥ V2, V7 ⊥ V1, V9 ⊥ V3, V7 ⊥ V5, V7 ⊥ V8, V2 ⊥ V6, V2 ⊥ V10
    """
    rng = np.random.default_rng(seed)

    # Latent confounder — NOT included in output DataFrame
    L1 = rng.normal(0, 1, N)

    # Root exogenous variable
    V1 = rng.normal(0, 1, N)

    # Linear causal: V1 → V2
    V2 = 0.7 * V1 + rng.normal(0, 0.3, N)

    # Linear causal chain: V1 → V3
    V3 = 0.5 * V1 + rng.normal(0, 0.3, N)

    # Confounded pair: L1 → V5, L1 → V6
    V5 = L1 + rng.normal(0, 0.3, N)
    V6 = 0.8 * L1 + rng.normal(0, 0.3, N)

    # Compositional: V10 = 1 − V5 − V6 + noise
    V10 = 1.0 - V5 - V6 + rng.normal(0, 0.05, N)

    # Synergistic causal: V4 depends on V3 (linear) AND V1*V5 (interaction)
    V4 = 0.6 * V3 + 0.3 * V1 * V5 + rng.normal(0, 0.3, N)

    # Equilibrium: Ornstein-Uhlenbeck process for V7
    V7 = np.zeros(N)
    theta, mu, sigma_ou = 0.5, 2.0, 0.3
    V7[0] = mu  # start at equilibrium
    for t in range(1, N):
        V7[t] = V7[t - 1] + theta * (mu - V7[t - 1]) + sigma_ou * rng.normal()

    # Competitive: V8 + V9 ≈ 5.0
    V8 = rng.normal(0, 1, N)
    V9 = 5.0 - V8 + rng.normal(0, 0.2, N)

    return pd.DataFrame({
        'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5,
        'V6': V6, 'V7': V7, 'V8': V8, 'V9': V9, 'V10': V10,
    })


def get_ground_truth_edges() -> list[dict]:
    """Return the complete ground truth edge list with relationship types."""
    return [
        {
            'source': 'V1', 'target': 'V2',
            'type': 'causal', 'directed': True,
            'description': 'Linear causal: V2 = 0.7*V1 + noise',
        },
        {
            'source': 'V1', 'target': 'V3',
            'type': 'causal', 'directed': True,
            'description': 'Linear causal chain: V3 = 0.5*V1 + noise',
        },
        {
            'source': 'V3', 'target': 'V4',
            'type': 'causal', 'directed': True,
            'description': 'Linear causal chain: V4 = 0.6*V3 + interaction + noise',
        },
        {
            'source': 'V1', 'target': 'V4',
            'type': 'synergistic', 'directed': True,
            'description': 'Interaction effect: V1*V5 contributes to V4',
        },
        {
            'source': 'V5', 'target': 'V4',
            'type': 'synergistic', 'directed': True,
            'description': 'Interaction effect: V1*V5 contributes to V4',
        },
        {
            'source': 'V5', 'target': 'V6',
            'type': 'correlational', 'directed': False,
            'description': 'Confounded by latent L1 — not truly causal',
        },
        {
            'source': 'V7', 'target': 'V7',
            'type': 'equilibrium', 'directed': False,
            'description': 'Mean-reverting OU process, equilibrium at mu=2.0',
        },
        {
            'source': 'V8', 'target': 'V9',
            'type': 'competitive', 'directed': False,
            'description': 'Zero-sum: V8 + V9 ≈ 5.0',
        },
        {
            'source': 'V5', 'target': 'V10',
            'type': 'compositional', 'directed': False,
            'description': 'Sum constraint: V5 + V6 + V10 ≈ 1.0',
        },
        {
            'source': 'V6', 'target': 'V10',
            'type': 'compositional', 'directed': False,
            'description': 'Sum constraint: V5 + V6 + V10 ≈ 1.0',
        },
        {
            'source': 'V5', 'target': 'V6',
            'type': 'compositional', 'directed': False,
            'description': 'Both part of sum constraint with V10',
        },
        {
            'source': 'V1', 'target': 'V3',
            'type': 'mediating', 'directed': True,
            'description': 'V1 → V3 → V4: V3 mediates effect of V1 on V4',
        },
    ]


def get_known_null_pairs() -> list[tuple[str, str]]:
    """Return variable pairs known to have NO relationship (false positive test)."""
    return [
        ('V8', 'V2'),   # competitive subsystem vs causal chain
        ('V7', 'V1'),   # independent OU vs root
        ('V9', 'V3'),   # competitive subsystem vs causal chain
        ('V7', 'V5'),   # independent OU vs confounded subsystem
        ('V7', 'V8'),   # independent OU vs competitive subsystem
        ('V2', 'V6'),   # causal chain vs confounded subsystem
        ('V2', 'V10'),  # causal chain vs compositional
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df = generate_ground_truth(N=100, seed=42)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDescriptive stats:\n{df.describe().round(3)}")
    print(f"\nCorrelation matrix:\n{df.corr().round(3)}")
    print(f"\nGround truth edges: {len(get_ground_truth_edges())}")
    print(f"Known null pairs: {len(get_known_null_pairs())}")

    assert df.shape[1] == 10, "Must have exactly 10 variables"
    assert not df.isnull().any().any(), "No NaN values allowed"
    assert abs(df['V8'] + df['V9'] - 5.0).mean() < 0.5, "V8+V9 should approximate 5.0"
    assert abs(df['V5'] + df['V6'] + df['V10'] - 1.0).mean() < 0.5, "Sum constraint check"
    print("\nAll sanity checks passed.")

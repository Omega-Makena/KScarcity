"""
Synthetic Data Generators for Testing 15 Relationship Types

Each generator creates data that exhibits a specific relationship type,
used to validate that scarcity can detect each type correctly.
"""

import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class SyntheticDataset:
    """Container for synthetic test data."""
    data: Dict[str, np.ndarray]
    ground_truth: str
    relationship_type: str
    expected_pairs: list  # [(source, target), ...]


def generate_causal(n: int = 200, lag: int = 2, strength: float = 0.8, 
                    noise: float = 0.1, seed: int = 42) -> SyntheticDataset:
    """
    1. CAUSAL: X causes Y with a time lag.
    
    X_t -> Y_{t+lag}
    
    Granger causality should detect X -> Y but not Y -> X.
    """
    rng = np.random.default_rng(seed)
    
    X = np.cumsum(rng.standard_normal(n))  # Random walk
    Y = np.zeros(n)
    Y[lag:] = strength * X[:-lag] + noise * rng.standard_normal(n - lag)
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y},
        ground_truth='X causes Y with lag',
        relationship_type='CAUSAL',
        expected_pairs=[('X', 'Y')]
    )


def generate_correlational(n: int = 200, noise: float = 0.2, 
                           seed: int = 42) -> SyntheticDataset:
    """
    2. CORRELATIONAL: X and Y both caused by hidden Z.
    
    Z -> X, Z -> Y (spurious correlation)
    
    Should detect correlation but NOT causation.
    """
    rng = np.random.default_rng(seed)
    
    Z = np.cumsum(rng.standard_normal(n))  # Hidden confounder
    X = 0.9 * Z + noise * rng.standard_normal(n)
    Y = 0.9 * Z + noise * rng.standard_normal(n)
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y},  # Z is hidden
        ground_truth='X and Y correlated via hidden Z',
        relationship_type='CORRELATIONAL',
        expected_pairs=[('X', 'Y')]  # Correlation, not causation
    )


def generate_structural(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    3. STRUCTURAL: Nested/hierarchical data.
    
    Level 2 (schools) -> Level 1 (classrooms) -> observations
    
    Mixed-effects structure should be detected.
    """
    rng = np.random.default_rng(seed)
    
    n_groups = 5
    obs_per_group = n // n_groups
    
    group_effects = rng.standard_normal(n_groups) * 2
    
    Group = np.repeat(np.arange(n_groups), obs_per_group)
    Within = rng.standard_normal(n_groups * obs_per_group)
    Y = group_effects[Group] + Within
    
    return SyntheticDataset(
        data={'Group': Group.astype(float), 'Within': Within, 'Y': Y},
        ground_truth='Y has hierarchical structure by Group',
        relationship_type='STRUCTURAL',
        expected_pairs=[('Group', 'Y')]
    )


def generate_temporal(n: int = 200, ar_order: int = 3, 
                      seed: int = 42) -> SyntheticDataset:
    """
    4. TEMPORAL: Autoregressive process.
    
    Y_t = sum(phi_i * Y_{t-i}) + noise
    
    Should detect self-lag relationships.
    """
    rng = np.random.default_rng(seed)
    
    phi = [0.5, 0.3, 0.1][:ar_order]
    Y = np.zeros(n)
    Y[:ar_order] = rng.standard_normal(ar_order)
    
    for t in range(ar_order, n):
        Y[t] = sum(phi[i] * Y[t-i-1] for i in range(ar_order))
        Y[t] += 0.1 * rng.standard_normal()
    
    return SyntheticDataset(
        data={'Y': Y},
        ground_truth=f'Y is AR({ar_order})',
        relationship_type='TEMPORAL',
        expected_pairs=[('Y', 'Y')]  # Self-lag
    )


def generate_functional(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    5. FUNCTIONAL: Exact mathematical relationship.
    
    Y = X^2 + 2*X + 3 (deterministic)
    
    Should detect exact functional form.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.uniform(-5, 5, n)
    Y = X**2 + 2*X + 3  # Exact, no noise
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y},
        ground_truth='Y = X^2 + 2X + 3',
        relationship_type='FUNCTIONAL',
        expected_pairs=[('X', 'Y')]
    )


def generate_probabilistic(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    6. PROBABILISTIC: X shifts the distribution of Y.
    
    P(Y | X=0) ~ N(0, 1), P(Y | X=1) ~ N(2, 1)
    
    Should detect distribution shift.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.choice([0, 1], size=n)
    Y = np.where(X == 0, 
                 rng.normal(0, 1, n),
                 rng.normal(2, 1, n))
    
    return SyntheticDataset(
        data={'X': X.astype(float), 'Y': Y},
        ground_truth='X shifts mean of Y by 2',
        relationship_type='PROBABILISTIC',
        expected_pairs=[('X', 'Y')]
    )


def generate_compositional(n: int = 200, noise: float = 0.0, 
                           seed: int = 42) -> SyntheticDataset:
    """
    7. COMPOSITIONAL: Parts sum to whole.
    
    Total = A + B + C (accounting identity)
    
    Should detect sum constraint.
    """
    rng = np.random.default_rng(seed)
    
    A = rng.uniform(0, 100, n)
    B = rng.uniform(0, 100, n)
    C = rng.uniform(0, 100, n)
    Total = A + B + C + noise * rng.standard_normal(n)
    
    return SyntheticDataset(
        data={'A': A, 'B': B, 'C': C, 'Total': Total},
        ground_truth='A + B + C = Total',
        relationship_type='COMPOSITIONAL',
        expected_pairs=[('A', 'Total'), ('B', 'Total'), ('C', 'Total')]
    )


def generate_competitive(n: int = 200, total: float = 100.0, 
                         seed: int = 42) -> SyntheticDataset:
    """
    8. COMPETITIVE: Trade-off between X and Y.
    
    X + Y = constant (zero-sum)
    
    Should detect negative constraint.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.uniform(0, total, n)
    Y = total - X + 0.01 * rng.standard_normal(n)  # Small noise
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y},
        ground_truth='X + Y = 100 (competition)',
        relationship_type='COMPETITIVE',
        expected_pairs=[('X', 'Y')]
    )


def generate_synergistic(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    9. SYNERGISTIC: Interaction effect (1+1=3).
    
    Y = X1 + X2 + 2*X1*X2 (synergy term)
    
    Should detect interaction.
    """
    rng = np.random.default_rng(seed)
    
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    Y = X1 + X2 + 2 * X1 * X2 + 0.1 * rng.standard_normal(n)
    
    return SyntheticDataset(
        data={'X1': X1, 'X2': X2, 'Y': Y},
        ground_truth='Y = X1 + X2 + 2*X1*X2',
        relationship_type='SYNERGISTIC',
        expected_pairs=[('X1', 'Y'), ('X2', 'Y')]
    )


def generate_mediating(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    10. MEDIATING: X -> M -> Y (indirect effect).
    
    X affects M, M affects Y.
    
    Should detect mediation path.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.standard_normal(n)
    M = 0.8 * X + 0.1 * rng.standard_normal(n)  # X -> M
    Y = 0.8 * M + 0.1 * rng.standard_normal(n)  # M -> Y
    
    return SyntheticDataset(
        data={'X': X, 'M': M, 'Y': Y},
        ground_truth='X -> M -> Y',
        relationship_type='MEDIATING',
        expected_pairs=[('X', 'M'), ('M', 'Y')]
    )


def generate_moderating(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    11. MODERATING: Z changes the X->Y relationship.
    
    Y = X * Z (Z moderates X's effect)
    
    Should detect conditional effect.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.standard_normal(n)
    Z = rng.choice([0.5, 2.0], size=n)  # Moderator (low/high)
    Y = X * Z + 0.1 * rng.standard_normal(n)
    
    return SyntheticDataset(
        data={'X': X, 'Z': Z, 'Y': Y},
        ground_truth='Effect of X on Y depends on Z',
        relationship_type='MODERATING',
        expected_pairs=[('X', 'Y'), ('Z', 'Y')]
    )


def generate_graph(n: int = 200, n_nodes: int = 20, n_edges: int = 50, 
                   seed: int = 42) -> SyntheticDataset:
    """
    12. GRAPH: Network structure.
    
    Nodes connected by edges (adjacency).
    
    Should detect graph communities.
    """
    rng = np.random.default_rng(seed)
    
    edges = []
    for _ in range(n_edges):
        src = rng.integers(0, n_nodes)
        dst = rng.integers(0, n_nodes)
        if src != dst:
            edges.append((src, dst))
    
    # Create adjacency representation
    Source = np.array([e[0] for e in edges], dtype=float)
    Target = np.array([e[1] for e in edges], dtype=float)
    
    return SyntheticDataset(
        data={'Source': Source, 'Target': Target},
        ground_truth=f'Graph with {n_nodes} nodes, {len(edges)} edges',
        relationship_type='GRAPH',
        expected_pairs=edges[:5]  # Sample
    )


def generate_similarity(n: int = 200, n_clusters: int = 3, 
                        seed: int = 42) -> SyntheticDataset:
    """
    13. SIMILARITY: Clustered observations.
    
    Points in 2D clustered around centers.
    
    Should detect cluster structure.
    """
    rng = np.random.default_rng(seed)
    
    centers = rng.uniform(-5, 5, (n_clusters, 2))
    labels = rng.integers(0, n_clusters, n)
    
    X = centers[labels, 0] + 0.5 * rng.standard_normal(n)
    Y = centers[labels, 1] + 0.5 * rng.standard_normal(n)
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y, 'Cluster': labels.astype(float)},
        ground_truth=f'{n_clusters} clusters in X-Y space',
        relationship_type='SIMILARITY',
        expected_pairs=[('X', 'Cluster'), ('Y', 'Cluster')]
    )


def generate_equilibrium(n: int = 200, mean: float = 0.0, 
                         reversion: float = 0.2, seed: int = 42) -> SyntheticDataset:
    """
    14. EQUILIBRIUM: Mean-reverting process.
    
    Y_t = Y_{t-1} - theta * (Y_{t-1} - mu) + noise
    
    Should detect equilibrium/attractor.
    """
    rng = np.random.default_rng(seed)
    
    Y = np.zeros(n)
    Y[0] = rng.standard_normal()
    
    for t in range(1, n):
        Y[t] = Y[t-1] - reversion * (Y[t-1] - mean) + 0.1 * rng.standard_normal()
    
    return SyntheticDataset(
        data={'Y': Y},
        ground_truth=f'Y reverts to {mean}',
        relationship_type='EQUILIBRIUM',
        expected_pairs=[('Y', 'Y')]
    )


def generate_logical(n: int = 200, seed: int = 42) -> SyntheticDataset:
    """
    15. LOGICAL: Boolean rule.
    
    Z = 1 if (X > 0 AND Y > 0) else 0
    
    Should detect logical rule.
    """
    rng = np.random.default_rng(seed)
    
    X = rng.standard_normal(n)
    Y = rng.standard_normal(n)
    Z = ((X > 0) & (Y > 0)).astype(float)
    
    return SyntheticDataset(
        data={'X': X, 'Y': Y, 'Z': Z},
        ground_truth='Z = 1 if X>0 AND Y>0',
        relationship_type='LOGICAL',
        expected_pairs=[('X', 'Z'), ('Y', 'Z')]
    )


# Convenience function to get all generators
ALL_GENERATORS = {
    'CAUSAL': generate_causal,
    'CORRELATIONAL': generate_correlational,
    'STRUCTURAL': generate_structural,
    'TEMPORAL': generate_temporal,
    'FUNCTIONAL': generate_functional,
    'PROBABILISTIC': generate_probabilistic,
    'COMPOSITIONAL': generate_compositional,
    'COMPETITIVE': generate_competitive,
    'SYNERGISTIC': generate_synergistic,
    'MEDIATING': generate_mediating,
    'MODERATING': generate_moderating,
    'GRAPH': generate_graph,
    'SIMILARITY': generate_similarity,
    'EQUILIBRIUM': generate_equilibrium,
    'LOGICAL': generate_logical,
}


def generate_all(n: int = 200, seed: int = 42) -> Dict[str, SyntheticDataset]:
    """Generate synthetic data for all 15 relationship types."""
    return {name: gen(n=n, seed=seed) for name, gen in ALL_GENERATORS.items()}

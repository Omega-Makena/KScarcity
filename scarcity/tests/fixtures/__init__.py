"""
Fixtures package for scarcity tests.
"""

from .synthetic import (
    SyntheticDataset,
    generate_causal,
    generate_correlational,
    generate_structural,
    generate_temporal,
    generate_functional,
    generate_probabilistic,
    generate_compositional,
    generate_competitive,
    generate_synergistic,
    generate_mediating,
    generate_moderating,
    generate_graph,
    generate_similarity,
    generate_equilibrium,
    generate_logical,
    generate_all,
    ALL_GENERATORS,
)

__all__ = [
    'SyntheticDataset',
    'generate_causal',
    'generate_correlational',
    'generate_structural',
    'generate_temporal',
    'generate_functional',
    'generate_probabilistic',
    'generate_compositional',
    'generate_competitive',
    'generate_synergistic',
    'generate_mediating',
    'generate_moderating',
    'generate_graph',
    'generate_similarity',
    'generate_equilibrium',
    'generate_logical',
    'generate_all',
    'ALL_GENERATORS',
]

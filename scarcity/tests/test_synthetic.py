"""
Test: Synthetic Data Generators

Validates that all 15 synthetic data generators produce valid data.
"""

import pytest
import numpy as np
from scarcity.tests.fixtures import (
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


class TestCausalGenerator:
    def test_creates_valid_data(self):
        dataset = generate_causal(n=100)
        assert 'X' in dataset.data
        assert 'Y' in dataset.data
        assert len(dataset.data['X']) == 100
        assert dataset.relationship_type == 'CAUSAL'
    
    def test_lag_relationship(self):
        dataset = generate_causal(n=100, lag=2, strength=0.8)
        X, Y = dataset.data['X'], dataset.data['Y']
        # Y[lag:] should correlate with X[:-lag]
        corr = np.corrcoef(Y[2:], X[:-2])[0, 1]
        assert corr > 0.7, f"Expected strong correlation, got {corr}"


class TestCorrelationalGenerator:
    def test_creates_spurious_correlation(self):
        dataset = generate_correlational(n=100)
        X, Y = dataset.data['X'], dataset.data['Y']
        # X and Y should be correlated (via hidden Z)
        corr = np.corrcoef(X, Y)[0, 1]
        assert corr > 0.7, f"Expected correlation, got {corr}"


class TestStructuralGenerator:
    def test_creates_hierarchical_data(self):
        dataset = generate_structural(n=100)
        assert 'Group' in dataset.data
        assert 'Y' in dataset.data


class TestTemporalGenerator:
    def test_creates_ar_process(self):
        dataset = generate_temporal(n=100)
        Y = dataset.data['Y']
        # Check autocorrelation
        auto_corr = np.corrcoef(Y[1:], Y[:-1])[0, 1]
        assert auto_corr > 0.0, f"Expected positive autocorrelation, got {auto_corr}"


class TestFunctionalGenerator:
    def test_exact_relationship(self):
        dataset = generate_functional(n=100)
        X, Y = dataset.data['X'], dataset.data['Y']
        # Y should be exactly X^2 + 2X + 3
        expected = X**2 + 2*X + 3
        np.testing.assert_array_almost_equal(Y, expected)


class TestProbabilisticGenerator:
    def test_distribution_shift(self):
        dataset = generate_probabilistic(n=1000)
        X, Y = dataset.data['X'], dataset.data['Y']
        # Mean of Y when X=1 should be higher
        mean_0 = Y[X == 0].mean()
        mean_1 = Y[X == 1].mean()
        assert mean_1 > mean_0 + 1.0, f"Expected shift, got {mean_0} vs {mean_1}"


class TestCompositionalGenerator:
    def test_sum_constraint(self):
        dataset = generate_compositional(n=100)
        A = dataset.data['A']
        B = dataset.data['B']
        C = dataset.data['C']
        Total = dataset.data['Total']
        np.testing.assert_array_almost_equal(A + B + C, Total)


class TestCompetitiveGenerator:
    def test_zero_sum(self):
        dataset = generate_competitive(n=100, total=100.0)
        X, Y = dataset.data['X'], dataset.data['Y']
        # X + Y should be approximately 100
        sums = X + Y
        assert np.abs(sums.mean() - 100) < 1.0


class TestSynergisticGenerator:
    def test_interaction_term(self):
        dataset = generate_synergistic(n=100)
        assert 'X1' in dataset.data
        assert 'X2' in dataset.data
        assert 'Y' in dataset.data


class TestMediatingGenerator:
    def test_mediation_path(self):
        dataset = generate_mediating(n=100)
        X, M, Y = dataset.data['X'], dataset.data['M'], dataset.data['Y']
        # X -> M correlation
        corr_xm = np.corrcoef(X, M)[0, 1]
        # M -> Y correlation
        corr_my = np.corrcoef(M, Y)[0, 1]
        assert corr_xm > 0.7
        assert corr_my > 0.7


class TestModeratingGenerator:
    def test_conditional_effect(self):
        dataset = generate_moderating(n=100)
        assert 'X' in dataset.data
        assert 'Z' in dataset.data
        assert 'Y' in dataset.data


class TestGraphGenerator:
    def test_creates_edges(self):
        dataset = generate_graph(n_nodes=20, n_edges=50)
        assert len(dataset.data['Source']) > 0
        assert len(dataset.data['Target']) > 0


class TestSimilarityGenerator:
    def test_creates_clusters(self):
        dataset = generate_similarity(n=100, n_clusters=3)
        assert 'Cluster' in dataset.data
        unique_clusters = np.unique(dataset.data['Cluster'])
        assert len(unique_clusters) == 3


class TestEquilibriumGenerator:
    def test_mean_reversion(self):
        dataset = generate_equilibrium(n=200, mean=0.0)
        Y = dataset.data['Y']
        # Should oscillate around 0
        assert np.abs(Y.mean()) < 1.0


class TestLogicalGenerator:
    def test_boolean_rule(self):
        dataset = generate_logical(n=100)
        X, Y, Z = dataset.data['X'], dataset.data['Y'], dataset.data['Z']
        # Z should be 1 only when X > 0 AND Y > 0
        expected = ((X > 0) & (Y > 0)).astype(float)
        np.testing.assert_array_equal(Z, expected)


class TestGenerateAll:
    def test_generates_all_15(self):
        all_data = generate_all(n=50)
        assert len(all_data) == 15
        for name in ALL_GENERATORS.keys():
            assert name in all_data

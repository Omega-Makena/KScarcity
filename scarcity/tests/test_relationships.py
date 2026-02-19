"""
Test: Relationship Hypothesis Classes

Tests that each hypothesis type correctly identifies its relationship type
using the synthetic data generators.
"""

import pytest
import numpy as np
from scarcity.tests.fixtures import (
    generate_causal,
    generate_correlational,
    generate_temporal,
    generate_functional,
    generate_equilibrium,
    generate_compositional,
    generate_competitive,
    generate_synergistic,
    generate_probabilistic,
    generate_structural,
    generate_mediating,
    generate_moderating,
    generate_graph,
    generate_similarity,
    generate_logical,
)

from scarcity.engine.relationships import (
    CausalHypothesis,
    CorrelationalHypothesis,
    TemporalHypothesis,
    FunctionalHypothesis,
    EquilibriumHypothesis,
    CompositionalHypothesis,
    CompetitiveHypothesis,
    SynergisticHypothesis,
    ProbabilisticHypothesis,
    StructuralHypothesis,
)

from scarcity.engine.relationships_extended import (
    MediatingHypothesis,
    ModeratingHypothesis,
    GraphHypothesis,
    SimilarityHypothesis,
    LogicalHypothesis,
)


class TestCausalHypothesis:
    def test_detects_causal_relationship(self):
        """Should detect X → Y causality."""
        dataset = generate_causal(n=200, lag=2, strength=0.8)
        hyp = CausalHypothesis('X', 'Y', lag=2)
        
        # Feed data
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result['direction'] == 1, "Should detect X→Y direction"
        assert result['confidence'] > 0.5, f"Confidence too low: {result['confidence']}"


class TestCorrelationalHypothesis:
    def test_detects_correlation(self):
        """Should detect correlation."""
        dataset = generate_correlational(n=200)
        hyp = CorrelationalHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert abs(result.get('correlation', 0)) > 0.5, "Should detect correlation"


class TestTemporalHypothesis:
    def test_detects_autocorrelation(self):
        """Should detect autoregressive structure."""
        dataset = generate_temporal(n=200)
        hyp = TemporalHypothesis('Y', lag=3)
        
        for i in range(len(dataset.data['Y'])):
            row = {'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Should have learned some coefficients
        assert len(result.get('coefficients', [])) > 0


class TestFunctionalHypothesis:
    def test_detects_functional_relationship(self):
        """Should detect deterministic Y = f(X)."""
        dataset = generate_functional(n=200)
        hyp = FunctionalHypothesis('X', 'Y', degree=2)
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result['fit_score'] > 0.9, "Should fit deterministic function well"
        assert result.get('deterministic', False), "Should detect deterministic relationship"


class TestEquilibriumHypothesis:
    def test_detects_mean_reversion(self):
        """Should detect mean-reverting process."""
        dataset = generate_equilibrium(n=300, mean=0.0, reversion=0.2)
        hyp = EquilibriumHypothesis('Y')
        
        for i in range(len(dataset.data['Y'])):
            row = {'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('is_reverting', False), "Should detect mean reversion"
        assert abs(result.get('equilibrium', 1.0)) < 1.0, "Equilibrium should be near 0"


class TestCompositionalHypothesis:
    def test_detects_sum_constraint(self):
        """Should detect A + B + C = Total."""
        dataset = generate_compositional(n=100)
        hyp = CompositionalHypothesis(['A', 'B', 'C'], 'Total')
        
        for i in range(len(dataset.data['A'])):
            row = {
                'A': dataset.data['A'][i],
                'B': dataset.data['B'][i],
                'C': dataset.data['C'][i],
                'Total': dataset.data['Total'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('constraint_holds', False), "Should detect sum constraint"


class TestCompetitiveHypothesis:
    def test_detects_trade_off(self):
        """Should detect X + Y = constant."""
        dataset = generate_competitive(n=100)
        hyp = CompetitiveHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('is_competitive', False), "Should detect competitive relationship"


class TestSynergisticHypothesis:
    def test_detects_interaction(self):
        """Should detect significant X1*X2 interaction."""
        dataset = generate_synergistic(n=200)
        hyp = SynergisticHypothesis('X1', 'X2', 'Y')
        
        for i in range(len(dataset.data['X1'])):
            row = {
                'X1': dataset.data['X1'][i],
                'X2': dataset.data['X2'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Interaction coefficient should be significant
        assert abs(result.get('interaction_coefficient', 0)) > 0.5


class TestProbabilisticHypothesis:
    def test_detects_distribution_shift(self):
        """Should detect X shifts distribution of Y."""
        dataset = generate_probabilistic(n=500)
        hyp = ProbabilisticHypothesis('X', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('effect_size', 0) > 0.5, f"Effect size too low: {result}"


class TestStructuralHypothesis:
    def test_detects_hierarchy(self):
        """Should detect group structure."""
        dataset = generate_structural(n=200)
        hyp = StructuralHypothesis('Group', 'Y')
        
        for i in range(len(dataset.data['Group'])):
            row = {
                'Group': dataset.data['Group'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('icc', 0) > 0.1, f"ICC too low: {result}"


class TestMediatingHypothesis:
    def test_detects_mediation(self):
        """Should detect X → M → Y path."""
        dataset = generate_mediating(n=200)
        hyp = MediatingHypothesis('X', 'M', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'M': dataset.data['M'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('has_mediation', False), f"Should detect mediation: {result}"


class TestModeratingHypothesis:
    def test_detects_moderation(self):
        """Should detect Z moderates X→Y."""
        dataset = generate_moderating(n=200)
        hyp = ModeratingHypothesis('X', 'Z', 'Y')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Z': dataset.data['Z'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        # Interaction should be detected
        assert abs(result.get('interaction', 0)) > 0.1


class TestGraphHypothesis:
    def test_detects_graph_structure(self):
        """Should track graph edges."""
        dataset = generate_graph(n_nodes=20, n_edges=50)
        hyp = GraphHypothesis('Source', 'Target')
        
        for i in range(len(dataset.data['Source'])):
            row = {
                'Source': dataset.data['Source'][i],
                'Target': dataset.data['Target'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('n_edges', 0) > 10, f"Should track edges: {result}"


class TestSimilarityHypothesis:
    def test_detects_clusters(self):
        """Should detect cluster structure."""
        dataset = generate_similarity(n=300, n_clusters=3)
        hyp = SimilarityHypothesis(['X', 'Y'], n_clusters=3)
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Y': dataset.data['Y'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('explained_variance', 0) > 0.2


class TestLogicalHypothesis:
    def test_detects_boolean_rule(self):
        """Should detect Z = X AND Y."""
        dataset = generate_logical(n=200)
        hyp = LogicalHypothesis('X', 'Y', 'Z')
        
        for i in range(len(dataset.data['X'])):
            row = {
                'X': dataset.data['X'][i],
                'Y': dataset.data['Y'][i],
                'Z': dataset.data['Z'][i]
            }
            hyp.fit_step(row)
        
        result = hyp.evaluate({})
        assert result.get('best_rule') == 'AND', f"Should detect AND rule: {result}"
        assert result.get('rule_accuracy', 0) > 0.9

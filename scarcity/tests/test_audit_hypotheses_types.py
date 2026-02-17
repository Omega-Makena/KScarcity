from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine import relationships as rel
from scarcity.engine import relationships_extended as relx


def test_engine_v2_initializes_all_hypothesis_types():
    engine = OnlineDiscoveryEngine()
    schema = {"fields": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    engine.initialize_v2(schema, use_causal=True)
    types = {type(h) for h in engine.hypotheses.population.values()}
    required = {
        rel.CausalHypothesis,
        rel.CorrelationalHypothesis,
        rel.TemporalHypothesis,
        rel.FunctionalHypothesis,
        rel.EquilibriumHypothesis,
        rel.CompositionalHypothesis,
        rel.CompetitiveHypothesis,
        rel.SynergisticHypothesis,
        rel.ProbabilisticHypothesis,
        rel.StructuralHypothesis,
        relx.MediatingHypothesis,
        relx.ModeratingHypothesis,
        relx.GraphHypothesis,
        relx.SimilarityHypothesis,
        relx.LogicalHypothesis,
    }
    assert required.issubset(types)

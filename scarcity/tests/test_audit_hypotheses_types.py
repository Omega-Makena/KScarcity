import numpy as np

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine import relationships as rel
from scarcity.engine import relationships_extended as relx


# initialize_v2 seeds these directly, for every variable / pair / triple.
INIT_SEEDED = {
    rel.CausalHypothesis,
    rel.CorrelationalHypothesis,
    rel.TemporalHypothesis,
    rel.FunctionalHypothesis,
    rel.EquilibriumHypothesis,
    rel.CompositionalHypothesis,
    rel.SynergisticHypothesis,
    relx.MediatingHypothesis,
    relx.ModeratingHypothesis,
    relx.SimilarityHypothesis,
    relx.LogicalHypothesis,
}

# These four are deliberately NOT seeded at init — _explore_step() adds them for
# pairs not already covered by a strong hypothesis, which keeps the initial pool
# under capacity. See the note in initialize_v2().
EXPLORE_SEEDED = {
    rel.CompetitiveHypothesis,
    rel.ProbabilisticHypothesis,
    rel.StructuralHypothesis,
    relx.GraphHypothesis,
}


def test_initialize_v2_seeds_the_direct_hypothesis_types():
    engine = OnlineDiscoveryEngine()
    schema = {"fields": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    engine.initialize_v2(schema, use_causal=True)

    types = {type(h) for h in engine.hypotheses.population.values()}
    assert INIT_SEEDED.issubset(types)

    # The exploration-gated types must NOT be present yet. Asserting all 15
    # here is what made the previous version of this test unpassable.
    assert not (EXPLORE_SEEDED & types)


def test_exploration_adds_the_remaining_types_given_enough_variables():
    """All 15 types become reachable once exploration has pairs to work with.

    _explore_step() only considers pairs absent from get_strongest(top_k=20).
    initialize_v2() seeds a hypothesis for every pair, so with few variables
    every pair is always "covered" and the exploration-gated types can never be
    created. They first appear at roughly 8 variables (28 pairs > top_k=20).
    """
    n_vars, n_rows = 10, 300
    names = [f"v{i}" for i in range(n_vars)]

    rng = np.random.default_rng(3)
    data = np.zeros((n_vars, n_rows))
    data[0] = rng.normal(size=n_rows)
    for k in range(1, n_vars):
        for t in range(1, n_rows):
            data[k, t] = 0.7 * data[k - 1, t - 1] + 0.3 * rng.normal()

    # Python backend: exploration operates on the Python pool, which the tensor
    # backend leaves without evidence.
    engine = OnlineDiscoveryEngine(vectorized=False)
    engine.initialize_v2({"fields": [{"name": n} for n in names]}, use_causal=True)
    for t in range(n_rows):
        engine.process_row({names[k]: data[k, t] for k in range(n_vars)})

    types = {type(h) for h in engine.hypotheses.population.values()}
    assert EXPLORE_SEEDED.issubset(types), (
        f"exploration never created: {sorted(t.__name__ for t in EXPLORE_SEEDED - types)}"
    )

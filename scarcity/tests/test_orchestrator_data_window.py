"""
Regression tests for the MPIEOrchestrator data-window path and the
Candidate-returning BanditRouter contract it depends on.

Prior to this, BanditRouter.propose returned arm-id ints, so the orchestrator
bailed at its Candidate guard and the whole data_window path was inert. These
tests pin the fixed contract.
"""

import asyncio

import numpy as np

from scarcity.engine.bandit_router import BanditRouter
from scarcity.engine.types import Candidate, Reward
from scarcity.engine.engine import MPIEOrchestrator


def _schema(n):
    return {'fields': {f'v{i}': {} for i in range(n)}}


def test_propose_returns_candidates():
    r = BanditRouter(rng=np.random.default_rng(0))
    cands = r.propose(n_proposals=8, context={'schema': _schema(5)})
    assert len(cands) == 8
    assert all(isinstance(c, Candidate) for c in cands)
    # Deterministic path_ids: stable across calls for the same structure.
    ids1 = {c.path_id for c in r.propose(200, context={'schema': _schema(5)})}
    ids2 = {c.path_id for c in r.propose(200, context={'schema': _schema(5)})}
    assert ids1 == ids2


def test_propose_empty_without_schema_vars():
    r = BanditRouter(rng=np.random.default_rng(0))
    # A single variable cannot form a directed pair.
    assert r.propose(5, context={'schema': _schema(1)}) == []


def test_diversity_score_in_unit_interval():
    r = BanditRouter(rng=np.random.default_rng(0))
    cands = r.propose(10, context={'schema': _schema(4)})
    for c in cands:
        d = r.diversity_score(c)
        assert 0.0 <= d <= 1.0


def test_diversity_drops_after_repeated_proposal():
    r = BanditRouter(rng=np.random.default_rng(0))
    c = r.propose(1, context={'schema': _schema(3)})[0]
    first = r.diversity_score(c)
    for _ in range(5):
        r.propose(200, context={'schema': _schema(3)})  # bump proposal counts
    assert r.diversity_score(c) < first


def test_apply_rewards_updates_arms():
    r = BanditRouter(rng=np.random.default_rng(0))
    cands = r.propose(6, context={'schema': _schema(4)})
    rewards = [
        Reward(path_id=c.path_id, arm_key=(c.root, c.depth), value=0.8,
               latency_penalty=0.0, diversity_bonus=0.0, accepted=True)
        for c in cands
    ]
    r.apply_rewards(rewards)
    total_obs = sum(s.observations for s in r.arms.values())
    assert total_obs == len(cands)


def test_handle_data_window_processes_window():
    async def drive():
        orch = MPIEOrchestrator()
        await orch.start()
        rng = np.random.default_rng(1)
        data = rng.normal(size=(400, 5))
        data[2:, 3] += 0.9 * data[:-2, 0]  # inject 0 -> 3 coupling
        payload = {
            'data': data.tolist(),
            'schema': _schema(5),
            'window_id': 1,
        }
        await orch._handle_data_window('data_window', payload)
        stats = orch.get_stats()
        await orch.stop()
        return stats

    stats = asyncio.run(drive())
    assert stats.get('windows_processed', 0) >= 1

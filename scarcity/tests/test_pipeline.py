"""End-to-end epistemic pipeline: a discovered edge climbs the ladder via real analyses."""
import numpy as np
import pandas as pd

from scarcity.engine.gpu_engine import GPUDiscoveryEngine
from scarcity.pipeline import EpistemicPipeline
from scarcity.epistemic import Rung


def _confounded(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.5, size=n)
    y = 1.2 * x + 0.7 * c + rng.normal(scale=0.5, size=n)   # true x->y, confounded by c
    w = rng.normal(size=n)                                   # pure noise
    return pd.DataFrame({"x": x, "y": y, "c": c, "w": w})


def _run(df):
    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": k} for k in df.columns]}, use_causal=True)
    for i in range(len(df)):
        e.process_row(df.iloc[i].to_dict())
    return EpistemicPipeline(alpha=0.05, rv_threshold=0.10).run(e, df)


def test_true_causal_edge_climbs_to_robust():
    ladder = _run(_confounded())
    xy = ladder.status("x", "y")
    assert xy.rung == Rung.ROBUST, ladder.explain("x", "y")
    ev = xy.evidence
    # the adjustment set should include the confounder c, and the estimate ~1.2
    assert "c" in ev["IDENTIFIED"]["adjustment_set"]
    assert 0.9 <= ev["ESTIMATED"]["effect"] <= 1.5


def test_noise_variable_does_not_reach_estimated():
    ladder = _run(_confounded())
    # w is pure noise; no edge into/out of w should be estimated as a causal effect
    for (s, t), ep in ladder._edges.items():
        if "w" in (s, t):
            assert ep.rung < Rung.ESTIMATED, f"{s}->{t} spuriously estimated: {ladder.explain(s, t)}"


def test_each_directed_pair_recorded_once():
    ladder = _run(_confounded())
    # dedup: a directed pair's log should not stack multiple HYPOTHESIZED entries
    log = ladder.status("x", "y").log
    assert sum("HYPOTHESIZED: reached" in ln for ln in log) == 1

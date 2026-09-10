"""End-to-end epistemic pipeline — the system climbing its own ladder.

Ties the pieces that otherwise sit as standalone modules into one flow: a
discovered edge is promoted rung by rung through real analyses, and the
EpistemicLadder records how far each edge actually earns its way.

  DISCOVERED / HYPOTHESIZED  the streaming engine's calibrated knowledge graph
                             (autocorrelation-robust gate) supplies edges + types
  PREDICTIVE                 a held-out train/test fit: does the edge predict
                             out-of-sample better than the mean baseline?
  IDENTIFIED                 a backdoor adjustment set (the edge's other
                             discovered parents) yields an adjusted estimate
  ESTIMATED                  the adjusted effect with a CI clear of 0
                             (FWL t-stat from scarcity.causal.sensitivity)
  ROBUST                     survives a placebo refuter AND an omitted-variable
                             robustness value above threshold (Cinelli-Hazlett)

Run it on a fitted engine + the DataFrame; get back a populated EpistemicLadder.
Every stage is defensive: a failed gate is recorded, not raised, and the edge
simply stops climbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from scarcity.epistemic import EpistemicLadder
from scarcity.causal.sensitivity import _adjusted_t_stat, robustness_value, _t_critical

_DIRECTIONAL = {"causal", "temporal", "functional", "probabilistic", "graph"}


@dataclass
class EpistemicPipeline:
    alpha: float = 0.05
    rv_threshold: float = 0.10
    holdout_frac: float = 0.30
    max_edges: Optional[int] = None

    def __post_init__(self):
        self.ladder = EpistemicLadder(predictive_alpha=self.alpha, rv_threshold=self.rv_threshold)

    # -- stage helpers -------------------------------------------------------
    def _predictive(self, df, src: str, tgt: str):
        """Held-out R² of tgt ~ src vs the mean baseline, with a significance p."""
        x = df[src].to_numpy(dtype=float)
        y = df[tgt].to_numpy(dtype=float)
        n = len(y)
        cut = int(n * (1.0 - self.holdout_frac))
        if cut < 10 or n - cut < 10:
            return 0.0, 0.0, 1.0
        xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
        sx = xtr.std()
        if sx < 1e-9:
            return 0.0, 0.0, 1.0
        b = np.cov(xtr, ytr)[0, 1] / (sx ** 2)
        a = ytr.mean() - b * xtr.mean()
        pred = a + b * xte
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum()) or 1e-12
        skill = 1.0 - ss_res / ss_tot                      # out-of-sample R²
        # significance of the held-out predictive correlation
        r = float(np.corrcoef(pred, yte)[0, 1]) if np.std(pred) > 1e-9 else 0.0
        m = len(yte)
        from scipy.stats import t as _t
        tt = abs(r) * np.sqrt(max(m - 2, 1) / max(1e-9, 1 - r * r))
        p = float(2 * (1 - _t.cdf(tt, max(m - 2, 1))))
        return skill, 0.0, p

    def _adjustment_set(self, kg_pairs, src: str, tgt: str) -> List[str]:
        """Other discovered parents of tgt (edges touching tgt, excluding src)."""
        adj = set()
        for a, b in kg_pairs:
            if tgt in (a, b):
                other = b if a == tgt else a
                if other != src:
                    adj.add(other)
        return sorted(adj)

    def _placebo_ok(self, df, src, tgt, confounders, seed=0) -> bool:
        """Permuting the treatment should destroy the adjusted effect."""
        rng = np.random.default_rng(seed)
        d = df.copy()
        d[src] = rng.permutation(d[src].to_numpy())
        r = _adjusted_t_stat(d, src, tgt, confounders)
        if r is None:
            return True
        return abs(r["t_stat"]) < _t_critical(self.alpha, int(r["dof"]))

    # -- main ----------------------------------------------------------------
    def run(self, engine, df, q: float = 0.05):
        kg = engine.get_knowledge_graph(top_k=400, calibrated=True, q=q)
        pairs = [tuple(h["variables"]) for h in kg if len(h["variables"]) == 2]
        edges = [h for h in kg if len(h["variables"]) == 2]

        # Deduplicate per directed (src, tgt): the same pair appears under several
        # types (correlational/functional/causal). Keep the strongest representative
        # (the graph is confidence-sorted), preferring a directional type so the
        # edge can claim a direction to estimate.
        seen = set()
        deduped = []
        for h in sorted(edges, key=lambda e: (e["type"] not in _DIRECTIONAL,
                                              -e["metrics"].get("confidence", 0.0))):
            v = h["variables"]
            src, tgt = (v[-1], v[0]) if h.get("direction") == -1 else (v[0], v[-1])
            if (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            deduped.append((src, tgt, h))
        if self.max_edges:
            deduped = deduped[: self.max_edges]

        for src, tgt, h in deduped:
            rt = h["type"]
            conf = float(h["metrics"].get("confidence", 0.0))

            self.ladder.record_discovery(src, tgt, conf)
            self.ladder.record_hypothesis(src, tgt, rt)

            # only directional types claim a causal direction to estimate; symmetric
            # types stop at HYPOTHESIZED (a correlation is not a causal claim)
            if rt not in _DIRECTIONAL:
                continue

            skill, base, p = self._predictive(df, src, tgt)
            if not self.ladder.record_predictive_validation(src, tgt, skill, base, p):
                continue

            confs = self._adjustment_set(pairs, src, tgt)
            adj = _adjusted_t_stat(df, src, tgt, confs)
            if adj is None:
                self.ladder.record_identification(src, tgt, identifiable=False)
                continue
            self.ladder.record_identification(src, tgt, identifiable=True, adjustment_set=confs)

            beta, se, t, dof = adj["estimate"], adj["se"], adj["t_stat"], int(adj["dof"])
            tc = _t_critical(self.alpha, dof)
            if not self.ladder.record_estimation(src, tgt, beta, beta - tc * se, beta + tc * se):
                continue

            rv = robustness_value(t, dof, q=1.0, alpha=self.alpha)
            placebo = self._placebo_ok(df, src, tgt, confs)
            self.ladder.record_refutation(src, tgt, refuters_passed=placebo, robustness_value=rv)

        return self.ladder

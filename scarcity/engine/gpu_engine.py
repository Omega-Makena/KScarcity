"""
GPUDiscoveryEngine — drop-in GPU-accelerated replacement for OnlineDiscoveryEngine.

Uses GPUBatchRLS + GPUHypothesisPool to process all hypotheses in parallel
on CUDA instead of iterating ~1565 Python Hypothesis objects per row.

API mirrors OnlineDiscoveryEngine:
    engine = GPUDiscoveryEngine()
    engine.initialize_v2(schema, use_causal=True)
    engine.process_row({col: val, ...})
    graph, edges = gpu_extract_graph(engine, conf_threshold=0.35, min_evidence=5)

Falls back to CPU tensors automatically when CUDA is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import hashlib

from .gpu_batch_rls import GPUBatchRLS
from .gpu_hypothesis_pool import GPUHypothesisPool, HypoSpec, LifecycleEmulator
from .types import Candidate

# Lifecycle int8 -> LifecycleState string (matches discovery.HypothesisState values).
_LC_STATE = {0: "tentative", 1: "active", 2: "decaying", 3: "dead"}

# ---------------------------------------------------------------------------
# Rel-type classification (mirrors graph_extractor.py logic)
# ---------------------------------------------------------------------------

_DIRECTIONAL = frozenset({
    'causal', 'functional', 'temporal', 'equilibrium',
    'structural', 'probabilistic', 'graph',
})
_SYMMETRIC = frozenset({
    'correlational', 'competitive', 'similarity',
})
_MULTI_VAR = frozenset({
    'synergistic', 'mediating', 'moderating', 'logical', 'compositional',
})

# Lifecycle state codes from LifecycleEmulator
_DEAD = 3


# ---------------------------------------------------------------------------
# GPUDiscoveryEngine
# ---------------------------------------------------------------------------

class GPUDiscoveryEngine:
    """
    Stateful streaming discovery engine backed by batch-tensor RLS.

    Replaces the per-hypothesis Python object loop in OnlineDiscoveryEngine with
    vectorized torch.einsum operations — 2-3× faster on CPU, much faster on GPU
    when batching many permutation resamples.

    Default device is 'cpu': no kernel-launch overhead, vectorization via BLAS.
    Set device='cuda' when running bootstrap calibration (B_perm ≥ 50).

    Compatible with gpu_extract_graph(); use extract_graph() adapter in
    graph_extractor.py for transparent drop-in replacement.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        small_dataset_mode: bool = False,
        forgetting_window: int = 0,
    ) -> None:
        # Default to CPU — for N<200, BLAS vectorisation beats CUDA kernel overhead
        if device is None:
            device = 'cpu'
        self.device = device
        self.small_dataset_mode = small_dataset_mode
        # Batched RLS forgetting factor. The GPU backend forgets by construction
        # (recursive least squares with lambda < 1); forgetting_window tunes it:
        # lambda = 1 - 1/window, so a smaller window ages out stale relationships
        # faster. 0 keeps the standard lambda = 0.99 (effective window ~100).
        self._rls_lambda = (min(max(1.0 - 1.0 / forgetting_window, 0.90), 0.9999)
                            if forgetting_window > 0 else 0.99)

        self._col_names: List[str] = []
        self._N: int = 0
        self._pool: Optional[GPUHypothesisPool] = None
        self._rls: Dict[Tuple, GPUBatchRLS] = {}
        self._group_order: List[Tuple] = []
        self._lc: Optional[LifecycleEmulator] = None
        self._data: Optional[torch.Tensor] = None   # (1, T, N)
        self.step_count: int = 0
        self._lc_interval: int = 10

    def initialize_v2(self, schema: Dict[str, Any], use_causal: bool = True) -> None:
        fields = schema.get('fields', [])
        self._col_names = [f['name'] for f in fields]
        self._N = len(self._col_names)

        self._pool = GPUHypothesisPool(self._col_names, device=self.device)
        groups = self._pool.groups()
        self._group_order = list(groups.keys())

        for key, spec_list in groups.items():
            M = len(spec_list)
            F = spec_list[0].F
            self._rls[key] = GPUBatchRLS(M, F, lam=self._rls_lambda, device=self.device)

        N_hyp = sum(len(sl) for sl in groups.values())
        self._lc = LifecycleEmulator(
            N_hyp=N_hyp, R=1, small_dataset=self.small_dataset_mode
        )

        # Preallocated ring-free buffer; grows by doubling so per-row append is
        # amortized O(1) instead of the O(n^2) torch.cat it replaced.
        self._data = torch.zeros(1, 256, self._N, device=self.device, dtype=torch.float64)
        self._n_rows = 0
        self.step_count = 0

    def _view(self) -> torch.Tensor:
        """The filled portion of the buffer, shape (1, n_rows, N)."""
        return self._data[:, :self._n_rows]

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        vals = [float(row.get(c, float('nan'))) for c in self._col_names]
        if self._n_rows >= self._data.shape[1]:                # grow by doubling
            grown = torch.zeros(1, self._data.shape[1] * 2, self._N,
                                device=self.device, dtype=torch.float64)
            grown[:, :self._n_rows] = self._data[:, :self._n_rows]
            self._data = grown
        self._data[0, self._n_rows] = torch.tensor(vals, dtype=torch.float64, device=self.device)
        self._n_rows += 1
        t = self._n_rows - 1
        self.step_count += 1

        data_view = self._view()
        groups = self._pool.groups()
        for key in self._group_order:
            spec_list = groups[key]
            X, Y = self._pool.extract_features_gpu(data_view, spec_list, t)
            self._rls[key].update(X.squeeze(0), Y.squeeze(0))

        if self.step_count % self._lc_interval == 0:
            self._run_lifecycle()

        return {'step': self.step_count}

    def _group_confidence(self, key, r):
        """Per-hypothesis, type-aware confidence for a (perm_col, F) group.

        Prediction-style types (causal / correlational / temporal / functional /
        graph) keep the RLS goodness-of-fit confidence. Types defined by a
        specific term or sign get their own statistic, computed from the batched
        RLS state and applied only to the hypotheses of that type (F=2 groups mix
        several types, so the override must be per-hypothesis, not per-group):

        - synergistic / moderating: significance of the interaction coefficient
          a*b (feature 3) — so an additive fit is not mistaken for synergy.
        - competitive: significance of a *negative* slope (feature 1) — a
          substitute relationship, not just any strong coupling.
        """
        specs = self._pool.groups()[key]
        conf = r.confidence.clone()                       # (M,) fit-based default
        types = np.array([self._rel_type_str(s.rel_type) for s in specs])

        def _mask(name):
            return torch.as_tensor(types == name, device=conf.device)

        if r.F >= 4:
            inter = _mask("synergistic") | _mask("moderating")
            if bool(inter.any()):
                conf = torch.where(inter, r.coef_significance(3), conf)

        if r.F >= 2:
            comp = _mask("competitive")
            if bool(comp.any()):
                neg = r.W[:, 1] < 0                        # substitute = negative slope
                comp_conf = torch.where(neg, r.coef_significance(1), torch.zeros_like(conf))
                conf = torch.where(comp, comp_conf, conf)

        if r.F >= 2:
            eq = _mask("equilibrium")
            if bool(eq.any()):
                # AR(1) v_t ~ [1, v_{t-1}]: W[:,1] is phi. Mean-reversion
                # (stationarity) means phi significantly < 1 (ADF-style unit-root
                # test); a random walk (phi ~ 1) is not an equilibrium.
                phi = r.W[:, 1]
                eq_conf = torch.where(
                    phi < 1.0, r.coef_significance(1, null=1.0), torch.zeros_like(conf))
                conf = torch.where(eq, eq_conf, conf)

        med = _mask("mediating")
        if bool(med.any()):
            for i in torch.nonzero(med, as_tuple=True)[0].tolist():
                conf[i] = self._sobel_confidence(specs[i])

        struc = _mask("structural")
        if bool(struc.any()):
            for i in torch.nonzero(struc, as_tuple=True)[0].tolist():
                conf[i] = self._anova_confidence(specs[i])

        if r.F >= 2:
            # Compositional: a near-exact linear identity (accounting relation),
            # i.e. R^2 ~ 1. Ramp on the fit above 0.9 so only near-perfect
            # relationships qualify — an approximate coupling is correlational,
            # not an identity.
            comp = _mask("compositional")
            if bool(comp.any()):
                r2 = ((r.fit_score - 0.9) / 0.1).clamp(0.0, 1.0)
                conf = torch.where(comp, r2, conf)

        prob = _mask("probabilistic")
        if bool(prob.any()):
            for i in torch.nonzero(prob, as_tuple=True)[0].tolist():
                conf[i] = self._conditional_confidence(specs[i])

        logic = _mask("logical")
        if bool(logic.any()):
            for i in torch.nonzero(logic, as_tuple=True)[0].tolist():
                conf[i] = self._logical_confidence(specs[i])

        return conf

    def _conditional_confidence(self, spec, d_floor: float = 0.30) -> torch.Tensor:
        """Probabilistic: P(outcome | predictor) shifts with the predictor.

        Splits the outcome at the predictor's median and measures the effect
        size (Cohen's d) between the two conditional groups. Effect-size gated:
        below d_floor (near-zero under independence, regardless of sample size)
        the confidence is zero, which defeats the uniform-under-null false
        positives of a bare significance test. Above the gate, the two-sample
        significance is returned.
        """
        d = self._data[0, :self._n_rows]
        zero = torch.zeros((), device=d.device, dtype=d.dtype)
        a, b = d[:, spec.col_a], d[:, spec.col_y]
        m = torch.isfinite(a) & torch.isfinite(b)
        a, b = a[m], b[m]
        if b.shape[0] < 20:
            return zero
        hi = a > a.median()
        lo = ~hi
        nh, nl = int(hi.sum()), int(lo.sum())
        if nh < 5 or nl < 5:
            return zero
        bh, bl = b[hi], b[lo]
        vh, vl = bh.var(), bl.var()
        pooled_sd = torch.sqrt((((nh - 1) * vh + (nl - 1) * vl) / max(nh + nl - 2, 1)).clamp(min=1e-12))
        cohen_d = (bh.mean() - bl.mean()).abs() / (pooled_sd + 1e-12)
        if float(cohen_d) < d_floor:
            return zero
        se = torch.sqrt((vh / nh + vl / nl).clamp(min=1e-12))
        t = (bh.mean() - bl.mean()).abs() / se
        return torch.erf(t / (2.0 ** 0.5)).clamp(0.0, 1.0)

    def _logical_confidence(self, spec, acc_floor: float = 0.15) -> torch.Tensor:
        """Logical: a Boolean rule over binarized variables predicts the outcome.

        Binarizes a, b, c at their medians and scores the best of AND / OR /
        IMPLIES / EQUIV (and their negations) at predicting c. Effect-size gated
        on how far the best accuracy beats chance (|acc - 0.5| >= acc_floor), then
        a binomial z with a Sidak correction for choosing the best of 8 rules —
        together these stop a rule that fits random data by chance from reading
        as significant.
        """
        d = self._data[0, :self._n_rows]
        zero = torch.zeros((), device=d.device, dtype=d.dtype)
        a, b, c = d[:, spec.col_a], d[:, spec.col_b], d[:, spec.col_y]
        m = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(c)
        a, b, c = a[m], b[m], c[m]
        n = a.shape[0]
        if n < 20:
            return zero
        ab, bb, cb = a > a.median(), b > b.median(), c > c.median()
        rules = [ab & bb, ab | bb, (~ab) | bb, ab == bb]   # AND, OR, IMPLIES, EQUIV
        best = max(float((r == cb).float().mean()) for r in rules)
        best = max(best, 1.0 - best)                       # a rule and its negation
        if (best - 0.5) < acc_floor:
            return zero
        z = (best - 0.5) / ((0.25 / n) ** 0.5)
        single = float(torch.erf(torch.tensor(max(z, 0.0)) / (2.0 ** 0.5)).clamp(0.0, 1.0))
        return torch.tensor(single ** 8, device=d.device, dtype=d.dtype)

    def _anova_confidence(self, spec, n_bins: int = 4) -> torch.Tensor:
        """One-way ANOVA: does the outcome differ across groups of the predictor.

        Bins the group variable into quantiles and tests whether the outcome's
        between-group variance is significant (eta-squared + an F-test). torch
        has no F CDF, so F is turned into a z-score via Fisher's chi-square
        approximation (chi^2 ~ F*df_between for large df_within) and mapped to
        [0, 1] as erf(|z|/sqrt(2)).
        """
        d = self._data[0, :self._n_rows]                                  # (T, N)
        zero = torch.zeros((), device=d.device, dtype=d.dtype)
        g = d[:, spec.col_a]                               # group variable
        y = d[:, spec.col_y]                               # outcome
        m = torch.isfinite(g) & torch.isfinite(y)
        g, y = g[m], y[m]
        n = y.shape[0]
        if n < 4 * n_bins:
            return zero
        edges = torch.quantile(g, torch.linspace(0, 1, n_bins + 1, device=d.device, dtype=d.dtype))
        bins = torch.bucketize(g, edges[1:-1].contiguous())   # 0..n_bins-1
        grand = y.mean()
        ss_tot = ((y - grand) ** 2).sum()
        if float(ss_tot) < 1e-12:
            return zero
        ss_between = zero
        k_eff = 0
        for gb in range(n_bins):
            sel = bins == gb
            ng = int(sel.sum())
            if ng > 0:
                ss_between = ss_between + ng * (y[sel].mean() - grand) ** 2
                k_eff += 1
        df_b = max(k_eff - 1, 1)
        df_w = max(n - k_eff, 1)
        ss_within = (ss_tot - ss_between).clamp(min=1e-12)
        F = (ss_between / df_b) / (ss_within / df_w)
        # Fisher chi-square -> z: chi2 ~ F*df_b; z = sqrt(2 chi2) - sqrt(2 df_b - 1).
        chi2 = F * df_b
        z = (torch.sqrt(2.0 * chi2) - (2.0 * df_b - 1.0) ** 0.5).clamp(min=0.0)
        return torch.erf(z / (2.0 ** 0.5)).clamp(0.0, 1.0)

    def _sobel_confidence(self, spec) -> torch.Tensor:
        """Mediation significance via the Sobel test on the indirect path.

        For a mediating hypothesis a -> b -> c: alpha is the a->b slope, beta is
        the b->c slope adjusting for a (from the c ~ [1, a, b] regression). The
        indirect effect is alpha*beta; the Sobel z = alpha*beta / SE, with
        SE = sqrt(beta^2 se_alpha^2 + alpha^2 se_beta^2), mapped to [0, 1] as
        erf(|z|/sqrt(2)). Computed from the stored series (the b->c effect is not
        recoverable from the pool's single c~[1,a,b] fit alone).
        """
        d = self._data[0, :self._n_rows]                                  # (T, N)
        zero = torch.zeros((), device=d.device, dtype=d.dtype)
        a = d[:, spec.col_a]
        b = d[:, spec.col_b]
        c = d[:, spec.col_y]
        mask = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(c)
        a, b, c = a[mask], b[mask], c[mask]
        n = a.shape[0]
        if n < 10:
            return zero
        am, bm, cm = a - a.mean(), b - b.mean(), c - c.mean()
        var_a = (am * am).sum()
        if float(var_a) < 1e-12:
            return zero
        # Path a->b
        alpha = (am * bm).sum() / var_a
        res_b = bm - alpha * am
        se_alpha = torch.sqrt((res_b * res_b).sum() / max(n - 2, 1) / var_a)
        # Path b->c adjusting for a: regress cm on [am, bm] (no intercept, centered)
        X = torch.stack([am, bm], dim=1)                   # (n, 2)
        XtX = X.t() @ X
        try:
            XtX_inv = torch.linalg.inv(XtX + 1e-9 * torch.eye(2, device=d.device, dtype=d.dtype))
        except Exception:
            return zero
        coef = XtX_inv @ (X.t() @ cm)                      # [a_effect, beta]
        beta = coef[1]
        resid_c = cm - X @ coef
        sigma2_c = (resid_c * resid_c).sum() / max(n - 3, 1)
        se_beta = torch.sqrt((sigma2_c * XtX_inv[1, 1]).clamp(min=1e-12))
        denom = torch.sqrt((beta ** 2 * se_alpha ** 2 + alpha ** 2 * se_beta ** 2).clamp(min=1e-12))
        z = (alpha * beta).abs() / denom
        return torch.erf(z / (2.0 ** 0.5)).clamp(0.0, 1.0)

    def _run_lifecycle(self) -> None:
        conf_p, stab_p, evid_p = [], [], []
        for key in self._group_order:
            r = self._rls[key]
            conf_p.append(self._group_confidence(key, r).cpu().numpy())
            stab_p.append(r.stability.cpu().numpy())
            evid_p.append(r.evidence.cpu().numpy())
        conf = np.concatenate(conf_p)[np.newaxis, :]   # (1, N_hyp)
        stab = np.concatenate(stab_p)[np.newaxis, :]
        evid = np.concatenate(evid_p)[np.newaxis, :]
        self._lc.update(conf, stab, evid)

    def get_state_counts(self) -> Dict[str, int]:
        """
        Lifecycle-state histogram for the tensor-backed pool.

        Cheap by design — reads the int8 state array directly and does no
        device sync or metric concatenation, so it is safe to call once per
        row from the hosting engine's reporting path.
        """
        names = ('tentative', 'active', 'decaying', 'dead')
        counts = {n: 0 for n in names}
        if self._lc is None:
            return counts
        state = self._lc.state[0]
        for code, name in enumerate(names):
            counts[name] = int(np.count_nonzero(state == code))
        return counts

    def get_hyp_metrics(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[HypoSpec]]:
        """
        Returns (conf, stab, evid, lc_state, specs) arrays of shape (N_hyp,).
        lc_state is int8: 0=tentative, 1=active, 2=decaying, 3=dead.
        """
        conf_p, stab_p, evid_p = [], [], []
        specs_ordered: List[HypoSpec] = []
        groups = self._pool.groups()
        for key in self._group_order:
            spec_list = groups[key]
            r = self._rls[key]
            conf_p.append(self._group_confidence(key, r).cpu().numpy())
            stab_p.append(r.stability.cpu().numpy())
            evid_p.append(r.evidence.cpu().numpy())
            specs_ordered.extend(spec_list)
        conf  = np.concatenate(conf_p)
        stab  = np.concatenate(stab_p)
        evid  = np.concatenate(evid_p)
        state = self._lc.state[0]    # (N_hyp,), int8; run=0
        return conf, stab, evid, state, specs_ordered

    @staticmethod
    def _rel_type_str(rel_type: Any) -> str:
        return rel_type if isinstance(rel_type, str) else getattr(rel_type, "value", str(rel_type))

    def get_knowledge_graph(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """Export discovered hypotheses in the same format as the CPU engine.

        Mirrors OnlineDiscoveryEngine.get_knowledge_graph: the strongest
        hypotheses (by confidence) serialized to the Hypothesis.to_dict() shape,
        so downstream consumers (bridges, benchmark, graph extraction) are
        backend-agnostic.
        """
        if self._pool is None:
            return []
        conf, stab, evid, state, specs = self.get_hyp_metrics()
        items: List[Dict[str, Any]] = []
        for i, s in enumerate(specs):
            variables = list(s.variables)
            rel = self._rel_type_str(s.rel_type)
            items.append({
                "id": f"{rel}:{'|'.join(map(str, variables))}",
                "type": rel,
                "state": _LC_STATE.get(int(state[i]), "tentative"),
                "created_at": 0.0,
                "generation": 0,
                "variables": variables,
                "metrics": {
                    "fit_score": 0.0,
                    "confidence": float(conf[i]),
                    "evidence": int(evid[i]),
                    "stability": float(stab[i]),
                },
            })
        # Similarity is a collective type excluded from the RLS groups; evaluate
        # it separately from the stored series and append it to the graph.
        for s in self._pool.specs:
            if getattr(s, "F", 0) > 0 or self._rel_type_str(s.rel_type) != "similarity":
                continue
            variables = list(s.variables)
            c = self._similarity_confidence(s)
            items.append({
                "id": f"similarity:{'|'.join(map(str, variables))}",
                "type": "similarity",
                "state": "active" if c > 0.55 else "tentative",
                "created_at": 0.0,
                "generation": 0,
                "variables": variables,
                "metrics": {
                    "fit_score": 0.0,
                    "confidence": c,
                    "evidence": int(self._n_rows),
                    "stability": 0.0,
                },
            })
        items.sort(key=lambda d: d["metrics"]["confidence"], reverse=True)
        return items[:top_k]

    def _similarity_confidence(self, spec) -> float:
        """Similarity: variables in the subset co-move (are redundant).

        Mean absolute pairwise Pearson correlation across the subset — an effect
        size in [0, 1] that is near zero for independent variables (regardless of
        sample size) and near one when they move together. No bare significance,
        so it does not false-fire on noise.
        """
        if self._data is None or self._n_rows < 20:
            return 0.0
        d = self._data[0, :self._n_rows]                                  # (T, N)
        idx = [i for i, name in enumerate(self._col_names) if name in spec.variables]
        if len(idx) < 2:
            return 0.0
        X = d[:, idx]
        X = X[torch.isfinite(X).all(1)]
        if X.shape[0] < 20:
            return 0.0
        Xc = X - X.mean(0)
        std = Xc.std(0, unbiased=True)
        if bool((std < 1e-9).any()):
            return 0.0
        k = X.shape[1]
        cov = (Xc.t() @ Xc) / (X.shape[0] - 1)
        corr = (cov / torch.outer(std, std)).abs()
        off_mean = (corr.sum() - corr.diag().sum()) / (k * (k - 1))
        return float(off_mean.clamp(0.0, 1.0))

    def get_candidate_paths(self, top_k: int = 30) -> List[Candidate]:
        """Export top hypotheses as Candidate objects (parity with CPU engine)."""
        if self._pool is None:
            return []
        var_index = {name: idx for idx, name in enumerate(self._col_names)}
        conf, stab, evid, state, specs = self.get_hyp_metrics()
        order = np.argsort(-conf)
        candidates: List[Candidate] = []
        for i in order:
            if int(state[i]) == _DEAD or float(conf[i]) < 0.25:
                continue
            variables = list(specs[i].variables)
            if len(variables) < 2:
                continue
            try:
                var_indices = tuple(var_index[v] for v in variables[:2] if v in var_index)
            except KeyError:
                continue
            if len(var_indices) < 2:
                continue
            rel = self._rel_type_str(specs[i].rel_type)
            path_key = f"{var_indices}:{rel}"
            path_id = hashlib.md5(path_key.encode()).hexdigest()[:16]
            candidates.append(Candidate(
                path_id=path_id, vars=var_indices, lags=(0, 0),
                ops=("identity", "identity"), root=var_indices[0], depth=1,
                domain=0, gen_reason=f"discovery:{rel}",
            ))
            if len(candidates) >= top_k:
                break
        return candidates


# ---------------------------------------------------------------------------
# Graph extraction from GPUDiscoveryEngine
# ---------------------------------------------------------------------------

def gpu_extract_graph(
    engine: GPUDiscoveryEngine,
    conf_threshold: float = 0.50,
    min_evidence: int = 5,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """
    Extract directed graph from a GPUDiscoveryEngine.

    Returns the same (graph, edges) format as extract_graph() so callers
    can use _top_k_graph() and the rest of the benchmark pipeline unchanged.
    """
    conf, stab, evid, state, specs = engine.get_hyp_metrics()

    graph: Dict[str, List[str]] = {}
    edges: List[Dict[str, Any]] = []

    for i, s in enumerate(specs):
        # Dead hypotheses never contribute
        if state[i] == _DEAD:
            continue

        c = float(conf[i])
        e = int(evid[i])

        if c < conf_threshold or e < min_evidence:
            continue

        rel = s.rel_type
        vs  = s.variables

        if len(vs) < 2:
            continue

        if rel in _MULTI_VAR and len(vs) >= 3:
            tgt = vs[-1]
            for src in vs[:-1]:
                _add_edge(graph, src, tgt)
            edges.append(_edge_dict(vs[0], tgt, s, c, stab[i], e, symmetric=False))

        elif rel in _DIRECTIONAL:
            src, tgt = vs[0], vs[1]
            _add_edge(graph, src, tgt)
            edges.append(_edge_dict(src, tgt, s, c, stab[i], e, symmetric=False))

        elif rel in _SYMMETRIC:
            src, tgt = vs[0], vs[1]
            _add_edge(graph, src, tgt)
            _add_edge(graph, tgt, src)
            edges.append(_edge_dict(src, tgt, s, c, stab[i], e, symmetric=True))

    return graph, edges


def _add_edge(graph: Dict[str, List[str]], src: str, tgt: str) -> None:
    graph.setdefault(tgt, [])
    if src not in graph[tgt]:
        graph[tgt].append(src)


def _edge_dict(
    src: str,
    tgt: str,
    s: HypoSpec,
    conf: float,
    stab: float,
    evid: int,
    symmetric: bool,
) -> Dict[str, Any]:
    return {
        'source':     src,
        'target':     tgt,
        'variables':  list(s.variables),
        'type':       s.rel_type,
        'confidence': round(conf, 4),
        'fit_score':  0.0,
        'evidence':   evid,
        'stability':  round(float(stab), 4),
        'state':      'active',
        'symmetric':  symmetric,
    }

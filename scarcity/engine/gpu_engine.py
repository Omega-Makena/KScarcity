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

from .gpu_batch_rls import GPUBatchRLS
from .gpu_hypothesis_pool import GPUHypothesisPool, HypoSpec, LifecycleEmulator

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
    ) -> None:
        # Default to CPU — for N<200, BLAS vectorisation beats CUDA kernel overhead
        if device is None:
            device = 'cpu'
        self.device = device
        self.small_dataset_mode = small_dataset_mode

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
            self._rls[key] = GPUBatchRLS(M, F, device=self.device)

        N_hyp = sum(len(sl) for sl in groups.values())
        self._lc = LifecycleEmulator(
            N_hyp=N_hyp, R=1, small_dataset=self.small_dataset_mode
        )

        self._data = torch.zeros(1, 0, self._N, device=self.device, dtype=torch.float64)
        self.step_count = 0

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        vals = [float(row.get(c, float('nan'))) for c in self._col_names]
        new_row = torch.tensor(vals, dtype=torch.float64, device=self.device).view(1, 1, self._N)
        self._data = torch.cat([self._data, new_row], dim=1)
        t = self._data.shape[1] - 1
        self.step_count += 1

        groups = self._pool.groups()
        for key in self._group_order:
            spec_list = groups[key]
            X, Y = self._pool.extract_features_gpu(self._data, spec_list, t)
            self._rls[key].update(X.squeeze(0), Y.squeeze(0))

        if self.step_count % self._lc_interval == 0:
            self._run_lifecycle()

        return {'step': self.step_count}

    def _run_lifecycle(self) -> None:
        conf_p, stab_p, evid_p = [], [], []
        for key in self._group_order:
            r = self._rls[key]
            conf_p.append(r.confidence.cpu().numpy())
            stab_p.append(r.stability.cpu().numpy())
            evid_p.append(r.evidence.cpu().numpy())
        conf = np.concatenate(conf_p)[np.newaxis, :]   # (1, N_hyp)
        stab = np.concatenate(stab_p)[np.newaxis, :]
        evid = np.concatenate(evid_p)[np.newaxis, :]
        self._lc.update(conf, stab, evid)

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
            conf_p.append(r.confidence.cpu().numpy())
            stab_p.append(r.stability.cpu().numpy())
            evid_p.append(r.evidence.cpu().numpy())
            specs_ordered.extend(spec_list)
        conf  = np.concatenate(conf_p)
        stab  = np.concatenate(stab_p)
        evid  = np.concatenate(evid_p)
        state = self._lc.state[0]    # (N_hyp,), int8; run=0
        return conf, stab, evid, state, specs_ordered


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

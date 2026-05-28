"""
Engine-based permutation p-values (Step 1 replacement).

Covers all 15 hypothesis types from scarcity.engine:
  Pairwise  (8): causal, correlational, functional, competitive,
                 compositional, probabilistic, structural, graph
  Univariate(2): temporal, equilibrium
  Triplet   (4): synergistic, mediating, moderating, logical
  Collective(1): similarity (all variables jointly)

T_obs = hypothesis.fit_score after feeding all original data rows.
T_perm = hypothesis.fit_score after feeding permuted data rows.

No scipy/numpy test statistics (Pearson r, Granger F, ADF, etc.) are computed
directly. All statistics come from hypothesis.fit_score in scarcity.engine.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable, IO, Optional

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ── Core 10 (scarcity.engine.relationships) ──────────────────────────────────
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

# ── Extended 5 (scarcity.engine.relationships_extended) ──────────────────────
from scarcity.engine.relationships_extended import (
    MediatingHypothesis,
    ModeratingHypothesis,
    GraphHypothesis,
    SimilarityHypothesis,
    LogicalHypothesis,
)

# ---------------------------------------------------------------------------
# Hypothesis registries
# ---------------------------------------------------------------------------

# PAIRWISE: constructor(src, tgt, buffer_size) → Hypothesis
PAIRWISE_ENGINES: dict[str, Callable] = {
    'causal':
        lambda a, b, bs: CausalHypothesis(a, b, lag=2, buffer_size=bs),
    'correlational':
        lambda a, b, bs: CorrelationalHypothesis(a, b, buffer_size=bs),
    'functional':
        lambda a, b, bs: FunctionalHypothesis(a, b, degree=1, buffer_size=bs),
    'competitive':
        lambda a, b, bs: CompetitiveHypothesis(a, b, buffer_size=bs),
    'compositional':
        lambda a, b, bs: CompositionalHypothesis([a], b, buffer_size=bs),
    'probabilistic':
        lambda a, b, bs: ProbabilisticHypothesis(a, b, buffer_size=bs),
    'structural':
        lambda a, b, bs: StructuralHypothesis(a, b, buffer_size=bs),
    'graph':
        lambda a, b, bs: GraphHypothesis(a, b, buffer_size=bs),
}

# UNIVARIATE: constructor(var, buffer_size) → Hypothesis
UNIVARIATE_ENGINES: dict[str, Callable] = {
    'temporal':
        lambda v, bs: TemporalHypothesis(v, lag=2, buffer_size=bs),
    'equilibrium':
        lambda v, bs: EquilibriumHypothesis(v, buffer_size=bs),
}

# TRIPLET: constructor(a, b, c, buffer_size) → Hypothesis
# Convention: (a, b) are the input/predictor/mediator variables; c is the target/output.
# Permutation permutes c (the target), keeping a and b intact.
TRIPLET_ENGINES: dict[str, Callable] = {
    'synergistic':
        lambda a, b, c, bs: SynergisticHypothesis(a, b, c, buffer_size=bs),
    'mediating':
        lambda a, b, c, bs: MediatingHypothesis(a, b, c, buffer_size=bs),
    'moderating':
        lambda a, b, c, bs: ModeratingHypothesis(a, b, c, buffer_size=bs),
    'logical':
        lambda a, b, c, bs: LogicalHypothesis(a, b, c, buffer_size=bs),
}

# Module map for trace artifact
_CLASS_MODULE: dict[str, str] = {
    'CausalHypothesis':        'scarcity.engine.relationships',
    'CorrelationalHypothesis': 'scarcity.engine.relationships',
    'TemporalHypothesis':      'scarcity.engine.relationships',
    'FunctionalHypothesis':    'scarcity.engine.relationships',
    'EquilibriumHypothesis':   'scarcity.engine.relationships',
    'CompositionalHypothesis': 'scarcity.engine.relationships',
    'CompetitiveHypothesis':   'scarcity.engine.relationships',
    'SynergisticHypothesis':   'scarcity.engine.relationships',
    'ProbabilisticHypothesis': 'scarcity.engine.relationships',
    'StructuralHypothesis':    'scarcity.engine.relationships',
    'MediatingHypothesis':     'scarcity.engine.relationships_extended',
    'ModeratingHypothesis':    'scarcity.engine.relationships_extended',
    'GraphHypothesis':         'scarcity.engine.relationships_extended',
    'SimilarityHypothesis':    'scarcity.engine.relationships_extended',
    'LogicalHypothesis':       'scarcity.engine.relationships_extended',
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _rows(df: pd.DataFrame) -> list[dict]:
    return [df.iloc[i].to_dict() for i in range(len(df))]


def _run_engine_hypothesis(constructor: Callable, rows: list[dict]) -> float:
    """
    Instantiate hypothesis, feed all rows via update(), return fit_score.
    fit_score is T (higher = stronger evidence for this hypothesis type).
    """
    hyp = constructor()
    for row in rows:
        try:
            hyp.update(row)
        except Exception:
            pass
    score = getattr(hyp, 'fit_score', 0.0)
    return float(score) if np.isfinite(score) else 0.0


def _permute_rows(
    rows_orig: list[dict],
    perm_col: str,
    test_type: str,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Build permuted rows: only perm_col is shuffled; all other columns unchanged.
    Strategy follows step1_permutation_pvalues.permute_for_test().
    """
    vals = np.array([r[perm_col] for r in rows_orig], dtype=np.float64)
    n = len(vals)

    if test_type == 'causal':
        shift = int(rng.integers(1, max(n, 2)))
        vals_perm = np.roll(vals, shift)
    elif test_type in ('temporal', 'equilibrium'):
        try:
            fft_v = np.fft.rfft(vals)
            phases = rng.uniform(0, 2 * np.pi, len(fft_v))
            vals_perm = np.fft.irfft(np.abs(fft_v) * np.exp(1j * phases), n=n)
        except Exception:
            vals_perm = np.roll(vals, int(rng.integers(1, max(n, 2))))
    else:
        vals_perm = vals.copy()
        rng.shuffle(vals_perm)

    rows_perm = []
    for i, row in enumerate(rows_orig):
        prow = dict(row)
        prow[perm_col] = float(vals_perm[i])
        rows_perm.append(prow)
    return rows_perm


def _permute_rows_all_cols(
    rows_orig: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """
    Shuffle ALL columns independently — for SimilarityHypothesis permutation.
    Destroys multivariate cluster structure while preserving marginal distributions.
    """
    cols = list(rows_orig[0].keys())
    perm_vals = {
        col: rng.permutation([r[col] for r in rows_orig]).tolist()
        for col in cols
    }
    return [{col: perm_vals[col][i] for col in cols} for i in range(len(rows_orig))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_pvalues_engine(
    df: pd.DataFrame,
    B: int = 200,
    seed: int = 42,
    include_pairwise: Optional[list[str]] = None,
    include_univariate: Optional[list[str]] = None,
    include_triplet: Optional[list[str]] = None,
    include_collective: bool = True,
    max_triplets: int = 969,
    verbose: bool = True,
    trace_fh: Optional[IO] = None,
    call_log_fh: Optional[IO] = None,
) -> list[dict]:
    """
    Compute engine-based permutation p-values for all 15 hypothesis types.

    All T_obs and T_perm come from hypothesis.fit_score of classes in
    scarcity.engine.relationships / scarcity.engine.relationships_extended.
    No scipy/numpy test statistics are computed.

    Hypothesis types tested:
      Pairwise  (8): ordered pairs (i,j), i≠j
      Univariate(2): each variable independently
      Triplet   (4): unordered C(K,3) combinations with target=third var
      Collective(1): SimilarityHypothesis on all variables jointly

    Args:
        max_triplets: Cap on number of C(K,3) triplets to test (default: all 969).
    """
    cols = list(df.columns)
    K = len(cols)
    n = len(df)
    buf_size = max(n, 50)

    pairwise_types = include_pairwise if include_pairwise else list(PAIRWISE_ENGINES.keys())
    univariate_types = include_univariate if include_univariate else list(UNIVARIATE_ENGINES.keys())
    triplet_types = include_triplet if include_triplet else list(TRIPLET_ENGINES.keys())

    ordered_pairs = [(cols[i], cols[j]) for i in range(K) for j in range(K) if i != j]
    all_triplets_full = list(itertools.combinations(cols, 3))
    triplets = all_triplets_full[:max_triplets]

    n_tests = (len(ordered_pairs) * len(pairwise_types)
               + K * len(univariate_types)
               + len(triplets) * len(triplet_types)
               + (1 if include_collective and K >= 3 else 0))

    if verbose:
        print(f'  [engine-15] {len(ordered_pairs)} pairs × {len(pairwise_types)} pairwise '
              f'+ {K} vars × {len(univariate_types)} univ '
              f'+ {len(triplets)} triplets × {len(triplet_types)} triplet '
              f'+ {"1 collective" if include_collective else "0 collective"} '
              f'= {n_tests} tests, B={B}')

    rng = np.random.default_rng(seed)
    rows_orig = _rows(df)
    results: list[dict] = []

    def _log(msg: str) -> None:
        if call_log_fh:
            call_log_fh.write(msg + '\n')

    def _trace(rec: dict) -> None:
        if trace_fh:
            trace_fh.write(json.dumps(rec) + '\n')

    def _compute_one(
        mk: Callable,
        test_type: str,
        src: str,
        tgt: str,
        perm_col: str,
        pair_key: tuple,
        extra_fields: Optional[dict] = None,
    ) -> dict:
        """Run one (pair/var/triplet, type) test and return result dict."""
        hyp_class = mk().__class__.__name__

        def mk_fresh(fn=mk):
            return fn()

        _log(f'INIT {hyp_class}({src!r},{tgt!r}) type={test_type}')
        t_obs = _run_engine_hypothesis(mk_fresh, rows_orig)
        _log(f'T_OBS {hyp_class}({src!r},{tgt!r}) type={test_type} fit_score={t_obs:.6f}')
        _trace({
            'source': src, 'target': tgt, 'test_type': test_type,
            'hypothesis_class': hyp_class,
            'hypothesis_module': _CLASS_MODULE.get(hyp_class, 'scarcity.engine.relationships'),
            'perm_idx': -1, 'T_value': t_obs,
        })

        null_dist = np.zeros(B, dtype=np.float64)
        for b in range(B):
            rows_perm = _permute_rows(rows_orig, perm_col, test_type, rng)
            t_perm = _run_engine_hypothesis(mk_fresh, rows_perm)
            null_dist[b] = t_perm
            _log(f'T_PERM {hyp_class} type={test_type} perm={b} fit_score={t_perm:.6f}')
            _trace({
                'source': src, 'target': tgt, 'test_type': test_type,
                'hypothesis_class': hyp_class,
                'hypothesis_module': _CLASS_MODULE.get(hyp_class, 'scarcity.engine.relationships'),
                'perm_idx': b, 'T_value': t_perm,
            })

        p = (1 + int(np.sum(null_dist >= t_obs))) / (1 + B)
        rec = {
            'source': src,
            'target': tgt,
            'pair': pair_key,
            'test_type': test_type,
            'T_obs': float(t_obs),
            'p_value': float(p),
            'null_mean': float(null_dist.mean()),
            'null_std': float(null_dist.std()),
            'B': B,
            'null_distribution': [],
            'hypothesis_class': hyp_class,
            'hypothesis_module': _CLASS_MODULE.get(hyp_class, 'scarcity.engine.relationships'),
        }
        if extra_fields:
            rec.update(extra_fields)
        return rec

    # ── PAIRWISE ─────────────────────────────────────────────────────────────
    for pair_idx, (src, tgt) in enumerate(ordered_pairs):
        for test_type in pairwise_types:
            fn = PAIRWISE_ENGINES[test_type]
            results.append(_compute_one(
                mk=lambda s=src, t=tgt, tt=test_type, bs=buf_size: PAIRWISE_ENGINES[tt](s, t, bs),
                test_type=test_type, src=src, tgt=tgt,
                perm_col=tgt, pair_key=(src, tgt),
            ))
        if verbose and (pair_idx + 1) % max(1, len(ordered_pairs) // 4) == 0:
            pct = 100 * (pair_idx + 1) / len(ordered_pairs)
            print(f'  [engine-15] Pairwise: {pair_idx + 1}/{len(ordered_pairs)} pairs ({pct:.0f}%)')

    # ── UNIVARIATE ────────────────────────────────────────────────────────────
    for var in cols:
        for test_type in univariate_types:
            results.append(_compute_one(
                mk=lambda v=var, tt=test_type, bs=buf_size: UNIVARIATE_ENGINES[tt](v, bs),
                test_type=test_type, src=var, tgt=var,
                perm_col=var, pair_key=(var, var),
            ))

    # ── TRIPLETS ──────────────────────────────────────────────────────────────
    for tri_idx, (a, b, c) in enumerate(triplets):
        for test_type in triplet_types:
            # c = target/output; permute c to break the relationship
            results.append(_compute_one(
                mk=lambda va=a, vb=b, vc=c, tt=test_type, bs=buf_size: TRIPLET_ENGINES[tt](va, vb, vc, bs),
                test_type=test_type, src=a, tgt=c,
                perm_col=c, pair_key=(a, c),
                extra_fields={'mediator': b},
            ))
        if verbose and len(triplets) > 0 and (tri_idx + 1) % max(1, len(triplets) // 4) == 0:
            pct = 100 * (tri_idx + 1) / len(triplets)
            print(f'  [engine-15] Triplets: {tri_idx + 1}/{len(triplets)} ({pct:.0f}%)')

    # ── COLLECTIVE: SimilarityHypothesis ─────────────────────────────────────
    if include_collective and K >= 3:
        n_clusters = min(3, K)
        hyp_class = SimilarityHypothesis(cols, n_clusters=n_clusters,
                                          buffer_size=buf_size).__class__.__name__

        def mk_sim(c=cols, nc=n_clusters, bs=buf_size):
            return SimilarityHypothesis(c, n_clusters=nc, buffer_size=bs)

        _log(f'INIT SimilarityHypothesis(all_{K}_vars) type=similarity')
        t_obs = _run_engine_hypothesis(mk_sim, rows_orig)
        _log(f'T_OBS SimilarityHypothesis type=similarity fit_score={t_obs:.6f}')
        _trace({
            'source': 'ALL', 'target': 'ALL', 'test_type': 'similarity',
            'hypothesis_class': hyp_class,
            'hypothesis_module': 'scarcity.engine.relationships_extended',
            'perm_idx': -1, 'T_value': t_obs,
        })

        null_dist = np.zeros(B, dtype=np.float64)
        for b in range(B):
            rows_perm = _permute_rows_all_cols(rows_orig, rng)
            t_perm = _run_engine_hypothesis(mk_sim, rows_perm)
            null_dist[b] = t_perm
            _log(f'T_PERM SimilarityHypothesis type=similarity perm={b} fit_score={t_perm:.6f}')
            _trace({
                'source': 'ALL', 'target': 'ALL', 'test_type': 'similarity',
                'hypothesis_class': hyp_class,
                'hypothesis_module': 'scarcity.engine.relationships_extended',
                'perm_idx': b, 'T_value': t_perm,
            })

        p = (1 + int(np.sum(null_dist >= t_obs))) / (1 + B)
        results.append({
            'source': 'ALL', 'target': 'ALL',
            'pair': ('ALL', 'ALL'),
            'test_type': 'similarity',
            'T_obs': float(t_obs),
            'p_value': float(p),
            'null_mean': float(null_dist.mean()),
            'null_std': float(null_dist.std()),
            'B': B,
            'null_distribution': [],
            'hypothesis_class': hyp_class,
            'hypothesis_module': 'scarcity.engine.relationships_extended',
        })
        if verbose:
            print(f'  [engine-15] SimilarityHypothesis: T_obs={t_obs:.4f}, p={p:.4f}')

    return results

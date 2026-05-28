"""Phase 5 — K-Scarcity ablation runners.

Each variant disables one component at a time to isolate its contribution.
Ablation is implemented by monkey-patching engine internals after construction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from scripts.experiments.synthetic_data import generate_ground_truth
from scripts.experiments.run_kscarcity import _build_schema, _engine_discoveries

# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: dict[str, dict] = {
    'full_system': {
        'description': 'All components enabled (baseline)',
        'disable': [],
    },
    'no_meta_learning': {
        'description': 'MetaController lifecycle disabled — hypotheses never promoted or killed',
        'disable': ['meta_controller'],
    },
    'no_bandit_routing': {
        'description': 'Exploration disabled — engine uses performance mode (no explore steps)',
        'disable': ['exploration'],
    },
    'no_vectorized_rls': {
        'description': 'VectorizedHypothesisPool disabled — scalar updates only',
        'disable': ['vectorized_rls'],
    },
    'causal_only': {
        'description': 'Only CausalHypothesis type retained after initialization',
        'disable': ['other_hypothesis_types'],
    },
    'no_federation': {
        'description': 'Single node, no peer sharing (default for all variants)',
        'disable': ['federation'],
    },
}


# ---------------------------------------------------------------------------
# Ablation builder
# ---------------------------------------------------------------------------

def run_kscarcity_ablated(
    df,
    variant: str,
    buffer_size: int = 25,
) -> list[dict]:
    """Run K-Scarcity with a specific component disabled.

    Monkey-patches engine internals after construction to isolate each component.

    Args:
        df: Input DataFrame.
        variant: Key from ABLATION_VARIANTS.
        buffer_size: RLS buffer size.

    Returns:
        Standardised edge list (same format as run_kscarcity_discovery).
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    from scarcity.engine.discovery import RelationshipType

    disable = ABLATION_VARIANTS[variant]['disable']
    schema = _build_schema(df)

    engine = OnlineDiscoveryEngine(
        explore_interval=5,
        mode='balanced',
        buffer_size=buffer_size,
    )
    engine.initialize_v2(schema, use_causal=True)

    # -----------------------------------------------------------------------
    # Apply ablations
    # -----------------------------------------------------------------------

    if 'meta_controller' in disable:
        # Disable lifecycle management: manage_lifecycle becomes a no-op.
        # Hypotheses remain in their initial state forever (TENTATIVE).
        # Confidence can still grow via fit_step, but promotion/killing is disabled.
        engine.meta_controller.manage_lifecycle = lambda pool: None

    if 'exploration' in disable:
        # Disable exploration steps (the _explore_step method that adds new
        # hypothesis types like Competitive, Graph, etc. based on observed signals).
        # Mode "performance" also disables exploration_enabled.
        engine.exploration_enabled = False
        engine._explore_step = lambda: None

    if 'vectorized_rls' in disable:
        # Constrain the buffer to a minimal size — this forces the RLS computation
        # into the regime where the VectorizedHypothesisPool operates on a 1-row
        # window, effectively reducing it to a scalar update.  The pool itself
        # still exists (avoiding the scoring differences of a complete bypass).
        engine.buffer_size = 1
        # Re-create hypotheses with buffer_size=1 so existing hypothesis buffers
        # are also constrained.
        for h in engine.hypotheses.population.values():
            try:
                h.buffer_size = 1
            except AttributeError:
                pass

    if 'other_hypothesis_types' in disable:
        # Keep only CausalHypothesis instances in the pool.
        # Also disable exploration so process_row doesn't re-add other types.
        causal_type = RelationshipType.CAUSAL
        to_remove = [
            hid for hid, h in engine.hypotheses.population.items()
            if h.rel_type != causal_type
        ]
        for hid in to_remove:
            engine.hypotheses.population.pop(hid, None)
        # Prevent _explore_step from adding non-causal hypotheses
        engine.exploration_enabled = False
        engine._explore_step = lambda: None

    if 'federation' in disable:
        # Federation is not active in the default single-engine setup anyway.
        # This variant serves as an explicit label for the control condition.
        pass

    # -----------------------------------------------------------------------
    # Feed data
    # -----------------------------------------------------------------------
    for _, row_s in df.iterrows():
        engine.process_row(row_s.to_dict())

    return _engine_discoveries(engine)


# ---------------------------------------------------------------------------
# Full ablation sweep
# ---------------------------------------------------------------------------

def run_ablation_sweep(
    n_values: list[int] | None = None,
    n_seeds: int = 10,
) -> dict[str, dict[int, list[list[dict]]]]:
    """Run all ablation variants across N values and seeds.

    Args:
        n_values: N values to test. Defaults to [10, 25, 50, 100].
        n_seeds: Seeds per N (do not reduce below 10).

    Returns:
        variant_name -> {N -> [edge_list_per_seed]}
    """
    if n_values is None:
        n_values = [10, 25, 50, 100]

    results: dict[str, dict[int, list[list[dict]]]] = {
        v: {} for v in ABLATION_VARIANTS
    }

    for variant in ABLATION_VARIANTS:
        for n in n_values:
            results[variant][n] = []
            for seed in range(n_seeds):
                t0 = time.perf_counter()
                df = generate_ground_truth(N=n, seed=seed)
                discoveries = run_kscarcity_ablated(df, variant, buffer_size=min(25, n))
                elapsed = time.perf_counter() - t0
                conf = [d for d in discoveries if d['confidence'] >= 0.25]
                print(f"  {variant:25s} N={n:4d} seed={seed+1}/{n_seeds} "
                      f"... {len(conf):4d} confident ({elapsed:.1f}s)")
                results[variant][n].append(discoveries)

    return results


# ---------------------------------------------------------------------------
# Phase 5 self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 5 -- Ablation check at N=50")
    print("=" * 60)

    df = generate_ground_truth(N=50, seed=42)

    for variant, meta in ABLATION_VARIANTS.items():
        t0 = time.perf_counter()
        discoveries = run_kscarcity_ablated(df, variant, buffer_size=25)
        elapsed = time.perf_counter() - t0
        conf = [d for d in discoveries if d['confidence'] >= 0.25]
        print(f"\n{variant} ({meta['description'][:45]})")
        print(f"  Total: {len(discoveries)}  Confident: {len(conf)}  Time: {elapsed:.1f}s")
        for d in sorted(conf, key=lambda x: -x['confidence'])[:3]:
            print(f"    {d['vars']} | {d['type']:15s} | conf={d['confidence']:.3f}")

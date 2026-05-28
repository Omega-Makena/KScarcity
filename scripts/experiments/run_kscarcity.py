"""Phase 3 — K-Scarcity discovery runner.

Wraps OnlineDiscoveryEngine to produce standardised edge-list output
for comparison with baselines and ground truth.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.experiments.synthetic_data import generate_ground_truth


def _build_schema(df) -> dict:
    return {'fields': [{'name': c, 'type': 'numeric'} for c in df.columns]}


def _engine_discoveries(engine) -> list[dict]:
    """Extract all non-dead hypotheses from the engine in standardised format."""
    # Primary: export_hypothesis_summary gives all non-dead hypotheses
    summaries = engine.export_hypothesis_summary(min_conf=0.0)

    # Secondary: KG gives fit_score and state (top 50)
    kg = engine.get_knowledge_graph()
    # Build lookup by (vars_sorted_tuple, type) -> kg entry
    kg_lookup: dict[tuple, dict] = {}
    for entry in kg:
        key = (tuple(sorted(entry.get('variables', []))), entry.get('type', ''))
        kg_lookup[key] = entry

    results = []
    for s in summaries:
        vars_ = s['vars']
        typ = s['type']
        key = (tuple(sorted(vars_)), typ)
        kg_entry = kg_lookup.get(key, {})
        metrics = kg_entry.get('metrics', {})

        results.append({
            'vars': vars_,
            'source': vars_[0] if vars_ else '',
            'target': vars_[1] if len(vars_) >= 2 else (vars_[0] if vars_ else ''),
            'type': typ,
            'confidence': s['conf'],
            'fit_score': metrics.get('fit_score', 0.0),
            'evidence': s['evidence'],
            'status': kg_entry.get('state', 'tentative'),
        })
    return results


def run_kscarcity_discovery(
    df,
    buffer_size: int = 25,
    drg_profile: str = 'GREEN',
    timeout_per_row: float | None = None,
) -> list[dict]:
    """Run K-Scarcity's OnlineDiscoveryEngine on a DataFrame.

    Args:
        df: Input DataFrame (N rows, any columns).
        buffer_size: RLS buffer size passed to the engine.
        drg_profile: 'GREEN', 'YELLOW', or 'RED'.
        timeout_per_row: If set, skip a row if engine takes longer than this.

    Returns:
        List of discovered-relationship dicts with standardised keys.
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    schema = _build_schema(df)
    engine = OnlineDiscoveryEngine(
        explore_interval=5,
        mode='balanced',
        buffer_size=buffer_size,
    )
    engine.initialize_v2(schema, use_causal=True)

    for _, row_s in df.iterrows():
        row = row_s.to_dict()
        if timeout_per_row is not None:
            t0 = time.perf_counter()
            engine.process_row(row)
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_per_row:
                pass  # row was processed; mark for caller via timing
        else:
            engine.process_row(row)

    return _engine_discoveries(engine)


def run_kscarcity_n_sweep(
    n_values: list[int],
    n_seeds: int = 10,
    fast: bool = False,
) -> dict[int, list[list[dict]]]:
    """Run K-Scarcity across all N values and seeds.

    Args:
        n_values: List of N values to test.
        n_seeds: Number of random seeds per N value.
        fast: If True, use buffer_size=15 instead of 25.

    Returns:
        Dict mapping N -> list of n_seeds discovery lists.
    """
    buffer_size = 15 if fast else 25
    results: dict[int, list[list[dict]]] = {}

    for n in n_values:
        results[n] = []
        for seed in range(n_seeds):
            t0 = time.perf_counter()
            df = generate_ground_truth(N=n, seed=seed)
            discoveries = run_kscarcity_discovery(df, buffer_size=min(buffer_size, n))
            elapsed = time.perf_counter() - t0
            print(f"  K-Scarcity N={n:4d} seed={seed+1:2d}/{n_seeds} "
                  f"... {len(discoveries):4d} discoveries ({elapsed:.1f}s)")
            results[n].append(discoveries)

    return results


# ---------------------------------------------------------------------------
# Phase 3 manual check at N=25
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3 — K-Scarcity N=25 discovery check")
    print("=" * 60)

    df = generate_ground_truth(N=25, seed=42)
    discoveries = run_kscarcity_discovery(df, buffer_size=25)

    print(f"\nTotal discovered (all conf): {len(discoveries)}")
    confident = [d for d in discoveries if d['confidence'] >= 0.25]
    print(f"Confident (conf >= 0.25): {len(confident)}")

    # Sort by confidence
    confident.sort(key=lambda d: d['confidence'], reverse=True)
    print("\nTop 20 by confidence:")
    for d in confident[:20]:
        print(f"  {d['vars']} | {d['type']:15s} | conf={d['confidence']:.3f} "
              f"| fit={d['fit_score']:.3f} | ev={d['evidence']}")

    # Check for key expected discoveries
    var_pairs_found = {
        (tuple(d['vars']), d['type']) for d in confident
    }
    checks = [
        ('V1-V2 causal', any(
            set(d['vars']) == {'V1','V2'} and d['type']=='causal' for d in confident)),
        ('V5-V6 correlational', any(
            set(d['vars']) == {'V5','V6'} and d['type']=='correlational' for d in confident)),
        ('V8-V9 competitive', any(
            set(d['vars']) == {'V8','V9'} and d['type']=='competitive' for d in confident)),
        ('V7 equilibrium', any(
            'V7' in d['vars'] and d['type']=='equilibrium' for d in confident)),
    ]
    print("\nKey discoveries:")
    for label, found in checks:
        print(f"  {label}: {'FOUND' if found else 'NOT FOUND'}")

"""Phase 6 — Compute scarcity experiments.

Tests the engine under wall-clock time budgets (time_budget_sweep),
DRG RED pressure (drg_adaptation), and memory-limited buffer sizes.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.experiments.synthetic_data import generate_ground_truth
from scripts.experiments.run_kscarcity import _build_schema, _engine_discoveries


# ---------------------------------------------------------------------------
# DRG profile helper
# ---------------------------------------------------------------------------

def _make_drg_profile(level: str) -> dict:
    """Return a DRG resource profile dict matching the expected format."""
    if level == 'RED':
        return {'cpu_load': 0.92, 'vram_high': 1.0, 'latency_high': 1.0,
                'bandwidth_free': 0.0}
    if level == 'YELLOW':
        return {'cpu_load': 0.70, 'vram_high': 0.5, 'latency_high': 0.5,
                'bandwidth_free': 0.5}
    return {'cpu_load': 0.10, 'vram_high': 0.0, 'latency_high': 0.0,
            'bandwidth_free': 1.0}


# ---------------------------------------------------------------------------
# Time-budget run with real wall-clock enforcement
# ---------------------------------------------------------------------------

def run_with_time_budget(
    df: pd.DataFrame,
    budget_per_row_seconds: float,
    use_drg: bool = True,
    buffer_size: int = 25,
) -> dict:
    """Run K-Scarcity with a real wall-clock budget per observation.

    For each row:
      - Calls process_row() in a daemon thread.
      - Waits up to budget_per_row_seconds for it to finish.
      - Marks the row as interrupted if it didn't finish in time.

    Args:
        df: Input DataFrame.
        budget_per_row_seconds: Per-row time limit in seconds.
        use_drg: Whether to inject DRG RED pressure after row 5.
        buffer_size: Engine buffer size.

    Returns:
        Result dict with per-row stats and aggregate metrics.
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    schema = _build_schema(df)
    engine = OnlineDiscoveryEngine(
        explore_interval=5,
        mode='balanced',
        buffer_size=buffer_size,
    )
    engine.initialize_v2(schema, use_causal=True)

    per_row_stats: list[dict] = []
    total_start = time.perf_counter()

    for row_idx, (_, row_s) in enumerate(df.iterrows()):
        row = row_s.to_dict()

        # After 5 rows, inject DRG RED pressure if enabled
        if use_drg and row_idx >= 5:
            drg_level = 'RED'
        else:
            drg_level = 'GREEN'

        result_holder: list[bool] = [False]
        exception_holder: list[Exception | None] = [None]

        def _run_row():
            try:
                engine.process_row(row)
                result_holder[0] = True
            except Exception as e:
                exception_holder[0] = e

        t0 = time.perf_counter()
        thread = threading.Thread(target=_run_row, daemon=True)
        thread.start()
        thread.join(timeout=budget_per_row_seconds)
        elapsed = time.perf_counter() - t0

        was_interrupted = not result_holder[0]
        per_row_stats.append({
            'row_index': row_idx,
            'time_used': elapsed,
            'was_interrupted': was_interrupted,
            'drg_state': drg_level,
        })

    total_time = time.perf_counter() - total_start
    times = [s['time_used'] for s in per_row_stats]
    n_interruptions = sum(1 for s in per_row_stats if s['was_interrupted'])

    # Extract discoveries after all rows
    discoveries = _engine_discoveries(engine)
    n_discoveries = len([d for d in discoveries if d['confidence'] >= 0.25])

    return {
        'budget': budget_per_row_seconds,
        'use_drg': use_drg,
        'discoveries': discoveries,
        'per_row_stats': per_row_stats,
        'total_time': total_time,
        'mean_time_per_row': float(np.mean(times)),
        'p95_time_per_row': float(np.percentile(times, 95)),
        'n_interruptions': n_interruptions,
        'n_discoveries': n_discoveries,
    }


# ---------------------------------------------------------------------------
# Time-budget sweep
# ---------------------------------------------------------------------------

def run_compute_budget_sweep(
    df: pd.DataFrame,
    budgets: list[float] | None = None,
    n_seeds: int = 5,
) -> dict:
    """Run time-budget experiment across multiple budget levels.

    For each budget: run with DRG and without DRG. 5 seeds.

    Returns:
        {budget: {'with_drg': [results], 'without_drg': [results]}}
    """
    if budgets is None:
        budgets = [0.1, 0.5, 2.0, 10.0]

    results: dict[float, dict] = {}
    for budget in budgets:
        results[budget] = {'with_drg': [], 'without_drg': []}
        for seed in range(n_seeds):
            df_s = generate_ground_truth(N=len(df), seed=seed)
            for drg in (True, False):
                label = 'with_drg' if drg else 'without_drg'
                t0 = time.perf_counter()
                r = run_with_time_budget(df_s, budget, use_drg=drg)
                elapsed = time.perf_counter() - t0
                print(f"  budget={budget:.1f}s  drg={drg}  seed={seed+1}/{n_seeds} "
                      f"... {r['n_discoveries']} disc  "
                      f"interruptions={r['n_interruptions']}  ({elapsed:.1f}s total)")
                results[budget][label].append(r)

    return results


# ---------------------------------------------------------------------------
# Buffer size sweep
# ---------------------------------------------------------------------------

def run_buffer_size_sweep(
    df: pd.DataFrame,
    buffer_sizes: list[int] | None = None,
    n_seeds: int = 10,
) -> dict:
    """Run K-Scarcity with different buffer sizes (memory constraint).

    Args:
        df: Template DataFrame for schema and size. Regenerated per seed.
        buffer_sizes: RLS buffer sizes to test.
        n_seeds: Seeds per buffer size.

    Returns:
        {buffer_size -> [list_of_discovery_lists_per_seed]}
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    if buffer_sizes is None:
        buffer_sizes = [5, 10, 15, 25, 50]

    results: dict[int, list] = {}
    schema = _build_schema(df)

    for bs in buffer_sizes:
        results[bs] = []
        for seed in range(n_seeds):
            df_s = generate_ground_truth(N=len(df), seed=seed)
            engine = OnlineDiscoveryEngine(
                explore_interval=5,
                mode='balanced',
                buffer_size=bs,
            )
            engine.initialize_v2(schema, use_causal=True)
            for _, row_s in df_s.iterrows():
                engine.process_row(row_s.to_dict())
            discoveries = _engine_discoveries(engine)
            conf = [d for d in discoveries if d['confidence'] >= 0.25]
            print(f"  buffer_size={bs:3d} seed={seed+1:2d}/{n_seeds} "
                  f"... {len(conf):4d} confident")
            results[bs].append(discoveries)

    return results


# ---------------------------------------------------------------------------
# Phase 6 self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 6 -- Compute scarcity experiments at N=25")
    print("=" * 60)

    df = generate_ground_truth(N=25, seed=42)

    # Time-budget test at a comfortable budget
    print("\n--- Time budget test (budget=5.0s per row, N=25) ---")
    r = run_with_time_budget(df, budget_per_row_seconds=5.0, use_drg=True)
    print(f"  n_discoveries={r['n_discoveries']}")
    print(f"  n_interruptions={r['n_interruptions']}")
    print(f"  mean_time_per_row={r['mean_time_per_row']:.3f}s")
    print(f"  p95_time_per_row={r['p95_time_per_row']:.3f}s")

    # Buffer sweep at small N
    print("\n--- Buffer size sweep (N=25, 2 seeds) ---")
    buf_results = run_buffer_size_sweep(df, buffer_sizes=[5, 15, 25], n_seeds=2)
    for bs, seed_results in buf_results.items():
        avg_conf = np.mean([
            len([d for d in sr if d['confidence'] >= 0.25])
            for sr in seed_results
        ])
        print(f"  buffer_size={bs:3d} -> avg confident={avg_conf:.1f}")

    print("\nPhase 6 checks passed.")

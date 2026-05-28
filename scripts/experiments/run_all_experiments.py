"""Phase 8 — Master orchestrator.

Runs all 8 phases end-to-end. Saves intermediate results as JSON after each
phase so a later crash doesn't require re-running everything from scratch.

Usage:
    python scripts/experiments/run_all_experiments.py              # Full run
    python scripts/experiments/run_all_experiments.py --phase 1    # Only Phase 1
    python scripts/experiments/run_all_experiments.py --fast       # 5 seeds, fewer N
    python scripts/experiments/run_all_experiments.py --no-baselines
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# Phase imports
from scripts.experiments.synthetic_data import (
    generate_ground_truth,
    get_ground_truth_edges,
    get_known_null_pairs,
)
from scripts.experiments.evaluation import (
    compute_n_sweep_metrics,
    compute_n_sweep_metrics_edge_only,
    compute_scarcity_gap,
    match_discovered_to_ground_truth,
)
from scripts.experiments.run_kscarcity import run_kscarcity_n_sweep
from scripts.experiments.run_baselines import run_all_baselines_n_sweep
from scripts.experiments.run_ablation import run_ablation_sweep, ABLATION_VARIANTS
from scripts.experiments.run_compute_scarcity import (
    run_compute_budget_sweep,
    run_buffer_size_sweep,
)
from scripts.experiments.plot_results import (
    plot_n_sweep_f1,
    plot_n_sweep_precision_recall,
    plot_typed_vs_edge_only,
    plot_ablation_heatmap,
    plot_compute_budget_degradation,
    generate_latex_tables,
)


# ---------------------------------------------------------------------------
# JSON serialisation helpers (numpy → python)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, cls=_NumpyEncoder, indent=2)
    print(f"  Saved: {path}")


def _load_json(path: str):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _phase_header(n: int, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Phase {n}: {name}")
    print('='*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='K-Scarcity Validation Experiments')
    parser.add_argument('--phase', type=int, choices=list(range(1, 9)),
                        help='Run only this phase (1-8)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: 5 seeds, N=[10,25,50,100]')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Skip baseline runs')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results',
                        help='Output directory for results and figures')
    args = parser.parse_args()

    out = args.output_dir
    fig_dir = os.path.join(out, 'figures')
    raw_dir = os.path.join(out, 'raw')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    n_seeds = 5 if args.fast else 10
    n_values = [10, 25, 50, 100] if args.fast else [5, 10, 15, 20, 25, 50, 100, 200]

    run_phase = args.phase  # None = run all

    gt = get_ground_truth_edges()
    null_pairs = get_known_null_pairs()

    # -----------------------------------------------------------------------
    # Phase 1 — Synthetic data verification
    # -----------------------------------------------------------------------
    if run_phase in (None, 1):
        _phase_header(1, 'Synthetic data generation and sanity checks')
        t0 = time.perf_counter()

        df100 = generate_ground_truth(N=100, seed=42)
        assert df100.shape == (100, 10)
        assert not df100.isnull().any().any()
        assert abs(df100['V8'] + df100['V9'] - 5.0).mean() < 0.5
        assert abs(df100['V5'] + df100['V6'] + df100['V10'] - 1.0).mean() < 0.5

        print(f"  Generated N=100 DataFrame shape: {df100.shape}")
        print(f"  Ground truth edges: {len(gt)}")
        print(f"  Known null pairs: {len(null_pairs)}")
        print(f"  V8+V9 mean: {(df100['V8']+df100['V9']).mean():.3f} (expected ~5.0)")
        print(f"Phase 1 complete in {time.perf_counter()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 3 — K-Scarcity N-sweep
    # -----------------------------------------------------------------------
    ks_raw_path = os.path.join(raw_dir, 'kscarcity_results.json')
    ks_results: dict[int, list] = {}

    if run_phase in (None, 3):
        _phase_header(3, 'K-Scarcity N-sweep')
        t0 = time.perf_counter()
        ks_results = run_kscarcity_n_sweep(n_values, n_seeds=n_seeds, fast=args.fast)
        _save_json(ks_results, ks_raw_path)
        elapsed = time.perf_counter() - t0
        print(f"Phase 3 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    elif os.path.exists(ks_raw_path):
        ks_results = {int(k): v for k, v in _load_json(ks_raw_path).items()}

    # -----------------------------------------------------------------------
    # Phase 4 — Baselines N-sweep
    # -----------------------------------------------------------------------
    bl_raw_path = os.path.join(raw_dir, 'baseline_results.json')
    bl_results: dict[str, dict[int, list]] = {}

    if run_phase in (None, 4) and not args.no_baselines:
        _phase_header(4, 'Baseline runners N-sweep')
        t0 = time.perf_counter()
        bl_results = run_all_baselines_n_sweep(n_values, n_seeds=n_seeds)
        _save_json(bl_results, bl_raw_path)
        elapsed = time.perf_counter() - t0
        print(f"Phase 4 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    elif os.path.exists(bl_raw_path):
        raw = _load_json(bl_raw_path)
        bl_results = {name: {int(k): v for k, v in nd.items()} for name, nd in raw.items()}

    # -----------------------------------------------------------------------
    # Phase 2 — Evaluate (typed and edge-only)
    # -----------------------------------------------------------------------
    ks_metrics_typed: pd.DataFrame = pd.DataFrame()
    ks_metrics_edge: pd.DataFrame = pd.DataFrame()
    bl_metrics_typed: dict[str, pd.DataFrame] = {}
    scarcity_gaps: dict = {}

    if run_phase in (None, 2, 3, 4):
        _phase_header(2, 'Evaluation (typed and edge-only)')
        t0 = time.perf_counter()

        if ks_results:
            ks_metrics_typed = compute_n_sweep_metrics(ks_results, gt, null_pairs)
            ks_metrics_edge = compute_n_sweep_metrics_edge_only(ks_results, gt, null_pairs)
            print("  K-Scarcity metrics (typed):")
            print(ks_metrics_typed[['N', 'f1_mean', 'f1_std',
                                    'precision_mean', 'recall_mean']].to_string(index=False))

        if bl_results:
            for name, nd in bl_results.items():
                bl_metrics_typed[name] = compute_n_sweep_metrics(nd, gt, null_pairs)

        if ks_metrics_typed is not None and not ks_metrics_typed.empty and bl_metrics_typed:
            scarcity_gaps = compute_scarcity_gap(ks_metrics_typed, bl_metrics_typed)
            print("\n  Scarcity gaps (integrated F1 difference, positive = K-Scarcity better):")
            for name, g in scarcity_gaps.items():
                print(f"    {name:15s}: gap={g['scarcity_gap']:.3f}  "
                      f"@N=10: {g['gap_at_n10']}  @N=25: {g['gap_at_n25']}")

        print(f"Phase 2/evaluation complete in {time.perf_counter()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 5 — Ablation
    # -----------------------------------------------------------------------
    abl_raw_path = os.path.join(raw_dir, 'ablation_results.json')
    abl_results: dict[str, dict] = {}
    abl_f1_by_variant: dict[str, dict[int, float]] = {}

    if run_phase in (None, 5):
        _phase_header(5, 'Ablation study')
        t0 = time.perf_counter()
        abl_n_values = [10, 25, 50, 100]
        abl_raw = run_ablation_sweep(n_values=abl_n_values, n_seeds=n_seeds)
        _save_json(abl_raw, abl_raw_path)

        # Compute F1 per variant per N
        for variant, nd in abl_raw.items():
            abl_f1_by_variant[variant] = {}
            for n, seed_edges_list in nd.items():
                seed_metrics = [
                    match_discovered_to_ground_truth(edges, gt, null_pairs, mode='typed')
                    for edges in seed_edges_list
                ]
                mean_f1 = float(np.mean([m['f1'] for m in seed_metrics]))
                abl_f1_by_variant[variant][int(n)] = mean_f1

        print("  Ablation F1 at N=25:")
        for variant, n_f1 in abl_f1_by_variant.items():
            print(f"    {variant:25s}: {n_f1.get(25, 0.0):.3f}")
        print(f"Phase 5 complete in {time.perf_counter()-t0:.1f}s")
    elif os.path.exists(abl_raw_path):
        abl_raw = {v: {int(k): edges for k, edges in nd.items()}
                   for v, nd in _load_json(abl_raw_path).items()}
        for variant, nd in abl_raw.items():
            abl_f1_by_variant[variant] = {}
            for n, seed_edges_list in nd.items():
                seed_metrics = [
                    match_discovered_to_ground_truth(edges, gt, null_pairs, mode='typed')
                    for edges in seed_edges_list
                ]
                mean_f1 = float(np.mean([m['f1'] for m in seed_metrics]))
                abl_f1_by_variant[variant][int(n)] = mean_f1

    # -----------------------------------------------------------------------
    # Phase 6 — Compute scarcity
    # -----------------------------------------------------------------------
    comp_raw_path = os.path.join(raw_dir, 'compute_results.json')
    compute_results: dict = {}
    ref_discoveries = 48  # from Phase 3 at N=25

    if run_phase in (None, 6):
        _phase_header(6, 'Compute scarcity experiments')
        t0 = time.perf_counter()

        df25 = generate_ground_truth(N=25, seed=42)
        budgets = [0.5, 2.0, 10.0] if args.fast else [0.1, 0.5, 2.0, 10.0]
        compute_results = run_compute_budget_sweep(
            df25, budgets=budgets, n_seeds=min(n_seeds, 5))
        _save_json(compute_results, comp_raw_path)

        if ks_results and 25 in ks_results:
            all_disc = [d for dl in ks_results[25] for d in dl
                        if d.get('confidence', d.get('conf', 0)) >= 0.25]
            if all_disc:
                ref_discoveries = len(all_disc) // len(ks_results[25])

        print(f"  Reference discoveries at N=25: {ref_discoveries}")
        print(f"Phase 6 complete in {time.perf_counter()-t0:.1f}s")
    elif os.path.exists(comp_raw_path):
        compute_results = {float(k): v for k, v in _load_json(comp_raw_path).items()}

    # -----------------------------------------------------------------------
    # Phase 7 — Plotting
    # -----------------------------------------------------------------------
    if run_phase in (None, 7):
        _phase_header(7, 'Generate plots and tables')
        t0 = time.perf_counter()

        if not ks_metrics_typed.empty:
            plot_n_sweep_f1(
                ks_metrics_typed, bl_metrics_typed,
                output_path=os.path.join(fig_dir, 'n_sweep_f1.pdf'),
            )
            plot_n_sweep_precision_recall(
                ks_metrics_typed, bl_metrics_typed,
                output_path=os.path.join(fig_dir, 'n_sweep_pr.pdf'),
            )

        if not ks_metrics_typed.empty and not ks_metrics_edge.empty:
            plot_typed_vs_edge_only(
                ks_metrics_typed, ks_metrics_edge,
                output_path=os.path.join(fig_dir, 'typed_vs_edge.pdf'),
            )

        if abl_f1_by_variant:
            plot_ablation_heatmap(
                abl_f1_by_variant,
                output_path=os.path.join(fig_dir, 'ablation_heatmap.pdf'),
            )

        if compute_results:
            plot_compute_budget_degradation(
                compute_results,
                reference_n_discoveries=ref_discoveries,
                output_path=os.path.join(fig_dir, 'compute_budget.pdf'),
            )

        generate_latex_tables(
            ks_metrics_typed if not ks_metrics_typed.empty else pd.DataFrame(),
            bl_metrics_typed,
            scarcity_gaps,
            ablation_f1=abl_f1_by_variant,
            output_path=os.path.join(fig_dir, 'tables.tex'),
        )

        print(f"Phase 7 complete in {time.perf_counter()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("All requested phases complete.")
    print(f"Results in: {os.path.abspath(out)}")
    if not ks_metrics_typed.empty:
        row25 = ks_metrics_typed[ks_metrics_typed['N'] == 25]
        if len(row25):
            print(f"\nK-Scarcity @ N=25: "
                  f"F1={float(row25['f1_mean'].iloc[0]):.3f} "
                  f"+/- {float(row25['f1_std'].iloc[0]):.3f}")
    print('='*60)


if __name__ == '__main__':
    main()

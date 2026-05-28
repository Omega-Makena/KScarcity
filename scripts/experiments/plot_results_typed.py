"""
Plotting for the typed discovery validation suite (v3 fixes).

Generates 5 figures:
  1. local_vs_fed_recall.png    -- paired bar chart: local vs federated per-type recall
  2. threshold_sweep.png        -- confidence threshold sweep: P/R/F1 for local vs fed
  3. specialist_calibration.png -- specialist discovery counts before/after calibration
  4. capability_unlock.png      -- which GT types are unlocked/improved by federation
  5. ablation_f1.png            -- F1 per ablation variant

All figures saved to results/typed_validation/plots/.

Usage:
    python scripts/experiments/plot_results_typed.py
    python scripts/experiments/plot_results_typed.py --fast
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

PLOT_DIR = _ROOT / 'results' / 'typed_validation' / 'plots'

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 200,
})

COLOR_LOCAL = '#1f77b4'
COLOR_FED   = '#ff7f0e'
COLOR_SPEC  = '#2ca02c'
COLOR_GREY  = '#aaaaaa'


def _save(fig, name: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: local vs federated per-type recall (paired bar)
# ---------------------------------------------------------------------------

def plot_local_vs_fed_recall(
    local_recall: dict[str, float],
    fed_recall: dict[str, float],
    title: str = 'Local vs Federated: Per-type GT Recall',
) -> Path:
    types = sorted(set(local_recall) | set(fed_recall))
    loc_vals = [local_recall.get(t, 0.0) for t in types]
    fed_vals = [fed_recall.get(t, 0.0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_loc = ax.bar(x - width / 2, loc_vals, width, label='Local', color=COLOR_LOCAL, alpha=0.85)
    bars_fed = ax.bar(x + width / 2, fed_vals, width, label='Federated', color=COLOR_FED, alpha=0.85)

    ax.set_xlabel('Relationship Type')
    ax.set_ylabel('GT Recall')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    # Annotate bars > 0
    for bar in bars_loc:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=7)
    for bar in bars_fed:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    return _save(fig, 'local_vs_fed_recall.png')


# ---------------------------------------------------------------------------
# Figure 2: confidence threshold sweep
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    sweep_data: dict[float, dict],
    title: str = 'Confidence Threshold Sweep: F1',
) -> Path:
    thresholds = sorted(sweep_data.keys())
    loc_f1 = [sweep_data[t]['local']['f1'] for t in thresholds]
    fed_f1 = [sweep_data[t]['federated']['f1'] for t in thresholds]
    loc_p  = [sweep_data[t]['local']['precision'] for t in thresholds]
    fed_p  = [sweep_data[t]['federated']['precision'] for t in thresholds]
    loc_r  = [sweep_data[t]['local']['recall'] for t in thresholds]
    fed_r  = [sweep_data[t]['federated']['recall'] for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, metric, loc_vals, fed_vals, ylabel in zip(
        axes,
        ['F1', 'Precision', 'Recall'],
        [loc_f1, loc_p, loc_r],
        [fed_f1, fed_p, fed_r],
        ['F1 Score', 'Precision', 'Recall'],
    ):
        ax.plot(thresholds, loc_vals, '-o', color=COLOR_LOCAL, label='Local', linewidth=2)
        ax.plot(thresholds, fed_vals, '-s', color=COLOR_FED, label='Federated', linewidth=2)
        ax.set_xlabel('Min Confidence')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric} vs Threshold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, 'threshold_sweep.png')


# ---------------------------------------------------------------------------
# Figure 3: specialist calibration (before/after discovery counts)
# ---------------------------------------------------------------------------

def plot_specialist_calibration(
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    title: str = 'Specialist Calibration: Discovery Count Reduction',
) -> Path:
    types = sorted(set(counts_before) | set(counts_after))
    before = [counts_before.get(t, 0) for t in types]
    after  = [counts_after.get(t, 0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, before, width, label='Before calibration', color=COLOR_GREY, alpha=0.8)
    ax.bar(x + width / 2, after, width, label='After calibration', color=COLOR_LOCAL, alpha=0.85)

    ax.set_xlabel('Specialist Type')
    ax.set_ylabel('Number of Discoveries')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha='right', fontsize=9)
    ax.legend()

    # Annotate reduction % for the three key types
    key_types = {'mediating', 'synergistic', 'functional'}
    for i, t in enumerate(types):
        b = counts_before.get(t, 0)
        a = counts_after.get(t, 0)
        if t in key_types and b > 0:
            pct = 100 * (b - a) / b
            ax.text(x[i] + width / 2, a + max(before) * 0.02,
                    f'-{pct:.0f}%', ha='center', va='bottom', fontsize=8,
                    color='darkred', fontweight='bold')

    fig.tight_layout()
    return _save(fig, 'specialist_calibration.png')


# ---------------------------------------------------------------------------
# Figure 4: capability unlock (horizontal bar showing type status)
# ---------------------------------------------------------------------------

def plot_capability_unlock(
    unlock_analysis: dict[str, dict],
    title: str = 'Federation Capability Unlock by Type',
) -> Path:
    types = sorted(unlock_analysis.keys())
    loc_recalls = [unlock_analysis[t]['local_recall'] for t in types]
    fed_recalls = [unlock_analysis[t]['fed_recall'] for t in types]

    statuses = [unlock_analysis[t].get('status', '') for t in types]
    colors = []
    for t in types:
        info = unlock_analysis[t]
        if info.get('unlocked', False):
            colors.append('#d62728')  # red = unlocked (new capability)
        elif info['fed_recall'] > info['local_recall']:
            colors.append('#ff7f0e')  # orange = improved
        elif info['fed_recall'] < info['local_recall']:
            colors.append('#7f7f7f')  # grey = regressed
        else:
            colors.append('#1f77b4')  # blue = same

    y = np.arange(len(types))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + height / 2, fed_recalls, height, label='Federated', color=COLOR_FED, alpha=0.85)
    ax.barh(y - height / 2, loc_recalls, height, label='Local', color=COLOR_LOCAL, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlabel('GT Recall')
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    fig.tight_layout()
    return _save(fig, 'capability_unlock.png')


# ---------------------------------------------------------------------------
# Figure 5: ablation F1 comparison
# ---------------------------------------------------------------------------

def plot_ablation_f1(
    ablation_results: dict[str, dict],
    title: str = 'Ablation Study: F1 Score per Variant',
) -> Path:
    variants = list(ablation_results.keys())
    f1_scores  = [ablation_results[v]['overall']['f1'] for v in variants]
    recall     = [ablation_results[v]['overall']['recall'] for v in variants]
    precision  = [ablation_results[v]['overall']['precision'] for v in variants]

    x = np.arange(len(variants))
    width = 0.25

    colors = [COLOR_LOCAL if v == 'full_system' else COLOR_GREY for v in variants]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x - width, precision, width, label='Precision', color='#aec7e8', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#ffbb78', alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1', color=colors, alpha=0.85)

    ax.set_xlabel('Ablation Variant')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, max(max(f1_scores + recall + precision) * 1.2, 0.15))
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    # Annotate F1 bars
    for bar in bars3:
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=0)

    fig.tight_layout()
    return _save(fig, 'ablation_f1.png')


# ---------------------------------------------------------------------------
# Data generation helpers (for standalone runs)
# ---------------------------------------------------------------------------

def _load_or_run_federation_data(fast: bool) -> dict:
    """Load existing federation results or run them fresh."""
    cached = _ROOT / 'results' / 'typed_validation' / 'federation_typed_results.json'
    if cached.exists():
        data = json.loads(cached.read_text(encoding='utf-8'))
        # Check if we have threshold_sweep in the format we need
        if 'threshold_sweep' in data and 'delta_recall' in data:
            return data
    # Run fresh
    from scripts.experiments.run_federation_typed import (
        run_kscarcity_local_typed, run_kscarcity_federated_typed,
        prepare_data, run_confidence_threshold_sweep,
        run_capability_unlock_analysis, compare_local_vs_federated_typed,
        GT_COLS,
    )
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth, get_known_null_relationships
    )
    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()
    _, df_ken_work, dfs_peers = prepare_data(fast=fast, verbose=True)
    local_disc = run_kscarcity_local_typed(df_ken_work, use_causal=False, verbose=False)
    fed_disc = run_kscarcity_federated_typed(df_ken_work, dfs_peers, use_causal=False, verbose=False)
    comparison = compare_local_vs_federated_typed(local_disc, fed_disc, gt, null_pairs, verbose=False)
    sweep = run_confidence_threshold_sweep(local_disc, fed_disc, gt, verbose=False)
    unlock = run_capability_unlock_analysis(local_disc, fed_disc, gt, verbose=False)
    return {
        'delta_recall': comparison['delta_recall'],
        'recall': comparison['recall'],
        'overall': comparison['overall'],
        'unlock_analysis': unlock,
        'threshold_sweep': {k: v for k, v in sweep.items()},
    }


def _load_or_run_ablation_data(fast: bool) -> dict:
    cached = _ROOT / 'results' / 'typed_validation' / 'ablation_typed_results.json'
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf-8'))
    from scripts.experiments.run_ablation_typed import run_full_ablation
    from scripts.experiments.data_loader import load_country_data
    df = load_country_data('KEN')
    from scripts.experiments.run_federation_typed import GT_COLS
    avail = [c for c in GT_COLS if c in df.columns]
    df_work = df[avail].dropna()
    if fast:
        df_work = df_work.head(15)
    results = run_full_ablation(df_work, verbose=False)
    out: dict = {}
    for var, info in results.items():
        out[var] = {'overall': info['overall'], 'n_discoveries': info['n_discoveries']}
    return out


def _run_specialist_calibration(fast: bool) -> tuple[dict, dict]:
    """Return (counts_before, counts_after) for specialist calibration plot."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.run_federation_typed import GT_COLS
    from scripts.experiments.specialist_baselines import (
        discover_mediating, discover_synergistic, discover_functional,
        discover_temporal, discover_causal, discover_correlational,
        discover_competitive, discover_compositional, discover_equilibrium,
        discover_structural,
    )

    df = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df.columns]
    df_work = df[avail].dropna()
    if fast:
        df_work = df_work.head(15)

    # Uncalibrated counts (old defaults)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        before = {
            'temporal':      len(discover_temporal(df_work)),
            'causal':        len(discover_causal(df_work)),
            'correlational': len(discover_correlational(df_work)),
            'competitive':   len(discover_competitive(df_work)),
            'compositional': len(discover_compositional(df_work)),
            'equilibrium':   len(discover_equilibrium(df_work)),
            'mediating':     len(discover_mediating(df_work, significance=0.20, min_r_prefilter=0.20, min_indirect=0.01)),
            'synergistic':   len(discover_synergistic(df_work, significance=0.10, min_r_main=0.15, min_interaction_coef=0.01)),
            'functional':    len(discover_functional(df_work, significance=0.10, min_r2_gain=0.05, min_r2_abs=0.0)),
            'structural':    len(discover_structural(df_work)),
        }
        after = {
            'temporal':      len(discover_temporal(df_work)),
            'causal':        len(discover_causal(df_work)),
            'correlational': len(discover_correlational(df_work)),
            'competitive':   len(discover_competitive(df_work)),
            'compositional': len(discover_compositional(df_work)),
            'equilibrium':   len(discover_equilibrium(df_work)),
            'mediating':     len(discover_mediating(df_work)),
            'synergistic':   len(discover_synergistic(df_work)),
            'functional':    len(discover_functional(df_work)),
            'structural':    len(discover_structural(df_work)),
        }
    return before, after


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print('Generating typed validation plots...')
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load / run federation data
    print('  Loading federation results...')
    fed_data = _load_or_run_federation_data(args.fast)

    # Figure 1: local vs fed recall
    local_recall = {
        t: fed_data['recall']['local'].get(t, {}).get('recall', 0.0)
        for t in fed_data.get('recall', {}).get('local', {})
    }
    fed_recall = {
        t: fed_data['recall']['federated'].get(t, {}).get('recall', 0.0)
        for t in fed_data.get('recall', {}).get('federated', {})
    }
    # Fallback: use delta_recall if 'recall' key absent
    if not local_recall and 'delta_recall' in fed_data:
        delta = fed_data['delta_recall']
        local_recall = {t: 0.0 for t in delta}
        fed_recall   = {t: delta[t] for t in delta}

    p = plot_local_vs_fed_recall(local_recall, fed_recall)
    print(f'  Saved: {p.name}')

    # Figure 2: threshold sweep
    sweep_raw = fed_data.get('threshold_sweep', {})
    if sweep_raw:
        # Convert keys back to float, values to full metrics dicts
        sweep: dict[float, dict] = {}
        for k, v in sweep_raw.items():
            thresh = float(k)
            if isinstance(v, dict) and 'local' in v and 'federated' in v:
                sweep[thresh] = v
            elif isinstance(v, dict) and 'local_f1' in v:
                sweep[thresh] = {
                    'local':    {'f1': v['local_f1'], 'precision': 0.0, 'recall': 0.0},
                    'federated': {'f1': v['fed_f1'], 'precision': 0.0, 'recall': 0.0},
                }
        if sweep:
            p = plot_threshold_sweep(sweep)
            print(f'  Saved: {p.name}')

    # Figure 3: specialist calibration
    print('  Running specialist calibration (before/after)...')
    try:
        counts_before, counts_after = _run_specialist_calibration(args.fast)
        p = plot_specialist_calibration(counts_before, counts_after)
        print(f'  Saved: {p.name}')
    except Exception as exc:
        print(f'  WARNING: specialist calibration plot failed: {exc}')

    # Figure 4: capability unlock
    unlock_raw = fed_data.get('unlock_analysis', {})
    if unlock_raw:
        p = plot_capability_unlock(unlock_raw)
        print(f'  Saved: {p.name}')

    # Figure 5: ablation F1
    print('  Loading ablation results...')
    try:
        ablation_data = _load_or_run_ablation_data(args.fast)
        if ablation_data:
            p = plot_ablation_f1(ablation_data)
            print(f'  Saved: {p.name}')
    except Exception as exc:
        print(f'  WARNING: ablation plot failed: {exc}')

    print(f'\nAll plots saved to {PLOT_DIR.relative_to(_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate typed validation plots'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Use fast/cached data where possible')
    args = parser.parse_args()
    main(args)

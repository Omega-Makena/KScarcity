"""
Master orchestrator for K-Scarcity Real-Data Typed Discovery Validation.

Runs the full typed validation suite on World Bank macro data (Kenya, Tanzania,
Uganda) using theory-grounded typed relationships from economic literature.

Evaluation questions:
  Q1  Per-type recall   (specialists vs K-Scarcity)
  Q2  Specialist comparison (which specialist works best per type)
  Q3  False positive cost (null-pair FP rate, sign-wrong fraction)
  Q4  Scarcity curves   (per-type F1 vs N)

Usage:
    python scripts/experiments/run_typed_validation.py             # full run
    python scripts/experiments/run_typed_validation.py --fast      # N-sweep only KEN, N=[8,15,21]
    python scripts/experiments/run_typed_validation.py --no-kscarcity  # skip engine
    python scripts/experiments/run_typed_validation.py --country TZA   # one country
    python scripts/experiments/run_typed_validation.py --list      # show GT summary

Exit codes: 0 = success, 1 = error
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OUT_DIR = _ROOT / 'results' / 'typed_validation'
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_FIG_DIR = _OUT_DIR / 'figures'
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GT_COLS = [
    'gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
    'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
    'current_account', 'gcf', 'electricity_access', 'internet_users',
    'school_enrollment', 'life_expectancy', 'broad_money',
]

N_SWEEP_FULL = [8, 12, 15, 20, 25, 30]
N_SWEEP_FAST = [8, 15, 21]

COUNTRIES_FULL = ['KEN', 'TZA', 'UGA']
COUNTRIES_FAST = ['KEN']


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def _save_fig(fig, name: str) -> Path:
    import matplotlib
    path = _FIG_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    matplotlib.pyplot.close(fig)
    return path


def plot_n_sweep_f1(
    sweep_spec: dict[int, dict],
    sweep_ksc: dict[int, dict] | None,
    country: str,
) -> Path:
    """Line chart: F1 vs N for specialists vs K-Scarcity."""
    import matplotlib.pyplot as plt

    ns_spec = sorted(k for k in sweep_spec if 'overall' in sweep_spec[k])
    ns_ksc = sorted(k for k in sweep_ksc if isinstance(sweep_ksc[k], list)) if sweep_ksc else []

    f1_spec = [sweep_spec[n]['overall']['f1'] for n in ns_spec]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns_spec, f1_spec, 'o-', label='Specialists (combined)', color='steelblue')

    if sweep_ksc is not None and ns_ksc:
        from scripts.experiments.evaluation_typed import compare_specialists
        from scripts.experiments.ground_truth_typed import get_typed_ground_truth
        gt = get_typed_ground_truth()
        f1_ksc = []
        for n in ns_ksc:
            disc_list = sweep_ksc[n]
            cmp = compare_specialists({'k_scarcity': disc_list}, gt)
            f1_ksc.append(cmp['k_scarcity']['f1'])
        ax.plot(ns_ksc, f1_ksc, 's--', label='K-Scarcity', color='darkorange')

    ax.set_xlabel('Observations (N)')
    ax.set_ylabel('F1 (strict type match)')
    ax.set_title(f'Typed Discovery F1 vs N - {country}')
    ax.set_ylim(0, 1)
    ax.legend()
    return _save_fig(fig, f'typed_f1_n_sweep_{country}')


def plot_per_type_recall_heatmap(
    recall_info: dict[str, dict],
    title: str,
    name: str,
) -> Path:
    """Horizontal bar chart showing recall per GT type."""
    import matplotlib.pyplot as plt

    types = sorted(recall_info.keys())
    recalls = [recall_info[t]['recall'] for t in types]
    n_gt = [recall_info[t]['n_gt'] for t in types]

    fig, ax = plt.subplots(figsize=(7, max(3, len(types) * 0.45)))
    bars = ax.barh(types, recalls, color='steelblue', alpha=0.8)
    for bar, n, r in zip(bars, n_gt, recalls):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{r:.2f}  (GT={n})', va='center', fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel('Recall')
    ax.set_title(title)
    return _save_fig(fig, name)


def plot_type_scarcity_curves(
    sweep: dict[int, dict],
    country: str,
    method: str,
) -> Path:
    """Per-type recall vs N curves."""
    import matplotlib.pyplot as plt
    from scripts.experiments.evaluation_typed import summarise_per_type_sweep

    df = summarise_per_type_sweep(sweep)
    if df.empty:
        return _FIG_DIR / 'empty.png'

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap('tab10')
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], 'o-', label=col, color=cmap(i % 10), alpha=0.8)
    ax.set_xlabel('Observations (N)')
    ax.set_ylabel('Recall per type')
    ax.set_title(f'Per-type recall vs N - {method} - {country}')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', ncol=2, fontsize=7)
    return _save_fig(fig, f'type_scarcity_{method}_{country}')


def plot_specialist_f1_bars(cmp: dict[str, dict], country: str) -> Path:
    """Bar chart of F1 per specialist."""
    import matplotlib.pyplot as plt

    specs = sorted(cmp.keys())
    f1_vals = [cmp[s]['f1'] for s in specs]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(specs, f1_vals, color='teal', alpha=0.8)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=7)
    ax.set_ylim(0, max(f1_vals) * 1.4 + 0.01)
    ax.set_ylabel('F1')
    ax.set_title(f'Specialist F1 vs Ground Truth - {country}')
    ax.set_xticklabels(specs, rotation=30, ha='right')
    return _save_fig(fig, f'specialist_f1_{country}')


# ---------------------------------------------------------------------------
# Per-country runner
# ---------------------------------------------------------------------------

def run_country(
    country: str,
    n_values: list[int],
    run_kscarcity: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full typed validation for one country.

    Returns a results dict with all Q1-Q4 outputs and figure paths.
    """
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import (
        compute_per_type_recall,
        compare_specialists,
        false_positive_analysis,
        n_sweep_typed,
        summarise_n_sweep,
        summarise_per_type_sweep,
    )

    print(f'\n{"="*60}')
    print(f'  Country: {country}')
    print(f'{"="*60}')

    # ---- Load data ----
    t0 = time.time()
    df_raw = load_country_data(country)
    cols_available = [c for c in GT_COLS if c in df_raw.columns]
    df = df_raw[cols_available].dropna()
    n_rows, n_cols = df.shape

    if verbose:
        print(f'  Data: {n_rows} complete rows, {n_cols} variables')
        missing_cols = [c for c in GT_COLS if c not in cols_available]
        if missing_cols:
            print(f'  Missing cols: {missing_cols}')

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    # Adjust n_values to not exceed actual data
    n_values_adj = [n for n in n_values if n <= n_rows] + [n_rows]
    n_values_adj = sorted(set(n_values_adj))

    # ---- Run specialists (full dataset) ----
    print(f'\n  [Specialists] Running all 10 specialists on N={n_rows}...')
    disc_spec = run_all_specialists(df, verbose=False)
    n_disc_spec = sum(len(v) for v in disc_spec.values())
    print(f'  [Specialists] {n_disc_spec} total discoveries')

    # Q1: Per-type recall - specialists
    recall_spec = compute_per_type_recall(disc_spec, gt)
    # Q2: Specialist comparison
    cmp_spec = compare_specialists(disc_spec, gt)
    # Q3: FP analysis - specialists
    fp_spec = false_positive_analysis(disc_spec, gt, null_pairs)

    # ---- Print Q1 ----
    print('\n  Q1 - Per-type recall (specialists, full dataset):')
    for t, info in sorted(recall_spec.items()):
        bar = '#' * int(info['recall'] * 20)
        print(f'    {t:15s} [{bar:<20}] {info["n_discovered"]}/{info["n_gt"]}  '
              f'recall={info["recall"]:.3f}')

    # ---- Print Q2 ----
    print('\n  Q2 - Specialist comparison:')
    best_f1 = 0.0
    best_spec = ''
    for s, m in sorted(cmp_spec.items(), key=lambda x: -x[1]['f1']):
        marker = '  *' if m['f1'] == max(x['f1'] for x in cmp_spec.values()) else '   '
        print(f'  {marker} {s:15s}  P={m["precision"]:.3f}  R={m["recall"]:.3f}  '
              f'F1={m["f1"]:.3f}  (#disc={m["n_discoveries"]})')
        if m['f1'] > best_f1:
            best_f1, best_spec = m['f1'], s

    # ---- Print Q3 ----
    print('\n  Q3 - False positive analysis:')
    print(f'    Null-pair FP rate  : {fp_spec["null_fp_rate"]:.3f}  '
          f'({fp_spec["null_fp_count"]}/{len(null_pairs)})')
    print(f'    Total FP (strict)  : {fp_spec["total_fp_all"]}')
    print(f'    Sign-wrong frac    : {fp_spec["sign_wrong_frac"]:.3f}  '
          f'({fp_spec["sign_wrong_count"]}/{fp_spec["gt_matched_total"]})')
    for detail in fp_spec['null_fp_details']:
        if detail['n_fires']:
            print(f'    ! null-fire: {detail["pair"]}  ->  {detail["fired_by"]}')

    # ---- Q4: N-sweep - specialists ----
    print(f'\n  Q4 - N-sweep specialists: {n_values_adj}')
    sweep_spec = n_sweep_typed(df, n_values_adj, gt, null_pairs)
    summary_spec = summarise_n_sweep(sweep_spec)
    print(summary_spec[['discoveries', 'tp_unique', 'fp', 'precision', 'recall', 'f1']].to_string())

    # ---- K-Scarcity runner ----
    ksc_results: dict = {}
    sweep_ksc: dict[int, list[dict]] | None = None

    if run_kscarcity:
        from scripts.experiments.run_kscarcity_typed import (
            run_kscarcity_on_df,
            run_kscarcity_n_sweep,
        )
        from scripts.experiments.evaluation_typed import (
            compute_per_type_recall as cptr,
            compare_specialists as cspec,
            false_positive_analysis as fpa,
        )

        print(f'\n  [K-Scarcity] Running engine on N={n_rows}...')
        disc_ksc = run_kscarcity_on_df(df, buffer_size=min(30, n_rows),
                                        min_conf=0.15, verbose=verbose)
        ksc_as_dict = {'k_scarcity': disc_ksc}

        recall_ksc = cptr(ksc_as_dict, gt)
        cmp_ksc = cspec(ksc_as_dict, gt)
        fp_ksc = fpa(ksc_as_dict, gt, null_pairs)
        m_ksc = cmp_ksc['k_scarcity']

        print(f'\n  [K-Scarcity] Full dataset metrics:')
        print(f'    TP={m_ksc["tp"]}  FP={m_ksc["fp"]}  FN={m_ksc["fn"]}')
        print(f'    P={m_ksc["precision"]:.3f}  R={m_ksc["recall"]:.3f}  '
              f'F1={m_ksc["f1"]:.3f}')
        print(f'    Null-pair FP rate: {fp_ksc["null_fp_rate"]:.3f}')

        print(f'\n  [K-Scarcity] Per-type recall:')
        for t, info in sorted(recall_ksc.items()):
            if info['n_gt'] > 0:
                print(f'    {t:15s}: {info["n_discovered"]}/{info["n_gt"]} '
                      f'recall={info["recall"]:.3f}')

        print(f'\n  [K-Scarcity] N-sweep: {n_values_adj}')
        sweep_ksc = run_kscarcity_n_sweep(
            df, n_values_adj, buffer_size=min(30, n_rows),
            min_conf=0.15, verbose=False,
        )

        ksc_results = {
            'disc_full': disc_ksc,
            'recall': recall_ksc,
            'comparison': m_ksc,
            'fp': fp_ksc,
            'sweep': {n: disc_ksc for n in n_values_adj},  # stored for JSON
        }

    # ---- Figures ----
    print('\n  Generating figures...')
    try:
        _setup_matplotlib()

        fig_f1 = plot_n_sweep_f1(sweep_spec, sweep_ksc, country)
        print(f'    {fig_f1}')

        fig_recall = plot_per_type_recall_heatmap(
            recall_spec,
            f'Specialist per-type recall - {country}',
            f'recall_by_type_spec_{country}',
        )
        print(f'    {fig_recall}')

        fig_curves = plot_type_scarcity_curves(sweep_spec, country, 'specialists')
        print(f'    {fig_curves}')

        fig_bars = plot_specialist_f1_bars(cmp_spec, country)
        print(f'    {fig_bars}')

        if run_kscarcity and sweep_ksc:
            ksc_sweep_for_plot = {}
            for n, disc_list in sweep_ksc.items():
                from scripts.experiments.evaluation_typed import (
                    compare_specialists as _cs,
                    compute_per_type_recall as _cpr,
                    false_positive_analysis as _fpa,
                )
                per_spec = _cs({'k_scarcity': disc_list}, gt)
                per_type = _cpr({'k_scarcity': disc_list}, gt)
                fp_i = _fpa({'k_scarcity': disc_list}, gt, null_pairs)
                all_gt_matched: set[int] = set()
                total_fp = 0
                from scripts.experiments.evaluation_typed import _any_gt_match
                for d in disc_list:
                    entry = _any_gt_match(d, gt, strict_type=True)
                    if entry is not None:
                        all_gt_matched.add(gt.index(entry))
                    else:
                        total_fp += 1
                tp_u = len(all_gt_matched)
                fn_t = len(gt) - tp_u
                prec = tp_u / (tp_u + total_fp) if (tp_u + total_fp) else 0.0
                rec = tp_u / len(gt) if gt else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
                ksc_sweep_for_plot[n] = {
                    'n_rows': n,
                    'n_cols': n_cols,
                    'n_discoveries_total': len(disc_list),
                    'per_specialist': per_spec,
                    'per_type_recall': per_type,
                    'fp_analysis': fp_i,
                    'overall': {
                        'tp_unique': tp_u, 'fp': total_fp, 'fn': fn_t,
                        'precision': round(prec, 4), 'recall': round(rec, 4),
                        'f1': round(f1, 4), 'null_fp_rate': fp_i['null_fp_rate'],
                    },
                }
            fig_ksc = plot_type_scarcity_curves(ksc_sweep_for_plot, country, 'k_scarcity')
            print(f'    {fig_ksc}')

    except Exception as exc:
        print(f'    Figure generation error: {exc}')

    elapsed = time.time() - t0

    # ---- Package results ----
    country_results = {
        'country': country,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_gt_relationships': len(gt),
        'n_null_pairs': len(null_pairs),
        'elapsed_sec': round(elapsed, 1),
        'specialists': {
            'n_discoveries_total': n_disc_spec,
            'per_type_recall': recall_spec,
            'comparison': cmp_spec,
            'fp_analysis': fp_spec,
        },
        'k_scarcity': ksc_results,
    }

    return country_results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_gt_summary() -> None:
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
        get_ground_truth_by_type,
        get_all_gt_variables,
    )
    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()
    by_type = get_ground_truth_by_type()
    vars_ = get_all_gt_variables()

    print(f'\nGround Truth Summary:')
    print(f'  Total relationships: {len(gt)}')
    print(f'  Known null pairs   : {len(null_pairs)}')
    print(f'  Distinct variables : {len(vars_)}')
    print(f'\n  By type:')
    for t, rels in sorted(by_type.items()):
        print(f'    {t:15s}: {len(rels)}')
    print(f'\n  Variables: {sorted(vars_)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='K-Scarcity Real-Data Typed Discovery Validation'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Fast run: KEN only, N=[8,15,21]')
    parser.add_argument('--no-kscarcity', dest='no_kscarcity', action='store_true',
                        help='Skip K-Scarcity engine (specialists only)')
    parser.add_argument('--country', type=str, default=None,
                        help='Single country code (KEN, TZA, UGA)')
    parser.add_argument('--list', action='store_true',
                        help='Print ground truth summary and exit')
    args = parser.parse_args(argv)

    if args.list:
        print_gt_summary()
        return 0

    # Select config
    if args.country:
        countries = [args.country.upper()]
    elif args.fast:
        countries = COUNTRIES_FAST
    else:
        countries = COUNTRIES_FULL

    n_values = N_SWEEP_FAST if args.fast else N_SWEEP_FULL
    run_kscarcity = not args.no_kscarcity

    print('K-Scarcity Real-Data Typed Discovery Validation')
    print(f'  Countries   : {countries}')
    print(f'  N sweep     : {n_values}')
    print(f'  K-Scarcity  : {run_kscarcity}')
    print_gt_summary()

    all_results: dict[str, dict] = {}
    errors: list[str] = []

    for country in countries:
        try:
            result = run_country(
                country,
                n_values=n_values,
                run_kscarcity=run_kscarcity,
                verbose=True,
            )
            all_results[country] = result
        except Exception as exc:
            print(f'\n  ERROR in {country}: {exc}')
            import traceback
            traceback.print_exc()
            errors.append(f'{country}: {exc}')

    # ---- Cross-country summary ----
    if len(all_results) > 0:
        print(f'\n{"="*60}')
        print('  Cross-country summary')
        print(f'{"="*60}')
        print(f'  {"Country":6s}  {"Rows":>5}  {"Spec-F1":>8}  {"Null-FP":>8}')
        for cc, res in sorted(all_results.items()):
            spec_f1s = [res['specialists']['comparison'][s]['f1']
                        for s in res['specialists']['comparison']]
            best_f1 = max(spec_f1s) if spec_f1s else 0.0
            null_fp = res['specialists']['fp_analysis']['null_fp_rate']
            print(f'  {cc:6s}  {res["n_rows"]:5d}  {best_f1:8.3f}  {null_fp:8.3f}')

    # ---- Save results ----
    out_path = _OUT_DIR / 'typed_validation_results.json'
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Results saved: {out_path}')
    except Exception as exc:
        print(f'\n  Could not save JSON: {exc}')

    print(f'\n  Figures: {_FIG_DIR}')

    if errors:
        print(f'\n  Errors: {len(errors)}')
        for e in errors:
            print(f'    {e}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

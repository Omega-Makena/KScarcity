"""
Per-Parent Causal Ablation Analysis (§48)

For each target where causal filtering affects forecasting, reconstructs which
specific parents were filtered by DoWhy and classifies them by Granger predictive
utility, producing the X%/Y%/Z% trichotomy:

  Spurious       --low Granger R2 (<0.05): DoWhy correctly filtered
  Predictively   --R2>=0.05, some DoWhy votes but below majority threshold
    useful despite failing
  Real-but-      --R2>=0.05, zero DoWhy votes: DoWhy failed to identify a real effect
    unidentified

Usage:
    python benchmark/scripts/benchmark_causal_ablation.py
    python benchmark/scripts/benchmark_causal_ablation.py --targets exports_gdp
    python benchmark/scripts/benchmark_causal_ablation.py --targets exports_gdp govt_consumption
    python benchmark/scripts/benchmark_causal_ablation.py --r2-threshold 0.10
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.real_data.world_bank_loader import prepare_multi_country_data

# ----------------------------------------------------------------------------─
# Config
# ----------------------------------------------------------------------------─

CAUSAL_ARTIFACT_ROOT = _ROOT / 'artifacts' / 'causal_benchmark' / 'runs'
EFFECT_THRESHOLD      = 0.5     # fallback |estimate| > this when CI unavailable
CAUSAL_VOTE_THRESHOLD = 0.5     # parent retained if sig/total >= this
R2_SPURIOUS           = 0.05    # Granger R2 below this → spurious
R2_USEFUL             = 0.05    # same threshold, useful if above this


# ----------------------------------------------------------------------------─
# Artifact parsing
# ----------------------------------------------------------------------------─

def _is_significant(record: dict) -> bool:
    """Mirror of benchmark_forecasting_causal._is_significant, applied to raw dict."""
    est = record.get('estimate')
    ci  = record.get('confidence_intervals')

    if est is None:
        return False
    if isinstance(est, list):
        valid = [x for x in est if x is not None and not np.isnan(float(x))]
        est = float(np.mean(valid)) if valid else float('nan')
    try:
        est = float(est)
    except (TypeError, ValueError):
        return False
    if np.isnan(est):
        return False

    if ci is not None:
        try:
            lower, upper = ci
            if isinstance(lower, list):
                lower = float(np.mean(lower))
            if isinstance(upper, list):
                upper = float(np.mean(upper))
            lower, upper = float(lower), float(upper)
            if not (np.isnan(lower) or np.isnan(upper)):
                return lower > 0 or upper < 0
        except Exception:
            pass

    return abs(est) > EFFECT_THRESHOLD


def _parse_effects(jsonl_path: Path) -> dict:
    """
    Parse one effects.jsonl → {parent: {'total': N, 'sig': M, 'estimands': {name: 'sig'|'ns'|'null'}}}
    """
    parent_votes = {}
    with open(jsonl_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            parent = rec.get('spec', {}).get('treatment') or _extract_treatment(rec.get('spec_id', ''))
            if not parent:
                continue
            ename = rec.get('estimand_type', 'UNK')

            if parent not in parent_votes:
                parent_votes[parent] = {'total': 0, 'sig': 0, 'estimands': {}}

            if rec.get('estimate') is None and rec.get('confidence_intervals') is None:
                parent_votes[parent]['estimands'][ename] = 'null'
                continue

            parent_votes[parent]['total'] += 1
            is_sig = _is_significant(rec)
            if is_sig:
                parent_votes[parent]['sig'] += 1
            parent_votes[parent]['estimands'][ename] = 'sig' if is_sig else 'ns'

    return parent_votes


def _extract_treatment(spec_id: str) -> str:
    """Fallback: parse treatment from spec_id string."""
    for part in spec_id.split('|'):
        if part.startswith('treatment='):
            return part.split('=', 1)[1]
    return ''


def load_all_votes(target: str) -> dict:
    """
    Aggregate per-parent vote stats across all cutoff years for the given target.
    Returns {parent: {'cutoffs_seen': N, 'total_votes': N, 'sig_votes': N,
                       'retained_cutoffs': N, 'filtered_cutoffs': N, 'sig_rate': float}}
    """
    pattern = f'bench_*_{target}'
    run_dirs = sorted(CAUSAL_ARTIFACT_ROOT.glob(pattern))
    if not run_dirs:
        print(f"  [warn] No artifacts found for {target} at {CAUSAL_ARTIFACT_ROOT}")
        return {}

    aggregate = {}  # parent -> accumulated stats

    for run_dir in run_dirs:
        effects_path = run_dir / 'effects.jsonl'
        if not effects_path.exists():
            continue

        cutoff_votes = _parse_effects(effects_path)

        for parent, stats in cutoff_votes.items():
            if parent not in aggregate:
                aggregate[parent] = {
                    'cutoffs_seen': 0,
                    'total_votes': 0,
                    'sig_votes': 0,
                    'retained_cutoffs': 0,
                    'filtered_cutoffs': 0,
                    'estimand_counts': {},
                }
            agg = aggregate[parent]
            agg['cutoffs_seen'] += 1
            agg['total_votes']  += stats['total']
            agg['sig_votes']    += stats['sig']

            # Per-cutoff retention decision
            if stats['total'] == 0:
                agg['retained_cutoffs'] += 1  # fallback: no votes → retain
            elif stats['sig'] / stats['total'] >= CAUSAL_VOTE_THRESHOLD:
                agg['retained_cutoffs'] += 1
            else:
                agg['filtered_cutoffs'] += 1

            # Accumulate estimand-level outcomes
            for ename, outcome in stats['estimands'].items():
                ec = agg['estimand_counts']
                if ename not in ec:
                    ec[ename] = {'sig': 0, 'ns': 0, 'null': 0}
                ec[ename][outcome] = ec[ename].get(outcome, 0) + 1

    # Compute aggregate sig_rate
    for parent, agg in aggregate.items():
        tv = agg['total_votes']
        agg['sig_rate'] = agg['sig_votes'] / tv if tv > 0 else 0.0

    return aggregate


# ----------------------------------------------------------------------------─
# Granger utility (univariate OLS lag-1 R2)
# ----------------------------------------------------------------------------─

def compute_granger_r2(ken_df: pd.DataFrame, parent: str, target: str) -> float | None:
    """
    Univariate OLS: parent[t] → target[t+1], using the full KEN series.
    Returns adjusted R2 or None if data insufficient.
    """
    if parent not in ken_df.columns or target not in ken_df.columns:
        return None

    series = ken_df[[parent, target]].dropna()
    if len(series) < 10:
        return None

    X = series[parent].iloc[:-1].values.reshape(-1, 1)
    y = series[target].iloc[1:].values

    if np.std(X) < 1e-10 or np.std(y) < 1e-10:
        return 0.0

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Adjusted R2 (1 predictor)
    n = len(y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2
    return float(adj_r2)


# ----------------------------------------------------------------------------─
# Classification
# ----------------------------------------------------------------------------─

def classify_parent(agg: dict, r2: float | None, r2_thresh: float = R2_SPURIOUS) -> str:
    """
    Trichotomy for causally-filtered parents (filtered_cutoffs > retained_cutoffs):
      Spurious          --R2 < r2_thresh
      Real-unidentified --R2 >= r2_thresh AND sig_rate == 0
      Useful-failed     --R2 >= r2_thresh AND 0 < sig_rate < CAUSAL_VOTE_THRESHOLD
    """
    if r2 is None or r2 < r2_thresh:
        return 'Spurious'
    if agg['sig_rate'] == 0.0:
        return 'Real-unidentified'
    return 'Useful-failed'


# ----------------------------------------------------------------------------─
# Reporting
# ----------------------------------------------------------------------------─

def _bar(v, width=20):
    filled = round(v * width)
    return '#' * filled + '.' * (width - filled)


def print_target_report(target: str, aggregate: dict, ken_df: pd.DataFrame, r2_thresh: float):
    print(f"\n{'='*70}")
    print(f"  TARGET: {target}")
    print(f"{'='*70}")

    if not aggregate:
        print("  No causal artifacts found.")
        return

    total_parents = len(aggregate)

    # Classify each parent
    rows = []
    for parent, agg in sorted(aggregate.items(), key=lambda x: -x[1]['cutoffs_seen']):
        r2 = compute_granger_r2(ken_df, parent, target)
        n_cutoffs = agg['cutoffs_seen']
        retained  = agg['retained_cutoffs']
        filtered  = agg['filtered_cutoffs']
        sig_rate  = agg['sig_rate']

        # Overall retention decision: majority of cutoffs
        is_retained = retained >= filtered

        rows.append({
            'parent':     parent,
            'cutoffs':    n_cutoffs,
            'retained':   retained,
            'filtered':   filtered,
            'is_retained': is_retained,
            'sig_rate':   sig_rate,
            'r2':         r2,
            'class':      classify_parent(agg, r2, r2_thresh) if not is_retained else 'RETAINED',
        })

    retained_rows = [r for r in rows if r['is_retained']]
    filtered_rows = [r for r in rows if not r['is_retained']]

    print(f"\n  Across {len(list(CAUSAL_ARTIFACT_ROOT.glob(f'bench_*_{target}')))} "
          f"cutoff years, {total_parents} unique parents evaluated")
    print(f"  Retained (majority vote): {len(retained_rows)}")
    print(f"  Filtered (majority vote): {len(filtered_rows)}")

    # -- Retained parents ------------------------------------------------------
    print(f"\n  RETAINED PARENTS ({len(retained_rows)})")
    print(f"  {'Parent':<22} {'Cutoffs':>7} {'Ret/Flt':>8} {'SigRate':>8} {'Granger R2':>11}")
    print(f"  {'-'*60}")
    for r in sorted(retained_rows, key=lambda x: -x['sig_rate']):
        r2_str = f"{r['r2']:.3f}" if r['r2'] is not None else '  N/A'
        print(f"  {r['parent']:<22} {r['cutoffs']:>7} {r['retained']:>3}/{r['filtered']:<3}   "
              f"{r['sig_rate']:>6.2f}   {r2_str:>11}")

    # -- Filtered parents ------------------------------------------------------
    print(f"\n  FILTERED PARENTS ({len(filtered_rows)})")
    print(f"  {'Parent':<22} {'Ret/Flt':>8} {'SigRate':>8} {'Granger R2':>11} {'Class'}")
    print(f"  {'-'*65}")
    for r in sorted(filtered_rows, key=lambda x: (x['class'], -x['sig_rate'])):
        r2_str = f"{r['r2']:.3f}" if r['r2'] is not None else '  N/A'
        print(f"  {r['parent']:<22} {r['retained']:>3}/{r['filtered']:<3}   "
              f"{r['sig_rate']:>6.2f}   {r2_str:>11}   {r['class']}")

    # -- Classification summary ------------------------------------------------
    if not filtered_rows:
        print("\n  No filtered parents to classify.")
        return

    counts = {'Spurious': 0, 'Real-unidentified': 0, 'Useful-failed': 0}
    for r in filtered_rows:
        counts[r['class']] = counts.get(r['class'], 0) + 1

    n_flt = len(filtered_rows)
    print(f"\n  CLASSIFICATION OF {n_flt} FILTERED PARENTS")
    print(f"  {'Category':<28}  {'Count':>5}  {'%':>5}  Bar")
    print(f"  {'-'*55}")
    labels = [
        ('Spurious',          'R2 < {:.2f}: DoWhy correctly filtered'.format(r2_thresh)),
        ('Useful-failed',     'R2>={:.2f}, some DoWhy votes but < 50%'.format(r2_thresh)),
        ('Real-unidentified', 'R2>={:.2f}, 0 DoWhy votes: identification failure'.format(r2_thresh)),
    ]
    for cat, desc in labels:
        n = counts.get(cat, 0)
        pct = 100 * n / n_flt
        print(f"  {cat:<28}  {n:>5}  {pct:>4.0f}%  {_bar(n/n_flt)}")
        print(f"    {desc}")

    x_pct = 100 * counts['Spurious'] / n_flt
    y_pct = 100 * counts['Real-unidentified'] / n_flt
    z_pct = 100 * counts['Useful-failed'] / n_flt
    print(f"\n  STATEMENT: Of {n_flt} causally-filtered parents for {target},")
    print(f"    {x_pct:.0f}% are spurious (Granger R2<{r2_thresh}),")
    print(f"    {y_pct:.0f}% are real but unidentified by DoWhy (R2>={r2_thresh}, 0 sig estimands),")
    print(f"    {z_pct:.0f}% are predictively useful despite failing causal validation.")

    return {
        'target': target,
        'n_retained': len(retained_rows),
        'n_filtered': n_flt,
        'spurious_pct': x_pct,
        'real_unidentified_pct': y_pct,
        'useful_failed_pct': z_pct,
        'retained_rows': retained_rows,
        'filtered_rows': filtered_rows,
    }


def print_estimand_breakdown(target: str, aggregate: dict):
    """Show how often each estimand type agreed/disagreed, overall."""
    print(f"\n  ESTIMAND AGREEMENT BREAKDOWN --{target}")
    print(f"  {'Estimand':<20} {'Sig':>6} {'NS':>6} {'Null':>6} {'Sig%':>6}")
    print(f"  {'-'*46}")

    estimand_totals = {}
    for agg in aggregate.values():
        for ename, cnts in agg.get('estimand_counts', {}).items():
            if ename not in estimand_totals:
                estimand_totals[ename] = {'sig': 0, 'ns': 0, 'null': 0}
            for k, v in cnts.items():
                estimand_totals[ename][k] = estimand_totals[ename].get(k, 0) + v

    for ename in ['ATE', 'ATT', 'ATC', 'CATE', 'LATE', 'MEDIATION_NDE', 'MEDIATION_NIE']:
        if ename not in estimand_totals:
            continue
        t = estimand_totals[ename]
        sig  = t.get('sig', 0)
        ns   = t.get('ns', 0)
        null = t.get('null', 0)
        total = sig + ns
        pct = 100 * sig / total if total > 0 else 0
        print(f"  {ename:<20} {sig:>6} {ns:>6} {null:>6} {pct:>5.0f}%")


# ----------------------------------------------------------------------------─
# Main
# ----------------------------------------------------------------------------─

def main():
    parser = argparse.ArgumentParser(description="Per-parent causal ablation (§48)")
    parser.add_argument('--targets', nargs='+',
                        default=['exports_gdp', 'govt_consumption'],
                        help='Targets to analyse')
    parser.add_argument('--r2-threshold', type=float, default=R2_SPURIOUS,
                        help='Granger R2 below which a parent is classified Spurious')
    parser.add_argument('--show-estimands', action='store_true',
                        help='Show per-estimand agreement breakdown')
    args = parser.parse_args()

    r2_thresh = args.r2_threshold

    print("=" * 70)
    print("  CAUSAL ABLATION -- PER-PARENT DoWhy FILTER ANALYSIS (S48)")
    print(f"  Targets: {', '.join(args.targets)}")
    print(f"  Granger R2 spurious threshold: {r2_thresh}")
    print(f"  Causal vote threshold: {CAUSAL_VOTE_THRESHOLD}")
    print("=" * 70)

    # Load Kenya data once
    print("\n  Loading Kenya macro data...", flush=True)
    try:
        data = prepare_multi_country_data(['KEN'])
        ken_df = data['KEN'].ffill().bfill()
        for col in ken_df.columns:
            if ken_df[col].isnull().any():
                ken_df[col] = ken_df[col].fillna(ken_df[col].mean())
        print(f"  KEN data loaded: {len(ken_df)} years × {len(ken_df.columns)} cols "
              f"({int(ken_df.index.min())}–{int(ken_df.index.max())})")
    except Exception as exc:
        print(f"  [error] Could not load KEN data: {exc}")
        ken_df = pd.DataFrame()

    all_summaries = []
    for target in args.targets:
        print(f"\n  Aggregating votes for {target}...", flush=True)
        aggregate = load_all_votes(target)
        if not aggregate:
            continue

        summary = print_target_report(target, aggregate, ken_df, r2_thresh)
        if summary:
            all_summaries.append(summary)

        if args.show_estimands:
            print_estimand_breakdown(target, aggregate)

    # -- Cross-target summary --------------------------------------------------
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("  CROSS-TARGET SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Target':<22} {'Ret':>4} {'Flt':>4} {'Spur%':>7} {'RealUnid%':>10} {'UsefulFail%':>12}")
        print(f"  {'-'*63}")
        for s in all_summaries:
            print(f"  {s['target']:<22} {s['n_retained']:>4} {s['n_filtered']:>4} "
                  f"{s['spurious_pct']:>6.0f}% {s['real_unidentified_pct']:>9.0f}% "
                  f"{s['useful_failed_pct']:>11.0f}%")

    print("\n  Done.\n")


if __name__ == '__main__':
    main()

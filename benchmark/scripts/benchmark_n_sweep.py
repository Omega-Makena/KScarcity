"""
Synthetic N x SNR benchmark: when does graph-conditioning help forecasting?

Sweeps N in {50,100,200,500,1000,3000} x SNR in {1,2,5,10}.
Produces a 2D surface: delta_MAE = blind_MAE - graph_MAE.
Positive delta = graph-conditioning helps; negative = hurts.

Graph discovery method: Granger causality (lag-1 OLS F-test, BH-FDR q=0.10).
This is fast, directly tests lag-1 predictability, and isolates the feature-
selection mechanism from engine-specific overhead.

Synthetic DAG (6 variables):
    X1 (exogenous) -> Y  (lag-1, coeff=1.0) [TRUE PARENT]
    X2 (exogenous) -> Y  (lag-1, coeff=1.0) [TRUE PARENT]
    X3 (exogenous)       -- NOT a parent of Y
    Z = 0.9*Y + noise    -- correlated with Y but not a cause (spurious)
    Y[t] = X1[t-1] + X2[t-1] + noise(sigma)

SNR = signal_power / noise_power = sqrt(2) / noise_std  (two unit-variance parents)
    SNR=1:  noise_std = sqrt(2)  ~ 1.41
    SNR=2:  noise_std = sqrt(2)/2 ~ 0.71
    SNR=5:  noise_std = sqrt(2)/5 ~ 0.28
    SNR=10: noise_std = sqrt(2)/10 ~ 0.14

Evaluation: leave-last-30% as test set; XGBoost h=1 direct prediction.
Seeds: 10 per condition (averaged) for stability at small N.

Usage:
    python benchmark/scripts/benchmark_n_sweep.py
    python benchmark/scripts/benchmark_n_sweep.py --seeds 3 --no-save
"""

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_VALUES   = [50, 100, 200, 500, 1000, 3000]
SNR_VALUES = [1, 2, 5, 10]
N_SEEDS    = 10
TEST_FRAC  = 0.30
MIN_TRAIN  = 10   # minimum training observations for XGBoost

VARS = ['X1', 'X2', 'X3', 'Z', 'Y']
TARGET = 'Y'
TRUE_PARENTS = {'X1', 'X2'}

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _snr_to_noise(snr: float) -> float:
    """noise_std such that SNR = sqrt(Var(X1)+Var(X2)) / noise_std = sqrt(2)/noise_std."""
    return float(np.sqrt(2) / snr)


def generate_data(n: int, snr: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise_std = _snr_to_noise(snr)

    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.standard_normal(n)
    Y  = np.zeros(n)
    Z  = np.zeros(n)

    for t in range(1, n):
        Y[t] = X1[t-1] + X2[t-1] + rng.normal(0, noise_std)

    Z = 0.9 * Y + rng.normal(0, 0.3, size=n)

    return pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Z': Z, 'Y': Y})

# ---------------------------------------------------------------------------
# Graph discovery: Granger lag-1 F-test with BH-FDR
# ---------------------------------------------------------------------------

def granger_parents(df: pd.DataFrame, target: str, candidates: list,
                    q_fdr: float = 0.10) -> set:
    """
    Return set of variables that Granger-cause `target` at lag=1.
    F-test on regression: target[t] ~ candidate[t-1] + intercept.
    BH-FDR correction at level q_fdr.
    """
    n = len(df)
    y = df[target].values[1:]      # t = 1..n-1
    pvals = {}
    for c in candidates:
        if c == target:
            continue
        x = df[c].values[:-1]      # t = 0..n-2
        if len(x) < 5:
            continue
        # OLS: y = a*x + b
        X_ = np.column_stack([x, np.ones(len(x))])
        try:
            beta, res, _, _ = np.linalg.lstsq(X_, y, rcond=None)
            ss_res = float(np.sum((y - X_ @ beta) ** 2))
            # Null: y = b only
            ss_null = float(np.sum((y - y.mean()) ** 2))
            df_num, df_den = 1, len(y) - 2
            if df_den < 1 or ss_res <= 0:
                continue
            f = ((ss_null - ss_res) / df_num) / (ss_res / df_den)
            p = float(1 - scipy_stats.f.cdf(f, df_num, df_den))
            pvals[c] = p
        except Exception:
            continue

    if not pvals:
        return set()

    # BH-FDR correction
    items  = sorted(pvals.items(), key=lambda x: x[1])
    m      = len(items)
    cutoff = 0.0
    for i, (_, p) in enumerate(items):
        if p <= (i + 1) / m * q_fdr:
            cutoff = p
    return {c for c, p in pvals.items() if p <= max(cutoff, 1e-10) or
            (cutoff == 0.0 and p <= q_fdr / m)}


def granger_parents_strict(df: pd.DataFrame, target: str, candidates: list,
                            q_fdr: float = 0.10) -> set:
    """Wrapper — returns empty set if fewer than 3 training rows."""
    if len(df) < 5:
        return set()
    return granger_parents(df, target, candidates, q_fdr)

# ---------------------------------------------------------------------------
# XGBoost prediction
# ---------------------------------------------------------------------------

def _xgb_predict(X_tr, y_tr, X_te):
    try:
        import xgboost as xgb
        if len(X_tr) < MIN_TRAIN or X_tr.shape[1] == 0:
            return np.full(len(X_te), float(np.mean(y_tr)))
        m = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    except Exception:
        return np.full(len(X_te), float(np.mean(y_tr)))


def _arima_predict(series_tr, n_pred):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = ARIMA(series_tr, order=(1, 1, 0)).fit()
            fc = m.forecast(steps=n_pred)
            return np.array(fc)
    except Exception:
        return np.full(n_pred, float(np.mean(series_tr)))

# ---------------------------------------------------------------------------
# Single (N, SNR, seed) evaluation
# ---------------------------------------------------------------------------

def evaluate_one(n: int, snr: float, seed: int) -> dict:
    df = generate_data(n, snr, seed)

    n_test  = max(1, int(n * TEST_FRAC))
    n_train = n - n_test

    if n_train < MIN_TRAIN + 1:
        return None

    train = df.iloc[:n_train]
    test  = df.iloc[n_train:]

    candidates = [c for c in VARS if c != TARGET]

    # Discover graph from training data
    parents_disc = granger_parents_strict(train, TARGET, candidates)

    # Build h=1 direct training pairs from training data
    # X[t], Y[t+1] for t in 0..n_train-2
    tr_feat = train.iloc[:-1]    # t = 0..n_train-2
    tr_y    = train[TARGET].values[1:]  # Y[t+1]

    # Test pairs: predict Y[n_train..n-1] from X[n_train-1..n-2]
    te_feat = df.iloc[n_train-1 : n-1]  # features at t-1 for each test point
    te_y    = test[TARGET].values

    # Blind: all features
    blind_cols = candidates
    X_tr_blind = tr_feat[blind_cols].values
    X_te_blind = te_feat[blind_cols].values

    pred_blind = _xgb_predict(X_tr_blind, tr_y, X_te_blind)
    mae_blind  = float(np.mean(np.abs(te_y - pred_blind)))

    # Graph: discovered parents only (fallback to blind if empty)
    if parents_disc:
        graph_cols = sorted(parents_disc)
    else:
        graph_cols = blind_cols  # no discovery → same as blind

    X_tr_graph = tr_feat[graph_cols].values
    X_te_graph = te_feat[graph_cols].values

    pred_graph = _xgb_predict(X_tr_graph, tr_y, X_te_graph)
    mae_graph  = float(np.mean(np.abs(te_y - pred_graph)))

    # Oracle: true parents only (upper bound on what graph discovery can achieve)
    oracle_cols = sorted(TRUE_PARENTS)
    X_tr_orac = tr_feat[oracle_cols].values
    X_te_orac = te_feat[oracle_cols].values
    pred_orac  = _xgb_predict(X_tr_orac, tr_y, X_te_orac)
    mae_oracle = float(np.mean(np.abs(te_y - pred_orac)))

    # Discovery quality metrics
    tp = len(parents_disc & TRUE_PARENTS)
    fp = len(parents_disc - TRUE_PARENTS)
    fn = len(TRUE_PARENTS - parents_disc)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float('nan'))

    return {
        'n': n, 'snr': snr, 'seed': seed,
        'n_train': n_train, 'n_test': n_test,
        'n_discovered': len(parents_disc),
        'precision': precision, 'recall': recall, 'f1': f1,
        'mae_blind': mae_blind,
        'mae_graph': mae_graph,
        'mae_oracle': mae_oracle,
        'delta': mae_blind - mae_graph,          # positive = graph helps
        'oracle_delta': mae_blind - mae_oracle,   # upper bound delta
    }

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(n_values, snr_values, n_seeds):
    results = []
    total = len(n_values) * len(snr_values) * n_seeds
    done  = 0
    t0    = time.time()

    for n in n_values:
        for snr in snr_values:
            seed_results = []
            for seed in range(n_seeds):
                r = evaluate_one(n, snr, seed)
                if r is not None:
                    seed_results.append(r)
                    results.append(r)
                done += 1

            # Print per-condition summary
            if seed_results:
                delta_mean = float(np.mean([r['delta'] for r in seed_results]))
                delta_std  = float(np.std( [r['delta'] for r in seed_results]))
                f1_mean    = float(np.nanmean([r['f1'] for r in seed_results]))
                elapsed    = time.time() - t0
                helps = 'HELPS' if delta_mean > 0.01 else ('HURTS' if delta_mean < -0.01 else 'NEUTRAL')
                print(f"  N={n:5d}  SNR={snr:2d}  delta={delta_mean:+.4f}+/-{delta_std:.3f}"
                      f"  disc_F1={f1_mean:.2f}  {helps}"
                      f"  [{done}/{total}  {elapsed:.0f}s]", flush=True)

    return results


def print_surface(results):
    """Print the 2D delta surface as a table."""
    df = pd.DataFrame(results)
    agg = df.groupby(['n', 'snr'])['delta'].mean().reset_index()

    print()
    print('=' * 80)
    print('2D SURFACE: delta_MAE = blind_MAE - graph_MAE')
    print('  Positive = graph-conditioning HELPS XGBoost forecasting')
    print('  Negative = graph-conditioning HURTS  (noise from spurious parents)')
    print('=' * 80)
    print(f"  {'N':>6}" + ''.join(f"  SNR={s:2d}" for s in SNR_VALUES))
    print('  ' + '-' * (8 + 8 * len(SNR_VALUES)))

    for n in N_VALUES:
        row = f"  {n:6d}"
        for snr in SNR_VALUES:
            sub = agg[(agg['n'] == n) & (agg['snr'] == snr)]
            if len(sub):
                v = float(sub['delta'].iloc[0])
                tag = '+' if v > 0.01 else ('-' if v < -0.01 else '~')
                row += f"  {v:+6.3f}{tag}"
            else:
                row += '      N/A'
        print(row)

    print()
    print('=' * 80)
    print('2D SURFACE: Granger-discovery F1 (True Parents = X1, X2)')
    print('=' * 80)
    agg_f1 = df.groupby(['n', 'snr'])['f1'].mean().reset_index()
    print(f"  {'N':>6}" + ''.join(f"  SNR={s:2d}" for s in SNR_VALUES))
    print('  ' + '-' * (8 + 8 * len(SNR_VALUES)))
    for n in N_VALUES:
        row = f"  {n:6d}"
        for snr in SNR_VALUES:
            sub = agg_f1[(agg_f1['n'] == n) & (agg_f1['snr'] == snr)]
            if len(sub):
                v = float(sub['f1'].iloc[0])
                row += f"    {v:.2f} "
            else:
                row += '      N/A'
        print(row)

    # Find crossover point
    print()
    print('Crossover analysis (N where delta changes sign at each SNR):')
    for snr in SNR_VALUES:
        sub = agg[agg['snr'] == snr].sort_values('n')
        crossover = None
        prev_sign = None
        for _, row in sub.iterrows():
            sign = 1 if row['delta'] > 0.01 else (-1 if row['delta'] < -0.01 else 0)
            if prev_sign == -1 and sign >= 0:
                crossover = int(row['n'])
                break
            prev_sign = sign
        print(f"  SNR={snr:2d}: crossover N ~ {crossover if crossover else '>3000 (no crossover observed)'}")


def main():
    parser = argparse.ArgumentParser(description='N x SNR sweep benchmark')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    print('=' * 80)
    print('SYNTHETIC N x SNR BENCHMARK')
    print(f'  N values:   {N_VALUES}')
    print(f'  SNR values: {SNR_VALUES}')
    print(f'  Seeds:      {args.seeds}')
    print(f'  DAG: X1->Y, X2->Y (true parents); X3, Z (spurious)')
    print(f'  Discovery: Granger lag-1 F-test, BH-FDR q=0.10')
    print(f'  Evaluation: XGBoost blind vs +graph, last {TEST_FRAC:.0%} held out')
    print('=' * 80)

    t0 = time.time()
    results = run_sweep(N_VALUES, SNR_VALUES, args.seeds)
    elapsed = time.time() - t0

    print(f'\n  Total: {len(results)} records, {elapsed:.0f}s')
    print_surface(results)

    if not args.no_save:
        out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        path = out_dir / 'n_snr_sweep.csv'
        df.to_csv(path, index=False, float_format='%.4f')
        print(f'\n  Saved to {path.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()

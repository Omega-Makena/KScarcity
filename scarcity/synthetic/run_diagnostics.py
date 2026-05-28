"""
Diagnostic Script — Steps 1 & 2 from the recovery plan.

Step 1: Extract raw hypothesis state from the engine after streaming.
Step 2: Validate generator output quality with statistical tests.

Outputs:
  - benchmark_results/raw_hypotheses.csv
  - benchmark_results/generator_validation.csv
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats

# ── Setup paths ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE, "benchmark_schema.json")
OUT_DIR = os.path.join(BASE, "benchmark_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate data ────────────────────────────────────────────────────────
from scarcity.synthetic.benchmark.generator import create_benchmark_generator

print("=" * 70, flush=True)
print("STEP 2: Validating Generator Output Quality", flush=True)
print("=" * 70, flush=True)

gen = create_benchmark_generator(SCHEMA_PATH, seed=42)
N = 2000
df = gen.generate(N)

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

validation_rows = []

for rel in schema['relationships']:
    rt = rel['type']
    row = {"type": rt, "status": "UNKNOWN", "metric": "", "value": 0.0, "expected": "", "pass": False}

    try:
        if rt == 'temporal':
            var = rel['variable']
            lag = rel['lags'][0]
            y = df[var].values
            corr = np.corrcoef(y[lag:], y[:-lag])[0, 1]
            row.update(metric="AR_corr", value=round(corr, 4),
                       expected=f"coeff={rel['coefficients'][0]}")
            row["pass"] = corr > 0.1
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'causal':
            src, tgt = rel['source'], rel['target']
            lag = rel['lags'][0]
            x, y = df[src].values, df[tgt].values
            corr = np.corrcoef(y[lag:], x[:-lag])[0, 1]
            # Granger F-test
            n = len(y) - lag
            y_dep = y[lag:]
            y_lag = y[:-lag][:n]
            x_lag = x[:-lag][:n]
            # Restricted model: y_t ~ y_{t-1}
            from numpy.linalg import lstsq
            X_r = np.column_stack([np.ones(n), y_lag])
            b_r, _, _, _ = lstsq(X_r, y_dep, rcond=None)
            sse_r = np.sum((y_dep - X_r @ b_r)**2)
            # Unrestricted: y_t ~ y_{t-1} + x_{t-1}
            X_u = np.column_stack([np.ones(n), y_lag, x_lag])
            b_u, _, _, _ = lstsq(X_u, y_dep, rcond=None)
            sse_u = np.sum((y_dep - X_u @ b_u)**2)
            f_stat = ((sse_r - sse_u) / 1) / (sse_u / (n - 3))
            p_val = 1 - stats.f.cdf(f_stat, 1, n - 3)
            row.update(metric="Granger_F", value=round(f_stat, 2),
                       expected=f"p={p_val:.4e}, corr={corr:.4f}")
            row["pass"] = p_val < 0.01
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'correlational':
            v1, v2 = rel['pair']
            corr = np.corrcoef(df[v1], df[v2])[0, 1]
            row.update(metric="Pearson_r", value=round(corr, 4),
                       expected=f"target={rel['correlation']}")
            row["pass"] = abs(corr - rel['correlation']) < 0.15
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'mediating':
            src, med, tgt = rel['source'], rel['mediator'], rel['target']
            la, lb = rel['path_a_lag'], rel['path_b_lag']
            # Path a
            corr_a = np.corrcoef(df[med].values[la:], df[src].values[:-la])[0, 1] if la > 0 else \
                     np.corrcoef(df[med], df[src])[0, 1]
            # Path b
            corr_b = np.corrcoef(df[tgt].values[lb:], df[med].values[:-lb])[0, 1] if lb > 0 else \
                     np.corrcoef(df[tgt], df[med])[0, 1]
            indirect = corr_a * corr_b
            row.update(metric="indirect_ab", value=round(indirect, 4),
                       expected=f"a={corr_a:.3f}, b={corr_b:.3f}")
            row["pass"] = abs(indirect) > 0.1
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'moderating':
            src, mod, tgt = rel['source'], rel['moderator'], rel['target']
            lag = rel['lag']
            n = len(df) - lag
            x = df[src].values[:-lag][:n]
            z = df[mod].values[:-lag][:n]
            y = df[tgt].values[lag:][:n]
            xz = x * z
            X_mat = np.column_stack([np.ones(n), x, z, xz])
            b, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            y_hat = X_mat @ b
            ss_res = np.sum((y - y_hat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            row.update(metric="interaction_R2", value=round(r2, 4),
                       expected=f"coeff_int={rel['coeff_interaction']}")
            row["pass"] = r2 > 0.05
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'synergistic':
            sources = rel['sources']
            tgt = rel['target']
            lag = rel['lag']
            n = len(df) - lag
            s_vals = [df[s].values[:-lag][:n] for s in sources]
            y = df[tgt].values[lag:][:n]
            interaction = np.prod(s_vals, axis=0)
            X_mat = np.column_stack([np.ones(n)] + s_vals + [interaction])
            b, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            y_hat = X_mat @ b
            ss_res = np.sum((y - y_hat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            row.update(metric="synergy_R2", value=round(r2, 4),
                       expected=f"int_coeff={rel['interaction_coeff']}")
            row["pass"] = r2 > 0.05
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'functional':
            src, tgt = rel['source'], rel['target']
            lag = rel['lag']
            n = len(df) - lag
            x = df[src].values[:-lag][:n]
            y = df[tgt].values[lag:][:n]
            func = rel.get('function', 'linear')
            if func == 'quadratic':
                X_mat = np.column_stack([np.ones(n), x, x**2])
            else:
                X_mat = np.column_stack([np.ones(n), x])
            b, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            y_hat = X_mat @ b
            ss_res = np.sum((y - y_hat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            row.update(metric="func_R2", value=round(r2, 4),
                       expected=f"func={func}, coeff={rel['coeff']}")
            row["pass"] = r2 > 0.05
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'equilibrium':
            var = rel['variable']
            y = df[var].values
            mean_val = y.mean()
            row.update(metric="mean_dev", value=round(abs(mean_val - rel['mean']), 4),
                       expected=f"mean={rel['mean']}, rev={rel['reversion_rate']}")
            # Also check AR(1) coefficient
            corr = np.corrcoef(y[1:], y[:-1])[0, 1]
            row["pass"] = abs(mean_val - rel['mean']) < 1.0 and corr > 0
            row["status"] = "PASS" if row["pass"] else "WEAK"
            row["expected"] += f", ar1={corr:.3f}"

        elif rt == 'compositional':
            total_var = rel['total']
            comps = rel['components']
            comp_sum = df[comps].sum(axis=1)
            corr = np.corrcoef(df[total_var], comp_sum)[0, 1]
            row.update(metric="sum_corr", value=round(corr, 4), expected="~1.0")
            row["pass"] = corr > 0.8
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'competitive':
            v1, v2 = rel['pair']
            sum_vals = df[v1] + df[v2]
            neg_corr = np.corrcoef(df[v1], df[v2])[0, 1]
            row.update(metric="neg_corr", value=round(neg_corr, 4),
                       expected=f"total={rel['total']}, sum_mean={sum_vals.mean():.2f}")
            row["pass"] = neg_corr < -0.5
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'probabilistic':
            src, tgt = rel['source'], rel['target']
            lag = rel['lag']
            x = df[src].values[:-lag]
            y = df[tgt].values[lag:]
            mean_pos = y[x > 0].mean()
            mean_neg = y[x <= 0].mean()
            diff = mean_pos - mean_neg
            row.update(metric="cond_shift", value=round(diff, 4),
                       expected=f"shift={rel['shift']}")
            row["pass"] = diff > 0.5
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'structural':
            var = rel['variable']
            y = df[var].values
            mid = len(y) // 2
            corr_before = np.corrcoef(y[1:mid], y[:mid-1])[0, 1] if mid > 2 else 0
            corr_after = np.corrcoef(y[mid+1:], y[mid:-1])[0, 1] if mid > 2 else 0
            diff = abs(corr_after - corr_before)
            row.update(metric="break_diff", value=round(diff, 4),
                       expected=f"before={rel['coeff_before']}, after={rel['coeff_after']}")
            row["pass"] = diff > 0.1
            row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'graph':
            for edge in rel['edges']:
                src, tgt = edge['source'], edge['target']
                lag = edge.get('lag', 1)
                corr = np.corrcoef(df[tgt].values[lag:], df[src].values[:-lag])[0, 1]
                row2 = dict(row)
                row2.update(type=f"graph({src}->{tgt})", metric="lag_corr",
                            value=round(corr, 4), expected=f"coeff={edge['coeff']}")
                row2["pass"] = abs(corr) > 0.1
                row2["status"] = "PASS" if row2["pass"] else "WEAK"
                validation_rows.append(row2)
            continue

        elif rt == 'similarity':
            group = rel['group']
            if len(group) >= 2:
                corr = np.corrcoef(df[group[0]], df[group[1]])[0, 1]
                row.update(metric="group_corr", value=round(corr, 4),
                           expected=f"base_std={rel['base_signal_std']}")
                row["pass"] = corr > 0.5
                row["status"] = "PASS" if row["pass"] else "WEAK"

        elif rt == 'logical':
            sources = rel['sources']
            tgt = rel['target']
            lag = rel.get('lag', 1)
            # Check if output is binary-ish
            y = df[tgt].values[lag:]
            unique_vals = len(np.unique(np.round(y, 1)))
            row.update(metric="n_unique", value=unique_vals,
                       expected=f"op={rel.get('operation','AND')}")
            row["pass"] = unique_vals <= 5  # should be near-binary
            row["status"] = "PASS" if row["pass"] else "WEAK"

    except Exception as e:
        row["status"] = f"ERROR: {e}"

    validation_rows.append(row)

val_df = pd.DataFrame(validation_rows)
val_path = os.path.join(OUT_DIR, "generator_validation.csv")
val_df.to_csv(val_path, index=False)
print(f"\nGenerator Validation Results ({val_path}):", flush=True)
print(val_df.to_string(index=False), flush=True)

n_pass = sum(1 for r in validation_rows if r.get("pass"))
n_total = len(validation_rows)
print(f"\nPassed: {n_pass}/{n_total}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Extract Raw Hypothesis State from Engine
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("STEP 1: Extracting Raw Engine Hypothesis State", flush=True)
print("=" * 70, flush=True)

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.discovery import HypothesisState

engine = OnlineDiscoveryEngine(buffer_size=150)
engine_schema = {"fields": [{"name": v} for v in gen.variables]}
engine.initialize_v2(engine_schema, use_causal=True)

print(f"  Initialized {len(engine.hypotheses.population)} hypotheses", flush=True)
print(f"  Streaming {N} rows...", flush=True)

for i in range(len(df)):
    row = df.iloc[i].to_dict()
    engine.process_row(row)

print(f"  After streaming: {len(engine.hypotheses.population)} alive, "
      f"{len(engine.hypotheses.graveyard)} dead", flush=True)

# Dump ALL hypotheses (alive + graveyard)
hyp_rows = []
all_hyps = list(engine.hypotheses.population.values())
# Also include dead ones from graveyard if accessible
for h in all_hyps:
    hyp_rows.append({
        "hypothesis_id": h.meta.id,
        "type": h.rel_type.value if hasattr(h.rel_type, 'value') else str(h.rel_type),
        "variables": str(h.variables),
        "source": getattr(h, 'source', h.variables[0] if h.variables else ''),
        "target": getattr(h, 'target', h.variables[-1] if len(h.variables) > 1 else ''),
        "fit_score": round(getattr(h, 'fit_score', 0.0), 6),
        "confidence": round(getattr(h, 'confidence', 0.0), 6),
        "stability": round(getattr(h, 'stability', 0.0), 6),
        "evidence": getattr(h, 'evidence', 0),
        "state": h.meta.state.value if hasattr(h.meta.state, 'value') else str(h.meta.state),
        "is_active": h.meta.state == HypothesisState.ACTIVE,
    })

hyp_df = pd.DataFrame(hyp_rows)
hyp_path = os.path.join(OUT_DIR, "raw_hypotheses.csv")
hyp_df.to_csv(hyp_path, index=False)

print(f"\nRaw Hypotheses ({hyp_path}):", flush=True)
print(f"  Total alive: {len(hyp_df)}", flush=True)
print(f"  Active: {hyp_df['is_active'].sum()}", flush=True)
print(f"  By state:", flush=True)
print(hyp_df['state'].value_counts().to_string(), flush=True)

# Focus: for each ground-truth relationship, find matching hypotheses
print(f"\n{'='*70}", flush=True)
print("Ground-Truth <-> Engine Hypothesis Matching", flush=True)
print(f"{'='*70}", flush=True)

gt_types_map = {
    'temporal': 'temporal', 'causal': 'causal', 'correlational': 'correlational',
    'mediating': 'mediating', 'moderating': 'moderating', 'synergistic': 'synergistic',
    'functional': 'functional', 'equilibrium': 'equilibrium', 'compositional': 'compositional',
    'competitive': 'competitive', 'probabilistic': 'probabilistic', 'structural': 'structural_break',
    'graph': 'graph', 'similarity': 'similarity', 'logical': 'logical',
}

for rel in schema['relationships']:
    rt = rel['type']
    # Find variables involved
    gt_vars = set()
    if 'variable' in rel: gt_vars.add(rel['variable'])
    if 'source' in rel: gt_vars.add(rel['source'])
    if 'target' in rel: gt_vars.add(rel['target'])
    if 'mediator' in rel: gt_vars.add(rel['mediator'])
    if 'moderator' in rel: gt_vars.add(rel['moderator'])
    if 'pair' in rel: gt_vars.update(rel['pair'])
    if 'components' in rel: gt_vars.update(rel['components'])
    if 'total' in rel and isinstance(rel['total'], str): gt_vars.add(rel['total'])
    if 'sources' in rel: gt_vars.update(rel['sources'])
    if 'group' in rel: gt_vars.update(rel['group'])
    if 'edges' in rel:
        for e in rel['edges']:
            gt_vars.update([e['source'], e['target']])

    # Search engine hypotheses that overlap with these variables
    matches = []
    for _, hrow in hyp_df.iterrows():
        h_vars = set(eval(hrow['variables']))
        if h_vars & gt_vars:  # any overlap
            engine_type = hrow['type']
            expected_type = gt_types_map.get(rt, rt)
            type_match = expected_type in engine_type or engine_type in expected_type
            matches.append({
                "engine_type": engine_type,
                "type_match": type_match,
                "vars": hrow['variables'],
                "conf": hrow['confidence'],
                "fit": hrow['fit_score'],
                "stab": hrow['stability'],
                "state": hrow['state'],
            })

    # Show best matching hypothesis
    type_matches = [m for m in matches if m['type_match']]
    best = max(type_matches, key=lambda m: m['conf']) if type_matches else None

    if best:
        print(f"\n  GT: {rt:15s} | vars={gt_vars}", flush=True)
        print(f"  -> Engine: {best['engine_type']:20s} conf={best['conf']:.4f} "
              f"fit={best['fit']:.4f} stab={best['stab']:.4f} state={best['state']}", flush=True)
        if best['conf'] < 0.10:
            print(f"     !! KILLED: conf < 0.10 (kill threshold)", flush=True)
        elif best['conf'] < 0.70:
            print(f"     !! STUCK TENTATIVE: conf < 0.70 (promotion threshold)", flush=True)
        if best['stab'] < 0.60:
            print(f"     !! LOW STABILITY: stab < 0.60", flush=True)
    else:
        print(f"\n  GT: {rt:15s} | vars={gt_vars}", flush=True)
        print(f"  -> NO TYPE MATCH FOUND in engine", flush=True)

print(f"\n{'='*70}", flush=True)
print("Diagnostics complete.", flush=True)

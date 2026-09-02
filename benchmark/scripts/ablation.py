"""Ablation harness — isolates each component's contribution to recovery.

Runs the synthetic benchmark's calibration once per seed, then re-derives the
significance decision under progressively weaker gates. Reports the delta in
recovery F1 / null FPR so the value of each component is quantified rather than
asserted.

Gates (calibration ladder, strongest to weakest):
  full        permutation null + BH-FDR (the shipped decision)
  perm_no_fdr permutation null, per-hypothesis p<alpha, no multiple-testing correction
  parametric  regression F-test p<alpha under the iid-normal null, no permutation
  raw         raw effect-size threshold, no null model at all (the naive analyst)

To expose the gates' FPR control the null set is augmented with many genuinely
unrelated variable pairs (multiplicity is where a null model earns its keep).

Usage:
  python benchmark/scripts/ablation.py --n_samples 3000 --seeds 42 43 44 --B_perm 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.synthetic.benchmark_generator import create_benchmark_generator
from benchmark.synthetic.calibration import BenchmarkCalibrator
from benchmark.synthetic.pipeline import SyntheticBenchmark

ALPHA = 0.05
RAW_CONF_THR = 0.5   # naive "|effect| looks big" cutoff, no null model


# predictors (excluding intercept) per relationship type
_K_BY_TYPE = {
    "correlational": 1, "functional": 1, "competitive": 1, "compositional": 1,
    "probabilistic": 1, "structural": 1, "equilibrium": 1, "null": 1,
    "causal": 2, "temporal": 2, "mediating": 2, "logical": 2, "graph": 2,
    "synergistic": 3, "moderating": 3,
}


def _parametric_p(r: dict) -> float:
    """Regression F-test p-value under the iid-normal null — the naive analyst's
    p, ignoring autocorrelation and multiplicity. R^2 = fit_obs, n = evid, and
    k predictors inferred from the relationship type."""
    r2 = float(min(max(r.get("fit_obs", 0.0), 0.0), 1.0 - 1e-12))
    n = int(r.get("evid", 0))
    k = _K_BY_TYPE.get(r.get("rel_type", ""), 1)
    if n <= k + 1:
        return 1.0
    f_stat = (r2 / k) / ((1.0 - r2) / (n - k - 1))
    return float(stats.f.sf(f_stat, k, n - k - 1))


def apply_gate(results: dict, gate: str) -> None:
    """Set results[k]['significant'] in-place according to the ablation gate.

    'full' is left as calibrate() decided (permutation null + BH-FDR).
    """
    if gate == "full":
        return
    for k, r in results.items():
        if gate == "perm_no_fdr":            # drop multiple-testing correction
            r["significant"] = bool(r["p_value"] < ALPHA)
        elif gate == "parametric":           # drop the permutation null entirely
            r["significant"] = bool(_parametric_p(r) < ALPHA)
        elif gate == "raw":                  # drop any null model — bare threshold
            r["significant"] = bool(r.get("conf_obs", 0.0) >= RAW_CONF_THR)
        else:
            raise ValueError(f"unknown gate {gate}")


def _shuffle_columns(data: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each column's rows -> a dataset where every pair of
    variables is truly independent (a global null). Any hypothesis a gate then
    declares significant is, by construction, a false positive."""
    rng = np.random.default_rng(seed + 1)
    out = data.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def _fpr_on_shuffled(cal, results_template_keys, df_values, seed, B_perm, gates) -> dict:
    """Clean FPR per gate: calibrate on globally-shuffled data (all-null) and
    report the fraction of the real relationship hypotheses flagged significant."""
    shuf = _shuffle_columns(df_values, seed)
    res = cal.calibrate(shuf)
    # exclude the designated null_ specs; the rest are relationship hyps that are
    # now null under the shuffle, so their significant-rate is the false-positive rate
    keys = [k for k in res if not k.startswith("null_")]
    fpr = {}
    for gate in gates:
        apply_gate(res, gate)
        flagged = sum(1 for k in keys if res[k].get("significant"))
        fpr[gate] = flagged / len(keys) if keys else 0.0
    return fpr


def run_seed(schema_path: str, seed: int, n: int, B_perm: int, gates) -> dict:
    gen = create_benchmark_generator(schema_path, seed)
    df = gen.generate(n)
    cal = BenchmarkCalibrator(col_names=gen.variables, schema=gen.schema, B_perm=B_perm)
    results = cal.calibrate(df.values)

    # Reuse the pipeline's exact recovery evaluation; it reads results[k]['significant'].
    bench = SyntheticBenchmark.__new__(SyntheticBenchmark)
    bench.generator = gen

    # Clean FPR from a fully-shuffled (all-null) replica of the data.
    fpr_shuf = _fpr_on_shuffled(cal, list(results.keys()), df.values, seed, B_perm, gates)

    out = {}
    for gate in gates:
        apply_gate(results, gate)
        m = bench._evaluate_recovery(results)
        out[gate] = {
            "strict_f1": m["strict"]["f1"],
            "precision": m["strict"]["precision"],
            "recall": m["strict"]["recall"],
            "shuffle_fpr": round(fpr_shuf[gate], 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--B_perm", type=int, default=100)
    args = ap.parse_args()

    gates = ["full", "perm_no_fdr", "parametric", "raw"]
    per_seed = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        per_seed[seed] = run_seed(args.schema, seed, args.n_samples, args.B_perm, gates)

    # Aggregate mean over seeds
    print("\n" + "=" * 66)
    print(f"ABLATION: calibration gate  (n={args.n_samples}, seeds={args.seeds}, B_perm={args.B_perm})")
    print("=" * 66)
    print(f"{'gate':<14} {'F1':>8} {'precision':>10} {'recall':>8} {'shuffleFPR':>11}")
    print("-" * 66)
    agg = {}
    for gate in gates:
        f1 = np.mean([per_seed[s][gate]["strict_f1"] for s in args.seeds])
        pr = np.mean([per_seed[s][gate]["precision"] for s in args.seeds])
        rc = np.mean([per_seed[s][gate]["recall"] for s in args.seeds])
        fpr = np.mean([per_seed[s][gate]["shuffle_fpr"] for s in args.seeds])
        agg[gate] = dict(f1=round(f1, 3), precision=round(pr, 3), recall=round(rc, 3), shuffle_fpr=round(fpr, 4))
        print(f"{gate:<14} {f1:>8.3f} {pr:>10.3f} {rc:>8.3f} {fpr:>11.4f}")
    print("=" * 66)
    print(json.dumps({"per_seed": per_seed, "mean": agg}))


if __name__ == "__main__":
    main()

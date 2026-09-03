"""Discovery baselines — what does the full Scarcity architecture buy over
simpler relationship-discovery methods, on the same ground truth?

Every method emits a set of undirected variable pairs it believes are related;
each is scored against the benchmark schema's true relationships (precision /
recall / F1) and its false-edge rate over all genuinely-unrelated pairs (FPR).

Baselines:
  pearson     |Pearson r| significant, Benjamini-Hochberg FDR at q
  spearman    Spearman rho significant, BH FDR
  mutual_info sklearn mutual information, thresholded at the 95th percentile of a
              column-shuffled null (a cheap distribution-free significance cut)
  granger     lag-1 Granger F-test either direction, BH FDR
  pc          PC algorithm (causal-learn) skeleton, adjacency -> undirected pairs
  scarcity    the full pipeline: calibrated permutation + BH over typed hypotheses

Usage:
  python benchmark/scripts/discovery_baselines.py --n 1500 --seeds 0 1 2
"""
import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.synthetic.benchmark_generator import create_benchmark_generator
from benchmark.synthetic.calibration import BenchmarkCalibrator

Q = 0.05


# ---------------------------------------------------------------- ground truth
def ground_truth(schema):
    rel = set()
    def add(a, b):
        if a != b:
            rel.add(frozenset((a, b)))
    for r in schema.get("relationships", []):
        vs = []
        for k in ("variable", "source", "target", "mediator", "moderator", "total"):
            if k in r:
                vs.append(r[k])
        for k in ("pair", "sources", "components", "group"):
            if k in r:
                vs.extend(r[k])
        if r["type"] == "graph":
            for e in r["edges"]:
                vs += [e["source"], e["target"]]
        for a, b in itertools.combinations(set(vs), 2):
            add(a, b)
    return rel


def _bh(pairs_pvals, q=Q):
    """Benjamini-Hochberg over {pair: p}; return the set of rejected pairs."""
    items = sorted(pairs_pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    keep = 0
    for k, (_, p) in enumerate(items, start=1):
        if p <= (k / m) * q:
            keep = k
    return {items[i][0] for i in range(keep)}


# ---------------------------------------------------------------- baselines
def baseline_corr(data, cols, method):
    from scipy.stats import pearsonr, spearmanr
    fn = pearsonr if method == "pearson" else spearmanr
    pv = {}
    for i, j in itertools.combinations(range(len(cols)), 2):
        _, p = fn(data[:, i], data[:, j])
        pv[frozenset((cols[i], cols[j]))] = float(p) if np.isfinite(p) else 1.0
    return _bh(pv)


def baseline_mutual_info(data, cols, rng):
    from sklearn.feature_selection import mutual_info_regression
    n, d = data.shape
    def mi_matrix(X):
        M = np.zeros((d, d))
        for j in range(d):
            M[:, j] = mutual_info_regression(X, X[:, j], random_state=0)
        return M
    obs = mi_matrix(data)
    shuf = data.copy()
    for j in range(d):                      # break all cross-dependence
        shuf[:, j] = shuf[rng.permutation(n), j]
    null = mi_matrix(shuf)
    thr = np.quantile(null[np.triu_indices(d, 1)], 0.95)
    edges = set()
    for i, j in itertools.combinations(range(d), 2):
        if max(obs[i, j], obs[j, i]) > thr:
            edges.add(frozenset((cols[i], cols[j])))
    return edges


def baseline_granger(data, cols):
    from statsmodels.tsa.stattools import grangercausalitytests
    pv = {}
    for i, j in itertools.combinations(range(len(cols)), 2):
        best = 1.0
        for a, b in ((i, j), (j, i)):
            try:
                res = grangercausalitytests(np.column_stack([data[:, b], data[:, a]]),
                                            maxlag=1, verbose=False)
                best = min(best, res[1][0]["ssr_ftest"][1])
            except Exception:
                pass
        pv[frozenset((cols[i], cols[j]))] = best
    return _bh(pv)


def baseline_pc(data, cols):
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(data, alpha=0.05, indep_test="fisherz", show_progress=False)
    g = cg.G.graph
    edges = set()
    d = len(cols)
    for i in range(d):
        for j in range(i + 1, d):
            if g[i, j] != 0 or g[j, i] != 0:       # any adjacency
                edges.add(frozenset((cols[i], cols[j])))
    return edges


def scarcity_edges(gen, data, B_perm):
    cal = BenchmarkCalibrator(col_names=gen.variables, schema=gen.schema, B_perm=B_perm)
    res = cal.calibrate(data)
    edges = set()
    for name, r in res.items():
        if r.get("significant") and r.get("rel_type") != "null":
            parts = name.split("_")
            if len(parts) >= 3:
                edges.add(frozenset((parts[-2], parts[-1])))
    return edges


# ---------------------------------------------------------------- scoring
def score(pred, gt, all_pairs):
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    nulls = all_pairs - gt
    fpr = len(pred & nulls) / len(nulls) if nulls else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr, n_pred=len(pred))


def run_seed(schema_path, seed, n, B_perm, methods):
    gen = create_benchmark_generator(schema_path, seed)
    df = gen.generate(n)
    data = df.values
    cols = list(gen.variables)
    gt = ground_truth(gen.schema)
    all_pairs = {frozenset(p) for p in itertools.combinations(cols, 2)}
    rng = np.random.default_rng(seed)

    preds = {}
    if "pearson" in methods:      preds["pearson"] = baseline_corr(data, cols, "pearson")
    if "spearman" in methods:     preds["spearman"] = baseline_corr(data, cols, "spearman")
    if "mutual_info" in methods:  preds["mutual_info"] = baseline_mutual_info(data, cols, rng)
    if "granger" in methods:      preds["granger"] = baseline_granger(data, cols)
    if "pc" in methods:
        try:
            preds["pc"] = baseline_pc(data, cols)
        except Exception as ex:
            print(f"    pc failed: {ex}")
    if "scarcity" in methods:     preds["scarcity"] = scarcity_edges(gen, data, B_perm)

    return {m: score(p, gt, all_pairs) for m, p in preds.items()}, len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--B_perm", type=int, default=100)
    ap.add_argument("--methods", nargs="+",
                    default=["pearson", "spearman", "mutual_info", "granger", "pc", "scarcity"])
    args = ap.parse_args()

    per = {}
    n_gt = 0
    for s in args.seeds:
        print(f"=== seed {s} ===", flush=True)
        res, n_gt = run_seed(args.schema, s, args.n, args.B_perm, args.methods)
        for m, sc in res.items():
            per.setdefault(m, []).append(sc)

    print(f"\nDiscovery baselines vs Scarcity  (n={args.n}, seeds={args.seeds}, "
          f"|ground truth|={n_gt} pairs)")
    print("=" * 72)
    print(f"{'method':>12} {'precision':>10} {'recall':>8} {'F1':>7} {'FPR':>7} {'n_pred':>7}")
    print("-" * 72)
    out = []
    order = ["pearson", "spearman", "mutual_info", "granger", "pc", "scarcity"]
    for m in [x for x in order if x in per]:
        rs = per[m]
        row = {k: float(np.mean([r[k] for r in rs])) for k in ("precision", "recall", "f1", "fpr", "n_pred")}
        print(f"{m:>12} {row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>7.3f} "
              f"{row['fpr']:>7.3f} {row['n_pred']:>7.1f}")
        out.append({"method": m, **{k: round(v, 4) for k, v in row.items()}})
    print("=" * 72)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

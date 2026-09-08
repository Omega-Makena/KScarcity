"""Online-engine recovery — does the STREAMING engine's own knowledge graph hold
up, not just the offline calibrator?

The headline recovery F1 elsewhere is the offline BenchmarkCalibrator's. This
script scores the GPU streaming engine's own graph, raw vs calibrated
(get_knowledge_graph(calibrated=True): analytic F-test p-value + BH-FDR), on two
footings:

  real data     precision/recall/F1 + FPR vs the schema's pairwise ground truth
  global null   each column independently shuffled -> every emitted edge is a true
                false positive; this is the gate's real false-positive rate

Reading it: the calibrated gate's quality is the GLOBAL-NULL FPR (should collapse
to ~0). The real-data "FPR" is inflated by genuine *indirect* correlations the
generator induces (X->M->Y makes X and Y correlate) which a pairwise-adjacency
ground truth miscounts as false — a scoring artifact, not an engine error. So
read global-null FPR for gate quality and recall for coverage; real-data
precision is a lower bound distorted by transitive dependence.

Usage:
  python benchmark/scripts/online_recovery.py --n 1500 --seeds 0 1 2
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.synthetic.benchmark_generator import create_benchmark_generator
from scarcity.engine.gpu_engine import GPUDiscoveryEngine
sys.path.insert(0, str(_ROOT / "benchmark" / "scripts"))
from discovery_baselines import ground_truth


def _pairs(kg):
    s = set()
    for h in kg:
        for a, b in itertools.combinations(h["variables"], 2):
            if a != b:
                s.add(frozenset((a, b)))
    return s


def _score(pred, gt, all_pairs):
    tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
    nulls = all_pairs - gt
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    fpr = len(pred & nulls) / len(nulls) if nulls else 0.0
    return dict(precision=P, recall=R, f1=F, fpr=fpr, n=len(pred))


def _stream(engine, rows, cols):
    for i in range(len(rows)):
        engine.process_row({c: float(rows[i, k]) for k, c in enumerate(cols)})


def run_seed(schema_path, seed, n):
    gen = create_benchmark_generator(schema_path, seed)
    df = gen.generate(n)
    cols = list(gen.variables)
    gt = ground_truth(gen.schema)
    all_pairs = {frozenset(p) for p in itertools.combinations(cols, 2)}

    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": c} for c in cols]}, use_causal=True)
    _stream(e, df.values, cols)
    raw = _pairs([h for h in e.get_knowledge_graph(top_k=600) if h["metrics"]["confidence"] >= 0.55])
    cal = _pairs(e.get_knowledge_graph(top_k=600, calibrated=True, q=0.05))

    # global-null replica: independent column shuffle
    rng = np.random.default_rng(seed + 777)
    shuf = df.values.copy()
    for j in range(shuf.shape[1]):
        shuf[:, j] = shuf[rng.permutation(shuf.shape[0]), j]
    en = GPUDiscoveryEngine(device="cpu")
    en.initialize_v2({"fields": [{"name": c} for c in cols]}, use_causal=True)
    _stream(en, shuf, cols)
    null_raw = _pairs([h for h in en.get_knowledge_graph(top_k=600) if h["metrics"]["confidence"] >= 0.55])
    null_cal = _pairs(en.get_knowledge_graph(top_k=600, calibrated=True, q=0.05))

    return {
        "raw": _score(raw, gt, all_pairs),
        "calibrated": _score(cal, gt, all_pairs),
        "global_null_fpr_raw": len(null_raw) / len(all_pairs),
        "global_null_fpr_calibrated": len(null_cal) / len(all_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    rows = [run_seed(args.schema, s, args.n) for s in args.seeds]

    def avg(path):
        return float(np.mean([_dig(r, path) for r in rows]))

    print(f"\nOnline-engine knowledge-graph recovery  (n={args.n}, seeds={args.seeds})")
    print("=" * 70)
    print(f"{'graph':>12} {'precision':>10} {'recall':>8} {'F1':>7} {'realFPR':>8} {'nullFPR':>8}")
    print("-" * 70)
    for g in ("raw", "calibrated"):
        P = avg([g, "precision"]); R = avg([g, "recall"]); F = avg([g, "f1"]); fpr = avg([g, "fpr"])
        nfpr = float(np.mean([r[f"global_null_fpr_{g}"] for r in rows]))
        print(f"{g:>12} {P:>10.3f} {R:>8.3f} {F:>7.3f} {fpr:>8.3f} {nfpr:>8.3f}")
    print("=" * 70)
    print("nullFPR = false-edge rate on a global-null replica = the gate's true "
          "quality;\nrealFPR is inflated by genuine indirect correlations (scoring "
          "artifact, see #5).")
    print(json.dumps(rows))


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


if __name__ == "__main__":
    main()

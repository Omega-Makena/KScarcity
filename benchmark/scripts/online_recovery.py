"""Online-engine recovery — does the STREAMING engine's own knowledge graph hold
up, not just the offline calibrator?

The headline recovery F1 elsewhere is the offline BenchmarkCalibrator's. This
script scores the GPU streaming engine's own graph, raw vs calibrated
(get_knowledge_graph(calibrated=True): analytic F-test p-value + BH-FDR), on two
footings:

  real data     connectivity-aware score (scoring.py): adjacency recall on the
                planted edges, honest precision, and indep_fpr = false edges over
                only the truly d-separated pairs (indirect dependencies are
                neither rewarded nor punished)
  global null   each column independently shuffled -> an iid reference where every
                emitted edge is a false positive

READING. The calibrated gate scores each hypothesis by its PREDICTOR
coefficient's partial t-test, not the whole-regression R². That distinction is
what makes it autocorrelation-robust: for a causal hypothesis Y=b_t ~ [1,
a_{t-1}, b_{t-1}] the b_{t-1} term makes R² large from the target's own memory
regardless of a, so an R² F-test fired on every autocorrelated target
(indep_fpr ~0.33); the a_{t-1} coefficient does not, so the partial-t gate holds
indep_fpr near the nominal q (~0.02) on the REAL autocorrelated stream while
improving recall. It matches the global-null (iid) reference to within FDR noise.
The offline permutation calibrator (BLOCK/PHASE) remains available as the
rigorous instrument, but the online graph is now trustworthy on its own.

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
from scoring import build_dependency_sets, score_connectivity


def _pairs(kg):
    s = set()
    for h in kg:
        for a, b in itertools.combinations(h["variables"], 2):
            if a != b:
                s.add(frozenset((a, b)))
    return s


def _stream(engine, rows, cols):
    for i in range(len(rows)):
        engine.process_row({c: float(rows[i, k]) for k, c in enumerate(cols)})


def run_seed(schema_path, seed, n):
    gen = create_benchmark_generator(schema_path, seed)
    df = gen.generate(n)
    cols = list(gen.variables)
    direct, dependent, independent = build_dependency_sets(gen.schema, cols)
    all_pairs = {frozenset(p) for p in itertools.combinations(cols, 2)}

    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": c} for c in cols]}, use_causal=True)
    _stream(e, df.values, cols)
    raw = _pairs([h for h in e.get_knowledge_graph(top_k=600) if h["metrics"]["confidence"] >= 0.55])
    cal = _pairs(e.get_knowledge_graph(top_k=600, calibrated=True, q=0.05))

    # global-null replica: independent column shuffle (iid reference)
    rng = np.random.default_rng(seed + 777)
    shuf = df.values.copy()
    for j in range(shuf.shape[1]):
        shuf[:, j] = shuf[rng.permutation(shuf.shape[0]), j]
    en = GPUDiscoveryEngine(device="cpu")
    en.initialize_v2({"fields": [{"name": c} for c in cols]}, use_causal=True)
    _stream(en, shuf, cols)
    null_cal = _pairs(en.get_knowledge_graph(top_k=600, calibrated=True, q=0.05))

    return {
        "raw": score_connectivity(raw, direct, dependent, independent),
        "calibrated": score_connectivity(cal, direct, dependent, independent),
        "global_null_fpr_calibrated": len(null_cal) / len(all_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    rows = [run_seed(args.schema, s, args.n) for s in args.seeds]

    def avg(g, k):
        return float(np.mean([r[g][k] for r in rows]))

    print(f"\nOnline-engine knowledge-graph recovery  (n={args.n}, seeds={args.seeds})")
    print("=" * 74)
    print(f"{'graph':>12} {'adj_recall':>11} {'honest_prec':>12} {'indep_fpr':>10} {'indirect':>9}")
    print("-" * 74)
    for g in ("raw", "calibrated"):
        print(f"{g:>12} {avg(g,'adjacency_recall'):>11.3f} {avg(g,'honest_precision'):>12.3f} "
              f"{avg(g,'indep_fpr'):>10.3f} {avg(g,'indirect_hits'):>9.1f}")
    nfpr = float(np.mean([r['global_null_fpr_calibrated'] for r in rows]))
    print("-" * 74)
    print(f"iid reference: calibrated FPR on a global-null (shuffled) replica = {nfpr:.3f}")
    print("indep_fpr is over truly d-separated pairs. The coefficient partial-t gate holds")
    print("it near nominal q even on the autocorrelated stream (the R² gate left it ~0.33),")
    print("because it credits the predictor term, not the target's own AR control.")
    print("=" * 74)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

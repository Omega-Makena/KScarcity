"""Full component ablation — turn each part of the architecture off and measure
what it was buying. Components live in different subsystems, so each is scored on
the metric it actually moves rather than forced through one number:

RECOVERY components (metric: strict F1 + FPR on a shuffled all-null replica; the
recovery decision is the calibrator, so these re-run calibration with one part
disabled):
  full            typed features + type-appropriate permutation + perm-null + BH-FDR
  -typed          strip type structure (lags -> 0, interactions off): every
                  relationship treated as a contemporaneous linear fit
  -typed_perm     force naive SHUFFLE null for every type (drop BLOCK/PHASE)
  -FDR            permutation p < alpha with no Benjamini-Hochberg correction
  -calibration    raw effect-size threshold, no null model at all

STREAMING components (metric: throughput + streaming-KG true-edge recovery; these
do not touch the calibrator, so they are measured on the engine directly):
  gpu vs cpu backend, lifecycle on/off, causal hypotheses on/off

Usage:
  python benchmark/scripts/ablation_components.py --n 1200 --seeds 0 1 --B_perm 80
"""
import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.synthetic.benchmark_generator import create_benchmark_generator
from benchmark.synthetic.calibration import BenchmarkCalibrator, PermStrategy
from benchmark.synthetic.pipeline import SyntheticBenchmark

ALPHA = 0.05
RAW_THR = 0.5


# ----------------------------------------------------------------- recovery side
def _transform(cal, mode):
    """Disable one recovery component by rewriting the spec list, then reindex the
    F-groups (the calibrator batches by feature-dim, so a changed F must reindex)."""
    if mode == "-typed":
        for s in cal.specs:
            if getattr(s, "interaction", False):
                s.F -= 1
                s.interaction = False
            s.lags = [0] * len(s.lags)
    elif mode == "-typed_perm":
        for s in cal.specs:
            s.perm_strategy = PermStrategy.SHUFFLE
    cal._groups = {}
    for idx, s in enumerate(cal.specs):
        cal._groups.setdefault(s.F, []).append(idx)


def _decide(results, gate):
    if gate in ("full", "-typed", "-typed_perm"):
        return  # significance already set by calibrate() (perm + BH-FDR)
    for r in results.values():
        if gate == "-FDR":
            r["significant"] = bool(r["p_value"] < ALPHA)
        elif gate == "-calibration":
            r["significant"] = bool(r.get("conf_obs", 0.0) >= RAW_THR)


def _shuffle_fpr(cal, data, seed, gate):
    rng = np.random.default_rng(seed + 1)
    shuf = data.copy()
    for j in range(shuf.shape[1]):
        shuf[:, j] = shuf[rng.permutation(shuf.shape[0]), j]
    res = cal.calibrate(shuf)
    _decide(res, gate)
    keys = [k for k in res if not k.startswith("null_")]
    return sum(1 for k in keys if res[k].get("significant")) / max(len(keys), 1)


def recovery_ablations(schema_path, seed, n, B_perm):
    gen = create_benchmark_generator(schema_path, seed)
    df = gen.generate(n)
    data = df.values
    bench = SyntheticBenchmark.__new__(SyntheticBenchmark)
    bench.generator = gen

    out = {}
    # spec-transforming configs each need a fresh calibrator (specs are mutated)
    for cfg in ["full", "-typed", "-typed_perm", "-FDR", "-calibration"]:
        cal = BenchmarkCalibrator(col_names=gen.variables, schema=gen.schema, B_perm=B_perm)
        _transform(cal, cfg if cfg in ("-typed", "-typed_perm") else "full")
        res = cal.calibrate(data)
        _decide(res, cfg)
        m = bench._evaluate_recovery(res)
        fpr = _shuffle_fpr(cal, data, seed, cfg)
        out[cfg] = {"f1": m["strict"]["f1"], "recall": m["strict"]["recall"],
                    "precision": m["strict"]["precision"], "shuffle_fpr": round(fpr, 4)}
    return out


# ----------------------------------------------------------------- streaming side
def _known_stream(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n); y = 0.9 * x + 0.2 * rng.normal(size=n); z = rng.normal(size=n)
    rows = [{"x": float(x[t]), "y": float(y[t]), "z": float(z[t])} for t in range(n)]
    return rows, {"fields": [{"name": c} for c in ("x", "y", "z")]}


def _edge_conf(engine, a, b):
    try:
        kg = engine.get_knowledge_graph(top_k=200, calibrated=False)
    except TypeError:
        kg = engine.get_knowledge_graph()          # CPU engine takes no top_k
    best = 0.0
    for h in kg:
        if set(h.get("variables", [])) == {a, b}:
            best = max(best, h["metrics"]["confidence"])
    return best


def streaming_ablations(n, seed):
    from scarcity.engine.gpu_engine import GPUDiscoveryEngine
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    rows, schema = _known_stream(n, seed)
    configs = []

    def _time(engine, use_causal=True, no_lifecycle=False):
        engine.initialize_v2(schema, use_causal=use_causal)
        if no_lifecycle:
            engine._lc_interval = 10 ** 9        # never runs lifecycle
        t0 = time.time()
        for r in rows:
            engine.process_row(r)
        dt = time.time() - t0
        return dt, _edge_conf(engine, "x", "y")

    dt, xy = _time(GPUDiscoveryEngine(device="cpu"))
    configs.append(("gpu_full", dt, xy))
    dt, xy = _time(GPUDiscoveryEngine(device="cpu"), no_lifecycle=True)
    configs.append(("gpu_-lifecycle", dt, xy))
    dt, xy = _time(GPUDiscoveryEngine(device="cpu"), use_causal=False)
    configs.append(("gpu_-causal", dt, xy))
    dt, xy = _time(OnlineDiscoveryEngine())
    configs.append(("cpu_full", dt, xy))
    return [{"config": c, "secs": round(d, 2), "rows_s": round(n / d, 1),
             "xy_conf": round(x, 3)} for c, d, x in configs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--B_perm", type=int, default=80)
    ap.add_argument("--stream_n", type=int, default=800)
    args = ap.parse_args()

    rec = {}
    for s in args.seeds:
        print(f"=== recovery, seed {s} ===", flush=True)
        r = recovery_ablations(args.schema, s, args.n, args.B_perm)
        for cfg, v in r.items():
            rec.setdefault(cfg, []).append(v)

    print(f"\nRECOVERY ablations  (n={args.n}, seeds={args.seeds}, B_perm={args.B_perm})")
    print("=" * 66)
    print(f"{'component':>14} {'F1':>7} {'precision':>10} {'recall':>8} {'shufFPR':>8}")
    print("-" * 66)
    rec_out = []
    for cfg in ["full", "-typed", "-typed_perm", "-FDR", "-calibration"]:
        vs = rec[cfg]
        row = {k: float(np.mean([v[k] for v in vs])) for k in ("f1", "precision", "recall", "shuffle_fpr")}
        print(f"{cfg:>14} {row['f1']:>7.3f} {row['precision']:>10.3f} {row['recall']:>8.3f} {row['shuffle_fpr']:>8.3f}")
        rec_out.append({"component": cfg, **{k: round(v, 4) for k, v in row.items()}})

    print(f"\nSTREAMING ablations  (n={args.stream_n}, seed={args.seeds[0]})")
    print("=" * 66)
    print(f"{'config':>16} {'secs':>7} {'rows/s':>8} {'xy_conf':>8}")
    print("-" * 66)
    strm = streaming_ablations(args.stream_n, args.seeds[0])
    for r in strm:
        print(f"{r['config']:>16} {r['secs']:>7.2f} {r['rows_s']:>8.1f} {r['xy_conf']:>8.3f}")
    print("-" * 66)
    print("note: cpu_full xy_conf is vestigial (the CPU engine's population KG is not")
    print("      the live-updated set); read cpu_full for cost only, not recovery.")
    print("=" * 66)
    print(json.dumps({"recovery": rec_out, "streaming": strm}))


if __name__ == "__main__":
    main()

"""Scaling benchmark — how cost grows with the number of variables.

Sweeps N_vars and measures, for both engines:
  * N_candidates  — hypotheses instantiated (the combinatorial growth, ~O(V^2))
  * obs/sec       — streaming throughput
  * hyp/sec       — N_candidates * obs/sec (work rate the engine sustains)

What this faithfully shows: candidate count grows quadratically in N_vars while
the GPU engine's sustained hyp/sec keeps *rising* — it absorbs the combinatorial
growth (throughput per row degrades far slower than candidates grow).

What it does NOT show: the decisive GPU-vs-Python gap. That gap comes from
batching the expensive *diverse-type* confidence tests (Sobel mediation, logical
rule search, ANOVA-over-history) that dominate on the real 15-type benchmark; a
simple correlation-chain stream barely activates them, so here CPU ~ GPU per row.
The decisive crossover is the full --use_gpu benchmark (streaming 195s at F1=1.0,
vs the pure-Python pipeline's structured-data rate of ~1.6 rows/s).

Usage:
  python benchmark/scripts/scaling.py --n_rows 250 --vars 5 10 20 34 50
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.gpu_engine import GPUDiscoveryEngine

# Only time the pure-Python engine where it stays tractable; above this N_vars
# its per-row Python loop over the candidate set makes a full sweep too slow.
CPU_MAX_VARS = 34


def _make_stream(v: int, n: int, seed: int = 0):
    """Structured stream, not noise: enough real relationships activate to fire the
    expensive per-hypothesis paths (Granger, ANOVA-over-history, mediation). On
    pure noise nothing activates and both engines look artificially fast/equal, so
    the workload would not be representative."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, v))
    data = z.copy()
    for i in range(0, v - 1):        # contemporaneous correlation chain
        data[:, i + 1] += 0.7 * data[:, i]
    for i in range(0, v - 2, 4):     # a few lagged (causal/temporal) links
        data[1:, i + 2] += 0.5 * data[:-1, i]
    cols = [f"v{i}" for i in range(v)]
    rows = [{c: float(data[t, i]) for i, c in enumerate(cols)} for t in range(n)]
    schema = {"fields": [{"name": c} for c in cols]}
    return rows, schema


def _time_engine(engine, rows, schema):
    engine.initialize_v2(schema, use_causal=True)
    gc.collect()
    t0 = time.time()
    for r in rows:
        engine.process_row(r)
    return time.time() - t0


def _n_candidates(engine):
    pool = getattr(engine, "_pool", None)
    if pool is not None:
        return len(pool.specs)
    hyp = getattr(engine, "hypotheses", None)
    return len(hyp.population) if hyp is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows", type=int, default=250)
    ap.add_argument("--vars", type=int, nargs="+", default=[5, 10, 20, 34, 50])
    args = ap.parse_args()

    print(f"{'N_vars':>6} {'N_cand':>8} {'gpu_row/s':>10} {'cpu_row/s':>10} "
          f"{'speedup':>8} {'gpu_hyp/s':>11}")
    print("-" * 60)
    rows_out = []
    for v in args.vars:
        rows, schema = _make_stream(v, args.n_rows)

        g = GPUDiscoveryEngine(device="cpu")
        gdt = _time_engine(g, rows, schema)
        ncand = _n_candidates(g)
        gps = args.n_rows / gdt

        if v <= CPU_MAX_VARS:
            c = OnlineDiscoveryEngine()
            cdt = _time_engine(c, rows, schema)
            cps = args.n_rows / cdt
            cps_s = f"{cps:>10.1f}"
        else:
            cps = None
            cps_s = f"{'(skip)':>10}"

        speedup = f"{gps / cps:>8.1f}" if cps else f"{'-':>8}"
        hps = ncand * gps
        print(f"{v:>6} {ncand:>8} {gps:>10.1f} {cps_s} {speedup} {hps:>11.0f}")
        rows_out.append(dict(n_vars=v, n_candidates=ncand, gpu_rows_s=round(gps, 1),
                             cpu_rows_s=round(cps, 1) if cps else None,
                             gpu_hyp_s=round(hps)))

    import json
    print("\n" + json.dumps(rows_out))


if __name__ == "__main__":
    main()

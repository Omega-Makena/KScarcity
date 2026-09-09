"""Dirty-data suite — graceful degradation under realistic stream corruption.

A clean stream has one true edge (x->y, r~0.9) and one null pair (x-z). Each
corruption is applied at increasing severity; we stream it through the GPU engine
and check that (a) it never crashes, (b) the true edge survives as long as it
reasonably can, and (c) no spurious x-z edge appears.

Corruptions:
  mcar        values dropped (NaN) completely at random
  mnar        y dropped when y is large (missing-not-at-random, value-dependent)
  outliers    cells replaced with 10-sigma spikes
  duplicates  a fraction of rows repeated (breaks i.i.d., inflates n)
  reorder     rows shuffled within local windows (out-of-order arrival)
  vanishing   a variable disappears from the schema partway through the stream

Reading the output: the true edge should stay recovered and crashes stay 0. The
`spurious` column is the raw streaming KG's UNCALIBRATED false-positive rate — it
is already non-zero on clean data (~0.67 on this null pair) and fluctuates around
that baseline rather than climbing with corruption; the calibration gate (see
benchmark/scripts/ablation.py) is what removes these, not the streaming engine.
So the finding is graceful degradation of the true signal, not FPR control.

Usage:
  python benchmark/scripts/dirty_data.py --n 800 --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.gpu_engine import GPUDiscoveryEngine

COLS = ("x", "y", "z")
CONF = 0.55


def _base(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)
    z = rng.normal(size=n)
    return np.column_stack([x, y, z]), rng


def _rows(arr, drop_after=None):
    """Array -> list of row dicts; NaN cells become missing keys. drop_after=(col,
    frac) removes that column from rows past the fraction (a vanishing variable)."""
    n = arr.shape[0]
    out = []
    cut = int(drop_after[1] * n) if drop_after else None
    for t in range(n):
        row = {}
        for i, c in enumerate(COLS):
            if drop_after and c == drop_after[0] and t >= cut:
                continue
            v = arr[t, i]
            if np.isfinite(v):
                row[c] = float(v)
        out.append(row)
    return out


def _corrupt(arr, rng, kind, sev):
    a = arr.copy()
    n = a.shape[0]
    if kind == "clean":
        return a, None
    if kind == "mcar":
        mask = rng.random(a.shape) < sev
        a[mask] = np.nan
    elif kind == "mnar":                       # drop y at its high tail
        thr = np.quantile(a[:, 1], 1 - sev)
        a[a[:, 1] >= thr, 1] = np.nan
    elif kind == "outliers":
        m = rng.random(a.shape) < sev
        spike = 10.0 * a.std(0)[None, :] * rng.choice([-1, 1], size=a.shape)
        a[m] += spike[m]
    elif kind == "duplicates":
        k = int(sev * n)
        idx = rng.choice(n, k, replace=False)
        a = np.insert(a, np.repeat(idx, 1), a[idx], axis=0)
    elif kind == "reorder":
        w = max(2, int(sev * n))
        for s in range(0, n, w):
            block = a[s:s + w]
            a[s:s + w] = block[rng.permutation(len(block))]
    elif kind == "vanishing":
        return a, ("z", 1 - sev)              # z vanishes after (1-sev) of the stream
    return a, None


def _edge_conf(engine, a, b):
    best = 0.0
    for h in engine.get_knowledge_graph(top_k=200, calibrated=False):
        if set(h["variables"]) == {a, b}:
            best = max(best, h["metrics"]["confidence"])
    return best


def run(kind, sev, n, seed):
    arr, rng = _base(n, seed)
    a, drop = _corrupt(arr, rng, kind, sev)
    rows = _rows(a, drop_after=drop)
    crashed = False
    xy = xz = 0.0
    try:
        e = GPUDiscoveryEngine(device="cpu")
        e.initialize_v2({"fields": [{"name": c} for c in COLS]}, use_causal=True)
        for r in rows:
            e.process_row(r)
        xy = _edge_conf(e, "x", "y")
        xz = _edge_conf(e, "x", "z")
    except Exception as ex:
        crashed = True
        xy = xz = float("nan")
        print(f"    CRASH {kind}@{sev}: {type(ex).__name__}: {str(ex)[:60]}")
    return dict(kind=kind, sev=sev, seed=seed, xy_conf=round(xy, 3),
                xz_conf=round(xz, 3), true_recovered=bool(xy >= CONF),
                spurious=bool(xz >= CONF), crashed=crashed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    plan = [("clean", 0.0)]
    for kind in ("mcar", "mnar", "outliers", "duplicates", "reorder", "vanishing"):
        for sev in (0.1, 0.3, 0.5):
            plan.append((kind, sev))

    rows = []
    for kind, sev in plan:
        rs = [run(kind, sev, args.n, s) for s in args.seeds]
        rows.append((kind, sev, rs))

    print(f"\nDirty-data suite  (n={args.n}, seeds={args.seeds})")
    print("=" * 68)
    print(f"{'corruption':>12} {'sev':>5} {'xy_conf':>8} {'recovered':>10} "
          f"{'spurious':>9} {'crashes':>8}")
    print("-" * 68)
    out = []
    for kind, sev, rs in rows:
        xy = np.nanmean([r["xy_conf"] for r in rs])
        rec = np.mean([r["true_recovered"] for r in rs])
        spur = np.mean([r["spurious"] for r in rs])
        cr = sum(r["crashed"] for r in rs)
        print(f"{kind:>12} {sev:>5.1f} {xy:>8.3f} {rec:>10.0%} {spur:>9.0%} {cr:>8}")
        out.append(dict(kind=kind, sev=sev, xy_conf=round(float(xy), 3),
                        recovered=round(float(rec), 3), spurious=round(float(spur), 3),
                        crashes=int(cr)))
    print("=" * 68)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

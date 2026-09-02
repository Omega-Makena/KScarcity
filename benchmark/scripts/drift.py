"""Drift benchmark — how the online estimator handles a regime change.

A stream with a hard break at t=T: an x->y relationship holds before T and dies
after, while a new x->w relationship is born at T. We measure, per forgetting
window, how quickly the estimator reacts:

  stale_survival   steps after T until the dead x->y edge decays (|r| < STALE_THR)
  detect_latency   steps after T until the new x->w edge is picked up (|r| > LIVE_THR)
  cusum_latency    steps after T until Page's CUSUM flags the x->y break from residuals
  p1_falarm        CUSUM firings DURING stable phase 1 (should be ~0)
  phase1_recall    |r_xy| at end of phase 1 (forgetting must not hurt live detection)

Cumulative statistics (window=0) are sticky: the dead edge lingers. A window
forgets it. This is the mechanism behind Scarcity's non-stationary adaptation;
the benchmark quantifies the trade (a smaller window reacts faster but on less
data). It runs on the CorrelationalHypothesis estimator directly — the level at
which the windowed-forgetting signal is exact and reproducible.

FINDING: the estimator-decay metrics (stale_survival, detect_latency) are the
reliable drift signals. The default RegimeTracker (Page's CUSUM, threshold=5.0,
drift=0.5, sigma EMA alpha=0.05) is far too sensitive on prediction residuals --
it false-alarms ~65 times across a stable 800-step phase, so its ~0 latency is
uninformative. The tracker needs recalibration before it can serve as a break
detector; see p1_falarm.

Usage:
  python benchmark/scripts/drift.py --n_half 800 --windows 0 100 300 --seeds 7 8 9
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.relationships import CorrelationalHypothesis
from scarcity.engine.relationship_config import CorrelationalConfig
from scarcity.engine.discovery import RegimeTracker

STALE_THR = 0.30    # |r| below this = the edge is considered gone
LIVE_THR = 0.50     # |r| above this = the edge is considered detected


def _stream(n_half: int, seed: int):
    rng = np.random.default_rng(seed)
    n = 2 * n_half
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    y = np.empty(n)
    y[:n_half] = 0.9 * x[:n_half] + 0.2 * rng.normal(size=n_half)   # phase 1: x->y
    y[n_half:] = 0.2 * rng.normal(size=n_half)                       # phase 2: x->y gone
    w[n_half:] = 0.9 * x[n_half:] + 0.2 * rng.normal(size=n_half)    # phase 2: x->w born
    return x, y, w, n


def run(window: int, n_half: int, seed: int) -> dict:
    x, y, w, n = _stream(n_half, seed)
    bs = max(150, window)
    h_xy = CorrelationalHypothesis("x", "y", buffer_size=bs, config=CorrelationalConfig(window=window))
    h_xw = CorrelationalHypothesis("x", "w", buffer_size=bs, config=CorrelationalConfig(window=window))
    tracker = RegimeTracker()

    stale = detect = cusum = None
    phase1_recall = None
    phase1_false_alarms = 0
    for t in range(n):
        rxy = {"x": float(x[t]), "y": float(y[t])}
        rxw = {"x": float(x[t]), "w": float(w[t])}
        h_xy.fit_step(rxy); m_xy = h_xy.evaluate(rxy)
        h_xw.fit_step(rxw); m_xw = h_xw.evaluate(rxw)

        # feed CUSUM the x->y one-step prediction residual (spikes when the model
        # keeps predicting a relationship that has died). update() returns the
        # break flag AND resets its accumulator, so capture the return value.
        pred = h_xy.predict_value(rxy)
        fired = tracker.update(float(y[t]) - pred[1]) if pred is not None else False

        if t == n_half - 1:
            phase1_recall = abs(m_xy.get("correlation") or 0.0)
        if t < n_half:
            phase1_false_alarms += int(fired)
        else:
            k = t - n_half
            if stale is None and abs(m_xy.get("correlation") or 0.0) < STALE_THR:
                stale = k
            if detect is None and abs(m_xw.get("correlation") or 0.0) > LIVE_THR:
                detect = k
            if cusum is None and fired:
                cusum = k
    cap = n - n_half
    return dict(window=window, seed=seed,
                phase1_recall=round(phase1_recall, 3),
                stale_survival=stale if stale is not None else cap,
                detect_latency=detect if detect is not None else cap,
                cusum_latency=cusum if cusum is not None else cap,
                phase1_false_alarms=phase1_false_alarms,
                stale_censored=stale is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_half", type=int, default=800)
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = ap.parse_args()

    per = {}
    for W in args.windows:
        per[W] = [run(W, args.n_half, s) for s in args.seeds]

    print(f"\nDrift benchmark  (n_half={args.n_half}, seeds={args.seeds})")
    print("=" * 72)
    print(f"{'window':>7} {'phase1_recall':>14} {'stale_surv':>11} {'detect_lat':>11} "
          f"{'cusum_lat':>10} {'p1_falarm':>10}")
    print("-" * 78)
    agg = {}
    for W in args.windows:
        rs = per[W]
        pr = np.mean([r["phase1_recall"] for r in rs])
        st = np.mean([r["stale_survival"] for r in rs])
        dt = np.mean([r["detect_latency"] for r in rs])
        cu = np.mean([r["cusum_latency"] for r in rs])
        fa = np.mean([r["phase1_false_alarms"] for r in rs])
        note = "  (cumulative: sticky)" if W == 0 else ""
        agg[W] = dict(phase1_recall=round(pr, 3), stale_survival=round(st, 1),
                      detect_latency=round(dt, 1), cusum_latency=round(cu, 1),
                      phase1_false_alarms=round(fa, 1))
        print(f"{W:>7} {pr:>14.3f} {st:>11.1f} {dt:>11.1f} {cu:>10.1f} {fa:>10.1f}{note}")
    print("=" * 72)
    print(json.dumps({"per_seed": {str(k): v for k, v in per.items()}, "mean": {str(k): v for k, v in agg.items()}}))


if __name__ == "__main__":
    main()

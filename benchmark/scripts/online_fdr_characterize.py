"""Characterize online FDR control: uncorrected vs batch BH vs LORD++.

Sweeps the fraction of true signals in a sequential p-value stream and reports,
for each method, the realized FDR and power averaged over many streams. The
point: a fixed threshold (uncorrected) does not control FDR as the stream grows;
batch BH controls it but needs the whole pool at once; LORD++ controls it
online, decision by decision, at a modest power cost.

Usage:
  python benchmark/scripts/online_fdr_characterize.py --n 200 --streams 800
"""
import argparse
import json

import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.online_fdr import LordOnlineFDR

ALPHA = 0.05


def _bh(p, alpha):
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    dec = np.zeros(n, bool)
    if k > 0:
        dec[order[:k]] = True
    return dec


def _metrics(dec, is_sig):
    R = int(dec.sum())
    V = int((dec & ~is_sig).sum())
    fdp = V / max(R, 1)
    power = int((dec & is_sig).sum()) / max(int(is_sig.sum()), 1)
    return fdp, power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--streams", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    fracs = [0.0, 0.05, 0.1, 0.2, 0.4]
    methods = ["uncorrected", "batch_BH", "LORD"]
    print(f"\nOnline FDR characterization  (n={args.n}, streams={args.streams}, alpha={ALPHA})")
    print("=" * 72)
    print(f"{'signal_frac':>11} {'method':>12} {'FDR':>8} {'power':>8}")
    print("-" * 72)
    out = []
    for frac in fracs:
        n_sig = int(frac * args.n)
        acc = {m: {"fdr": [], "power": []} for m in methods}
        for _ in range(args.streams):
            is_sig = np.zeros(args.n, bool)
            is_sig[:n_sig] = True
            rng.shuffle(is_sig)
            p = np.where(is_sig, rng.beta(0.1, 30, args.n), rng.random(args.n))

            dec_u = p < ALPHA
            dec_b = _bh(p, ALPHA)
            lord = LordOnlineFDR(alpha=ALPHA)
            dec_l = np.array([lord.test(pi) for pi in p])

            for m, dec in zip(methods, (dec_u, dec_b, dec_l)):
                f, pw = _metrics(dec, is_sig)
                acc[m]["fdr"].append(f)
                acc[m]["power"].append(pw)
        for m in methods:
            fdr = float(np.mean(acc[m]["fdr"]))
            pw = float(np.mean(acc[m]["power"]))
            print(f"{frac:>11.2f} {m:>12} {fdr:>8.3f} {pw:>8.3f}")
            out.append(dict(signal_frac=frac, method=m, fdr=round(fdr, 4), power=round(pw, 4)))
        print("-" * 72)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

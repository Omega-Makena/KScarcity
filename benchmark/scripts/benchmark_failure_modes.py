"""
Failure-mode taxonomy benchmark.

Runs the adversarial suite (scarcity/synthetic/failure_modes.py) and reports what
Scarcity does on each deliberately-hard dataset: no-relationship, confounding,
reverse causality, nonstationary decay, collider bias, weak signal, feedback,
and Simpson's paradox. Each scenario is wrapped in the experiment layer so its
seed, code version, hardware, runtime, and verdict are recorded to JSON.

    python benchmark/scripts/benchmark_failure_modes.py --seed 0 \
        --out benchmark/reports/outputs/failure_modes
"""
import argparse
import json
import os

from scarcity.synthetic.failure_modes import SCENARIOS
from scarcity.experiment import experiment, DatasetSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark/reports/outputs/failure_modes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for scen in SCENARIOS:
        ds = DatasetSpec(name=f"failure_mode:{scen.name}", version="1")
        with experiment(f"failure_mode_{scen.name}", dataset=ds,
                        config={"scenario": scen.name, "category": scen.category},
                        seed=args.seed, out_dir=args.out) as rec:
            result = scen.run(args.seed)
            rec.metrics = {"verdict": result.verdict, **result.detail}
        rows.append({
            "scenario": result.name, "category": result.category,
            "verdict": result.verdict, "expected": result.expected,
            "observed": result.observed, "runtime_s": rec.runtime_seconds,
        })

    width = max(len(r["scenario"]) for r in rows) + 2
    print(f"\n{'scenario':<{width}}{'category':<14}{'verdict':<12}observed")
    print("-" * 100)
    for r in rows:
        print(f"{r['scenario']:<{width}}{r['category']:<14}{r['verdict']:<12}{r['observed']}")

    summary_path = os.path.join(args.out, "failure_modes_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRecords + summary written to {args.out}/")


if __name__ == "__main__":
    main()

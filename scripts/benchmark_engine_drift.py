import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allow running as a standalone script: include workspace root for local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine


@dataclass
class RunOutcome:
    magnitude: float
    detected: bool
    latency_steps: Optional[int]
    stable_mean: float
    drift_mean: float
    pressure_delta: float
    split_seen: bool


def _export_results(
    output_dir: Path,
    mode: str,
    repeats: int,
    stable_steps: int,
    drift_steps: int,
    seed: int,
    results_by_mag: Dict[float, Dict[str, object]],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"engine_drift_{mode}_{ts}"

    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / f"{base}.csv"

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "mode": mode,
            "repeats": repeats,
            "stable_steps": stable_steps,
            "drift_steps": drift_steps,
            "seed": seed,
        },
        "results": {
            str(mag): {
                "summary": data["summary"],
                "runs": data["runs"],
            }
            for mag, data in results_by_mag.items()
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "magnitude",
                "run_index",
                "detected",
                "latency_steps",
                "stable_mean",
                "drift_mean",
                "pressure_delta",
                "split_seen",
                "detect_rate",
                "split_rate",
                "latency_mean",
                "latency_median",
                "latency_p95",
                "avg_pressure_delta",
            ],
        )
        writer.writeheader()
        for mag, data in results_by_mag.items():
            summary = data["summary"]
            runs = data["runs"]
            for idx, run in enumerate(runs):
                writer.writerow(
                    {
                        "magnitude": mag,
                        "run_index": idx,
                        "detected": run["detected"],
                        "latency_steps": run["latency_steps"],
                        "stable_mean": run["stable_mean"],
                        "drift_mean": run["drift_mean"],
                        "pressure_delta": run["pressure_delta"],
                        "split_seen": run["split_seen"],
                        "detect_rate": summary["detect_rate"],
                        "split_rate": summary["split_rate"],
                        "latency_mean": summary["latency_mean"],
                        "latency_median": summary["latency_median"],
                        "latency_p95": summary["latency_p95"],
                        "avg_pressure_delta": summary["avg_pressure_delta"],
                    }
                )

    return {"json": json_path, "csv": csv_path}


def _run_once(
    magnitude: float,
    stable_steps: int,
    drift_steps: int,
    seed: int,
    mode: str,
) -> RunOutcome:
    rng = np.random.default_rng(seed)

    engine = OnlineDiscoveryEngine(mode=mode)
    engine.initialize_v2({"fields": [{"name": "X"}, {"name": "Y"}]}, use_causal=False)

    # Force one coarse group so we can observe split behavior directly.
    engine.grouper.groups.clear()
    engine.grouper.var_to_group.clear()
    engine.grouper._create_group({"X", "Y"})
    engine.grouper.split_threshold = 1.0

    stable_pressures: List[float] = []
    for t in range(stable_steps):
        x = float(t) / 50.0
        # Stable regime: near-identity with tiny noise.
        y = x + float(rng.normal(0.0, 0.01))
        res = engine.process_row({"X": x, "Y": y})
        stable_pressures.append(float(res.get("drift_pressure", 0.0)))

    stable_tail = stable_pressures[-min(60, len(stable_pressures)) :]
    stable_mean = float(np.mean(stable_tail)) if stable_tail else 0.0

    # Calibrate threshold slightly above stable pressure.
    engine.grouper.split_threshold = stable_mean + 0.03
    base_groups = int(res.get("groups", 0))

    first_detect_step: Optional[int] = None
    split_seen = False
    drift_pressures: List[float] = []

    for dt in range(drift_steps):
        t = stable_steps + dt
        x = float(t) / 50.0
        # Drift regime: slope inversion and offset change scaled by magnitude.
        y = (-magnitude * x) + (4.0 + 0.5 * magnitude) + float(rng.normal(0.0, 0.02))

        res = engine.process_row({"X": x, "Y": y})
        pressure = float(res.get("drift_pressure", 0.0))
        drift_pressures.append(pressure)

        if first_detect_step is None and bool(res.get("drift_alert")):
            first_detect_step = dt + 1

        if int(res.get("groups", 0)) > base_groups:
            split_seen = True

    drift_tail = drift_pressures[-min(60, len(drift_pressures)) :]
    drift_mean = float(np.mean(drift_tail)) if drift_tail else 0.0

    return RunOutcome(
        magnitude=magnitude,
        detected=first_detect_step is not None,
        latency_steps=first_detect_step,
        stable_mean=stable_mean,
        drift_mean=drift_mean,
        pressure_delta=drift_mean - stable_mean,
        split_seen=split_seen,
    )


def _summarize(outcomes: List[RunOutcome]) -> Dict[str, float]:
    detected_latencies = [o.latency_steps for o in outcomes if o.latency_steps is not None]
    detect_rate = len(detected_latencies) / max(1, len(outcomes))

    summary = {
        "runs": float(len(outcomes)),
        "detect_rate": detect_rate,
        "split_rate": sum(1 for o in outcomes if o.split_seen) / max(1, len(outcomes)),
        "avg_pressure_delta": float(np.mean([o.pressure_delta for o in outcomes])) if outcomes else 0.0,
    }

    if detected_latencies:
        summary["latency_mean"] = float(np.mean(detected_latencies))
        summary["latency_median"] = float(statistics.median(detected_latencies))
        summary["latency_p95"] = float(np.percentile(detected_latencies, 95))
    else:
        summary["latency_mean"] = float("nan")
        summary["latency_median"] = float("nan")
        summary["latency_p95"] = float("nan")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OnlineDiscoveryEngine drift detection latency.")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.3, 0.6, 1.0, 1.4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stable-steps", type=int, default=120)
    parser.add_argument("--drift-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["balanced", "performance"], default="balanced")
    parser.add_argument("--output-dir", default="artifacts/benchmarks")
    args = parser.parse_args()

    print("Drift benchmark: OnlineDiscoveryEngine")
    print(
        f"mode={args.mode} repeats={args.repeats} stable_steps={args.stable_steps} "
        f"drift_steps={args.drift_steps}"
    )
    print("")

    header = (
        f"{'magnitude':>10}  {'detect_rate':>11}  {'split_rate':>10}  "
        f"{'lat_mean':>8}  {'lat_p95':>8}  {'delta_mean':>10}"
    )
    print(header)
    print("-" * len(header))

    results_by_mag: Dict[float, Dict[str, object]] = {}

    for i, mag in enumerate(args.magnitudes):
        outcomes = [
            _run_once(
                magnitude=mag,
                stable_steps=args.stable_steps,
                drift_steps=args.drift_steps,
                seed=args.seed + i * 1000 + r,
                mode=args.mode,
            )
            for r in range(args.repeats)
        ]

        s = _summarize(outcomes)
        results_by_mag[mag] = {
            "summary": s,
            "runs": [o.__dict__ for o in outcomes],
        }
        lat_mean = s["latency_mean"]
        lat_p95 = s["latency_p95"]

        lat_mean_str = f"{lat_mean:8.2f}" if np.isfinite(lat_mean) else "     n/a"
        lat_p95_str = f"{lat_p95:8.2f}" if np.isfinite(lat_p95) else "     n/a"

        print(
            f"{mag:10.2f}  {s['detect_rate']:11.2%}  {s['split_rate']:10.2%}  "
            f"{lat_mean_str}  {lat_p95_str}  {s['avg_pressure_delta']:10.4f}"
        )

    exported = _export_results(
        output_dir=Path(args.output_dir),
        mode=args.mode,
        repeats=args.repeats,
        stable_steps=args.stable_steps,
        drift_steps=args.drift_steps,
        seed=args.seed,
        results_by_mag=results_by_mag,
    )
    print("")
    print(f"Saved JSON: {exported['json']}")
    print(f"Saved CSV:  {exported['csv']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.relationships import EquilibriumHypothesis, FunctionalHypothesis, TemporalHypothesis


def _summary(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys()) if rows else []
    out: Dict[str, float] = {}
    for k in keys:
        arr = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


def _bench_temporal(seed: int, n: int = 260) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    y[0] = rng.normal()
    phi = 0.82
    for i in range(1, n):
        y[i] = phi * y[i - 1] + 0.12 * rng.normal()

    hyp = TemporalHypothesis("Y", lag=1)
    pred_err = []
    for i in range(n):
        row = {"Y": float(y[i])}
        hyp.fit_step(row)
        pred = hyp.predict_value(row)
        if pred and i < n - 1:
            pred_err.append(abs(float(pred[1]) - float(y[i + 1])))

    res = hyp.evaluate({})
    fit = float(res.get("fit_score", 0.0))
    mae = float(np.mean(pred_err)) if pred_err else 1.0
    acc = max(0.0, 1.0 - min(1.0, mae))
    return {"fit_score": fit, "mae": mae, "accuracy_like": acc}


def _bench_functional(seed: int, n: int = 240) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + 5.0 + 0.10 * rng.normal(size=n)

    hyp = FunctionalHypothesis("X", "Y", degree=1)
    for i in range(n):
        hyp.fit_step({"X": float(x[i]), "Y": float(y[i])})

    res = hyp.evaluate({})
    coeff = res.get("coefficients", [0.0, 0.0])
    intercept = float(coeff[0]) if len(coeff) > 0 else 0.0
    slope = float(coeff[1]) if len(coeff) > 1 else 0.0
    fit = float(res.get("fit_score", 0.0))
    param_err = abs(intercept - 5.0) + abs(slope - 2.0)
    acc = max(0.0, 1.0 - min(1.0, param_err / 3.0))
    return {"fit_score": fit, "parameter_error": param_err, "accuracy_like": acc}


def _bench_equilibrium(seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y1 = rng.normal(loc=0.0, scale=1.0, size=60)
    y2 = rng.normal(loc=10.0, scale=1.0, size=170)
    y = np.concatenate([y1, y2]).astype(np.float64)

    hyp = EquilibriumHypothesis("Y")
    for v in y:
        hyp.fit_step({"Y": float(v)})

    res = hyp.evaluate({})
    eq = float(res.get("equilibrium", 0.0))
    shift_err = abs(eq - 10.0)
    acc = max(0.0, 1.0 - min(1.0, shift_err / 10.0))
    return {"equilibrium": eq, "shift_error": shift_err, "accuracy_like": acc}


def _bench_engine_latency(seed: int, rows: int = 220) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    eng = OnlineDiscoveryEngine(mode="balanced")
    eng.initialize_v2({"fields": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}, use_causal=False)

    latencies = []
    for _ in range(rows):
        row = {"A": float(rng.normal()), "B": float(rng.normal()), "C": float(rng.normal())}
        t0 = time.perf_counter()
        eng.process_row(row)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p99_ms = float(np.percentile(latencies, 99))
    # Normalize to an accuracy-like score where lower latency is better.
    score = max(0.0, 1.0 - min(1.0, avg_ms / 50.0))
    return {"avg_latency_ms": avg_ms, "p99_latency_ms": p99_ms, "accuracy_like": score}


def _write(payload: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"online_model_accuracy_{ts}.json"
    csv_path = out_dir / f"online_model_accuracy_{ts}.csv"
    md_path = out_dir / f"online_model_accuracy_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    rows = []
    for r in payload.get("results", []):
        if not isinstance(r, dict):
            continue
        summary = r.get("summary", {}) if isinstance(r.get("summary"), dict) else {}
        rows.append({"model": r.get("model"), **summary})

    if rows:
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# Online Model Accuracy Benchmark",
        "",
        f"Generated at (UTC): {payload.get('generated_at_utc')}",
        "",
        "| Model | Accuracy-like | Key Metric 1 | Key Metric 2 |",
        "|---|---:|---:|---:|",
    ]
    for r in payload.get("results", []):
        if not isinstance(r, dict):
            continue
        s = r.get("summary", {}) if isinstance(r.get("summary"), dict) else {}
        numeric_keys = [k for k in s.keys() if k.endswith("_mean")]
        k1 = numeric_keys[0] if numeric_keys else "n/a"
        k2 = numeric_keys[1] if len(numeric_keys) > 1 else "n/a"
        lines.append(
            f"| {r.get('model')} | {float(s.get('accuracy_like_mean', 0.0)):.4f} | "
            f"{k1}={float(s.get(k1, 0.0)):.4f} | {k2}={float(s.get(k2, 0.0)):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark online model family metrics and publish reports.")
    parser.add_argument("--repeats", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    benchmarks = {
        "temporal_hypothesis": _bench_temporal,
        "functional_hypothesis": _bench_functional,
        "equilibrium_hypothesis": _bench_equilibrium,
        "online_engine_latency": _bench_engine_latency,
    }

    results: List[Dict[str, object]] = []
    for name, fn in benchmarks.items():
        rows = [fn(args.seed + i) for i in range(max(6, args.repeats))]
        summ = _summary(rows)
        results.append({"model": name, "summary": summ})
        print(f"{name:22s} accuracy_like={summ.get('accuracy_like_mean', 0.0):.4f}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": max(6, args.repeats),
        "results": results,
    }
    out_paths = _write(payload, Path(args.output_dir))
    print("Published reports:")
    print(f"  JSON: {out_paths['json']}")
    print(f"  CSV:  {out_paths['csv']}")
    print(f"  MD:   {out_paths['md']}")


if __name__ == "__main__":
    main()

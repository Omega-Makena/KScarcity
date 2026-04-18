from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scarcity.meta.cross_meta import CrossDomainMetaAggregator, CrossMetaConfig
from scarcity.meta.domain_meta import DomainMetaLearner
from scarcity.meta.optimizer import OnlineReptileOptimizer


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _simulate_once(method: str, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    dim = 4
    keys = [f"p{i}" for i in range(dim)]

    true_target = rng.normal(0.0, 0.8, size=dim).astype(np.float32)

    learner = DomainMetaLearner()
    domain_updates = []
    domain_count = 8
    for i in range(domain_count):
        domain_id = f"d{i}"
        is_outlier = i in {domain_count - 1}
        base = true_target.copy()
        noise = rng.normal(0.0, 0.08 if not is_outlier else 0.9, size=dim).astype(np.float32)
        final_params_vec = base + noise

        gain = float(0.70 + rng.normal(0.0, 0.08))
        stability = float(0.82 + rng.normal(0.0, 0.06))
        if is_outlier:
            gain = float(0.20 + rng.normal(0.0, 0.05))
            stability = float(0.15 + rng.normal(0.0, 0.03))

        for t in range(5):
            scale = 0.30 + (0.14 * t)
            params_t = {k: float(v) for k, v in zip(keys, final_params_vec * scale)}
            learner.observe(
                domain_id=domain_id,
                metrics={"gain_p50": gain + 0.01 * t, "stability_avg": stability},
                parameters=params_t,
            )

        # Final observation uses full target-aligned parameters to emit a meaningful delta.
        params_final = {k: float(v) for k, v in zip(keys, final_params_vec)}

        upd = learner.observe(
            domain_id=domain_id,
            metrics={"gain_p50": gain + 0.08, "stability_avg": stability},
            parameters=params_final,
        )
        domain_updates.append(upd)

    agg = CrossDomainMetaAggregator(
        CrossMetaConfig(method=method, trim_alpha=0.1, min_confidence=0.05)
    )
    agg_vec, _, meta = agg.aggregate(domain_updates)
    if agg_vec.size == 0:
        agg_vec = np.zeros(dim, dtype=np.float32)

    agg_error = float(np.linalg.norm(agg_vec - true_target))
    zero_error = float(np.linalg.norm(true_target))
    transfer_gain = float(zero_error - agg_error)
    transfer_gain_rate = _safe_div(transfer_gain, max(1e-9, zero_error))

    sign_match = float(np.mean((np.sign(agg_vec) == np.sign(true_target)).astype(np.float32)))

    optimizer = OnlineReptileOptimizer()
    prior = optimizer.apply(
        aggregated_vector=agg_vec,
        keys=keys,
        reward=0.78,
        drg_profile={"vram_high": 0.0, "latency_high": 0.0, "bandwidth_free": 1.0},
    )
    prior_vec = np.array([prior[k] for k in keys], dtype=np.float32)
    post_error = float(np.linalg.norm(prior_vec - true_target))

    # Controlled rollback detection episodes.
    tp = tn = fp = fn = 0
    for j in range(16):
        crash = bool(j % 2 == 0)
        reward = 0.03 if crash else 0.75
        pred = optimizer.should_rollback(reward)
        if crash and pred:
            tp += 1
        elif crash and not pred:
            fn += 1
        elif (not crash) and pred:
            fp += 1
        else:
            tn += 1

        optimizer.apply(
            aggregated_vector=agg_vec * (1.5 if crash else 0.7),
            keys=keys,
            reward=reward,
            drg_profile={"vram_high": 1.0 if crash else 0.0, "latency_high": 1.0 if crash else 0.0, "bandwidth_free": 0.0 if crash else 1.0},
        )

    rollback_accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    rollback_precision = _safe_div(tp, tp + fp)
    rollback_recall = _safe_div(tp, tp + fn)

    return {
        "participants": float(meta.get("participants", 0)),
        "confidence_mean": float(meta.get("confidence_mean", 0.0)),
        "sign_accuracy": sign_match,
        "transfer_gain": transfer_gain,
        "transfer_gain_rate": transfer_gain_rate,
        "agg_error": agg_error,
        "post_prior_error": post_error,
        "rollback_accuracy": rollback_accuracy,
        "rollback_precision": rollback_precision,
        "rollback_recall": rollback_recall,
    }


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys()) if rows else []
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _write(outputs: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"meta_model_accuracy_{ts}.json"
    csv_path = out_dir / f"meta_model_accuracy_{ts}.csv"
    md_path = out_dir / f"meta_model_accuracy_{ts}.md"

    json_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=True), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        rows: List[Dict[str, object]] = []
        for block in outputs.get("results", []):
            if not isinstance(block, dict):
                continue
            row = {"model": block.get("model")}
            row.update(block.get("summary", {}))
            rows.append(row)
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    lines = [
        "# Meta Model Accuracy Benchmark",
        "",
        f"Generated at (UTC): {outputs.get('generated_at_utc')}",
        "",
        "| Model | Sign Acc | Transfer Gain Rate | Rollback Acc | Rollback Precision | Rollback Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for block in outputs.get("results", []):
        if not isinstance(block, dict):
            continue
        s = block.get("summary", {}) if isinstance(block.get("summary"), dict) else {}
        lines.append(
            f"| {block.get('model')} | {float(s.get('sign_accuracy_mean', 0.0)):.4f} | "
            f"{float(s.get('transfer_gain_rate_mean', 0.0)):.4f} | {float(s.get('rollback_accuracy_mean', 0.0)):.4f} | "
            f"{float(s.get('rollback_precision_mean', 0.0)):.4f} | {float(s.get('rollback_recall_mean', 0.0)):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark meta-model accuracy and rollback reliability.")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    methods = ["trimmed_mean", "median"]
    results: List[Dict[str, object]] = []

    for method in methods:
        trials: List[Dict[str, float]] = []
        for i in range(max(4, args.repeats)):
            trials.append(_simulate_once(method=method, seed=args.seed + (1000 * (methods.index(method) + 1)) + i))

        summary = _summarize(trials)
        results.append({"model": method, "summary": summary})
        print(
            f"{method:14s} sign_acc={summary.get('sign_accuracy_mean', 0.0):.4f} "
            f"transfer_gain_rate={summary.get('transfer_gain_rate_mean', 0.0):.4f} "
            f"rollback_acc={summary.get('rollback_accuracy_mean', 0.0):.4f}"
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": max(4, args.repeats),
        "results": results,
    }
    out_paths = _write(payload, Path(args.output_dir))
    print("Published reports:")
    print(f"  JSON: {out_paths['json']}")
    print(f"  CSV:  {out_paths['csv']}")
    print(f"  MD:   {out_paths['md']}")


if __name__ == "__main__":
    main()

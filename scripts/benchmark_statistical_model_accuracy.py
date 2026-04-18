from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scarcity.engine.forecasting import _compute_garch_varx_forecast


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _simulate_series(steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((steps, 2), dtype=np.float32)
    out[0] = np.array([6.0, 4.0], dtype=np.float32)

    for t in range(1, steps):
        shock = 0.0
        if 120 <= t < 136:
            shock = 0.75
        noise = rng.normal(0.0, 0.22, size=2).astype(np.float32)
        prev = out[t - 1]
        infl = 0.82 * prev[0] + 0.08 * prev[1] + shock + noise[0]
        gdp = 0.10 * prev[0] + 0.86 * prev[1] - 0.25 * shock + noise[1]
        out[t] = np.array([infl, gdp], dtype=np.float32)
    return out


def _train_varx_weights(train: np.ndarray) -> np.ndarray:
    x = train[:-1]
    y = train[1:]
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    return w.T.astype(np.float32)


def _train_ols_weights(train: np.ndarray) -> np.ndarray:
    x_prev = train[:-1]
    x = np.concatenate([np.ones((x_prev.shape[0], 1), dtype=np.float32), x_prev], axis=1)
    y = train[1:]
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    return b.astype(np.float32)


def _direction_accuracy(prev: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    # Match backtest semantics: direction is correct when (pred-prev) and (actual-prev)
    # have non-negative product (includes flat/no-change predictions).
    ok = ((pred - prev) * (actual - prev)) >= 0.0
    return float(np.mean(ok.astype(np.float32)))


def _magnitude_score(prev: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    den = np.abs(actual - prev)
    den = np.where(den < 1e-6, 1.0, den)
    err = np.abs(pred - actual) / den
    score = 1.0 - np.minimum(1.0, err)
    return float(np.mean(score))


def _metrics(prev: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    dir_acc = _direction_accuracy(prev, pred, actual)
    mag_acc = _magnitude_score(prev, pred, actual)
    overall = 0.6 * dir_acc + 0.4 * mag_acc
    return {
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": dir_acc,
        "magnitude_accuracy": mag_acc,
        "overall_accuracy": overall,
    }


def _run_ar1(test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev = test[:-1]
    actual = test[1:]
    pred = prev.copy()
    return prev, pred, actual


def _run_varx_garch(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _train_varx_weights(train)
    sigma2 = np.ones(2, dtype=np.float32) * 0.1

    prev_rows = []
    preds = []
    actuals = []
    for i in range(1, len(test)):
        prev = test[i - 1].astype(np.float32)
        f, v = _compute_garch_varx_forecast(
            W=w,
            X_t=prev,
            exogenous_shock=0.0,
            sigma2_t=sigma2,
            max_steps=1,
        )
        sigma2 = v[0].astype(np.float32)
        prev_rows.append(prev)
        preds.append(f[0].astype(np.float32))
        actuals.append(test[i].astype(np.float32))

    return np.asarray(prev_rows), np.asarray(preds), np.asarray(actuals)


def _run_ols(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = _train_ols_weights(train)
    prev = test[:-1]
    x = np.concatenate([np.ones((prev.shape[0], 1), dtype=np.float32), prev], axis=1)
    pred = x @ b
    actual = test[1:]
    return prev, pred.astype(np.float32), actual


def _summary(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys()) if rows else []
    out: Dict[str, float] = {}
    for k in keys:
        arr = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


def _write(payload: Dict[str, object], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"statistical_model_accuracy_{ts}.json"
    csv_path = out_dir / f"statistical_model_accuracy_{ts}.csv"
    md_path = out_dir / f"statistical_model_accuracy_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    rows = []
    for r in payload.get("results", []):
        if not isinstance(r, dict):
            continue
        s = r.get("summary", {}) if isinstance(r.get("summary"), dict) else {}
        rows.append({"model": r.get("model"), **s})

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    lines = [
        "# Statistical Model Accuracy Benchmark",
        "",
        f"Generated at (UTC): {payload.get('generated_at_utc')}",
        "",
        "| Model | Overall Acc | Direction Acc | Magnitude Acc | RMSE | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload.get("results", []):
        if not isinstance(r, dict):
            continue
        s = r.get("summary", {}) if isinstance(r.get("summary"), dict) else {}
        lines.append(
            f"| {r.get('model')} | {float(s.get('overall_accuracy_mean', 0.0)):.4f} | "
            f"{float(s.get('direction_accuracy_mean', 0.0)):.4f} | {float(s.get('magnitude_accuracy_mean', 0.0)):.4f} | "
            f"{float(s.get('rmse_mean', 0.0)):.4f} | {float(s.get('mae_mean', 0.0)):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark statistical model accuracy and publish reports.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    models = {
        "ar1_persistence": _run_ar1,
        "varx_garch": _run_varx_garch,
        "ols_lag1": _run_ols,
    }

    outputs: List[Dict[str, object]] = []
    for model_name, runner in models.items():
        trials: List[Dict[str, float]] = []
        for i in range(max(6, args.repeats)):
            series = _simulate_series(steps=max(220, args.steps), seed=args.seed + i)
            split = int(0.7 * len(series))
            train = series[:split]
            test = series[split - 1 :]
            prev, pred, actual = runner(train if model_name != "ar1_persistence" else test, test) if model_name != "ar1_persistence" else runner(test)
            trials.append(_metrics(prev, pred, actual))

        summary = _summary(trials)
        outputs.append({"model": model_name, "summary": summary})
        print(
            f"{model_name:16s} overall={summary.get('overall_accuracy_mean', 0.0):.4f} "
            f"dir={summary.get('direction_accuracy_mean', 0.0):.4f} "
            f"mag={summary.get('magnitude_accuracy_mean', 0.0):.4f}"
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": max(6, args.repeats),
        "results": outputs,
    }
    out_paths = _write(payload, Path(args.output_dir))
    print("Published reports:")
    print(f"  JSON: {out_paths['json']}")
    print(f"  CSV:  {out_paths['csv']}")
    print(f"  MD:   {out_paths['md']}")


if __name__ == "__main__":
    main()

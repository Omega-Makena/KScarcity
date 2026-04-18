from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_databases.model_registry import FLModelRegistry


FEATURE_COUNT = 6
FEATURE_KEYS = [
    "threat_score",
    "escalation_score",
    "coordination_score",
    "urgency_rate",
    "imperative_rate",
    "policy_severity",
]


@dataclass
class FoldMetrics:
    fold: int
    model_name: str
    threshold: float
    samples_fit: int
    samples_val: int
    samples_test: int
    fit_positive_rate: float
    test_positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    tpr: float
    tnr: float
    brier: float
    ece: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = _safe_div(tp + tn, len(y_true))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    tpr = recall
    tnr = _safe_div(tn, tn + fp)
    balanced_accuracy = 0.5 * (tpr + tnr)

    brier = float(np.mean((y_true.astype(float) - y_score.astype(float)) ** 2))
    ece = _expected_calibration_error(y_true.astype(float), y_score.astype(float), bins=10)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "brier": float(brier),
        "ece": float(ece),
    }


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    y_true = y_true.astype(float)
    y_score = np.clip(y_score.astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = max(1, len(y_true))
    ece = 0.0
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i == bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc_bin = float(np.mean(y_true[mask]))
        conf_bin = float(np.mean(y_score[mask]))
        ece += abs(acc_bin - conf_bin) * (count / n)
    return float(ece)


def _generate_dataset(samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(low=0.0, high=1.0, size=(samples, FEATURE_COUNT)).astype(np.float64)

    # Add smooth time trend to support temporal split validation.
    time_index = np.arange(samples, dtype=np.float64)
    drift = 0.15 * np.sin(2.0 * np.pi * time_index / max(200.0, samples / 3.0))

    # Structured, mildly imbalanced signal.
    score = (
        0.35 * x[:, 0]
        + 0.22 * x[:, 1]
        + 0.18 * x[:, 3]
        + 0.10 * x[:, 4]
        - 0.17 * x[:, 2]
        - 0.08 * x[:, 5]
        + drift
        + rng.normal(0.0, 0.05, size=samples)
    )
    prob = _sigmoid(2.8 * (score - 0.58))
    y = rng.binomial(1, np.clip(prob, 0.01, 0.99)).astype(np.float64)
    return x, y, time_index


def _rolling_splits(total: int, folds: int, min_fit: int, test_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if total < (min_fit + test_size + 20):
        split = int(0.7 * total)
        train_idx = np.arange(0, max(1, split))
        test_idx = np.arange(max(1, split), total)
        return [(train_idx, test_idx)]

    usable = total - min_fit - test_size
    step = max(1, usable // max(1, folds - 1))
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(folds):
        train_end = min_fit + i * step
        test_start = train_end
        test_end = min(total, test_start + test_size)
        if test_end - test_start < 30:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 50:
            continue
        out.append((train_idx, test_idx))
    return out or [(np.arange(0, int(0.7 * total)), np.arange(int(0.7 * total), total))]


def _rebalance_binary(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y_int = y.astype(int)
    pos_idx = np.where(y_int == 1)[0]
    neg_idx = np.where(y_int == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x, y

    rng = np.random.default_rng(seed)
    target = max(len(pos_idx), len(neg_idx))
    pos_sample = rng.choice(pos_idx, size=target, replace=True)
    neg_sample = rng.choice(neg_idx, size=target, replace=True)
    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return x[idx], y[idx]


def _find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, objective: str) -> float:
    grid = np.linspace(0.05, 0.95, 37)
    best_thr = 0.5
    best_val = -1.0
    for thr in grid:
        y_pred = (y_score >= thr).astype(np.float64)
        m = _classification_metrics(y_true, y_pred, y_score)
        value = float(m.get(objective, 0.0))
        if value > best_val:
            best_val = value
            best_thr = float(thr)
    return best_thr


def _model_predict(model_name: str, weights: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if model_name == "hypothesis_ensemble":
        if weights.size >= FEATURE_COUNT + 1:
            rls_w = weights[-(FEATURE_COUNT + 1) : -1]
            bias = float(weights[-1])
            raw = x_test @ rls_w + bias
            return np.clip(raw, 0.0, 1.0)
        return np.full(shape=(len(x_test),), fill_value=0.5, dtype=np.float64)

    if weights.size < FEATURE_COUNT:
        return np.full(shape=(len(x_test),), fill_value=0.5, dtype=np.float64)

    raw = x_test @ weights[:FEATURE_COUNT]
    return _sigmoid(raw)


def _mean_std_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * (std / np.sqrt(max(1, arr.size)))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def _run_fold(
    fold_id: int,
    model_name: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    objective: str,
) -> FoldMetrics:
    model = FLModelRegistry.create(model_name, n_features=FEATURE_COUNT, feature_names=FEATURE_KEYS)
    weights = np.zeros(FEATURE_COUNT, dtype=np.float64)

    for _ in range(max(1, epochs)):
        update = model.train_local(x_fit, y_fit, global_weights=weights)
        weights = np.asarray(update.weights, dtype=np.float64)

    y_val_score = _model_predict(model_name, weights, x_val)
    threshold = _find_best_threshold(y_val, y_val_score, objective=objective)

    y_test_score = _model_predict(model_name, weights, x_test)
    y_test_pred = (y_test_score >= threshold).astype(np.float64)
    m = _classification_metrics(y_test, y_test_pred, y_test_score)

    return FoldMetrics(
        fold=fold_id,
        model_name=model_name,
        threshold=float(threshold),
        samples_fit=len(x_fit),
        samples_val=len(x_val),
        samples_test=len(x_test),
        fit_positive_rate=float(np.mean(y_fit)) if len(y_fit) else 0.0,
        test_positive_rate=float(np.mean(y_test)) if len(y_test) else 0.0,
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        balanced_accuracy=m["balanced_accuracy"],
        tpr=m["tpr"],
        tnr=m["tnr"],
        brier=m["brier"],
        ece=m["ece"],
    )


def evaluate_model(
    model_name: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    folds: int,
    test_size: int,
    min_fit: int,
    epochs: int,
    objective: str,
    rebalance_train: bool,
    seed: int,
) -> Dict[str, object]:
    split_defs = _rolling_splits(total=len(x_all), folds=folds, min_fit=min_fit, test_size=test_size)
    fold_rows: List[FoldMetrics] = []

    for fold_id, (train_idx, test_idx) in enumerate(split_defs, start=1):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]
        y_test = y_all[test_idx]

        val_size = max(30, int(0.2 * len(x_train)))
        if len(x_train) <= val_size + 10:
            continue

        x_fit = x_train[:-val_size]
        y_fit = y_train[:-val_size]
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]

        if rebalance_train:
            x_fit, y_fit = _rebalance_binary(x_fit, y_fit, seed=seed + fold_id)

        fold_result = _run_fold(
            fold_id=fold_id,
            model_name=model_name,
            x_fit=x_fit,
            y_fit=y_fit,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            epochs=epochs,
            objective=objective,
        )
        fold_rows.append(fold_result)

    if not fold_rows:
        raise RuntimeError(f"No valid folds produced for model: {model_name}")

    def _vals(field: str) -> List[float]:
        return [float(getattr(r, field)) for r in fold_rows]

    metrics_keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "tpr", "tnr", "brier", "ece"]
    metrics_mean = {k: float(np.mean(_vals(k))) for k in metrics_keys}
    metrics_std = {k: float(np.std(_vals(k), ddof=1)) if len(fold_rows) > 1 else 0.0 for k in metrics_keys}
    metrics_ci95 = {
        k: {
            "low": _mean_std_ci(_vals(k))["ci95_low"],
            "high": _mean_std_ci(_vals(k))["ci95_high"],
        }
        for k in metrics_keys
    }

    return {
        "model_name": model_name,
        "folds": [r.__dict__ for r in fold_rows],
        "objective_metric": objective,
        "best_threshold_mean": float(np.mean(_vals("threshold"))),
        "best_threshold_std": float(np.std(_vals("threshold"), ddof=1)) if len(fold_rows) > 1 else 0.0,
        "samples_fit_avg": float(np.mean(_vals("samples_fit"))),
        "samples_val_avg": float(np.mean(_vals("samples_val"))),
        "samples_test_avg": float(np.mean(_vals("samples_test"))),
        "fit_positive_rate_avg": float(np.mean(_vals("fit_positive_rate"))),
        "test_positive_rate_avg": float(np.mean(_vals("test_positive_rate"))),
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "metrics_ci95": metrics_ci95,
    }


def _write_outputs(results: List[Dict[str, object]], out_dir: Path, config: Dict[str, object]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"fl_model_accuracy_{ts}.json"
    csv_path = out_dir / f"fl_model_accuracy_{ts}.csv"
    md_path = out_dir / f"fl_model_accuracy_{ts}.md"

    payload = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "models_evaluated": len(results),
                "config": config,
                "results": results,
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "objective_metric",
                "best_threshold_mean",
                "samples_fit_avg",
                "samples_test_avg",
                "fit_positive_rate_avg",
                "test_positive_rate_avg",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "tpr",
                "tnr",
                "brier",
                "ece",
                "accuracy_ci95_low",
                "accuracy_ci95_high",
                "f1_ci95_low",
                "f1_ci95_high",
                "balanced_accuracy_ci95_low",
                "balanced_accuracy_ci95_high",
            ],
        )
        writer.writeheader()
        for row in results:
            mm = row.get("metrics_mean", {}) if isinstance(row.get("metrics_mean"), dict) else {}
            ci = row.get("metrics_ci95", {}) if isinstance(row.get("metrics_ci95"), dict) else {}
            writer.writerow(
                {
                    "model_name": row.get("model_name"),
                    "objective_metric": row.get("objective_metric"),
                    "best_threshold_mean": row.get("best_threshold_mean"),
                    "samples_fit_avg": row.get("samples_fit_avg"),
                    "samples_test_avg": row.get("samples_test_avg"),
                    "fit_positive_rate_avg": row.get("fit_positive_rate_avg"),
                    "test_positive_rate_avg": row.get("test_positive_rate_avg"),
                    "accuracy": mm.get("accuracy"),
                    "precision": mm.get("precision"),
                    "recall": mm.get("recall"),
                    "f1": mm.get("f1"),
                    "balanced_accuracy": mm.get("balanced_accuracy"),
                    "tpr": mm.get("tpr"),
                    "tnr": mm.get("tnr"),
                    "brier": mm.get("brier"),
                    "ece": mm.get("ece"),
                    "accuracy_ci95_low": (ci.get("accuracy", {}) or {}).get("low"),
                    "accuracy_ci95_high": (ci.get("accuracy", {}) or {}).get("high"),
                    "f1_ci95_low": (ci.get("f1", {}) or {}).get("low"),
                    "f1_ci95_high": (ci.get("f1", {}) or {}).get("high"),
                    "balanced_accuracy_ci95_low": (ci.get("balanced_accuracy", {}) or {}).get("low"),
                    "balanced_accuracy_ci95_high": (ci.get("balanced_accuracy", {}) or {}).get("high"),
                }
            )

    lines = [
        "# FL Model Accuracy Benchmark",
        "",
        f"Generated at (UTC): {payload['generated_at_utc']}",
        f"Models evaluated: {len(results)}",
        f"Objective metric for threshold search: {config.get('objective')}",
        f"Rebalance train data: {config.get('rebalance_train')}",
        "",
        "| Model | Thr | Accuracy | Precision | Recall | F1 | Balanced Acc | TPR | TNR | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        mm = r.get("metrics_mean", {}) if isinstance(r.get("metrics_mean"), dict) else {}
        lines.append(
            f"| {r.get('model_name')} | {float(r.get('best_threshold_mean', 0.5)):.3f} | "
            f"{float(mm.get('accuracy', 0.0)):.4f} | {float(mm.get('precision', 0.0)):.4f} | "
            f"{float(mm.get('recall', 0.0)):.4f} | {float(mm.get('f1', 0.0)):.4f} | "
            f"{float(mm.get('balanced_accuracy', 0.0)):.4f} | {float(mm.get('tpr', 0.0)):.4f} | "
            f"{float(mm.get('tnr', 0.0)):.4f} | {float(mm.get('brier', 0.0)):.4f} | {float(mm.get('ece', 0.0)):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run accuracy metrics for all registered FL models and publish reports.")
    parser.add_argument("--samples", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=280)
    parser.add_argument("--min-fit", type=int, default=700)
    parser.add_argument("--objective", type=str, default="f1", choices=["f1", "balanced_accuracy", "recall", "precision"])
    parser.add_argument("--rebalance-train", action="store_true", default=True)
    parser.add_argument("--no-rebalance-train", action="store_false", dest="rebalance_train")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    model_names = FLModelRegistry.list_models()
    if not model_names:
        raise SystemExit("No registered FL models were found.")

    x_all, y_all, _t = _generate_dataset(samples=max(1000, args.samples), seed=args.seed)

    results: List[Dict[str, object]] = []
    for name in model_names:
        result = evaluate_model(
            model_name=name,
            x_all=x_all,
            y_all=y_all,
            folds=max(1, args.folds),
            test_size=max(60, args.test_size),
            min_fit=max(120, args.min_fit),
            epochs=max(1, args.epochs),
            objective=args.objective,
            rebalance_train=bool(args.rebalance_train),
            seed=args.seed,
        )
        results.append(result)
        mm = result.get("metrics_mean", {}) if isinstance(result.get("metrics_mean"), dict) else {}
        print(
            f"{name:20s} thr={float(result.get('best_threshold_mean', 0.5)):.3f} "
            f"acc={float(mm.get('accuracy', 0.0)):.4f} rec={float(mm.get('recall', 0.0)):.4f} "
            f"f1={float(mm.get('f1', 0.0)):.4f} bacc={float(mm.get('balanced_accuracy', 0.0)):.4f} "
            f"ece={float(mm.get('ece', 0.0)):.4f}"
        )

    results = sorted(
        results,
        key=lambda r: float((r.get("metrics_mean", {}) or {}).get("balanced_accuracy", 0.0)),
        reverse=True,
    )
    out_paths = _write_outputs(
        results,
        Path(args.output_dir),
        config={
            "samples": args.samples,
            "seed": args.seed,
            "epochs": args.epochs,
            "folds": args.folds,
            "test_size": args.test_size,
            "min_fit": args.min_fit,
            "objective": args.objective,
            "rebalance_train": bool(args.rebalance_train),
        },
    )
    print("Published reports:")
    print(f"  JSON: {out_paths['json']}")
    print(f"  CSV:  {out_paths['csv']}")
    print(f"  MD:   {out_paths['md']}")


if __name__ == "__main__":
    main()

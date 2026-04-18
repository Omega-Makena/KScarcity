from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_databases import ScarcityFederationManager
from federated_databases.model_registry import FLModelRegistry


def _health_sample(uid: str, county: str, severity: float, label: int) -> dict:
    sev = float(max(0.0, min(1.0, severity)))
    return {
        "sample_uid": uid,
        "timestamp": "2026-03-01T00:00:00+00:00",
        "county": county,
        "sector": "health",
        "criticality": sev,
        "threat_score": min(1.0, 0.20 + 0.75 * sev),
        "escalation_score": min(1.0, 0.10 + 0.70 * sev),
        "coordination_score": min(1.0, 0.05 + 0.50 * sev),
        "urgency_rate": min(1.0, 0.15 + 0.65 * sev),
        "imperative_rate": min(1.0, 0.10 + 0.55 * sev),
        "policy_severity": min(1.0, 0.10 + 0.80 * sev),
        "label": int(label),
    }


def _register_three_health_nodes(manager: ScarcityFederationManager) -> None:
    manager.register_node("health_turkana", county_filter="Turkana")
    manager.register_node("health_garissa", county_filter="Garissa")
    manager.register_node("health_marsabit", county_filter="Marsabit")


def _append_round_samples(manager: ScarcityFederationManager, round_idx: int, n_per_node: int = 60) -> None:
    rng = np.random.default_rng(100 + round_idx)

    turkana = []
    garissa = []
    marsabit = []
    for i in range(n_per_node):
        sev_t = float(np.clip(rng.normal(0.78 - 0.02 * round_idx, 0.10), 0.01, 0.99))
        sev_g = float(np.clip(rng.normal(0.66 + 0.01 * round_idx, 0.09), 0.01, 0.99))
        sev_m = float(np.clip(rng.normal(0.52 + 0.03 * round_idx, 0.12), 0.01, 0.99))

        turkana.append(
            _health_sample(
                uid=f"turkana-r{round_idx}-{i}",
                county="Turkana",
                severity=sev_t,
                label=int(sev_t >= 0.60),
            )
        )
        garissa.append(
            _health_sample(
                uid=f"garissa-r{round_idx}-{i}",
                county="Garissa",
                severity=sev_g,
                label=int(sev_g >= 0.60),
            )
        )
        marsabit.append(
            _health_sample(
                uid=f"marsabit-r{round_idx}-{i}",
                county="Marsabit",
                severity=sev_m,
                label=int(sev_m >= 0.60),
            )
        )

    manager._node_store("health_turkana").add_samples(turkana)
    manager._node_store("health_garissa").add_samples(garissa)
    manager._node_store("health_marsabit").add_samples(marsabit)


_REGISTERED_MODELS = FLModelRegistry.list_models()


@pytest.mark.parametrize("model_name", _REGISTERED_MODELS)
def test_registered_models_handle_edge_cases(model_name: str) -> None:
    model = FLModelRegistry.create(model_name, n_features=6, learning_rate=0.1)

    edge_cases = [
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.0, 1.0], dtype=np.float64),
        ),
        (
            np.array(
                [
                    [1e6, -1e6, 5e5, -5e5, 1e3, -1e3],
                    [-1e6, 1e6, -5e5, 5e5, -1e3, 1e3],
                    [2e6, 2e6, -2e6, -2e6, 1e4, -1e4],
                    [-2e6, -2e6, 2e6, 2e6, -1e4, 1e4],
                ],
                dtype=np.float64,
            ),
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        ),
        (
            np.array(
                [[0.55, 0.51, 0.50, 0.49, 0.52, 0.48]] * 16,
                dtype=np.float64,
            ),
            np.ones(16, dtype=np.float64),
        ),
    ]

    for x, y in edge_cases:
        update = model.train_local(x, y, global_weights=np.zeros(6, dtype=np.float64))

        assert update.weights is not None
        assert update.weights.size > 0
        assert bool(np.isfinite(update.weights).all())
        assert np.isfinite(update.loss)
        assert np.isfinite(update.gradient_norm)
        assert isinstance(update.metrics, dict)
        assert "samples" in update.metrics


@pytest.mark.parametrize("model_name", _REGISTERED_MODELS)
def test_registered_models_emit_finite_metrics_across_time(tmp_path: Path, model_name: str) -> None:
    manager = ScarcityFederationManager(base_dir=tmp_path / f"runtime_{model_name}")
    _register_three_health_nodes(manager)

    rounds = 4
    series_loss = []
    series_grad = []

    for round_idx in range(rounds):
        _append_round_samples(manager, round_idx, n_per_node=50)
        result = manager.run_sync_round(
            lookback_hours=24,
            source_path=tmp_path / "missing.csv",
            model_name=model_name,
        )

        assert result.participants == 3
        assert result.total_samples >= 150
        assert np.isfinite(result.global_loss)
        assert np.isfinite(result.global_gradient_norm)

        series_loss.append(float(result.global_loss))
        series_grad.append(float(result.global_gradient_norm))

    history = manager.get_round_history(limit=rounds)
    assert len(history) == rounds

    # History is reverse-chronological (latest first)
    observed_rounds = [int(h["round_number"]) for h in history]
    assert observed_rounds == list(range(rounds, 0, -1))

    # Validate that the same metrics are available for each round across time.
    for row in history:
        assert np.isfinite(float(row["global_loss"]))
        assert np.isfinite(float(row["global_gradient_norm"]))
        assert int(row["participants"]) == 3
        assert int(row["total_samples"]) > 0

        metrics = row.get("metrics", {})
        assert "local_metrics" in metrics
        local = metrics["local_metrics"]
        assert len(local) == 3

        for lm in local:
            assert int(lm["sample_count"]) > 0
            assert int(lm["feature_count"]) == 6
            assert np.isfinite(float(lm["loss"]))
            assert np.isfinite(float(lm["gradient_norm"]))
            assert np.isfinite(float(lm["mean_criticality"]))

    # Across-time metric tracks must be finite and non-empty for dashboard plotting.
    assert len(series_loss) == rounds
    assert len(series_grad) == rounds
    assert bool(np.isfinite(np.array(series_loss, dtype=np.float64)).all())
    assert bool(np.isfinite(np.array(series_grad, dtype=np.float64)).all())

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
from scarcity.engine.vectorized_core import VectorizedRLS
from scarcity.federation.aggregator import (
    AggregationConfig,
    AggregationMethod,
    FederatedAggregator,
)
from scarcity.meta.domain_meta import DomainMetaConfig, DomainMetaLearner


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


def _seed_health_nodes(manager: ScarcityFederationManager) -> None:
    manager.register_node("health_turkana", county_filter="Turkana")
    manager.register_node("health_garissa", county_filter="Garissa")
    manager.register_node("health_marsabit", county_filter="Marsabit")

    rng = np.random.default_rng(42)

    turkana = []
    garissa = []
    marsabit = []
    for i in range(80):
        sev_t = float(np.clip(rng.normal(0.82, 0.08), 0.02, 0.99))
        sev_g = float(np.clip(rng.normal(0.68, 0.09), 0.02, 0.99))
        sev_m = float(np.clip(rng.normal(0.55, 0.10), 0.02, 0.99))
        turkana.append(_health_sample(f"turkana-{i}", "Turkana", sev_t, int(sev_t >= 0.60)))
        garissa.append(_health_sample(f"garissa-{i}", "Garissa", sev_g, int(sev_g >= 0.60)))
        marsabit.append(_health_sample(f"marsabit-{i}", "Marsabit", sev_m, int(sev_m >= 0.60)))

    manager._node_store("health_turkana").add_samples(turkana)
    manager._node_store("health_garissa").add_samples(garissa)
    manager._node_store("health_marsabit").add_samples(marsabit)


def test_health_sector_pipeline_end_to_end(tmp_path: Path) -> None:
    manager = ScarcityFederationManager(base_dir=tmp_path / "health_runtime")
    _seed_health_nodes(manager)

    result = manager.run_sync_round(
        lookback_hours=24,
        source_path=tmp_path / "missing.csv",
        model_name="logistic",
    )

    assert result.participants == 3
    assert result.total_samples > 0
    assert np.isfinite(result.global_loss)
    assert np.isfinite(result.global_gradient_norm)

    history = manager.get_round_history(limit=1)
    assert len(history) == 1
    latest = history[0]
    assert latest["participants"] == 3
    assert latest["total_samples"] >= 200
    assert "local_metrics" in latest["metrics"]
    assert len(latest["metrics"]["local_metrics"]) == 3

    for metric in latest["metrics"]["local_metrics"]:
        assert metric["sample_count"] > 0
        assert np.isfinite(metric["loss"])
        assert np.isfinite(metric["gradient_norm"])

    exchanges = manager.get_exchange_log(limit=20)
    # 3 uplinks + 3 downlinks minimum for a 3-node round.
    assert len(exchanges) >= 6


_REGISTERED_MODELS = FLModelRegistry.list_models()


@pytest.mark.parametrize("model_name", _REGISTERED_MODELS)
def test_registered_fl_models_emit_finite_metrics(model_name: str) -> None:
    model = FLModelRegistry.create(model_name, n_features=6, learning_rate=0.1)

    x = np.array(
        [
            [0.81, 0.78, 0.66, 0.72, 0.74, 0.70],
            [0.25, 0.22, 0.30, 0.28, 0.20, 0.24],
            [0.68, 0.64, 0.59, 0.63, 0.58, 0.67],
            [0.34, 0.29, 0.40, 0.31, 0.35, 0.33],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 0, 1, 0], dtype=np.float64)

    update = model.train_local(x, y, global_weights=np.zeros(6, dtype=np.float64))

    assert update.weights is not None
    assert update.weights.size > 0
    assert bool(np.isfinite(update.weights).all())
    assert np.isfinite(update.loss)
    assert np.isfinite(update.gradient_norm)
    assert isinstance(update.metrics, dict)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        (AggregationMethod.TRIMMED_MEAN, {"trim_alpha": 0.2}),
        (AggregationMethod.KRUM, {}),
        (AggregationMethod.BULYAN, {"trim_alpha": 0.15, "multi_krum_m": 5}),
    ],
)
def test_poisoning_defense_robust_aggregation(method: AggregationMethod, kwargs: dict) -> None:
    rng = np.random.default_rng(7)
    honest = [rng.normal(loc=1.0, scale=0.02, size=24) for _ in range(7)]
    attackers = [
        np.ones(24, dtype=np.float64) * 1200.0,
        np.ones(24, dtype=np.float64) * -900.0,
    ]
    all_updates = honest + attackers

    honest_center = np.mean(np.vstack(honest), axis=0)
    fedavg = np.mean(np.vstack(all_updates), axis=0)
    dist_fedavg = float(np.linalg.norm(fedavg - honest_center))

    config = AggregationConfig(method=method, **kwargs)
    robust, meta = FederatedAggregator(config).aggregate(all_updates)
    dist_robust = float(np.linalg.norm(robust - honest_center))

    assert dist_robust < dist_fedavg * 0.20
    assert meta.get("participants") == len(all_updates)


def test_online_vectorized_rls_large_pool_edge_case() -> None:
    n_models = 1_000_000
    n_active = 2048

    rls = VectorizedRLS(n_models=n_models, n_features=2, lambda_forget=0.995)
    rng = np.random.default_rng(11)

    indices = rng.choice(n_models, size=n_active, replace=False)
    x = np.column_stack(
        [
            np.ones(n_active, dtype=np.float32),
            rng.normal(0.0, 1.0, size=n_active).astype(np.float32),
        ]
    )
    y = (0.4 + 1.7 * x[:, 1] + rng.normal(0.0, 0.03, size=n_active)).astype(np.float32)

    rls.update_subset(indices, x, y)

    updated = rls.W[indices]
    assert bool(np.isfinite(updated).all())
    assert float(np.mean(np.abs(updated[:, 1]))) > 0.01


def test_meta_learning_remember_forget_cycle_health() -> None:
    learner = DomainMetaLearner(
        DomainMetaConfig(
            max_history=5,
            confidence_decay=0.85,
            confidence_boost=0.08,
            meta_lr_min=0.02,
            meta_lr_max=0.20,
        )
    )

    for i in range(8):
        learner.observe(
            "healthcare",
            metrics={"gain_p50": 0.20 + 0.06 * i, "stability_avg": 0.88},
            parameters={"lr": 0.01 + i * 0.001, "alpha": 0.20},
        )

    state_good = learner.state("healthcare")
    good_conf = state_good.confidence
    assert len(state_good.history) == 5

    for i in range(6):
        learner.observe(
            "healthcare",
            metrics={"gain_p50": -0.10 - 0.05 * i, "stability_avg": 0.15},
            parameters={"lr": 0.02, "alpha": 0.25},
        )

    state_after = learner.state("healthcare")
    assert len(state_after.history) == 5
    assert state_after.confidence < good_conf
    assert np.isfinite(state_after.confidence)
    assert np.isfinite(state_after.last_score)

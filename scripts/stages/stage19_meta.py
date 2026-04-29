"""stage19_meta.py — Stages 19.1–19.3: Meta-learning benchmarks.

OnlineReptileOptimizer + EpisodicMemory, CrossDomainMetaAggregator,
MetaIntegrativeLayer policy verification.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result

_VEC_DIM = 8


# ---------------------------------------------------------------------------
# Stage 19.1 — Reptile optimizer + EpisodicMemory
# ---------------------------------------------------------------------------

def run_stage_19_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "19.1", "OnlineReptileOptimizer"
    try:
        from scarcity.meta.optimizer import OnlineReptileOptimizer
        from scarcity.meta.memory import EpisodicMemory
    except ImportError as e:
        return skip_result(stage_id, name, f"OnlineReptileOptimizer import failed: {e}")

    try:
        optimizer = OnlineReptileOptimizer()
        memory = EpisodicMemory()

        rng = np.random.default_rng(42)
        n_updates = 6 if fast else 12

        # Simulate domain updates: reward goes from 0.3 to 0.9
        keys = [f"feat_{i}" for i in range(_VEC_DIM)]
        prior_before = None
        rewards = np.linspace(0.3, 0.9, n_updates)

        for i, reward in enumerate(rewards):
            agg_vec = rng.standard_normal(_VEC_DIM).astype(np.float32)
            drg_profile = {"level": "GREEN", "cpu": 0.3}
            result = optimizer.apply(
                aggregated_vector=agg_vec,
                keys=keys,
                reward=float(reward),
                drg_profile=drg_profile,
            )
            if i == 0:
                prior_before = dict(optimizer.state.prior) if hasattr(optimizer, "state") else None

            # Store episode in memory
            key_vec = rng.standard_normal(_VEC_DIM).astype(np.float32)
            memory.store(
                key=key_vec,
                value={"reward": float(reward)},
                context={"domain": f"domain_{i % 3}"},
                delta={"conf_delta": float(reward * 0.1)},
                policy={"action": "update"},
            )

        prior_after = dict(optimizer.state.prior) if hasattr(optimizer, "state") else None

        # Prior should have moved (Reptile updates it)
        prior_moved = False
        if prior_before is not None and prior_after is not None:
            diffs = [abs(prior_after.get(k, 0) - prior_before.get(k, 0))
                     for k in prior_after]
            prior_moved = max(diffs) > 0.0 if diffs else False

        # Retrieve top-1 from memory
        query = rng.standard_normal(_VEC_DIM).astype(np.float32)
        retrieved = memory.retrieve(query_key=query, top_k=1, min_similarity=0.0)
        retrieval_works = isinstance(retrieved, list)

        wall = time.time() - t0
        status = "PASS" if (prior_moved and retrieval_works) else ("WARN" if retrieval_works else "FAIL")

        return make_result(stage_id, name, status,
                           "Reptile prior moves after updates; EpisodicMemory retrieval works",
                           {"prior_moved": prior_moved,
                            "n_episodes_stored": n_updates,
                            "retrieval_works": retrieval_works,
                            "n_retrieved": len(retrieved) if retrieval_works else 0},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "Reptile prior moves; EpisodicMemory stores/retrieves",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 19.2 — CrossDomainMetaAggregator Byzantine robustness
# ---------------------------------------------------------------------------

def run_stage_19_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "19.2", "CrossDomainMetaAggregator"
    try:
        from scarcity.meta.cross_meta import CrossDomainMetaAggregator, CrossMetaConfig
        from scarcity.meta.domain_meta import DomainMetaUpdate
    except ImportError as e:
        return skip_result(stage_id, name, f"CrossDomainMetaAggregator import failed: {e}")

    try:
        import time as _time
        config = CrossMetaConfig(method="trimmed_mean", trim_alpha=0.2, min_confidence=0.05)
        agg = CrossDomainMetaAggregator(config=config)
        rng = np.random.default_rng(42)
        keys = [f"k{i}" for i in range(_VEC_DIM)]

        # 4 coherent updates (confidence=0.8) + 1 Byzantine (100x scale, confidence=0.8)
        clean_vecs = [rng.standard_normal(_VEC_DIM).astype(np.float32) for _ in range(4)]
        byzantine_vec = np.ones(_VEC_DIM, dtype=np.float32) * 100.0
        clean_mean = np.mean(clean_vecs, axis=0)

        updates = []
        for i, vec in enumerate(clean_vecs):
            updates.append(DomainMetaUpdate(
                domain_id=f"domain_{i}", vector=vec, keys=keys,
                confidence=0.8, timestamp=_time.time(), score_delta=0.1,
            ))
        updates.append(DomainMetaUpdate(
            domain_id="domain_byz", vector=byzantine_vec, keys=keys,
            confidence=0.8, timestamp=_time.time(), score_delta=-0.5,
        ))

        result_vec, result_keys, meta = agg.aggregate(updates)
        result_ok = result_vec is not None and len(result_vec) == _VEC_DIM

        if result_ok:
            diff_from_clean = float(np.linalg.norm(result_vec - clean_mean))
            byzantine_bounded = diff_from_clean < 5.0
        else:
            diff_from_clean = -1.0
            byzantine_bounded = False

        wall = time.time() - t0
        status = "PASS" if byzantine_bounded else ("WARN" if result_ok else "FAIL")

        return make_result(stage_id, name, status,
                           "CrossDomainAgg result within 5.0 of clean mean despite Byzantine",
                           {"result_ok": result_ok, "diff_from_clean_mean": round(diff_from_clean, 4),
                            "byzantine_bounded": byzantine_bounded,
                            "n_participants": int(meta.get("participants", 0))},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "CrossDomainAggregator trims Byzantine outlier",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 19.3 — MetaIntegrativeLayer policy changes with governance signal
# ---------------------------------------------------------------------------

def run_stage_19_3(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "19.3", "MetaIntegrativeLayer"
    try:
        from scarcity.meta.integrative_meta import MetaIntegrativeLayer
    except ImportError as e:
        return skip_result(stage_id, name, f"MetaIntegrativeLayer import failed: {e}")

    try:
        layer = MetaIntegrativeLayer()

        # Telemetry with low system pressure (GREEN-like)
        telemetry_low = {
            "meta_reward": 0.85, "reward_avg": 0.80,
            "cpu_util": 0.2, "mem_util": 0.15,
            "vram_util": 0.1, "latency_ms": 30.0,
            "fps": 60.0,
        }
        # Telemetry with high system pressure (RED-like)
        telemetry_high = {
            "meta_reward": 0.3, "reward_avg": 0.35,
            "cpu_util": 0.95, "mem_util": 0.92,
            "vram_util": 0.9, "latency_ms": 250.0,
            "fps": 10.0,
        }

        output_low = layer.update(telemetry_low)
        output_high = layer.update(telemetry_high)

        # Both calls should return non-null dicts
        outputs_non_null = output_low is not None and output_high is not None

        # Resource profile hints should differ between low and high pressure
        hint_low = output_low.get("resource_profile_hint") if outputs_non_null else None
        hint_high = output_high.get("resource_profile_hint") if outputs_non_null else None
        hints_differ = hint_low != hint_high if (hint_low is not None and hint_high is not None) else False

        # Meta scores should differ
        score_low = output_low.get("meta_score", 0.0) if outputs_non_null else 0.0
        score_high = output_high.get("meta_score", 0.0) if outputs_non_null else 0.0
        scores_differ = abs(score_low - score_high) > 0.01

        wall = time.time() - t0
        status = "PASS" if (outputs_non_null and scores_differ) else (
            "WARN" if outputs_non_null else "FAIL")

        return make_result(stage_id, name, status,
                           "MetaIntegrativeLayer.update() returns non-null output; meta_score differs under stress",
                           {"outputs_non_null": outputs_non_null,
                            "meta_score_low": round(float(score_low), 4),
                            "meta_score_high": round(float(score_high), 4),
                            "scores_differ": scores_differ,
                            "hints_differ": hints_differ},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "MetaIntegrativeLayer.update() varies with system pressure",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)

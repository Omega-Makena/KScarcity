"""stage22_scarcity.py — Stages 22.1–22.2: Scarcity dimension sweeps.

Data scarcity N sweep (CausalHypothesis readiness threshold) and compute
scarcity DRG coupling (BanditRouter decay + Reptile beta adaptation).
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result


def _gen_causal_rows(n: int, seed: int = 42) -> List[Dict[str, float]]:
    """X=AR(0.7), Y=0.6*X_lag1+noise — same as stage12 generator."""
    rng = np.random.default_rng(seed)
    rows = []
    x_prev = 0.0
    for _ in range(n):
        x = 0.7 * x_prev + rng.standard_normal() * 0.5
        y = 0.6 * x_prev + rng.standard_normal() * 0.3
        rows.append({"X": float(x), "Y": float(y)})
        x_prev = x
    return rows


# ---------------------------------------------------------------------------
# Stage 22.1 — Data scarcity N sweep
# ---------------------------------------------------------------------------

def run_stage_22_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "22.1", "data_scarcity_N_sweep"

    try:
        from scarcity.engine.relationships import CausalHypothesis
    except ImportError as e:
        return skip_result(stage_id, name, f"CausalHypothesis import failed: {e}")

    try:
        n_values = [5, 10, 20, 30, 50, 80] if fast else [5, 10, 20, 30, 50, 80, 100]
        all_rows = _gen_causal_rows(max(n_values), seed=42)

        results_by_n = {}
        first_ready_n = None

        for n in n_values:
            hyp = CausalHypothesis("X", "Y")
            last_row: Dict[str, float] = {}
            for row in all_rows[:n]:
                hyp.fit_step(row)
                last_row = row
            ev = hyp.evaluate(last_row)
            if isinstance(ev, dict):
                fs = float(ev.get("fit_score", 0.0))
                ready = bool(ev.get("ready", False))
                evidence = int(ev.get("evidence", 0))
            else:
                fs = 0.0
                ready = False
                evidence = 0
            results_by_n[n] = {"fit_score": round(fs, 4), "ready": ready, "evidence": evidence}
            if ready and first_ready_n is None:
                first_ready_n = n

        # N=5 should NOT be ready (too few observations)
        n5_not_ready = not results_by_n[5]["ready"]

        # Coarse monotonicity: max N should have higher score than N=10
        max_n = max(n_values)
        score_grows = results_by_n[max_n]["fit_score"] >= results_by_n[10]["fit_score"] * 0.8

        wall = time.time() - t0
        status = "PASS" if (n5_not_ready and score_grows) else (
            "WARN" if n5_not_ready else "FAIL")

        return make_result(stage_id, name, status,
                           "N=5 not ready; fit_score at max_N >= 80% of N=10 score × growth",
                           {"results_by_n": results_by_n,
                            "first_ready_n": first_ready_n,
                            "n5_not_ready": n5_not_ready,
                            "score_grows": score_grows,
                            f"score_n{max_n}": results_by_n[max_n]["fit_score"],
                            "score_n10": results_by_n[10]["fit_score"]},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "CausalHypothesis N sweep: N=5 not ready; score grows with N",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 22.2 — Compute scarcity with DRG coupling
# ---------------------------------------------------------------------------

def run_stage_22_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "22.2", "compute_scarcity_DRG_loop"

    try:
        from scarcity.engine.relationships import CausalHypothesis
        from scarcity.engine.bandit_router import BanditRouter, BanditConfig, BanditAlgorithm
        from scarcity.meta.optimizer import OnlineReptileOptimizer, MetaOptimizerConfig
    except ImportError as e:
        return skip_result(stage_id, name, f"Component import failed: {e}")

    try:
        rng = np.random.default_rng(42)
        n_rows = 50 if fast else 100
        all_rows = _gen_causal_rows(n_rows, seed=42)

        # RED DRG profile: high vram + high latency → beta decay
        red_profile = {"vram_high": 1.0, "latency_high": 1.0, "bandwidth_free": 0.0}

        # BanditRouter decay — must be callable and not raise
        router = BanditRouter(config=BanditConfig(algorithm=BanditAlgorithm.THOMPSON), n_arms=5)
        router.register_arms(5)
        decay_called = False
        try:
            router.decay()
            decay_called = True
        except Exception:
            decay_called = False

        # OnlineReptileOptimizer beta decreases under RED
        cfg = MetaOptimizerConfig(beta_init=0.1, beta_decay_rate=0.8, beta_growth_rate=1.1)
        optimizer = OnlineReptileOptimizer(config=cfg)
        keys = [f"k{i}" for i in range(4)]
        agg = rng.standard_normal(4).astype(np.float32)

        # Warm up with green first
        green_profile = {"vram_high": 0.0, "latency_high": 0.0, "bandwidth_free": 1.0}
        optimizer.apply(agg, keys, reward=0.7, drg_profile=green_profile)
        beta_before_red = optimizer.state.beta

        # Apply RED signal
        n_red_steps = 3 if fast else 5
        for _ in range(n_red_steps):
            optimizer.apply(agg, keys, reward=0.7, drg_profile=red_profile)
        beta_after_red = optimizer.state.beta

        beta_reduced = beta_after_red < beta_before_red

        # Hypothesis processing: buffer sizes [5, 20, 50, 100] under RED
        buffer_sizes = [5, 20, 50] if fast else [5, 20, 50, 100]
        fit_scores_by_n = {}
        no_crash = True

        for n in buffer_sizes:
            try:
                hyp = CausalHypothesis("X", "Y")
                last_row: Dict[str, float] = {}
                for row in all_rows[:n]:
                    hyp.fit_step(row)
                    last_row = row
                ev = hyp.evaluate(last_row)
                fs = float(ev.get("fit_score", 0.0)) if isinstance(ev, dict) else 0.0
                fit_scores_by_n[n] = round(fs, 4)
            except Exception:
                no_crash = False
                fit_scores_by_n[n] = -1.0

        # MAE non-increasing as N grows (score should not decrease significantly)
        scores_list = [fit_scores_by_n[n] for n in sorted(buffer_sizes)]
        monotone = all(
            scores_list[i + 1] >= scores_list[i] * 0.7
            for i in range(len(scores_list) - 1)
        )

        wall = time.time() - t0
        status = "PASS" if (beta_reduced and decay_called and no_crash) else (
            "WARN" if (decay_called and no_crash) else "FAIL")

        return make_result(stage_id, name, status,
                           "beta decreases after RED signal; BanditRouter.decay() works; no crash at any N",
                           {"beta_before_red": round(beta_before_red, 4),
                            "beta_after_red": round(beta_after_red, 4),
                            "beta_reduced": beta_reduced,
                            "decay_called": decay_called,
                            "no_crash": no_crash,
                            "fit_scores_by_n": fit_scores_by_n,
                            "score_monotone": monotone},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "DRG RED: beta decays; BanditRouter.decay() ok; no crash",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)

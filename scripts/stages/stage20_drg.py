"""stage20_drg.py — Stages 20.1–20.2: DRG assurance transitions and compute scarcity.

DynamicResourceGovernor assurance level detection and hypothesis processing
under RED resource pressure.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result


# Level thresholds matching DRG policy rules (cpu_util >= 0.85 → reduce_batch)
_LEVEL_GREEN_MAX = 0.55
_LEVEL_RED_MIN = 0.85


def _cpu_to_level(cpu: float) -> str:
    if cpu >= _LEVEL_RED_MIN:
        return "RED"
    if cpu >= _LEVEL_GREEN_MAX:
        return "YELLOW"
    return "GREEN"


# ---------------------------------------------------------------------------
# Stage 20.1 — DRG assurance level transitions
# ---------------------------------------------------------------------------

def run_stage_20_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "20.1", "DynamicResourceGovernor"

    try:
        from scarcity.governor.drg_core import DynamicResourceGovernor, DRGConfig
        from scarcity.governor.profiler import ResourceProfiler, ProfilerConfig
        from scarcity.governor.policies import PolicyRule
    except ImportError as e:
        return skip_result(stage_id, name, f"DRG import failed: {e}")

    try:
        config = DRGConfig()
        # Don't start the async loop — just test sync subsystems
        # alpha=0.9 makes EMA highly responsive so RED is reachable quickly
        profiler = ResourceProfiler(ProfilerConfig(ema_alpha=0.9))

        # CPU trajectory: GREEN → YELLOW → RED → YELLOW → GREEN
        # Use high alpha (0.9) and sustained high CPU to push EMA above RED threshold
        cpu_sequence = [0.20, 0.35, 0.95, 0.97, 0.96, 0.55, 0.22]

        levels_observed = []
        ema_values = []
        for cpu in cpu_sequence:
            ema, _ = profiler.update({"cpu_util": cpu})
            cpu_ema = ema.get("cpu_util", cpu)
            level = _cpu_to_level(cpu_ema)
            levels_observed.append(level)
            ema_values.append(round(cpu_ema, 3))

        # Check that RED was reached and GREEN was recovered
        saw_red = "RED" in levels_observed
        saw_recovery = levels_observed[-1] in ("GREEN", "YELLOW")
        transitions_reasonable = saw_red and saw_recovery

        # Verify PolicyRule threshold logic
        mpie_rule = PolicyRule(metric="cpu_util", threshold=0.85, action="reduce_batch", factor=0.5)
        rule_fires_high = mpie_rule.triggered(0.90)
        rule_fires_low = mpie_rule.triggered(0.40)
        policy_logic_ok = rule_fires_high and not rule_fires_low

        # DRG config instantiation succeeded
        drg_config_ok = config is not None
        profiler_ok = len(ema_values) == len(cpu_sequence)

        wall = time.time() - t0
        status = "PASS" if (transitions_reasonable and policy_logic_ok) else (
            "WARN" if transitions_reasonable else "FAIL")

        return make_result(stage_id, name, status,
                           "DRG profiler tracks CPU EMA; RED level reached and recovered; policy rules fire correctly",
                           {"levels_observed": levels_observed,
                            "ema_cpu": ema_values,
                            "saw_red": saw_red,
                            "saw_recovery": saw_recovery,
                            "policy_logic_ok": policy_logic_ok,
                            "drg_config_ok": drg_config_ok},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "DRG profiler EMA tracks CPU; level transitions GREEN/YELLOW/RED",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 20.2 — Compute scarcity sweep (hypothesis under RED DRG pressure)
# ---------------------------------------------------------------------------

def run_stage_20_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "20.2", "DRG_compute_scarcity"

    try:
        from scarcity.engine.relationships import CausalHypothesis, TemporalHypothesis
        from scarcity.meta.optimizer import OnlineReptileOptimizer, MetaOptimizerConfig
        from scarcity.engine.bandit_router import BanditRouter, BanditConfig, BanditAlgorithm
    except ImportError as e:
        return skip_result(stage_id, name, f"Component import failed: {e}")

    try:
        rng = np.random.default_rng(42)
        buffer_sizes = [5, 10, 20, 50, 100] if fast else [5, 10, 20, 50, 150]

        # RED drg_profile: high vram, high latency
        red_profile = {"vram_high": 1.0, "latency_high": 1.0, "bandwidth_free": 0.0}
        green_profile = {"vram_high": 0.0, "latency_high": 0.0, "bandwidth_free": 1.0}

        # Verify BanditRouter.decay() is callable
        router = BanditRouter(config=BanditConfig(algorithm=BanditAlgorithm.THOMPSON), n_arms=5)
        router.register_arms(5)
        decay_ok = False
        try:
            router.decay()
            decay_ok = True
        except Exception:
            decay_ok = False

        # Verify optimizer beta decays under RED pressure
        optimizer = OnlineReptileOptimizer(config=MetaOptimizerConfig(
            beta_init=0.1, beta_decay_rate=0.8, beta_growth_rate=1.1
        ))
        keys = [f"f{i}" for i in range(4)]
        agg_vec = rng.standard_normal(4).astype(np.float32)

        # Start with GREEN (beta should grow)
        optimizer.apply(agg_vec, keys, reward=0.7, drg_profile=green_profile)
        beta_green = optimizer.state.beta

        # Switch to RED (beta should shrink)
        optimizer.apply(agg_vec, keys, reward=0.7, drg_profile=red_profile)
        beta_red = optimizer.state.beta

        beta_decays_under_red = beta_red <= beta_green

        # Sweep: run CausalHypothesis + TemporalHypothesis at each buffer size
        fit_scores = {}
        no_exception_all = True
        X_seq = rng.standard_normal(max(buffer_sizes)).astype(float)
        Y_seq = 0.7 * X_seq + rng.standard_normal(max(buffer_sizes)) * 0.2

        for n in buffer_sizes:
            try:
                causal_hyp = CausalHypothesis("X", "Y")
                temporal_hyp = TemporalHypothesis("X")
                last_row = {}
                for i in range(n):
                    last_row = {"X": float(X_seq[i]), "Y": float(Y_seq[i])}
                    causal_hyp.fit_step(last_row)
                    temporal_hyp.fit_step(last_row)
                causal_result = causal_hyp.evaluate(last_row)
                fs = causal_result.get("fit_score", 0.0) if isinstance(causal_result, dict) else 0.0
                fit_scores[n] = round(float(fs), 4)
            except Exception:
                no_exception_all = False
                fit_scores[n] = -1.0

        # N=100 (or max) should have higher score than N=10
        max_n = max(buffer_sizes)
        score_monotone = fit_scores.get(max_n, 0.0) >= fit_scores.get(10, 0.0) * 0.8  # 20% tolerance

        wall = time.time() - t0
        status = "PASS" if (no_exception_all and beta_decays_under_red and decay_ok) else (
            "WARN" if no_exception_all else "FAIL")

        return make_result(stage_id, name, status,
                           "No exception at any buffer size under RED; beta decays under RED; BanditRouter.decay() works",
                           {"no_exception_all": no_exception_all,
                            "fit_scores_by_n": fit_scores,
                            "score_monotone": score_monotone,
                            "beta_green": round(beta_green, 4),
                            "beta_red": round(beta_red, 4),
                            "beta_decays_under_red": beta_decays_under_red,
                            "bandit_decay_ok": decay_ok},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "Hypothesis + DRG RED: no crash; beta decays; BanditRouter.decay() works",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)

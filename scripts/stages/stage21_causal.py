"""stage21_causal.py — Stages 21.1–21.3: Causal pipeline (DoWhy + EconML + refutation).

Tests the production causal inference pipeline end-to-end:
sign recovery, cross-backend agreement, and refutation sensitivity.
"""
from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result


def _make_causal_dataframe(n: int = 200, seed: int = 42):
    """Generate DataFrame with treatment X → outcome Y (positive effect ~0.8) and a confounder W."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, n)
    X = 0.3 * W + rng.normal(0, 1, n)
    Y = 0.8 * X + 0.3 * W + rng.normal(0, 0.3, n)
    return pd.DataFrame({"treatment": X, "outcome": Y, "confounder": W})


# ---------------------------------------------------------------------------
# Stage 21.1 — DoWhy default backend sign recovery
# ---------------------------------------------------------------------------

def run_stage_21_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "21.1", "run_causal_DoWhy"

    try:
        from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec
        from scarcity.causal.engine import run_causal
    except ImportError as e:
        return skip_result(stage_id, name, f"Causal pipeline import failed: {e}")

    try:
        import dowhy  # noqa: F401
    except ImportError:
        return skip_result(stage_id, name, "DoWhy not installed")

    try:
        n = 150 if fast else 300
        df = _make_causal_dataframe(n=n, seed=42)

        spec = EstimandSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["confounder"],
            type=EstimandType.ATE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeSpec(
                refutation_simulations=5 if fast else 15,
                seed=42,
                parallelism="none",
                artifact_root=tmpdir,
                refute_random_common_cause=False,
                refute_placebo_treatment=False,
                refute_data_subset=False,
            )
            result = run_causal(df, spec, runtime)

        has_results = bool(result.results)
        if has_results:
            estimate_value = float(result.results[0].estimate)
            sign_positive = estimate_value > 0
        else:
            estimate_value = None
            sign_positive = False

        wall = time.time() - t0
        status = "PASS" if sign_positive else ("WARN" if has_results else "FAIL")

        return make_result(stage_id, name, status,
                           "sign(estimate) == +1 for Y=0.8*X+noise",
                           {"has_results": has_results,
                            "estimate": round(estimate_value, 4) if estimate_value is not None else None,
                            "sign_positive": sign_positive,
                            "n_errors": len(result.errors)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "DoWhy sign recovery for positive causal effect",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 21.2 — EconML CausalForestDML sign recovery
# ---------------------------------------------------------------------------

def run_stage_21_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "21.2", "run_causal_EconML"

    try:
        from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec
        from scarcity.causal.engine import run_causal
    except ImportError as e:
        return skip_result(stage_id, name, f"Causal pipeline import failed: {e}")

    try:
        import dowhy  # noqa: F401
    except ImportError:
        return skip_result(stage_id, name, "DoWhy not installed")

    try:
        import econml  # noqa: F401
    except ImportError:
        return skip_result(stage_id, name, "EconML not installed")

    try:
        n = 150 if fast else 300
        df = _make_causal_dataframe(n=n, seed=42)

        spec = EstimandSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["confounder"],
            effect_modifiers=["confounder"],
            type=EstimandType.ATE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeSpec(
                estimator_method="backdoor.econml.dml.CausalForestDML",
                estimator_params={
                    "init_params": {
                        "n_estimators": 100,
                        "min_samples_leaf": 10,
                        "random_state": 42,
                    },
                    "fit_params": {},
                },
                refutation_simulations=3 if fast else 8,
                seed=42,
                parallelism="none",
                artifact_root=tmpdir,
                refute_random_common_cause=False,
                refute_placebo_treatment=False,
                refute_data_subset=False,
            )
            result = run_causal(df, spec, runtime)

        has_results = bool(result.results)
        if has_results:
            estimate_value = float(result.results[0].estimate) if not isinstance(result.results[0].estimate, list) else float(np.mean(result.results[0].estimate))
            sign_positive = estimate_value > 0
        else:
            estimate_value = None
            sign_positive = False

        wall = time.time() - t0
        status = "PASS" if sign_positive else ("WARN" if has_results else "FAIL")

        return make_result(stage_id, name, status,
                           "EconML CausalForestDML sign(estimate) == +1",
                           {"has_results": has_results,
                            "estimate": round(estimate_value, 4) if estimate_value is not None else None,
                            "sign_positive": sign_positive,
                            "n_errors": len(result.errors)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "EconML CausalForestDML sign recovery",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 21.3 — Refutation tests (placebo + random common cause)
# ---------------------------------------------------------------------------

def run_stage_21_3(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "21.3", "Validator_refutation"

    try:
        from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec
        from scarcity.causal.engine import run_causal
    except ImportError as e:
        return skip_result(stage_id, name, f"Causal pipeline import failed: {e}")

    try:
        import dowhy  # noqa: F401
    except ImportError:
        return skip_result(stage_id, name, "DoWhy not installed")

    try:
        n = 200 if fast else 400
        df = _make_causal_dataframe(n=n, seed=42)

        spec = EstimandSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["confounder"],
            type=EstimandType.ATE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeSpec(
                refutation_simulations=10 if fast else 30,
                seed=42,
                parallelism="none",
                artifact_root=tmpdir,
                refute_random_common_cause=True,
                refute_placebo_treatment=True,
                refute_data_subset=False,
            )
            result = run_causal(df, spec, runtime)

        has_results = bool(result.results)
        if not has_results:
            wall = time.time() - t0
            return fail_result(stage_id, name, "Refutation tests require successful estimation",
                               f"No results returned. Errors: {result.errors}", wall)

        effect = result.results[0]
        real_estimate = float(effect.estimate)
        refuter_results = effect.refuter_results

        # Check refuters ran
        placebo_ran = "placebo_treatment" in refuter_results
        rcc_ran = "random_common_cause" in refuter_results

        # Placebo: new_effect should be near zero (< 0.5 × real)
        placebo_ok = False
        placebo_new_effect = None
        if placebo_ran and refuter_results["placebo_treatment"].get("status") == "ok":
            placebo_new_effect = refuter_results["placebo_treatment"].get("new_effect")
            if placebo_new_effect is not None:
                try:
                    placebo_abs = abs(float(placebo_new_effect))
                    placebo_ok = placebo_abs < 0.5 * abs(real_estimate)
                except Exception:
                    placebo_ok = False

        # Random common cause: new_effect should not flip sign vs real estimate
        rcc_ok = False
        rcc_new_effect = None
        if rcc_ran and refuter_results["random_common_cause"].get("status") == "ok":
            rcc_new_effect = refuter_results["random_common_cause"].get("new_effect")
            if rcc_new_effect is not None:
                try:
                    rcc_same_sign = (float(rcc_new_effect) * real_estimate) > 0
                    rcc_ok = rcc_same_sign
                except Exception:
                    rcc_ok = False

        wall = time.time() - t0
        both_ran = placebo_ran and rcc_ran
        status = "PASS" if (both_ran and placebo_ok and rcc_ok) else (
            "WARN" if both_ran else "FAIL")

        return make_result(stage_id, name, status,
                           "placebo |new_effect| < 0.5×real; RCC does not flip sign",
                           {"real_estimate": round(real_estimate, 4),
                            "placebo_ran": placebo_ran,
                            "placebo_new_effect": round(float(placebo_new_effect), 4) if placebo_new_effect is not None else None,
                            "placebo_ok": placebo_ok,
                            "rcc_ran": rcc_ran,
                            "rcc_new_effect": round(float(rcc_new_effect), 4) if rcc_new_effect is not None else None,
                            "rcc_ok": rcc_ok},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "Refutation: placebo near-zero; RCC no sign flip",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)

"""Standalone refutation battery (DoWhy-version-independent).

Verifies the reimplemented refuters — placebo_treatment, random_common_cause,
data_subset — run without DoWhy's refute_estimate and return correct robustness
verdicts on data with a known effect.
"""
import numpy as np
import pandas as pd
import pytest

from scarcity.causal.validation import (
    Validator,
    refute_data_subset,
    refute_placebo_treatment,
    refute_random_common_cause,
)
from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec
from scarcity.causal.sensitivity import (
    partial_r2,
    robustness_value,
    sensitivity_analysis,
)


def _confounded(n=3000, effect=2.0, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    t = 0.7 * c + rng.normal(size=n)
    y = effect * t + 3.0 * c + rng.normal(0, 0.1, n)
    return pd.DataFrame({"t": t, "y": y, "c": c})


def test_placebo_collapses_and_flags_real_effect_robust():
    df = _confounded(effect=2.0)
    rng = np.random.default_rng(1)
    r = refute_placebo_treatment(df, "t", "y", ["c"], observed=2.0, n_sim=200, rng=rng)
    assert abs(r["new_effect"]) < 0.05          # placebo effect collapses to ~0
    assert r["p_value"] < 0.05                   # observed stands out of the null
    assert r["is_robust"] is True


def test_random_common_cause_leaves_effect_unchanged():
    df = _confounded(effect=2.0)
    rng = np.random.default_rng(2)
    r = refute_random_common_cause(df, "t", "y", ["c"], observed=2.0, n_sim=50, rng=rng)
    assert abs(r["new_effect"] - 2.0) < 0.05     # irrelevant confounder cannot move it
    assert r["is_robust"] is True


def test_data_subset_effect_is_stable():
    df = _confounded(effect=2.0)
    rng = np.random.default_rng(3)
    r = refute_data_subset(df, "t", "y", ["c"], observed=2.0, n_sim=50, rng=rng)
    assert abs(r["new_effect"] - 2.0) < 0.05
    assert r["std"] < 0.05                        # low dispersion across subsets
    assert r["is_robust"] is True


def test_placebo_flags_a_null_effect_not_robust():
    # No real effect: placebo should NOT single the observed out (large p).
    rng = np.random.default_rng(4)
    n = 3000
    c = rng.normal(size=n)
    t = 0.7 * c + rng.normal(size=n)
    y = 3.0 * c + rng.normal(0, 0.1, n)          # y depends on c only, not t
    df = pd.DataFrame({"t": t, "y": y, "c": c})
    r = refute_placebo_treatment(df, "t", "y", ["c"], observed=0.0, n_sim=200, rng=rng)
    assert r["is_robust"] is False               # a zero effect is not "robust"


def test_validator_noop_without_any_checks():
    # No refuter budget and sensitivity disabled -> empty result.
    df = _confounded()
    spec = EstimandSpec(treatment="t", outcome="y", confounders=["c"], type=EstimandType.ATE)
    runtime = RuntimeSpec(refutation_simulations=0, sensitivity_analysis=False,
                          parallelism="none", seed=0)
    assert Validator.validate(spec, df, 2.0, runtime) == {}


def test_validator_runs_requested_refuters():
    df = _confounded(effect=2.0)
    spec = EstimandSpec(treatment="t", outcome="y", confounders=["c"], type=EstimandType.ATE)
    runtime = RuntimeSpec(refutation_simulations=100, parallelism="none", seed=0)
    out = Validator.validate(spec, df, 2.0, runtime)
    # Three refuters plus the always-on sensitivity analysis.
    assert {"placebo_treatment", "random_common_cause", "data_subset"} <= set(out)
    for name in ("placebo_treatment", "random_common_cause", "data_subset"):
        assert out[name]["status"] == "ok" and out[name]["is_robust"]
    assert out["sensitivity"]["status"] == "ok"


# --- Sensitivity to unobserved confounding (Cinelli-Hazlett OVB) --------------

def test_robustness_value_matches_sensemakr_darfur_benchmark():
    # Canonical published values (sensemakr Darfur: t=4.18, df=783).
    assert abs(robustness_value(4.18, 783) - 0.139) < 0.002
    assert abs(robustness_value(4.18, 783, alpha=0.05) - 0.076) < 0.002
    assert abs(partial_r2(4.18, 783) - 0.022) < 0.002


def test_rv_to_insignificance_never_exceeds_rv_to_zero():
    for t in (2.5, 4.0, 8.0, 20.0):
        assert robustness_value(t, 500, alpha=0.05) <= robustness_value(t, 500) + 1e-12


def test_rv_rises_with_effect_strength():
    weak = robustness_value(2.1, 500)
    strong = robustness_value(12.0, 500)
    assert 0.0 <= weak < strong <= 1.0


def test_rv_zero_for_null_effect():
    assert robustness_value(0.0, 500) == 0.0
    assert robustness_value(1.0, 500, alpha=0.05) == 0.0   # not even significant


def test_sensitivity_analysis_reports_expected_fields():
    df = _confounded(effect=2.0)
    s = sensitivity_analysis(df, "t", "y", ["c"])
    assert s["status"] == "ok"
    assert 0.0 <= s["robustness_value"] <= 1.0
    assert s["robustness_value_alpha"] <= s["robustness_value"] + 1e-12
    assert abs(s["estimate"] - 2.0) < 0.05          # recovers the adjusted effect
    assert "summary" in s


def test_sensitivity_skips_on_too_few_rows():
    df = _confounded(n=4)
    assert sensitivity_analysis(df, "t", "y", ["c"])["status"] == "skipped"


def test_validator_includes_sensitivity_without_simulations():
    df = _confounded(effect=2.0)
    spec = EstimandSpec(treatment="t", outcome="y", confounders=["c"], type=EstimandType.ATE)
    runtime = RuntimeSpec(refutation_simulations=0, parallelism="none", seed=0)
    out = Validator.validate(spec, df, 2.0, runtime)
    assert "sensitivity" in out and out["sensitivity"]["status"] == "ok"

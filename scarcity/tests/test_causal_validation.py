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


def test_validator_noop_without_simulation_budget():
    df = _confounded()
    spec = EstimandSpec(treatment="t", outcome="y", confounders=["c"], type=EstimandType.ATE)
    runtime = RuntimeSpec(refutation_simulations=0, parallelism="none", seed=0)
    assert Validator.validate(spec, df, 2.0, runtime) == {}


def test_validator_runs_requested_refuters():
    df = _confounded(effect=2.0)
    spec = EstimandSpec(treatment="t", outcome="y", confounders=["c"], type=EstimandType.ATE)
    runtime = RuntimeSpec(refutation_simulations=100, parallelism="none", seed=0)
    out = Validator.validate(spec, df, 2.0, runtime)
    assert set(out) == {"placebo_treatment", "random_common_cause", "data_subset"}
    assert all(v["status"] == "ok" and v["is_robust"] for v in out.values())

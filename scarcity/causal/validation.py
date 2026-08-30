"""
Validation Layer.

Runs refutation checks to validate the robustness of a causal estimate.

These refuters are implemented standalone on the underlying data (backdoor /
Frisch-Waugh-Lovell adjustment), not through DoWhy's ``refute_estimate`` — that
API's signature drifts across releases (it crashes on DoWhy 0.14). The
standalone form is version-independent, fast enough to permute many times, and
directly testable. For a linear backdoor model the adjusted slope equals DoWhy's
``linear_regression`` ATE, so the refuters probe the same estimand the engine
reports.

Three checks, the standard causal-inference battery:

- **placebo_treatment** — replace the treatment with a permutation (no real
  link) and re-estimate; a robust effect collapses toward zero, and the observed
  effect sits in the tail of the placebo null (small p).
- **random_common_cause** — add an independent random confounder to the
  adjustment set; a robust effect barely moves (an irrelevant control cannot
  explain it).
- **data_subset** — re-estimate on random subsets; a robust effect is stable
  across them (low dispersion, mean near the observed effect).
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scarcity.causal.specs import RuntimeSpec

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _ols_residual(X: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Residual of y after regressing on X (with intercept)."""
    if X is not None and getattr(X, "size", 0):
        Xi = np.column_stack([np.ones(len(y)), X])
    else:
        Xi = np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    return y - Xi @ beta


def _backdoor_effect(
    df: pd.DataFrame, treatment: str, outcome: str, confounders: Sequence[str]
) -> float:
    """Backdoor-adjusted linear effect of treatment on outcome (FWL slope).

    Identical to DoWhy's ``backdoor.linear_regression`` ATE for a linear model,
    but two numpy regressions — cheap to permute thousands of times.
    """
    t = df[treatment].to_numpy(float)
    y = df[outcome].to_numpy(float)
    confs = [c for c in confounders if c in df.columns]
    C = df[confs].to_numpy(float) if confs else None
    rt = _ols_residual(C, t)
    ry = _ols_residual(C, y)
    denom = float(rt @ rt)
    return float(rt @ ry / denom) if denom > 0 else 0.0


def refute_placebo_treatment(
    df, treatment, outcome, confounders, observed, *, n_sim, rng, alpha=0.05
) -> Dict[str, Any]:
    """Permute the treatment; a robust effect collapses and the observed effect
    lies in the tail of the placebo null."""
    placebo = np.empty(n_sim)
    t = df[treatment].to_numpy(float)
    for i in range(n_sim):
        d = df.copy()
        d[treatment] = rng.permutation(t)
        placebo[i] = _backdoor_effect(d, treatment, outcome, confounders)
    p_value = float((np.abs(placebo) >= abs(observed)).mean())
    return {
        "status": "ok",
        "new_effect": float(placebo.mean()),          # expected ~0
        "p_value": p_value,                            # P(|placebo| >= |observed|)
        "is_robust": bool(p_value < alpha),            # observed stands out of the null
        "summary": (f"placebo mean_effect={placebo.mean():.3e} "
                    f"(observed={observed:.3e}), p={p_value:.3f}"),
    }


def refute_random_common_cause(
    df, treatment, outcome, confounders, observed, *, n_sim, rng, tol=0.10
) -> Dict[str, Any]:
    """Add an independent random confounder; a robust effect barely moves."""
    n = len(df)
    effects = np.empty(n_sim)
    base = list(confounders)
    for i in range(n_sim):
        d = df.copy()
        d["_rcc"] = rng.standard_normal(n)
        effects[i] = _backdoor_effect(d, treatment, outcome, base + ["_rcc"])
    new_effect = float(effects.mean())
    rel = abs(new_effect - observed) / (abs(observed) + _EPS)
    return {
        "status": "ok",
        "new_effect": new_effect,
        "relative_change": float(rel),
        "is_robust": bool(rel < tol),
        "summary": (f"random-common-cause new_effect={new_effect:.3e} "
                    f"(observed={observed:.3e}), rel_change={rel:.2%}"),
    }


def refute_data_subset(
    df, treatment, outcome, confounders, observed, *, n_sim, rng, fraction=0.9, tol=0.20
) -> Dict[str, Any]:
    """Re-estimate on random subsets; a robust effect is stable across them."""
    n = len(df)
    k = max(10, int(n * fraction))
    effects = np.empty(n_sim)
    for i in range(n_sim):
        idx = rng.choice(n, size=k, replace=False)
        effects[i] = _backdoor_effect(df.iloc[idx], treatment, outcome, confounders)
    new_effect = float(effects.mean())
    rel = abs(new_effect - observed) / (abs(observed) + _EPS)
    return {
        "status": "ok",
        "new_effect": new_effect,
        "std": float(effects.std()),
        "relative_change": float(rel),
        "is_robust": bool(rel < tol),
        "summary": (f"data-subset ({fraction:.0%}) mean_effect={new_effect:.3e} "
                    f"+/- {effects.std():.3e} (observed={observed:.3e})"),
    }


class Validator:
    """Executes the refutation battery defined in the RuntimeSpec, standalone."""

    @staticmethod
    def validate(
        spec,
        data: pd.DataFrame,
        observed_effect: float,
        runtime: RuntimeSpec,
    ) -> Dict[str, Any]:
        """Run the requested refuters on ``data`` for ``spec``'s estimand.

        ``observed_effect`` is the engine's point estimate (e.g. ``estimate.value``);
        refuters compare against it. No-ops (returns empty) when no refuter is
        requested or the simulation budget is zero.
        """
        results: Dict[str, Any] = {}
        n_sim = int(getattr(runtime, "refutation_simulations", 0) or 0)
        if n_sim <= 0:
            return results

        treatment = spec.treatment
        outcome = spec.outcome
        confounders: List[str] = list(getattr(spec, "confounders", []) or [])
        rng = np.random.default_rng(runtime.resolved_seed())

        if not np.isfinite(observed_effect):
            logger.warning("Observed effect is not finite; skipping refutation.")
            return results

        d = data.dropna(subset=[treatment, outcome, *confounders])
        if len(d) < 20:
            logger.warning("Too few rows for refutation; skipping.")
            return results

        checks = []
        if getattr(runtime, "refute_placebo_treatment", False):
            checks.append(("placebo_treatment", refute_placebo_treatment))
        if getattr(runtime, "refute_random_common_cause", False):
            checks.append(("random_common_cause", refute_random_common_cause))
        if getattr(runtime, "refute_data_subset", False):
            checks.append(("data_subset", refute_data_subset))

        for name, fn in checks:
            logger.info(f"Running refuter: {name}")
            try:
                results[name] = fn(d, treatment, outcome, confounders, float(observed_effect),
                                   n_sim=n_sim, rng=rng)
            except Exception as exc:
                logger.warning(f"Refuter {name} failed: {exc}")
                results[name] = {"status": "error", "error": str(exc)}

        return results

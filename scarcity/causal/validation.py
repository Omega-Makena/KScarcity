"""
Validation Layer.

Runs sensitivity analyses and refutation checks to validate the robustness
of the causal estimate.
"""
import logging
from typing import Any, Dict

from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate

from scarcity.causal.specs import RuntimeSpec

logger = logging.getLogger(__name__)


class Validator:
    """Executes refutation tests defined in the RuntimeSpec."""

    @staticmethod
    def validate(
        model: CausalModel,
        estimate: CausalEstimate,
        runtime: RuntimeSpec,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        def run_refuter(method_name: str, name: str) -> None:
            logger.info(f"Running refuter: {name} ({method_name})")
            refutation = model.refute_estimate(
                estimate,
                method_name=method_name,
                num_simulations=runtime.refutation_simulations,
                random_seed=runtime.resolved_seed(),
            )
            results[name] = {
                "status": "ok",
                "is_robust": refutation.refutation_result,
                "p_value": refutation.refutation_result if isinstance(refutation.refutation_result, float) else getattr(refutation, "p_value", None),
                "new_effect": refutation.new_effect,
                "summary": str(refutation),
            }

        def fail_refuter(name: str, exc: Exception) -> None:
            results[name] = {
                "status": "error",
                "error": str(exc),
            }

        if runtime.refute_random_common_cause:
            try:
                run_refuter("random_common_cause", "random_common_cause")
            except Exception as exc:
                logger.warning(f"Refuter random_common_cause failed: {exc}")
                fail_refuter("random_common_cause", exc)

        if runtime.refute_placebo_treatment:
            try:
                run_refuter("placebo_treatment_refuter", "placebo_treatment")
            except Exception as exc:
                logger.warning(f"Refuter placebo_treatment failed: {exc}")
                fail_refuter("placebo_treatment", exc)

        if runtime.refute_data_subset:
            try:
                run_refuter("data_subset_refuter", "data_subset")
            except Exception as exc:
                logger.warning(f"Refuter data_subset failed: {exc}")
                fail_refuter("data_subset", exc)

        return results

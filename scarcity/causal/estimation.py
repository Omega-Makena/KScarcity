"""
Estimation Layer.

Factory and backend abstraction that binds the identified statistical functional
to an actual estimation algorithm (DoWhy native or EconML).
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier import IdentifiedEstimand

from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec

logger = logging.getLogger(__name__)


@dataclass
class EstimatorResult:
    estimate: CausalEstimate
    backend: str
    method_name: str
    method_params: Dict[str, Any]


class EstimatorBackend:
    name = "base"

    def supports(self, spec: EstimandSpec, runtime: RuntimeSpec) -> bool:
        raise NotImplementedError

    def estimate(
        self,
        model: CausalModel,
        identified_estimand: IdentifiedEstimand,
        spec: EstimandSpec,
        runtime: RuntimeSpec,
    ) -> EstimatorResult:
        raise NotImplementedError


class DoWhyBackend(EstimatorBackend):
    name = "dowhy"

    def supports(self, spec: EstimandSpec, runtime: RuntimeSpec) -> bool:
        method = runtime.estimator_method
        return method is None or "econml" not in method

    def estimate(
        self,
        model: CausalModel,
        identified_estimand: IdentifiedEstimand,
        spec: EstimandSpec,
        runtime: RuntimeSpec,
    ) -> EstimatorResult:
        method_name, method_params = _resolve_method(spec, runtime)
        logger.info(f"Estimating {spec.type} using {method_name}")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method_name,
            target_units=spec.target_units,
            method_params=method_params,
            test_significance=None,
        )
        return EstimatorResult(
            estimate=estimate,
            backend=self.name,
            method_name=method_name,
            method_params=method_params,
        )


class EconMLBackend(EstimatorBackend):
    name = "econml"

    def supports(self, spec: EstimandSpec, runtime: RuntimeSpec) -> bool:
        if runtime.estimator_method and "econml" in runtime.estimator_method:
            return True
        return spec.type in (EstimandType.CATE, EstimandType.ITE)

    def estimate(
        self,
        model: CausalModel,
        identified_estimand: IdentifiedEstimand,
        spec: EstimandSpec,
        runtime: RuntimeSpec,
    ) -> EstimatorResult:
        method_name, method_params = _resolve_method(spec, runtime)
        if "econml" not in method_name:
            method_name = "backdoor.econml.dml.CausalForestDML"
        logger.info(f"Estimating {spec.type} using {method_name}")
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method_name,
                target_units=spec.target_units,
                method_params=method_params,
                test_significance=None,
            )
        except ImportError as exc:
            logger.error("EconML backend requested but not installed.")
            raise ImportError("EconML is required for CATE/CausalForest estimation.") from exc

        return EstimatorResult(
            estimate=estimate,
            backend=self.name,
            method_name=method_name,
            method_params=method_params,
        )


class EstimatorFactory:
    """Selects and executes the appropriate estimator backend."""

    _backends = [EconMLBackend(), DoWhyBackend()]

    @classmethod
    def estimate(
        cls,
        model: CausalModel,
        identified_estimand: IdentifiedEstimand,
        spec: EstimandSpec,
        runtime: RuntimeSpec,
    ) -> EstimatorResult:
        runtime.normalize()
        backend = cls._select_backend(spec, runtime)
        return backend.estimate(model, identified_estimand, spec, runtime)

    @classmethod
    def _select_backend(cls, spec: EstimandSpec, runtime: RuntimeSpec) -> EstimatorBackend:
        for backend in cls._backends:
            if backend.supports(spec, runtime):
                return backend
        return DoWhyBackend()


def _resolve_method(spec: EstimandSpec, runtime: RuntimeSpec) -> tuple[str, Dict[str, Any]]:
    method_name = "backdoor.linear_regression"
    method_params: Dict[str, Any] = {}

    if spec.type == EstimandType.LATE:
        method_name = "iv.instrumental_variable"
    elif spec.type in (EstimandType.CATE, EstimandType.ITE):
        method_name = "backdoor.econml.dml.CausalForestDML"
        method_params = {
            "init_params": {
                "n_estimators": 100,
                "criterion": "mse",
                "min_samples_leaf": 10,
                "random_state": runtime.resolved_seed(),
            },
            "fit_params": {},
        }
    elif spec.type in (EstimandType.ATE, EstimandType.ATT, EstimandType.ATC):
        method_name = "backdoor.linear_regression"
    elif spec.type in (EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE):
        method_name = "mediation.two_stage_regression"

    if runtime.estimator_method:
        method_name = runtime.estimator_method
    if runtime.estimator_params:
        method_params = {**method_params, **runtime.estimator_params}

    return method_name, method_params

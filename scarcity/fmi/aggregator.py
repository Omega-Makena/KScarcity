"""
Aggregation logic for FMI cohorts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

import numpy as np  # type: ignore

from .contracts import (
    ConceptCausalSummary,
    FMIContractRegistry,
    MetaPolicyHint,
    MetaPriorUpdate,
    MetaSignalPack,
    PacketBase,
    PacketType,
    PolicyOutcomePack,
    WarmStartProfile,
)


@dataclass
class AggregationConfig:
    metrics_trim_alpha: float = 0.1
    vote_min_sites: int = 3
    metrics_aggregation: str = "trimmed_mean"  # trimmed_mean | mean | fedavg | fedprox | scarcityw
    dp_noise_sigma: float = 0.0
    dp_epsilon: float = 0.0
    dp_delta: float = 0.0
    dp_sensitivity: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AggregationConfig":
        return cls(
            metrics_trim_alpha=float(data.get("metrics_trim_alpha", 0.1)),
            vote_min_sites=int(data.get("vote_min_sites", 3)),
            metrics_aggregation=str(data.get("metrics_aggregation", "trimmed_mean")),
            dp_noise_sigma=float(data.get("dp_noise_sigma", 0.0)),
            dp_epsilon=float(data.get("dp_epsilon", 0.0)),
            dp_delta=float(data.get("dp_delta", 0.0)),
            dp_sensitivity=float(data.get("dp_sensitivity", 1.0)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics_trim_alpha": self.metrics_trim_alpha,
            "vote_min_sites": self.vote_min_sites,
            "metrics_aggregation": self.metrics_aggregation,
            "dp_noise_sigma": self.dp_noise_sigma,
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "dp_sensitivity": self.dp_sensitivity,
        }


@dataclass
class AggregationResult:
    prior_update: Optional[MetaPriorUpdate] = None
    warm_start: Optional[WarmStartProfile] = None
    policy_hint: Optional[MetaPolicyHint] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def has_output(self) -> bool:
        return any([self.prior_update, self.warm_start, self.policy_hint])


class FMIAggregator:
    """
    Merges validated packets into meta priors.
    """

    def __init__(
        self,
        config: AggregationConfig,
        registry: FMIContractRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or FMIContractRegistry()
        self._last_prior: Optional[MetaPriorUpdate] = None

    def aggregate(self, cohort: str, packets: Sequence[PacketBase]) -> AggregationResult:
        msp_packets = [p for p in packets if p.type == PacketType.MSP]
        pop_packets = [p for p in packets if p.type == PacketType.POP]
        ccs_packets = [p for p in packets if p.type == PacketType.CCS]

        telemetry = {
            "packets": len(packets),
            "msp": len(msp_packets),
            "pop": len(pop_packets),
            "ccs": len(ccs_packets),
        }

        prior_update = self._build_prior(cohort, msp_packets, ccs_packets)
        warm_start = self._build_warm_start(cohort, prior_update, msp_packets)
        policy_hint = self._build_policy_hint(cohort, pop_packets, msp_packets)

        result = AggregationResult(
            prior_update=prior_update,
            warm_start=warm_start,
            policy_hint=policy_hint,
            telemetry=telemetry,
        )
        if prior_update:
            self._last_prior = prior_update
        return result

    # ------------------------------------------------------------------ #
    # Prior aggregation
    # ------------------------------------------------------------------ #

    def _build_prior(
        self,
        cohort: str,
        msps: Sequence[MetaSignalPack],
        ccs_packets: Sequence[ConceptCausalSummary],
    ) -> Optional[MetaPriorUpdate]:
        if not msps and not ccs_packets:
            return None

        controller_prior = self._aggregate_field(msps, "controller")
        evaluator_prior = self._aggregate_field(msps, "evaluator")
        operator_prior = self._aggregate_field(msps, "operators")

        metrics = self._aggregate_field(msps, "metrics")
        latency = metrics.get("latency_ms", 0.0)
        accept_rate = metrics.get("accept_rate", 0.0)

        contexts = self._aggregate_contexts(ccs_packets)
        confidence = self._blend_confidence(msps)

        prior = MetaPriorUpdate(
            rev=(self._last_prior.rev + 1) if self._last_prior else 1,
            prior={
                "controller": controller_prior,
                "evaluator": evaluator_prior,
                "operators": operator_prior,
                "metrics": metrics,
            },
            contexts=contexts,
            confidence=confidence,
            cohorts=[cohort],
        )

        # Attach coarse telemetry hints for downstream consumption.
        prior.prior["signals"] = {
            "latency_ms": latency,
            "accept_rate": accept_rate,
        }

        return prior

    def _aggregate_field(
        self,
        packets: Sequence[MetaSignalPack],
        attribute: str,
    ) -> Dict[str, float]:
        values: Dict[str, List[float]] = defaultdict(list)
        weights: Dict[str, List[float]] = defaultdict(list)
        for packet in packets:
            payload = getattr(packet, attribute, None) or {}
            if not isinstance(payload, Mapping):
                continue
            weight = self._packet_weight(packet)
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    values[key].append(float(value))
                    weights[key].append(weight)
        aggregated: Dict[str, float] = {}
        for key, series in values.items():
            method = self.config.metrics_aggregation.lower()
            if method in {"mean", "fedavg", "fedprox"}:
                agg_value = float(np.mean(series))
            elif method in {"scarcityw", "weighted"}:
                agg_value = self._weighted_mean(series, weights.get(key))
            else:
                agg_value = float(self._trimmed_mean(series, self.config.metrics_trim_alpha))
            aggregated[key] = self._apply_dp(agg_value)
        return aggregated

    def _aggregate_contexts(
        self,
        packets: Sequence[ConceptCausalSummary],
    ) -> List[Dict[str, Any]]:
        if not packets:
            return []

        vector_store: Dict[str, List[float]] = defaultdict(list)
        regime_store: Dict[str, str] = {}

        for packet in packets:
            regime = packet.provenance.get("regime_tag") if isinstance(packet.provenance, Mapping) else None
            regime = regime or packet.domain_id
            regime_store[regime] = regime
            concepts = packet.concepts or []
            # Build deterministic vectors sorted by concept id.
            concepts_sorted = sorted(
                [c for c in concepts if isinstance(c, Mapping) and "score" in c and "id" in c],
                key=lambda item: str(item["id"]),
            )
            vector = [float(c["score"]) for c in concepts_sorted[:16]]
            if vector:
                vector_store[regime].append(vector)

        contexts: List[Dict[str, Any]] = []
        for regime, vectors in vector_store.items():
            stacked = self._median_vector(vectors)
            contexts.append({"regime": regime_store.get(regime, regime), "vector": stacked})
        return contexts

    def _blend_confidence(self, packets: Sequence[MetaSignalPack]) -> float:
        confidences: List[float] = []
        for packet in packets:
            evidence = packet.evidence or {}
            conf = evidence.get("confidence")
            if conf is None:
                continue
            try:
                conf_float = float(conf)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            if conf_float > 0:
                confidences.append(conf_float)

        if not confidences:
            return 0.0

        reciprocal_sum = sum(1.0 / value for value in confidences if value > 0)
        if reciprocal_sum == 0.0:
            return float(np.clip(np.mean(confidences), 0.0, 1.0))
        harmonic = len(confidences) / reciprocal_sum
        return float(np.clip(harmonic, 0.0, 1.0))

    def _packet_weight(self, packet: MetaSignalPack) -> float:
        weight = 1.0
        if isinstance(packet.provenance, Mapping):
            trust = packet.provenance.get("trust") or packet.provenance.get("trust_score")
            if trust is not None:
                try:
                    weight = float(trust)
                except (TypeError, ValueError):
                    pass
        metrics = packet.metrics if isinstance(packet.metrics, Mapping) else {}
        for key in ("gain_p50", "accept_rate", "gain_p90"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                weight = max(weight, float(value))
        return max(weight, 1e-6)

    def _resolve_sigma(self) -> float:
        if self.config.dp_noise_sigma > 0:
            return self.config.dp_noise_sigma
        if self.config.dp_epsilon > 0 and self.config.dp_delta > 0:
            c = math.sqrt(2 * math.log(1.25 / self.config.dp_delta))
            return self.config.dp_sensitivity * c / self.config.dp_epsilon
        return 0.0

    def _apply_dp(self, value: float) -> float:
        sigma = self._resolve_sigma()
        if sigma <= 0:
            return value
        clipped = float(np.clip(value, -self.config.dp_sensitivity, self.config.dp_sensitivity))
        return clipped + float(np.random.normal(0.0, sigma))

    @staticmethod
    def _weighted_mean(series: Sequence[float], weights: Optional[Sequence[float]]) -> float:
        if not series:
            return 0.0
        if not weights:
            return float(np.mean(series))
        w = np.asarray(weights, dtype=np.float32)
        total = float(np.sum(w))
        if total <= 0:
            return float(np.mean(series))
        return float(np.sum(np.asarray(series, dtype=np.float32) * (w / total)))

    # ------------------------------------------------------------------ #
    # Warm start
    # ------------------------------------------------------------------ #

    def _build_warm_start(
        self,
        cohort: str,
        prior_update: Optional[MetaPriorUpdate],
        msps: Sequence[MetaSignalPack],
    ) -> Optional[WarmStartProfile]:
        if not prior_update:
            return None

        cohort_parts = cohort.split("/")
        fallback_profile = cohort_parts[1] if len(cohort_parts) > 1 else cohort_parts[0]
        profile_class = msps[0].profile_class if msps else fallback_profile
        controller = prior_update.prior.get("controller", {})
        evaluator = prior_update.prior.get("evaluator", {})

        context_selector: Dict[str, Any] = {}
        if prior_update.contexts:
            context_selector["nearest_regime"] = prior_update.contexts[0]["regime"]

        return WarmStartProfile(
            profile_class=profile_class,
            init={"controller": controller, "evaluator": evaluator},
            context_selector=context_selector,
        )

    # ------------------------------------------------------------------ #
    # Policy hints
    # ------------------------------------------------------------------ #

    def _build_policy_hint(
        self,
        cohort: str,
        pops: Sequence[PolicyOutcomePack],
        msps: Sequence[MetaSignalPack],
    ) -> Optional[MetaPolicyHint]:
        bundle, confidence = self._vote_policy_bundle(pops)
        if bundle is None:
            return None

        bounds = self._hint_bounds(bundle)
        reason = self._hint_reason(msps)

        hint_id = f"FMI-{abs(hash((cohort, bundle_tuple(bundle)))) % 10000:04d}"
        return MetaPolicyHint(
            hint_id=hint_id,
            bundle=bundle,
            bounds=bounds,
            reason=reason,
            confidence=confidence,
        )

    def _vote_policy_bundle(
        self,
        pops: Sequence[PolicyOutcomePack],
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        if not pops:
            return None, 0.0

        vote_book: Dict[str, Dict[str, Any]] = {}
        vote_support: Dict[str, set[str]] = defaultdict(set)
        vote_gain: Dict[str, List[float]] = defaultdict(list)

        for pop in pops:
            site_id = "unknown"
            if isinstance(pop.provenance, Mapping):
                site_id = str(pop.provenance.get("site_id", site_id))

            delta = self._pop_gain(pop)
            key = json_safe_key(pop.bundle)

            vote_book[key] = pop.bundle
            vote_support[key].add(site_id)
            vote_gain[key].append(delta)

        best_key = None
        best_gain = float("-inf")
        for key, support in vote_support.items():
            if len(support) < self.config.vote_min_sites:
                continue
            gain = float(np.mean(vote_gain[key])) if vote_gain[key] else 0.0
            if gain > best_gain:
                best_gain = gain
                best_key = key

        if best_key is None:
            return None, 0.0

        confidence = float(np.clip(best_gain, 0.0, 1.0))
        return vote_book[best_key], confidence

    @staticmethod
    def _pop_gain(pop: PolicyOutcomePack) -> float:
        before = pop.before or {}
        after = pop.after or {}
        gain_before = float(before.get("accept_rate", 0.0))
        gain_after = float(after.get("accept_rate", gain_before))
        return gain_after - gain_before

    def _hint_bounds(self, bundle: Mapping[str, Any]) -> Dict[str, Any]:
        bounds: Dict[str, Any] = {}
        for block, params in bundle.items():
            if not isinstance(params, Mapping):
                continue
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    span = max(0.1 * abs(value), 0.1)
                    bounds.setdefault(block, {})[key] = [
                        float(value - span),
                        float(value + span),
                    ]
        return bounds

    def _hint_reason(self, msps: Sequence[MetaSignalPack]) -> str:
        if not msps:
            return "policy voted by POP packets"

        metrics = self._aggregate_field(msps, "metrics")
        latency = metrics.get("latency_ms", 0.0)
        accept_rate = metrics.get("accept_rate", 0.0)

        if latency > 100 and accept_rate < 0.1:
            return "latency high, accept rate low; safe rollback suggested"
        if accept_rate > 0.2:
            return "cohort improving; reinforce high-performing bundle"
        return "policy bundle selected by cohort vote"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trimmed_mean(values: Sequence[float], alpha: float) -> float:
        if not values:
            return 0.0
        if len(values) < 3:
            return float(np.mean(values))
        sorted_values = sorted(values)
        trim = int(len(values) * alpha)
        if trim == 0:
            trimmed = sorted_values
        else:
            trimmed = sorted_values[trim:-trim] or sorted_values
        return float(np.mean(trimmed))

    @staticmethod
    def _median_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
        if not vectors:
            return []
        max_len = max(len(vec) for vec in vectors)
        padded = [list(vec) + [0.0] * (max_len - len(vec)) for vec in vectors]
        median_vector = [float(median([vec[i] for vec in padded])) for i in range(max_len)]
        return median_vector


def json_safe_key(payload: Mapping[str, Any]) -> str:
    """
    Produce a stable string key for dictionaries with non-serialisable values.
    """

    def normalise(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): normalise(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        if isinstance(value, (int, float, str, type(None))):
            return value
        return str(value)

    normalised = normalise(payload)
    return repr(normalised)


def bundle_tuple(bundle: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    flattened: List[Tuple[str, Any]] = []
    for block, params in sorted(bundle.items(), key=lambda item: item[0]):
        if isinstance(params, Mapping):
            for key, value in sorted(params.items(), key=lambda item: item[0]):
                flattened.append((f"{block}.{key}", value))
        else:
            flattened.append((block, params))
    return tuple(flattened)


__all__ = [
    "AggregationConfig",
    "AggregationResult",
    "FMIAggregator",
]



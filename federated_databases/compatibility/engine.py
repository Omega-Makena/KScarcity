"""Production-oriented compatibility scoring and basket formation.

Assumptions:
- Heterogeneity is default.
- Compatibility is computed per node relative to a project + dataset.
- Baskets are formed dynamically from compatibility and pairwise similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from k_collab.common.versioned_store import VersionedJSONStore
from federated_databases.contracts.models import CLASSIFICATION_ORDER


WEIGHTS = {
    "schema": 0.30,
    "temporal": 0.15,
    "statistical": 0.20,
    "quality": 0.15,
    "policy": 0.10,
    "operational": 0.10,
}

FULL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50
PAIRWISE_THRESHOLD = 0.65


@dataclass
class NodeCompatibility:
    node_id: str
    connector_id: str
    score: float
    tier: str
    components: Dict[str, float]
    reasoning: Dict[str, Any]


class CompatibilityEngine:
    """Computes compatibility scores, pairwise similarity, and baskets."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "compatibility_versions.jsonl", kind="compatibility")

    def analyze(
        self,
        *,
        project: Dict[str, Any],
        dataset_id: str,
        operation: str,
        connectors: List[Dict[str, Any]],
        contract: Dict[str, Any] | None,
        canonical_mapping: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        node_scores: List[NodeCompatibility] = []

        for connector in connectors:
            if dataset_id not in set(connector.get("dataset_ids", [])):
                continue
            node_scores.append(
                self._score_node(
                    connector=connector,
                    project=project,
                    contract=contract or {},
                    canonical_mapping=canonical_mapping or {},
                    operation=operation,
                )
            )

        node_score_dict = {n.node_id: n for n in node_scores}

        pairwise = self._pairwise(node_scores, canonical_mapping or {})
        baskets, excluded_nodes = self._form_baskets(project=project, node_scores=node_scores, pairwise=pairwise)

        report = {
            "dataset_id": dataset_id,
            "operation": operation,
            "node_scores": [
                {
                    "node_id": n.node_id,
                    "connector_id": n.connector_id,
                    "score": round(n.score, 4),
                    "tier": n.tier,
                    "components": {k: round(v, 4) for k, v in n.components.items()},
                    "reasoning": n.reasoning,
                }
                for n in sorted(node_scores, key=lambda x: x.node_id)
            ],
            "pairwise": pairwise,
            "baskets": baskets,
            "excluded_nodes": excluded_nodes,
            "summary": {
                "compatible_nodes": len([n for n in node_scores if n.score >= PARTIAL_THRESHOLD]),
                "excluded_nodes": len(excluded_nodes),
                "full_baskets": len([b for b in baskets if b.get("tier") == "full"]),
                "partial_baskets": len([b for b in baskets if b.get("tier") == "partial"]),
                "mean_score": round(
                    sum(n.score for n in node_scores) / max(1, len(node_scores)),
                    4,
                ),
            },
        }

        rec = self._store.save(report, actor="system", message=f"compatibility:{project.get('project_id','unknown')}:{dataset_id}")
        report["version_id"] = rec.version_id
        return report

    def latest(self) -> Dict[str, Any] | None:
        rec = self._store.latest()
        return rec.payload if rec else None

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [x.__dict__ for x in self._store.list(limit)]

    def _score_node(
        self,
        *,
        connector: Dict[str, Any],
        project: Dict[str, Any],
        contract: Dict[str, Any],
        canonical_mapping: Dict[str, Any],
        operation: str,
    ) -> NodeCompatibility:
        node_id = str(connector.get("node_id", ""))
        connector_id = str(connector.get("connector_id", ""))
        options = dict(connector.get("options", {}))
        governance = dict(project.get("governance", {}))

        schema_score, schema_reason = self._schema_compatibility(contract, canonical_mapping, governance)
        temporal_score, temporal_reason = self._temporal_compatibility(options, canonical_mapping, governance)
        statistical_score, statistical_reason = self._statistical_compatibility(options, governance)
        quality_score, quality_reason = self._quality_compatibility(options, canonical_mapping, governance)
        policy_score, policy_reason = self._policy_compatibility(options, contract, governance, operation)
        operational_score, operational_reason = self._operational_capability(options, connector, operation, governance)

        components = {
            "schema": schema_score,
            "temporal": temporal_score,
            "statistical": statistical_score,
            "quality": quality_score,
            "policy": policy_score,
            "operational": operational_score,
        }

        total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
        total = max(0.0, min(1.0, float(total)))

        tier = "excluded"
        if total >= FULL_THRESHOLD:
            tier = "full"
        elif total >= PARTIAL_THRESHOLD:
            tier = "partial"

        reasoning = {
            "schema": schema_reason,
            "temporal": temporal_reason,
            "statistical": statistical_reason,
            "quality": quality_reason,
            "policy": policy_reason,
            "operational": operational_reason,
        }

        return NodeCompatibility(
            node_id=node_id,
            connector_id=connector_id,
            score=total,
            tier=tier,
            components=components,
            reasoning=reasoning,
        )

    def _schema_compatibility(
        self,
        contract: Dict[str, Any],
        canonical_mapping: Dict[str, Any],
        governance: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        schema = dict(contract.get("schema", {}))
        fields = list(canonical_mapping.get("fields", []))
        required = set(governance.get("required_fields", []))

        mapped = 0
        type_ok = 0
        required_ok = 0

        for f in fields:
            canonical = str(f.get("canonical_name", "")).strip()
            local = str(f.get("local_name", "")).strip()
            expected_dtype = str(f.get("dtype", "text")).strip().lower()
            if not canonical or not local:
                continue
            if local in schema:
                mapped += 1
                actual_type = str(schema.get(local, "text")).lower()
                if self._types_compatible(expected_dtype, actual_type):
                    type_ok += 1
            if canonical in required and local in schema:
                required_ok += 1

        total_fields = max(1, len(fields))
        required_total = max(1, len(required)) if required else 1

        mapped_ratio = mapped / total_fields
        type_ratio = type_ok / max(1, mapped)
        required_ratio = (required_ok / required_total) if required else 1.0

        score = (mapped_ratio + type_ratio + required_ratio) / 3.0
        return score, {
            "mapped_ratio": round(mapped_ratio, 4),
            "type_ratio": round(type_ratio, 4),
            "required_ratio": round(required_ratio, 4),
        }

    def _temporal_compatibility(
        self,
        options: Dict[str, Any],
        canonical_mapping: Dict[str, Any],
        governance: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        node_grain = str(options.get("time_grain", "day")).lower()
        expected_grain = str(governance.get("time_grain", "day")).lower()
        timestamp_completeness = float((options.get("quality") or {}).get("timestamp_completeness", 0.8))

        grain_score = 1.0 if node_grain == expected_grain else 0.0
        if grain_score == 0.0 and self._grain_convertible(node_grain, expected_grain):
            grain_score = 0.8

        if not any(str(f.get("canonical_name", "")).strip().lower() == "timestamp" for f in canonical_mapping.get("fields", [])):
            timestamp_completeness = min(timestamp_completeness, 0.5)

        score = (grain_score + max(0.0, min(1.0, timestamp_completeness))) / 2.0
        return score, {
            "node_grain": node_grain,
            "expected_grain": expected_grain,
            "grain_score": round(grain_score, 4),
            "timestamp_completeness": round(timestamp_completeness, 4),
        }

    def _statistical_compatibility(self, options: Dict[str, Any], governance: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        profile = dict(options.get("profile", {}))
        ref_profile = dict(governance.get("reference_profile", {}))
        required_features = set(governance.get("required_features", []))

        feature_overlap = 1.0
        overlap_count = 0
        if required_features:
            available = set(profile.get("features", []) or profile.get("feature_stats", {}).keys())
            overlap_count = len(required_features & available)
            feature_overlap = overlap_count / max(1, len(required_features))

        dist_similarity = 0.5
        compared = 0
        if ref_profile and profile.get("feature_stats"):
            sims = []
            for feat, ref_stats in ref_profile.items():
                node_stats = (profile.get("feature_stats") or {}).get(feat)
                if not node_stats:
                    continue
                compared += 1
                ref_mean = float(ref_stats.get("mean", 0.0))
                ref_std = float(ref_stats.get("std", 1.0) or 1.0)
                node_mean = float(node_stats.get("mean", 0.0))
                node_std = float(node_stats.get("std", 1.0) or 1.0)
                diff = abs(ref_mean - node_mean)
                denom = max(1e-6, abs(ref_std) + abs(node_std))
                sim = max(0.0, 1.0 - (diff / denom))
                sims.append(sim)
            if sims:
                dist_similarity = sum(sims) / len(sims)

        iid_score = float(profile.get("iid_score", 0.5))
        iid_score = max(0.0, min(1.0, iid_score))

        score = (dist_similarity + feature_overlap + iid_score) / 3.0
        return score, {
            "distribution_similarity": round(dist_similarity, 4),
            "feature_overlap_ratio": round(feature_overlap, 4),
            "iid_score": round(iid_score, 4),
            "features_compared": compared,
            "feature_overlap_count": overlap_count,
        }

    def _quality_compatibility(
        self,
        options: Dict[str, Any],
        canonical_mapping: Dict[str, Any],
        governance: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        quality_opts = dict(options.get("quality", {}))
        mapping_quality = dict(canonical_mapping.get("quality", {}))

        completeness = float(quality_opts.get("completeness", mapping_quality.get("completeness_min", 0.8)))
        missing_rate = float(quality_opts.get("missing_rate", 1.0 - completeness))
        freshness_hours = float(quality_opts.get("freshness_hours", mapping_quality.get("freshness_sla_hours", 24)))
        max_freshness = float(governance.get("max_freshness_hours", mapping_quality.get("freshness_sla_hours", 48) or 48))

        freshness_score = 1.0
        if freshness_hours > 0 and max_freshness > 0:
            freshness_score = max(0.0, min(1.0, max_freshness / freshness_hours))

        score = (completeness + (1.0 - max(0.0, min(1.0, missing_rate))) + freshness_score) / 3.0
        return score, {
            "completeness": round(completeness, 4),
            "missing_rate": round(missing_rate, 4),
            "freshness_hours": round(freshness_hours, 4),
            "freshness_score": round(freshness_score, 4),
        }

    def _policy_compatibility(
        self,
        options: Dict[str, Any],
        contract: Dict[str, Any],
        governance: Dict[str, Any],
        operation: str,
    ) -> Tuple[float, Dict[str, Any]]:
        required_clearance = str(governance.get("required_clearance", "INTERNAL")).upper()
        node_clearance = str(options.get("max_classification", contract.get("classification", "INTERNAL"))).upper()

        clearance_ok = 1.0 if CLASSIFICATION_ORDER.get(node_clearance, -1) >= CLASSIFICATION_ORDER.get(required_clearance, -1) else 0.0

        contract_ops = {str(x).lower() for x in contract.get("allowed_operations", [])}
        adapter_ops = {str(x).lower() for x in options.get("supported_operations", ["aggregate", "time_bucket"])}
        overlap = contract_ops & adapter_ops
        op_required = str(operation or "aggregate").lower()
        operation_ok = 1.0 if op_required in overlap else 0.0
        overlap_ratio = len(overlap) / max(1, len(contract_ops | adapter_ops))

        score = (clearance_ok + operation_ok + overlap_ratio) / 3.0
        return score, {
            "required_clearance": required_clearance,
            "node_clearance": node_clearance,
            "clearance_ok": bool(clearance_ok),
            "operation_ok": bool(operation_ok),
            "operation_overlap_ratio": round(overlap_ratio, 4),
        }

    def _operational_capability(
        self,
        options: Dict[str, Any],
        connector: Dict[str, Any],
        operation: str,
        governance: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        source_type = str(connector.get("source_type", "")).lower()
        defaults = {
            "sqlite": ["aggregation", "group_by", "filters"],
            "postgres": ["aggregation", "group_by", "filters", "time_bucket"],
        }
        supported_pushdowns = {str(x).lower() for x in options.get("supported_pushdowns", defaults.get(source_type, ["aggregation"]))}
        required_pushdowns = {str(x).lower() for x in governance.get("required_pushdowns", ["aggregation", "group_by", "filters"])}

        pushdown_ratio = len(required_pushdowns & supported_pushdowns) / max(1, len(required_pushdowns))
        supports_agg = bool(options.get("supports_aggregation", True))
        operation_supported = str(operation).lower() in {str(x).lower() for x in options.get("supported_operations", ["aggregate", "time_bucket"])}

        score = (pushdown_ratio + (1.0 if supports_agg else 0.0) + (1.0 if operation_supported else 0.0)) / 3.0
        return score, {
            "supported_pushdowns": sorted(supported_pushdowns),
            "required_pushdowns": sorted(required_pushdowns),
            "pushdown_ratio": round(pushdown_ratio, 4),
            "supports_aggregation": supports_agg,
            "operation_supported": operation_supported,
        }

    def _pairwise(self, nodes: List[NodeCompatibility], canonical_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        node_count = len(nodes)
        canonical_fields = {str(x.get("canonical_name", "")) for x in canonical_mapping.get("fields", []) if str(x.get("canonical_name", ""))}

        for i in range(node_count):
            for j in range(i + 1, node_count):
                a = nodes[i]
                b = nodes[j]
                diffs = []
                for key in WEIGHTS.keys():
                    diffs.append(abs(float(a.components.get(key, 0.0)) - float(b.components.get(key, 0.0))))
                component_similarity = 1.0 - (sum(diffs) / max(1, len(diffs)))

                # If mappings are global canonical, field Jaccard is 1; keep generic hook.
                field_similarity = 1.0 if canonical_fields else 0.7

                score = max(0.0, min(1.0, 0.6 * component_similarity + 0.4 * field_similarity))
                output.append(
                    {
                        "node_a": a.node_id,
                        "node_b": b.node_id,
                        "score": round(score, 4),
                        "component_similarity": round(component_similarity, 4),
                        "field_similarity": round(field_similarity, 4),
                    }
                )
        return output

    def _form_baskets(
        self,
        *,
        project: Dict[str, Any],
        node_scores: List[NodeCompatibility],
        pairwise: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        by_node = {n.node_id: n for n in node_scores}
        compatible_nodes = [n.node_id for n in node_scores if n.score >= PARTIAL_THRESHOLD]

        graph: Dict[str, Set[str]] = {n: set() for n in compatible_nodes}
        for p in pairwise:
            if p["score"] >= PAIRWISE_THRESHOLD:
                a = p["node_a"]
                b = p["node_b"]
                if a in graph and b in graph:
                    graph[a].add(b)
                    graph[b].add(a)

        components: List[List[str]] = []
        visited: Set[str] = set()
        for node in compatible_nodes:
            if node in visited:
                continue
            stack = [node]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nxt in graph.get(cur, set()):
                    if nxt not in visited:
                        stack.append(nxt)
            components.append(sorted(comp))

        # Include isolated nodes as one-member components.
        for node in compatible_nodes:
            if not any(node in c for c in components):
                components.append([node])

        overrides = list((project.get("governance") or {}).get("basket_overrides", []))
        mode = str((project.get("governance") or {}).get("basket_override_mode", "merge")).lower()
        manual_baskets: List[Dict[str, Any]] = []
        if overrides:
            for idx, override in enumerate(overrides, start=1):
                nodes = sorted({str(x).strip() for x in override.get("nodes", []) if str(x).strip() and str(x).strip() in by_node})
                if not nodes:
                    continue
                manual_baskets.append(
                    self._build_basket(
                        basket_id=str(override.get("basket_id") or f"manual_{idx}"),
                        node_ids=nodes,
                        by_node=by_node,
                        reason="manual_override",
                    )
                )

        baskets: List[Dict[str, Any]] = []
        if mode == "replace" and manual_baskets:
            baskets = manual_baskets
        else:
            for idx, comp in enumerate(sorted(components), start=1):
                baskets.append(
                    self._build_basket(
                        basket_id=f"auto_{idx}",
                        node_ids=comp,
                        by_node=by_node,
                        reason="auto_cluster",
                    )
                )
            if manual_baskets:
                manual_nodes = {n for b in manual_baskets for n in b.get("members", [])}
                baskets = [b for b in baskets if not (set(b.get("members", [])) & manual_nodes)] + manual_baskets

        excluded_nodes = []
        for node in node_scores:
            if node.score < PARTIAL_THRESHOLD:
                excluded_nodes.append(
                    {
                        "node_id": node.node_id,
                        "connector_id": node.connector_id,
                        "score": round(node.score, 4),
                        "reason": "compatibility_below_threshold",
                        "components": {k: round(v, 4) for k, v in node.components.items()},
                    }
                )

        baskets.sort(key=lambda b: b.get("basket_id", ""))
        return baskets, excluded_nodes

    def _build_basket(self, *, basket_id: str, node_ids: List[str], by_node: Dict[str, NodeCompatibility], reason: str) -> Dict[str, Any]:
        full_members = [n for n in node_ids if by_node[n].score >= FULL_THRESHOLD]
        partial_members = [n for n in node_ids if PARTIAL_THRESHOLD <= by_node[n].score < FULL_THRESHOLD]
        tier = "full" if partial_members == [] else "partial"
        avg_score = sum(by_node[n].score for n in node_ids) / max(1, len(node_ids))
        return {
            "basket_id": basket_id,
            "tier": tier,
            "members": sorted(node_ids),
            "full_members": sorted(full_members),
            "partial_members": sorted(partial_members),
            "average_score": round(avg_score, 4),
            "formation_reason": reason,
        }

    def _types_compatible(self, expected: str, actual: str) -> bool:
        expected = expected.lower()
        actual = actual.lower()
        if expected == actual:
            return True
        numeric = {"real", "float", "double", "numeric", "integer", "int", "bigint", "smallint"}
        texty = {"text", "varchar", "char", "string"}
        if expected in numeric and actual in numeric:
            return True
        if expected in texty and actual in texty:
            return True
        return False

    def _grain_convertible(self, src: str, dst: str) -> bool:
        conversions = {
            ("minute", "hour"),
            ("hour", "day"),
            ("day", "week"),
            ("day", "month"),
            ("hour", "week"),
            ("hour", "month"),
            ("minute", "day"),
        }
        return src == dst or (src, dst) in conversions or (dst, src) in conversions

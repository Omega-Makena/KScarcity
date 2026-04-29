"""
Stage 8 — Integration.

8.1  EventBus wiring audit — static analysis of publish/subscribe pairs,
     then live instrumentation if EventBus is accessible.
"""
from __future__ import annotations

import ast
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import fail_result, make_result, save_artifact, skip_result

logger = logging.getLogger(__name__)

SCARCITY_ROOT = PROJECT_ROOT / "scarcity"

# Topics documented in ARCHITECTURE_DIAGRAMS.md §5.2 that should have both
# a publisher and a subscriber.
EXPECTED_TOPIC_PAIRS = [
    "data_window",
    "resource_profile",
    "meta_policy_update",
    "meta_prior_update",
    "meta_rollback_active",
    "processing_metrics",
    "engine.insight",
    "inference.path_pack",
    "federation.policy_pack",
    "federation.path_pack",
    "federation.edge_delta",
    "federation.causal_pack",
    "federation.health",
    "federation_update",
    "fmi.meta_prior_update",
    "fmi.meta_policy_hint",
    "fmi.warm_start_profile",
    "fmi.telemetry",
]


# ---------------------------------------------------------------------------
# Static audit: grep source for publish/subscribe calls
# ---------------------------------------------------------------------------

def _extract_string_arg(call_node: ast.Call) -> Optional[str]:
    """Try to extract the first string argument of a call node."""
    if call_node.args:
        arg = call_node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    if call_node.keywords:
        for kw in call_node.keywords:
            if kw.arg in ("topic", "event", "channel", "name"):
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value
    return None


def _audit_file(py_file: Path) -> Dict[str, List[str]]:
    """Return {"publish": [...topics...], "subscribe": [...topics...]} for one file."""
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return {"publish": [], "subscribe": []}

    published, subscribed = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        method = None
        if isinstance(func, ast.Attribute):
            method = func.attr
        elif isinstance(func, ast.Name):
            method = func.id

        if method in ("publish",):
            t = _extract_string_arg(node)
            if t:
                published.append(t)
        elif method in ("subscribe", "on", "listen"):
            t = _extract_string_arg(node)
            if t:
                subscribed.append(t)

    return {"publish": published, "subscribe": subscribed}


def _static_eventbus_audit() -> Dict[str, Any]:
    """Scan all scarcity .py files and build topic → publishers/subscribers maps."""
    publishers: Dict[str, List[str]] = defaultdict(list)
    subscribers: Dict[str, List[str]] = defaultdict(list)

    py_files = list(SCARCITY_ROOT.rglob("*.py"))
    for py_file in py_files:
        rel = str(py_file.relative_to(PROJECT_ROOT))
        result = _audit_file(py_file)
        for topic in result["publish"]:
            publishers[topic].append(rel)
        for topic in result["subscribe"]:
            subscribers[topic].append(rel)

    all_topics = sorted(set(publishers) | set(subscribers))
    orphan_publishers = []  # published but no subscriber
    orphan_subscribers = []  # subscribed but no publisher

    for topic in all_topics:
        if topic in publishers and topic not in subscribers:
            orphan_publishers.append(topic)
        if topic in subscribers and topic not in publishers:
            orphan_subscribers.append(topic)

    # Check expected topics
    expected_coverage = {}
    for topic in EXPECTED_TOPIC_PAIRS:
        expected_coverage[topic] = {
            "has_publisher": topic in publishers,
            "has_subscriber": topic in subscribers,
            "publishers": publishers.get(topic, []),
            "subscribers": subscribers.get(topic, []),
        }

    return {
        "n_files_scanned": len(py_files),
        "n_topics_found": len(all_topics),
        "orphan_publishers": orphan_publishers,
        "orphan_subscribers": orphan_subscribers,
        "expected_coverage": expected_coverage,
        "n_expected_fully_covered": sum(
            1 for t in expected_coverage.values()
            if t["has_publisher"] and t["has_subscriber"]
        ),
    }


# ---------------------------------------------------------------------------
# Live instrumentation: monkey-patch EventBus if accessible
# ---------------------------------------------------------------------------

def _try_live_eventbus_audit() -> Optional[Dict[str, Any]]:
    try:
        from scarcity.runtime.bus import EventBus
    except ImportError:
        try:
            from scarcity.engine.event_bus import EventBus  # noqa: F401
        except ImportError:
            return None

    try:
        log: List[Dict] = []
        original_publish = EventBus.publish

        def patched_publish(self_or_topic, topic_or_payload=None, payload=None):
            entry = {"ts": time.time()}
            if isinstance(self_or_topic, str):
                entry["topic"] = self_or_topic
                entry["payload_type"] = type(topic_or_payload).__name__
            else:
                entry["topic"] = topic_or_payload
                entry["payload_type"] = type(payload).__name__
            log.append(entry)
            return original_publish(self_or_topic, topic_or_payload, payload)

        EventBus.publish = patched_publish

        # Run a few windows through the engine
        from scripts.stages.utils import build_hub, make_structured_data, rows_to_yearly, stream_rows
        rows = make_structured_data(n_obs=10, seed=42)
        yearly = rows_to_yearly(rows)
        hub = build_hub("KEN")
        stream_rows(hub, "KEN", {yr: yearly[yr] for yr in list(sorted(yearly.keys()))[:5]})

        EventBus.publish = original_publish

        topics_seen = list({e["topic"] for e in log})
        topic_counts = {}
        for e in log:
            topic_counts[e["topic"]] = topic_counts.get(e["topic"], 0) + 1

        return {
            "mode": "live_instrumented",
            "n_events_captured": len(log),
            "topics_seen": sorted(topics_seen),
            "topic_counts": dict(sorted(topic_counts.items())),
        }
    except Exception as e:
        return {"mode": "live_failed", "error": str(e)}


# Topics that require distributed runtime or external publishers/consumers.
# Excluded from the adjusted coverage denominator.
NOT_APPLICABLE_IN_BENCHMARK = {
    # Publisher is external data pipeline; benchmark uses process_row() directly
    "data_window",
    # HierarchicalFederation distributed protocol — not wired in FederationHub path
    "federation.path_pack",
    "federation.edge_delta",
    "federation.causal_pack",
    "federation.health",
    # FMI emitter requires HierarchicalFederation runtime
    "fmi.meta_prior_update",
    "fmi.meta_policy_hint",
    "fmi.warm_start_profile",
    "fmi.telemetry",
    # Published by exporter/client_agent; consumed by external services (dashboard, coordinator)
    "inference.path_pack",
    "federation_update",
}


def run_stage_8_1() -> Dict[str, Any]:
    start = time.time()

    try:
        static_result = _static_eventbus_audit()
    except Exception as e:
        return fail_result("8.1", "eventbus_audit", "EventBus topics audited", str(e),
                           time.time() - start)

    live_result = _try_live_eventbus_audit()

    n_fully_covered = static_result["n_expected_fully_covered"]
    n_expected = len(EXPECTED_TOPIC_PAIRS)

    # Adjusted coverage: exclude topics not applicable in benchmark mode
    applicable = [t for t in EXPECTED_TOPIC_PAIRS if t not in NOT_APPLICABLE_IN_BENCHMARK]
    cov = static_result.get("expected_coverage", {})
    n_applicable_covered = sum(
        1 for t in applicable
        if cov.get(t, {}).get("has_publisher") and cov.get(t, {}).get("has_subscriber")
    )
    n_applicable = len(applicable)
    adjusted_coverage_rate = n_applicable_covered / max(n_applicable, 1)

    # Categorize the not-applicable topics by reason
    not_applicable_detail = {}
    for t in EXPECTED_TOPIC_PAIRS:
        if t in NOT_APPLICABLE_IN_BENCHMARK:
            has_pub = bool(cov.get(t, {}).get("publishers"))
            has_sub = bool(cov.get(t, {}).get("subscribers"))
            if not has_pub and not has_sub:
                reason = "distributed_runtime_required"
            elif has_pub and not has_sub:
                reason = "external_consumer"
            elif has_sub and not has_pub:
                reason = "external_publisher"
            else:
                reason = "not_applicable"
            not_applicable_detail[t] = reason

    status = "PASS" if adjusted_coverage_rate >= 0.80 else "WARN"

    return make_result(
        stage="8.1", name="eventbus_audit", status=status,
        target=">=80% of applicable EventBus topics (excl. distributed-runtime topics) covered",
        result={
            "static_audit": {
                "n_files_scanned": static_result["n_files_scanned"],
                "n_topics_found": static_result["n_topics_found"],
                "n_expected_fully_covered": n_fully_covered,
            },
            "live_audit": live_result,
            "n_expected_topics": n_expected,
            "n_not_applicable": len(NOT_APPLICABLE_IN_BENCHMARK),
            "n_applicable": n_applicable,
            "n_applicable_covered": n_applicable_covered,
            "adjusted_coverage_rate": round(adjusted_coverage_rate, 4),
            "raw_coverage_rate": round(n_fully_covered / n_expected, 4),
            "not_applicable_detail": not_applicable_detail,
            "applicable_topics": {
                t: {
                    "covered": bool(cov.get(t, {}).get("has_publisher") and cov.get(t, {}).get("has_subscriber")),
                    "has_publisher": bool(cov.get(t, {}).get("publishers")),
                    "has_subscriber": bool(cov.get(t, {}).get("subscribers")),
                }
                for t in applicable
            },
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = [run_stage_8_1()]
    for r in results:
        save_artifact(f"stage8_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    for r in run_all():
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")

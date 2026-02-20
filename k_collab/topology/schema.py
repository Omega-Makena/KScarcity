"""Topology schema and validation for nested federation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class TopologyValidationError(ValueError):
    """Raised when topology schema checks fail."""


CLEARANCE_ORDER = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "RESTRICTED": 2,
    "SECRET": 3,
}


@dataclass
class NodeRef:
    node_id: str
    level: int
    node_type: str
    parent_id: str | None
    agency_id: str
    domains: List[str]
    clearance: str


def _normalize_nodes(payload: Dict[str, Any]) -> List[NodeRef]:
    nodes = payload.get("nodes") or []
    result: List[NodeRef] = []
    for raw in nodes:
        node_id = str(raw.get("node_id", "")).strip()
        level = int(raw.get("level", 0) or 0)
        node_type = str(raw.get("node_type", "")).strip().lower()
        parent_id = raw.get("parent_id")
        parent_id = str(parent_id).strip() if parent_id else None
        agency_id = str(raw.get("agency_id") or raw.get("node_id") or "").strip()
        clearance = str(raw.get("clearance", "INTERNAL")).strip().upper()
        domains = [str(d).strip().lower() for d in (raw.get("domains") or []) if str(d).strip()]
        result.append(
            NodeRef(
                node_id=node_id,
                level=level,
                node_type=node_type,
                parent_id=parent_id,
                agency_id=agency_id,
                domains=domains,
                clearance=clearance,
            )
        )
    return result


def validate_topology(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate nested topology and return normalized payload."""
    nodes = _normalize_nodes(payload)
    if not nodes:
        raise TopologyValidationError("Topology must include at least one node")

    by_id: Dict[str, NodeRef] = {}
    for node in nodes:
        if not node.node_id:
            raise TopologyValidationError("Node missing node_id")
        if node.node_id in by_id:
            raise TopologyValidationError(f"Duplicate node_id: {node.node_id}")
        if node.level not in {1, 2, 3}:
            raise TopologyValidationError(f"Invalid level for {node.node_id}: {node.level}")
        if node.node_type not in {"agency", "department", "site"}:
            raise TopologyValidationError(f"Invalid node_type for {node.node_id}: {node.node_type}")
        if node.clearance not in CLEARANCE_ORDER:
            raise TopologyValidationError(f"Invalid clearance for {node.node_id}: {node.clearance}")
        by_id[node.node_id] = node

    for node in nodes:
        if node.level == 1 and node.parent_id:
            raise TopologyValidationError(f"Level-1 node cannot have parent: {node.node_id}")
        if node.level > 1 and not node.parent_id:
            raise TopologyValidationError(f"Level-{node.level} node must have parent: {node.node_id}")
        if node.parent_id:
            parent = by_id.get(node.parent_id)
            if not parent:
                raise TopologyValidationError(f"Parent not found for {node.node_id}: {node.parent_id}")
            if parent.level >= node.level:
                raise TopologyValidationError(f"Parent level must be lower than child for {node.node_id}")

    edges = payload.get("trust_edges") or []
    for edge in edges:
        src = str(edge.get("source", "")).strip()
        dst = str(edge.get("target", "")).strip()
        if not src or not dst:
            raise TopologyValidationError("Trust edge missing source/target")
        if src not in by_id or dst not in by_id:
            raise TopologyValidationError(f"Trust edge references unknown node: {src}->{dst}")

    normalized_nodes: List[Dict[str, Any]] = []
    for node in sorted(nodes, key=lambda n: (n.level, n.node_id)):
        normalized_nodes.append(
            {
                "node_id": node.node_id,
                "level": node.level,
                "node_type": node.node_type,
                "parent_id": node.parent_id,
                "agency_id": node.agency_id,
                "domains": sorted(set(node.domains)),
                "clearance": node.clearance,
            }
        )

    normalized_edges: List[Dict[str, Any]] = []
    for edge in edges:
        normalized_edges.append(
            {
                "source": str(edge.get("source", "")).strip(),
                "target": str(edge.get("target", "")).strip(),
                "channel": str(edge.get("channel", "standard")).strip().lower(),
            }
        )

    return {
        "name": str(payload.get("name", "k_collab_topology")).strip() or "k_collab_topology",
        "nodes": normalized_nodes,
        "trust_edges": sorted(normalized_edges, key=lambda e: (e["source"], e["target"], e["channel"])),
    }


def topology_preview(payload: Dict[str, Any]) -> str:
    """Return a text graph preview for UI fallback."""
    by_parent: Dict[str, List[str]] = {}
    for node in payload.get("nodes", []):
        parent = node.get("parent_id") or "ROOT"
        by_parent.setdefault(parent, []).append(node["node_id"])

    lines: List[str] = [f"Topology: {payload.get('name', 'unnamed')}"]

    def _walk(parent: str, depth: int) -> None:
        for node_id in sorted(by_parent.get(parent, [])):
            node = next((n for n in payload["nodes"] if n["node_id"] == node_id), None)
            if not node:
                continue
            label = f"{node['node_id']} [L{node['level']}:{node['node_type']}] domains={','.join(node.get('domains', [])) or '-'}"
            lines.append(f"{'  ' * depth}- {label}")
            _walk(node_id, depth + 1)

    _walk("ROOT", 0)

    if payload.get("trust_edges"):
        lines.append("Trust boundaries:")
        for edge in payload["trust_edges"]:
            lines.append(f"- {edge['source']} -> {edge['target']} ({edge['channel']})")

    return "\n".join(lines)


def diff_topologies(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Return minimal diff between two topology payloads."""
    old_nodes = {n["node_id"]: n for n in old.get("nodes", [])}
    new_nodes = {n["node_id"]: n for n in new.get("nodes", [])}

    added = sorted(set(new_nodes) - set(old_nodes))
    removed = sorted(set(old_nodes) - set(new_nodes))

    changed: List[Dict[str, Any]] = []
    for node_id in sorted(set(old_nodes) & set(new_nodes)):
        prev = old_nodes[node_id]
        curr = new_nodes[node_id]
        if prev != curr:
            changed.append({"node_id": node_id, "before": prev, "after": curr})

    old_edges = sorted(old.get("trust_edges", []), key=lambda e: (e.get("source"), e.get("target"), e.get("channel")))
    new_edges = sorted(new.get("trust_edges", []), key=lambda e: (e.get("source"), e.get("target"), e.get("channel")))

    return {
        "added_nodes": added,
        "removed_nodes": removed,
        "changed_nodes": changed,
        "trust_edges_changed": old_edges != new_edges,
    }

"""Convert K-Collab topology into hierarchical federation participant mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ParticipantBinding:
    client_id: str
    agency_id: str
    department_id: str
    domain_id: str


class NestedTopologyAdapter:
    """Maps L1/L2/L3 topology nodes to hierarchical FL client/domain identifiers."""

    def __init__(self, topology_payload: Dict[str, object]):
        self.topology = topology_payload or {"nodes": []}

    def participants(self, selected_nodes: List[str] | None = None) -> List[ParticipantBinding]:
        nodes = self.topology.get("nodes", [])
        by_id = {n["node_id"]: n for n in nodes}
        selected = set(selected_nodes or [])

        out: List[ParticipantBinding] = []

        for node in nodes:
            node_id = str(node.get("node_id"))
            node_type = str(node.get("node_type", "")).lower()
            level = int(node.get("level", 0) or 0)

            if selected and node_id not in selected:
                continue

            if node_type == "site" and level == 3:
                parent = by_id.get(str(node.get("parent_id", "")))
                if not parent:
                    continue
                agency_id = str(parent.get("agency_id") or parent.get("parent_id") or "unknown")
                department_id = str(parent.get("node_id"))
                out.append(
                    ParticipantBinding(
                        client_id=node_id,
                        agency_id=agency_id,
                        department_id=department_id,
                        domain_id=f"{agency_id}:{department_id}",
                    )
                )

            if node_type == "department" and level == 2:
                agency_id = str(node.get("agency_id") or node.get("parent_id") or "unknown")
                department_id = node_id
                has_sites = any(
                    int(c.get("level", 0) or 0) == 3 and str(c.get("parent_id", "")) == node_id
                    for c in nodes
                )
                if not has_sites:
                    out.append(
                        ParticipantBinding(
                            client_id=node_id,
                            agency_id=agency_id,
                            department_id=department_id,
                            domain_id=f"{agency_id}:{department_id}",
                        )
                    )

        return out

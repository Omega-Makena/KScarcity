"""K-Shield artifact store for causal adapter outputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional

from scarcity.causal.reporting import EffectArtifact, SpecError

from .types import KnowledgeGraphEdge, SimulationParameterUpdate


@dataclass
class ArtifactPaths:
    run_dir: Path
    effects_path: Path
    errors_path: Path
    edges_path: Path
    updates_path: Path
    summary_path: Path


class KShieldArtifactStore:
    def __init__(self, root: Path, run_id: str) -> None:
        run_dir = root / "runs" / run_id
        self.paths = ArtifactPaths(
            run_dir=run_dir,
            effects_path=run_dir / "effects.jsonl",
            errors_path=run_dir / "errors.jsonl",
            edges_path=run_dir / "edges.jsonl",
            updates_path=run_dir / "simulation_updates.jsonl",
            summary_path=run_dir / "summary.json",
        )

    def prepare(self) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)

    def write_effects(self, effects: Iterable[EffectArtifact]) -> None:
        with self.paths.effects_path.open("w", encoding="utf-8") as handle:
            for effect in effects:
                handle.write(json.dumps(effect.to_dict(), default=str))
                handle.write("\n")

    def write_errors(self, errors: Iterable[SpecError]) -> None:
        with self.paths.errors_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error.to_dict(), default=str))
                handle.write("\n")

    def write_edges(self, edges: Iterable[KnowledgeGraphEdge]) -> None:
        with self.paths.edges_path.open("w", encoding="utf-8") as handle:
            for edge in edges:
                payload = {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "sign": edge.sign,
                    "weight": edge.weight,
                    "confidence": edge.confidence,
                    "lag": edge.lag,
                    "window": edge.window,
                    "metadata": edge.metadata,
                }
                handle.write(json.dumps(payload, default=str))
                handle.write("\n")

    def write_updates(self, updates: Iterable[SimulationParameterUpdate]) -> None:
        with self.paths.updates_path.open("w", encoding="utf-8") as handle:
            for update in updates:
                payload = {
                    "parameter": update.parameter,
                    "delta": update.delta,
                    "reason": update.reason,
                    "edge_id": update.edge_id,
                    "metadata": update.metadata,
                }
                handle.write(json.dumps(payload, default=str))
                handle.write("\n")

    def write_summary(self, summary: dict) -> None:
        with self.paths.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)

    def write_all(
        self,
        effects: Iterable[EffectArtifact],
        errors: Iterable[SpecError],
        edges: Iterable[KnowledgeGraphEdge],
        updates: Iterable[SimulationParameterUpdate],
        summary: dict,
    ) -> None:
        self.prepare()
        self.write_effects(effects)
        self.write_errors(errors)
        self.write_edges(edges)
        self.write_updates(updates)
        self.write_summary(summary)

"""Artifact generation for the causal pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from scarcity.causal.graph import build_dot
from scarcity.causal.reporting import CausalRunResult, EffectArtifact, SpecError


@dataclass
class ArtifactPaths:
    root: str
    run_dir: str
    graphs_dir: str
    summary_path: str
    effects_path: str
    errors_path: str
    input_graph_path: str
    learned_graph_path: str


class ArtifactWriter:
    def __init__(self, root: str, run_id: str) -> None:
        run_dir = os.path.join(root, "runs", run_id)
        graphs_dir = os.path.join(run_dir, "graphs")
        self.paths = ArtifactPaths(
            root=root,
            run_dir=run_dir,
            graphs_dir=graphs_dir,
            summary_path=os.path.join(run_dir, "summary.json"),
            effects_path=os.path.join(run_dir, "effects.jsonl"),
            errors_path=os.path.join(run_dir, "errors.jsonl"),
            input_graph_path=os.path.join(graphs_dir, "input.dot"),
            learned_graph_path=os.path.join(graphs_dir, "learned.dot"),
        )

    def prepare(self) -> None:
        os.makedirs(self.paths.graphs_dir, exist_ok=True)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        with open(self.paths.summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def write_effects(self, effects: Iterable[EffectArtifact]) -> None:
        with open(self.paths.effects_path, "w", encoding="utf-8") as handle:
            for effect in effects:
                handle.write(json.dumps(effect.to_dict(), default=str))
                handle.write("\n")

    def write_errors(self, errors: Iterable[SpecError]) -> None:
        with open(self.paths.errors_path, "w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error.to_dict(), default=str))
                handle.write("\n")

    def write_graphs(self, input_dot: Optional[str], learned_edges: Iterable[tuple[str, str]]) -> None:
        input_payload = input_dot if input_dot is not None else build_dot([])
        with open(self.paths.input_graph_path, "w", encoding="utf-8") as handle:
            handle.write(input_payload)

        learned_dot = build_dot(list(learned_edges))
        with open(self.paths.learned_graph_path, "w", encoding="utf-8") as handle:
            handle.write(learned_dot)

    def write_run(self, result: CausalRunResult, input_dot: Optional[str], learned_edges: Iterable[tuple[str, str]]) -> None:
        self.prepare()
        summary_payload = {
            "summary": result.summary.to_dict() if result.summary else None,
            "metadata": result.metadata.to_dict() if result.metadata else None,
        }
        self.write_summary(summary_payload)
        self.write_effects(result.results)
        self.write_errors(result.errors)
        self.write_graphs(input_dot, learned_edges)


def compute_data_signature(data: pd.DataFrame, time_column: Optional[str] = None, sample_size: int = 50) -> Dict[str, Any]:
    head = data.head(sample_size)
    tail = data.tail(sample_size)
    sample = pd.concat([head, tail]) if len(data) > sample_size else data

    digest = hashlib.sha256()
    digest.update("|".join(map(str, data.columns)).encode("utf-8"))
    digest.update("|".join(map(str, data.dtypes)).encode("utf-8"))
    digest.update(str(data.shape).encode("utf-8"))
    digest.update(sample.to_csv(index=False).encode("utf-8"))

    signature = {
        "rows": int(data.shape[0]),
        "columns": list(map(str, data.columns)),
        "dtypes": {str(k): str(v) for k, v in data.dtypes.items()},
        "hash": digest.hexdigest(),
    }

    if time_column and time_column in data.columns:
        series = data[time_column]
        signature["time_min"] = str(series.min())
        signature["time_max"] = str(series.max())

    return signature

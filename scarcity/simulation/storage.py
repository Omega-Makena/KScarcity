"""
Simulation storage utilities: trajectory logs and what-if results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class SimulationStorageConfig:
    root: Path = Path("artifacts/sim")
    trajectories_dir: str = "trajectories"
    whatif_dir: str = "whatif"
    compression: str = "zstd"


class SimulationStorage:
    def __init__(self, config: SimulationStorageConfig):
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / config.trajectories_dir).mkdir(exist_ok=True)
        (self.root / config.whatif_dir).mkdir(exist_ok=True)

    def save_trajectory(self, trajectory: Iterable[Dict[str, float]], tag: str) -> Path:
        path = self.root / self.config.trajectories_dir / f"{tag}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for tick, snapshot in enumerate(trajectory):
                fh.write(json.dumps({"t": tick, "state": snapshot}) + "\n")
        return path

    def save_whatif(self, result: Dict[str, any]) -> Path:
        tag = result.get("scenario_id", "scenario")
        path = self.root / self.config.whatif_dir / f"{tag}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        return path


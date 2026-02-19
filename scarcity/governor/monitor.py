"""
DRG monitor and diagnostics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class MonitorConfig:
    log_dir: Path = Path("logs/drg")
    level: str = "INFO"


class DRGMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, float]] = []

    def record(self, metrics: Dict[str, float]) -> None:
        entry = {"timestamp": time.time(), **metrics}
        self._history.append(entry)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def dump(self) -> Path:
        path = self.config.log_dir / "drg_metrics.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._history, fh, indent=2)
        return path


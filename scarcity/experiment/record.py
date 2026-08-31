"""
First-class experiment and provenance records.

An experiment is reproducible only if everything that shaped its result is
captured with it: the data, the configuration, the seed, the exact code, the
hardware, the cost, and every discovered relationship with the evidence behind
it. These dataclasses are that record — JSON-serializable, saved and reloaded so
a result can be reconstructed and audited months later.

- ``DatasetSpec``   — what data (name, version, hash, shape, observation range).
- ``EnvInfo``       — which code (package version, git commit, dirty flag).
- ``HardwareInfo``  — which machine (platform, CPU, RAM, GPU).
- ``EdgeProvenance``— why a relationship is believed (metrics, calibration,
  first/last seen, and the causal analysis if one was run).
- ``ExperimentRecord`` — the envelope tying it all together, with metrics,
  warnings, and failures.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetSpec:
    """What data an experiment ran on."""
    name: str
    version: str = "unknown"
    n_observations: int = 0
    n_variables: int = 0
    variables: List[str] = field(default_factory=list)
    observation_range: Optional[Tuple[Any, Any]] = None
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.observation_range is not None:
            d["observation_range"] = list(self.observation_range)
        return d


@dataclass
class EnvInfo:
    """Which code produced the result."""
    scarcity_version: str = "unknown"
    git_commit: str = "unknown"
    git_dirty: bool = False
    python_version: str = ""

    @classmethod
    def capture(cls, repo_hint: Optional[str] = None) -> "EnvInfo":
        version = "unknown"
        try:
            import scarcity  # noqa
            version = getattr(scarcity, "__version__", "unknown")
        except Exception:
            pass
        if version == "unknown":
            try:
                from importlib.metadata import version as _v
                version = _v("scarcity")
            except Exception:
                pass
        commit, dirty = "unknown", False
        cwd = repo_hint or os.path.dirname(os.path.abspath(__file__))
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd,
                stderr=subprocess.DEVNULL, text=True).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd,
                stderr=subprocess.DEVNULL, text=True)
            dirty = bool(status.strip())
        except Exception:
            pass
        return cls(scarcity_version=str(version), git_commit=commit,
                   git_dirty=dirty, python_version=sys.version.split()[0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareInfo:
    """Which machine produced the result."""
    platform: str = ""
    cpu: str = ""
    cpu_count: int = 0
    ram_gb: float = 0.0
    gpu: str = ""

    @classmethod
    def capture(cls) -> "HardwareInfo":
        ram_gb = 0.0
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass
        gpu = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return cls(
            platform=platform.platform(),
            cpu=platform.processor() or platform.machine(),
            cpu_count=os.cpu_count() or 0,
            ram_gb=ram_gb,
            gpu=gpu,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeProvenance:
    """Why Scarcity believes one relationship — the auditable record of an edge."""
    source: str
    target: str
    rel_type: str
    state: str = "tentative"
    variables: List[str] = field(default_factory=list)
    # discovery metrics (fit_score, confidence, evidence, stability)
    metrics: Dict[str, Any] = field(default_factory=dict)
    # calibration evidence (permutation p-value, BH q-value, null_max, ...)
    calibration: Dict[str, Any] = field(default_factory=dict)
    # lifecycle timing
    first_detected: Any = None
    generation: int = 0
    # causal analysis, if one was run for this edge
    causal: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def explain(self) -> str:
        """A human-readable 'why do you believe this?' summary."""
        m = self.metrics or {}
        lines = [f"Edge: {self.source} -> {self.target}",
                 f"Type: {self.rel_type}   State: {self.state}",
                 "Evidence:"]
        for k in ("confidence", "fit_score", "evidence", "stability"):
            if k in m:
                lines.append(f"  {k}: {m[k]}")
        for k, v in (self.calibration or {}).items():
            lines.append(f"  {k}: {v}")
        if self.first_detected is not None:
            lines.append(f"  first_detected: {self.first_detected}")
        if self.causal:
            lines.append("Causal analysis:")
            for k, v in self.causal.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


@dataclass
class ExperimentRecord:
    """The reproducible envelope for one experiment."""
    name: str
    experiment_id: str = ""
    created_at: str = field(default_factory=_utc_now)
    seed: int = 0
    dataset: Optional[DatasetSpec] = None
    config: Dict[str, Any] = field(default_factory=dict)
    env: Optional[EnvInfo] = None
    hardware: Optional[HardwareInfo] = None
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_used: bool = False
    edges: List[EdgeProvenance] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "config": self.config,
            "env": self.env.to_dict() if self.env else None,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_used": self.gpu_used,
            "edges": [e.to_dict() for e in self.edges],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "failures": self.failures,
            "status": self.status,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        """Load a saved record as a plain dict (the audit view)."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

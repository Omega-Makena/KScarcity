"""Simple file-based model registry with version metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from k_collab.common.versioned_store import VersionedJSONStore


class FederatedModelRegistry:
    """Versioned model artifact metadata and storage."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(self.base_dir / "model_registry_versions.jsonl", kind="model_registry")

    def register(self, model_vector: np.ndarray, metadata: Dict[str, Any], actor: str = "system") -> Dict[str, Any]:
        model_id = metadata.get("model_id") or f"model_{len(self.list_models()) + 1}"
        artifact_path = self.artifacts_dir / f"{model_id}.npy"
        np.save(artifact_path, model_vector.astype(float))

        payload = {
            "models": self.list_models() + [
                {
                    "model_id": model_id,
                    "artifact_path": str(artifact_path),
                    "metadata": metadata,
                }
            ]
        }
        rec = self._store.save(payload, actor=actor, message=f"register_model:{model_id}")
        return rec.__dict__

    def list_models(self) -> List[Dict[str, Any]]:
        latest = self._store.latest()
        if not latest:
            return []
        return list(latest.payload.get("models", []))

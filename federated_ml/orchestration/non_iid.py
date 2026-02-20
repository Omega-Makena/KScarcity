"""Non-IID diagnostics for federated learning updates."""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np


def update_heterogeneity(updates: Dict[str, List[float]]) -> Dict[str, Any]:
    if not updates:
        return {
            "client_count": 0,
            "mean_norm": 0.0,
            "norm_cv": 0.0,
            "mean_cosine_distance": 0.0,
            "is_highly_non_iid": False,
        }

    vectors = []
    for vec in updates.values():
        arr = np.asarray(vec, dtype=float)
        if arr.size == 0:
            continue
        vectors.append(arr)

    if not vectors:
        return {
            "client_count": 0,
            "mean_norm": 0.0,
            "norm_cv": 0.0,
            "mean_cosine_distance": 0.0,
            "is_highly_non_iid": False,
        }

    norms = np.array([float(np.linalg.norm(v)) for v in vectors], dtype=float)
    mean_norm = float(np.mean(norms)) if norms.size else 0.0
    std_norm = float(np.std(norms)) if norms.size else 0.0
    norm_cv = float(std_norm / mean_norm) if mean_norm > 1e-12 else 0.0

    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a = vectors[i]
            b = vectors[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-12:
                continue
            cosine = float(np.dot(a, b) / denom)
            distances.append(float(1.0 - cosine))

    mean_distance = float(np.mean(distances)) if distances else 0.0

    return {
        "client_count": len(vectors),
        "mean_norm": mean_norm,
        "norm_cv": norm_cv,
        "mean_cosine_distance": mean_distance,
        "is_highly_non_iid": bool(norm_cv > 0.75 or mean_distance > 0.9),
    }

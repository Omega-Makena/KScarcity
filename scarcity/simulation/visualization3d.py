"""
3D visualization engine for SCARCITY simulation.

This module prefers PyTorch3D but falls back to a numpy-based layout if the GPU
stack is unavailable. The return value of `render_frame` is a lightweight
dictionary that can be consumed by a separate UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from pytorch3d.renderer import FoVPerspectiveCameras  # type: ignore

    _HAS_PT3D = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PT3D = False
    torch = None  # type: ignore


@dataclass
class VisualizationConfig:
    max_fps: int = 60
    color_scheme: str = "stability"
    lod_profiles: Optional[np.ndarray] = None


class VisualizationEngine:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        if config.lod_profiles is None:
            self.lod_profiles = np.asarray([1.0, 0.75, 0.5], dtype=np.float32)
        else:
            self.lod_profiles = np.asarray(config.lod_profiles, dtype=np.float32)
        self._frame_id = 0

    def render_frame(
        self,
        node_positions: np.ndarray,
        node_values: np.ndarray,
        edge_adjacency: np.ndarray,
        stability: np.ndarray,
        lod: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Returns GPU-ready buffers for external consumption. If PyTorch3D is not
        present, the buffers remain numpy arrays.
        """
        self._frame_id += 1
        lod_factor = float(np.clip(lod, self.lod_profiles.min(), self.lod_profiles.max()))

        node_positions = self._apply_lod_positions(node_positions, lod_factor)
        node_colors = self._compute_node_colors(node_values, stability)
        edges = self._extract_edge_segments(edge_adjacency, node_positions, lod_factor)

        if _HAS_PT3D:  # pragma: no cover - optional
            return {
                "positions": torch.from_numpy(node_positions),
                "colors": torch.from_numpy(node_colors),
                "edges": torch.from_numpy(edges),
                "frame_id": self._frame_id,
            }
        return {
            "positions": node_positions,
            "colors": node_colors,
            "edges": edges,
            "frame_id": self._frame_id,
        }

    def _apply_lod_positions(self, positions: np.ndarray, lod: float) -> np.ndarray:
        target = int(np.ceil(len(positions) * lod))
        if target <= 0:
            return positions[:0]
        return positions[:target]

    def _compute_node_colors(self, values: np.ndarray, stability: np.ndarray) -> np.ndarray:
        """
        Map node values and stability into a smooth, readable colour map.

        - Value drives a blue→cyan ramp.
        - Stability brightens/desaturates the colour so unstable regions are dimmer.
        """
        if values.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        norm_values = (values - values.min()) / (np.ptp(values) + 1e-6)
        norm_stability = np.clip(stability.mean(axis=0), 0.0, 1.0)

        # Base blue→cyan gradient from value.
        r = np.zeros_like(norm_values)
        g = 0.5 * norm_values + 0.2
        b = 0.8 + 0.2 * norm_values

        # Stability modulates brightness.
        brightness = 0.4 + 0.6 * norm_stability
        colors = np.stack([r * brightness, g * brightness, b * brightness], axis=1)
        return np.clip(colors, 0.0, 1.0).astype(np.float32)

    def _extract_edge_segments(
        self,
        adjacency: np.ndarray,
        positions: np.ndarray,
        lod: float,
    ) -> np.ndarray:
        # Identify candidate edges by non-zero adjacency entries.
        idx = np.nonzero(np.abs(adjacency) > 0)
        src_idx = idx[0]
        dst_idx = idx[1]
        if len(src_idx) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Score edges by absolute weight so that we can keep the strongest
        # relationships and avoid a fully-saturated mesh.
        strengths = np.abs(adjacency[src_idx, dst_idx])
        order = np.argsort(strengths)[::-1]

        # Target a limited number of edges relative to node count so the
        # visual reads as a constellation rather than a lattice.
        base_max = max(64, positions.shape[0] * 4)
        max_edges = int(base_max * float(np.clip(lod, 0.25, 1.0)))
        max_edges = min(max_edges, len(order))
        if max_edges <= 0:
            return np.zeros((0, 6), dtype=np.float32)

        keep = order[:max_edges]
        src_idx = src_idx[keep]
        dst_idx = dst_idx[keep]

        segments = np.zeros((len(src_idx), 6), dtype=np.float32)
        for i, (s, d) in enumerate(zip(src_idx, dst_idx)):
            if s >= len(positions) or d >= len(positions):
                continue
            segments[i, :3] = positions[s]
            segments[i, 3:] = positions[d]
        return segments


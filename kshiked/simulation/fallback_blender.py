"""
FallbackBlender — confidence-weighted blending of learned vs parametric predictions.

Design principle:
    final_value = confidence × learned_value + (1 - confidence) × fallback_value

    - confidence 0.0 → 100% fallback (no data)
    - confidence 0.5 → 50/50 blend
    - confidence 1.0 → 100% learned (full data coverage)

Fallbacks are safety nets, not ground truth.
As scarcity learns more from data, fallbacks naturally fade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kshield.blender")


@dataclass
class BlendResult:
    """Result of blending learned and fallback predictions."""
    blended: Dict[str, float]
    learned: Dict[str, float]
    fallback: Dict[str, float]
    confidence_used: Dict[str, float]
    blend_ratio: float  # Overall: how much came from learned (0–1)


class FallbackBlender:
    """
    Blends learned predictions with parametric fallbacks.

    Per-variable confidence weighting ensures that variables with
    strong learned support use the discovered values, while variables
    with sparse or conflicting data cling to conservative fallbacks.
    """

    def __init__(
        self,
        confidence_map: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.05,
        smoothing: float = 0.0,
    ):
        """
        Args:
            confidence_map: Per-variable confidence (0–1). Updated from bridge.
            min_confidence: Floor — even with zero data, allow tiny learning signal.
            smoothing: Temporal smoothing factor for confidence transitions (0 = instant).
        """
        self._confidence_map = confidence_map or {}
        self._min_confidence = min_confidence
        self._smoothing = smoothing
        self._ema_confidence: Dict[str, float] = {}

    def update_confidence(self, new_map: Dict[str, float]):
        """Update confidence map (e.g., after new data arrives)."""
        if self._smoothing > 0:
            for key, val in new_map.items():
                prev = self._ema_confidence.get(key, val)
                self._ema_confidence[key] = self._smoothing * prev + (1 - self._smoothing) * val
            self._confidence_map = dict(self._ema_confidence)
        else:
            self._confidence_map = dict(new_map)

    def blend(
        self,
        learned: Dict[str, float],
        fallback: Dict[str, float],
        override_confidence: Optional[Dict[str, float]] = None,
    ) -> BlendResult:
        """
        Blend learned and fallback predictions.

        Args:
            learned: Values from the learned simulator (PolicySimulator).
            fallback: Values from the parametric SFC.
            override_confidence: If provided, use these confidences instead of stored map.

        Returns:
            BlendResult with blended values and metadata.
        """
        conf_map = override_confidence or self._confidence_map
        result: Dict[str, float] = {}
        conf_used: Dict[str, float] = {}

        all_keys = set(learned.keys()) | set(fallback.keys())
        total_weight_learned = 0.0
        total_weight = 0.0

        for key in all_keys:
            c = max(self._min_confidence, conf_map.get(key, 0.0))
            l_val = learned.get(key, 0.0)
            f_val = fallback.get(key, l_val)  # If no fallback, use learned

            # Weighted blend
            blended_val = c * l_val + (1.0 - c) * f_val
            result[key] = blended_val
            conf_used[key] = c

            total_weight_learned += c
            total_weight += 1.0

        blend_ratio = total_weight_learned / max(total_weight, 1.0)

        return BlendResult(
            blended=result,
            learned=dict(learned),
            fallback=dict(fallback),
            confidence_used=conf_used,
            blend_ratio=blend_ratio,
        )

    def blend_trajectory(
        self,
        learned_trajectory: List[Dict[str, float]],
        fallback_trajectory: List[Dict[str, float]],
    ) -> List[BlendResult]:
        """
        Blend two full trajectories step by step.

        If trajectories differ in length, truncates to the shorter one.
        """
        min_len = min(len(learned_trajectory), len(fallback_trajectory))
        results = []
        for i in range(min_len):
            results.append(self.blend(learned_trajectory[i], fallback_trajectory[i]))
        return results

    def get_confidence_summary(self) -> Dict[str, Any]:
        """Summary statistics about current confidence levels."""
        if not self._confidence_map:
            return {"overall": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        vals = list(self._confidence_map.values())
        return {
            "overall": float(np.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "count": len(vals),
            "high_confidence": [k for k, v in self._confidence_map.items() if v > 0.7],
            "low_confidence": [k for k, v in self._confidence_map.items() if v < 0.3],
        }

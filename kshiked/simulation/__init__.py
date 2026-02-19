"""
K-SHIELD Simulation Layer

Kenya-specific economic simulation built on top of the
scarcity.simulation.sfc framework. Contains:
- kenya_calibration: Data-driven parameter derivation from World Bank data
- scenario_templates: Named real-world shock scenarios and policy templates
- fallback_blender: Confidence-weighted blending of learned vs parametric predictions
- validation: Historical episode detection and accuracy scoring
- compiler: Scenario compilation
- controller: Simulation orchestration
"""

from kshiked.simulation.kenya_calibration import get_kenya_config, calibrate_from_data
from kshiked.simulation.scenario_templates import (
    SCENARIO_LIBRARY, POLICY_TEMPLATES,
    get_scenario_by_id, build_custom_scenario,
)

try:
    from kshiked.simulation.fallback_blender import FallbackBlender
except ImportError:
    FallbackBlender = None

try:
    from kshiked.simulation.validation import ValidationRunner, EpisodeDetector
except ImportError:
    ValidationRunner = None
    EpisodeDetector = None


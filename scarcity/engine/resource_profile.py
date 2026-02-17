"""
Shared resource profile utilities for SCARCITY engine.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any

DEFAULT_RESOURCE_PROFILE: Dict[str, Any] = {
    'n_paths': 200,
    'precision': 'fp16',
    'sketch_dim': 512,
    'window_size': 256,
    'resamples': 8,
    'export_interval': 10,
    'branch_width': 1,
    'tier2_enabled': True,
    'tier3_topk': 5,
}


def clone_default_profile() -> Dict[str, Any]:
    """
    Creates multiple independent copies of the default resource profile.

    This function is used to instantiate a new resource profile dictionary based
    on the system defaults. This ensures that modifications to one profile instance
    do not affect the global defaults.

    Returns:
        A new dictionary containing the default resource configuration.
    """
    return deepcopy(DEFAULT_RESOURCE_PROFILE)

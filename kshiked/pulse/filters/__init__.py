# Kenya filters package
"""
Kenya-focused content filters for KShield Pulse.

Provides keyword lists and filtering utilities for
detecting Kenya-related content across platforms.
"""

from .kenya_keywords import (
    KENYA_POLITICAL,
    KENYA_ECONOMIC,
    KENYA_THREAT_SIGNALS,
    KENYA_LOCATIONS,
    KENYA_ETHNIC_SENSITIVE,
    get_all_kenya_keywords,
    contains_kenya_keyword,
    get_matched_keywords,
    is_threat_related,
    get_threat_level_keywords,
)

__all__ = [
    "KENYA_POLITICAL",
    "KENYA_ECONOMIC",
    "KENYA_THREAT_SIGNALS",
    "KENYA_LOCATIONS",
    "KENYA_ETHNIC_SENSITIVE",
    "get_all_kenya_keywords",
    "contains_kenya_keyword",
    "get_matched_keywords",
    "is_threat_related",
    "get_threat_level_keywords",
]

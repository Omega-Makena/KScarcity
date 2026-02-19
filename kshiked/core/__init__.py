"""
Core KShield logic.
"""

try:
    from kshiked.core.scarcity_bridge import ScarcityBridge, TrainingReport
except ImportError:
    ScarcityBridge = None
    TrainingReport = None


# LLM integration package
"""
LLM integration layer for KShield Pulse.

Provides swappable LLM provider interface with:
- Gemini implementation (default)
- Threat tier classification
- Role identification
- Narrative analysis
- Fine-tuning data preparation

The design allows easy swapping between Gemini, OpenAI, Claude,
or local models without changing the rest of the codebase.
"""

from .base import (
    LLMProvider,
    ThreatClassification,
    RoleClassification,
    NarrativeAnalysis
)
from .gemini import GeminiProvider, GeminiConfig, create_gemini_provider
from .ollama import OllamaProvider
from .fine_tuning import FineTuningDataPreparer, create_fine_tuning_preparer

__all__ = [
    "LLMProvider",
    "ThreatClassification",
    "RoleClassification",
    "ThreatTier",
    "RoleType",
    "GeminiProvider",
    "create_gemini_provider",
    "OllamaProvider",
    "FineTuningDataPreparer",
    "create_fine_tuning_preparer",
]

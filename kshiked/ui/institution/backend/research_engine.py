"""
Unified Research Engine for KShield Dashboard

This module provides a unified orchestration layer over the existing Scarcity and KShield ML functions:
- Predict: Trend forecasting and projections (scarcity.engine.predict_value)
- Classify: Sector/Signal threat classification (kshiked.pulse.llm.ollama)
- Recommend: Intervention recommendations (kshiked.pulse.llm.policy_predictor)
- Research: Conversational context-aware queries

It enforces role-based access control based on the EngineContext.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import json

from scarcity.engine.relationships import RelationshipType, Hypothesis
from kshiked.pulse.llm.ollama import OllamaProvider
from kshiked.pulse.llm.config import OllamaConfig
from kshiked.pulse.llm.policy_predictor import PolicyPredictor
from kshiked.pulse.llm.signals import ThreatSignal, ThreatTier, RoleType

logger = logging.getLogger(__name__)


@dataclass
class EngineContext:
  role: str # "spoke", "admin", "executive"
  user_id: str
  sector_id: Optional[str] = None
  project_id: Optional[str] = None


@dataclass
class ResearchResult:
  type: str # "prediction", "classification", "recommendation", "research"
  generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
  confidence: str = "medium"
  plain_language_summary: str = ""
  # Optional fields depending on the result type
  data: Dict[str, Any] = field(default_factory=dict)
  
  def to_dict(self) -> Dict[str, Any]:
    return {
      "type": self.type,
      "generated_at": self.generated_at,
      "confidence": self.confidence,
      "plain_language_summary": self.plain_language_summary,
      **self.data
    }


class ScopeViolationError(Exception):
  """Raised when a user queries outside their permitted scope."""
  pass


class ResearchEngine:
  def __init__(self, context: EngineContext):
    self.context = context
    
    # Initialize KShield Pulse ML providers
    # We reuse the existing ollama configuration
    self.ollama_config = OllamaConfig(base_url="http://localhost:11434")
    self.ollama_provider = OllamaProvider(config=self.ollama_config)
    self.policy_predictor = PolicyPredictor(provider=self.ollama_provider)
    
  def _check_scope(self, query_scope: str):
    """Enforce role-based access control rules."""
    role = self.context.role.lower()
    if role == "spoke":
      if query_scope in ["all_sectors", "cross_sector"]:
        raise ScopeViolationError(f"Spoke role cannot access scope: {query_scope}")
    elif role == "admin":
      if query_scope in ["all_sectors", "cross_sector"]:
        raise ScopeViolationError(f"Admin role cannot access scope: {query_scope}. Limit to sector.")
    elif role == "executive":
      # Executive has full access
      pass

  # async def predict(self, indicator: str, horizon: str = "30d", scope: str = "sector") -> ResearchResult:
  #   \"\"\"
  #   Wraps predictive forecasting logic.
  #   Uses Scarcity numerical predictors where possible.
  #   \"\"\"
  #   self._check_scope(scope)
  #   
  #   predicted_val = 0.0
  #   current_val = 0.0
  #   trend = "stable"
  #   
  #   result = ResearchResult(
  #     type="prediction",
  #     confidence="high",
  #     plain_language_summary=f"Projected {horizon} forecast for {indicator} remains {trend}.",
  #     data={
  #       "horizon": horizon,
  #       "indicator": indicator,
  #       "current_value": current_val,
  #       "predicted_value": predicted_val,
  #       "trend": trend,
  #       "scenarios": {
  #         "optimistic": { "value": predicted_val * 1.05, "assumption": "Favorable conditions" },
  #         "most_likely": { "value": predicted_val, "assumption": "Baseline trajectory" },
  #         "pessimistic": { "value": predicted_val * 0.95, "assumption": "Adverse conditions" }
  #       },
  #       "recommended_action": "Continue monitoring."
  #     }
  #   )
  #   return result

  # async def classify(self, text: str, entity: str, scope: str = "sector") -> ResearchResult:
  #   \"\"\"
  #   Wraps classification logic using KShield Pulse Ollama models.
  #   \"\"\"
  #   self._check_scope(scope)
  #   
  #   # Use existing Ollama provider
  #   signal = await self.ollama_provider.full_analysis(text=text, source_id=entity)
  #   
  #   classification = "stable"
  #   factors = []
  #   if signal and signal.threat:
  #     if signal.threat.tier in [ThreatTier.TIER_1, ThreatTier.TIER_2]:
  #       classification = "critical"
  #     elif signal.threat.tier == ThreatTier.TIER_3:
  #       classification = "at_risk"
  #     factors.append(signal.threat.category.value)
  #     
  #   result = ResearchResult(
  #     type="classification",
  #     confidence="high" if signal else "low",
  #     plain_language_summary=f"{entity.capitalize()} is currently classified as {classification}.",
  #     data={
  #       "entity": entity,
  #       "classification": classification,
  #       "contributing_factors": factors,
  #       "suggested_next_step": "Review signal feed."
  #     }
  #   )
  #   return result

  # async def recommend(self, text: str, scope: str = "sector") -> ResearchResult:
  #   \"\"\"
  #   Wraps recommendation generation using the PolicyPredictor.
  #   \"\"\"
  #   self._check_scope(scope)
  #   
  #   result = ResearchResult(
  #     type="recommendation",
  #     confidence="medium",
  #     plain_language_summary="Intervention recommended to stabilize current trend.",
  #     data={
  #       "priority": "high",
  #       "recommendations": [
  #         {
  #           "action": "Increase resource allocation",
  #           "rationale": "Mitigate emerging risk factors.",
  #           "expected_impact": "Stabilization within timeline.",
  #           "timeframe": "short_term",
  #           "responsible_role": "admin",
  #           "linked_to": { "type": "indicator", "id": "unknown" }
  #         }
  #       ]
  #     }
  #   )
  #   return result
    
  async def research(self, query: str, scope: str = "sector") -> ResearchResult:
    """
    Handles open-ended research queries.
    """
    self._check_scope(scope)
    
    # Convert open query into a generated response using the LLM text provider
    response = await self.ollama_provider._generate_text(
      prompt=query,
      system_prompt="You are a Kenyan policy intelligence analyst. Provide concise research."
    )
    
    return ResearchResult(
      type="research_query",
      confidence="high",
      plain_language_summary=response[:200] + "...",
      data={
        "query": query,
        "full_response": response
      }
    )

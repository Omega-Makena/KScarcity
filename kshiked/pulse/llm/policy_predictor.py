"""
Policy Impact Predictor — Forecast Social Response to Policy

Takes a BillAnalysis + search evidence and predicts:
- Mobilization probability per provision
- County risk heatmap (47 counties)
- Timeline prediction (when would protests peak?)
- Historical comparison (closest past event)
- Narrative forecast (which archetypes will emerge?)
- Counter-narrative suggestions

Uses qwen2.5:3b for prediction with rich context injection.

Usage:
    predictor = PolicyPredictor(provider)
    prediction = await predictor.predict(bill, evidence)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config import AnalysisTask
from .prompts_kenya import (
    KENYA_COUNTIES,
    KENYA_POLITICAL_CONTEXT_2025,
    SHENG_GLOSSARY,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProvisionImpact:
    """Predicted impact for a single bill provision."""
    clause_id: str = ""
    description: str = ""
    mobilization_probability: float = 0.0
    predicted_timeline: str = ""      # "24h", "72h", "14d", etc.
    risk_level: str = "low"           # low, moderate, high, critical
    likely_mobilizers: List[str] = field(default_factory=list)
    narrative_archetypes: List[str] = field(default_factory=list)
    counter_narratives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactPrediction:
    """Full impact prediction for a bill."""
    bill_title: str = ""
    overall_mobilization: float = 0.0
    overall_risk_level: str = "low"
    predicted_phase: str = ""           # Current/predicted PolicyPhase
    predicted_peak_timeline: str = ""   # When protests would peak
    county_risks: Dict[str, float] = field(default_factory=dict)
    top_risk_counties: List[str] = field(default_factory=list)
    provision_impacts: List[ProvisionImpact] = field(default_factory=list)
    historical_match: str = ""
    historical_similarity: float = 0.0
    historical_outcome: str = ""
    dominant_narratives: List[str] = field(default_factory=list)
    likely_hashtags: List[str] = field(default_factory=list)
    likely_mobilizers: List[str] = field(default_factory=list)
    counter_narrative_suggestions: List[str] = field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provision_impacts"] = [p.to_dict() for p in self.provision_impacts]
        return d

    @property
    def risk_emoji(self) -> str:
        levels = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}
        return levels.get(self.overall_risk_level, "⚪")


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Prompts
# ═══════════════════════════════════════════════════════════════════════════

PREDICTION_SYSTEM = """You are a Kenyan political risk analyst predicting social response 
to new legislation and government policy.

{political_context}

{counties}

You understand:
- Kenya's 47 counties with distinct political dynamics
- 7-phase policy lifecycle: LEAK → ANNOUNCE → REACT → MOBILIZE → IMPLEMENT → IMPACT → SETTLE
- Youth mobilization patterns (Gen Z, TikTok/Twitter organizing)
- Sheng/Swahili framing of policy opposition
- Tribal and regional political allegiances
- Historical precedents (Finance Bill 2024 protests, Housing Levy disputes, SHIF rollout)

NARRATIVE ARCHETYPES (common in Kenya):
1. "Serikali vs Mwananchi" — Government vs ordinary people
2. "Tribal Allocation" — ethnic favoritism
3. "Cost of Living Crisis" — economic suffering
4. "Foreign Debt Trap" — IMF/World Bank narrative
5. "Youth Betrayal" — broken promises to Gen Z
6. "Security Failure" — government can't protect citizens
7. "Corruption Cycle" — every govt is same corrupt system
8. "Devolution Promise" — counties deserve more
9. "Digital Resistance" — KOT as accountability force
10. "Ethnic Persecution" — our tribe is targeted

Return JSON with your predictions.""".format(
    political_context=KENYA_POLITICAL_CONTEXT_2025,
    counties=KENYA_COUNTIES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Predictor Engine
# ═══════════════════════════════════════════════════════════════════════════

class PolicyPredictor:
    """Predict social impact of policy bills using LLM + evidence."""

    def __init__(self, provider):
        """
        Args:
            provider: OllamaProvider instance
        """
        self.provider = provider

    async def predict(
        self,
        bill,
        evidence=None,
    ) -> ImpactPrediction:
        """
        Generate full impact prediction for a bill.
        
        Args:
            bill: BillAnalysis from PolicyExtractor
            evidence: Optional SearchResults from PolicySearchEngine
            
        Returns:
            ImpactPrediction with county risks, timelines, narratives
        """
        # Build rich context for the LLM
        provisions_text = ""
        for i, p in enumerate(bill.provisions[:8], 1):
            provisions_text += (
                f"\n{i}. {p.clause_id}: {p.description} "
                f"(sector={p.sector}, severity={p.severity:.2f}, "
                f"affects={', '.join(p.affected_groups[:5])})"
            )

        evidence_text = ""
        if evidence:
            evidence_text = f"\n\nSUPPORTING EVIDENCE ({evidence.total_found} items found):\n"
            evidence_text += evidence.summary_text(max_items=15)

        # Historical match info
        hist_text = ""
        if evidence and evidence.policy_events:
            top_match = evidence.policy_events[0]
            hist_text = (
                f"\n\nCLOSEST HISTORICAL MATCH:\n"
                f"Event: {top_match.text}\n"
                f"Similarity: {top_match.similarity:.2f}\n"
                f"Severity: {top_match.metadata.get('severity', 'unknown')}\n"
                f"Sector: {top_match.metadata.get('sector', 'unknown')}"
            )

        prompt = f"""Predict the social impact of this Kenyan policy:

BILL: {bill.title}
SUMMARY: {bill.summary}
OVERALL SEVERITY: {bill.total_severity:.2f}
SECTORS: {', '.join(bill.sectors)}

PROVISIONS:{provisions_text}
{hist_text}
{evidence_text}

Predict the following (return JSON):
{{
  "overall_mobilization": 0.0-1.0,
  "overall_risk_level": "low" | "moderate" | "high" | "critical",
  "predicted_phase": "LEAK" | "ANNOUNCE" | "REACT" | "MOBILIZE" | "IMPLEMENT" | "IMPACT" | "SETTLE",
  "predicted_peak_timeline": "timeline description (e.g., '48-72h after announcement')",
  "top_risk_counties": ["county1", "county2", ...],
  "county_risks": {{"Nairobi": 0.9, "Mombasa": 0.7, ...}},
  "dominant_narratives": ["archetype1", "archetype2"],
  "likely_hashtags": ["#tag1", "#tag2"],
  "likely_mobilizers": ["youth", "unions", "opposition", etc.],
  "counter_narrative_suggestions": ["suggestion1", "suggestion2"],
  "historical_match": "closest historical event name",
  "historical_similarity": 0.0-1.0,
  "historical_outcome": "what happened with the historical event",
  "confidence": 0.0-1.0,
  "provision_impacts": [
    {{
      "clause_id": "Section X",
      "mobilization_probability": 0.0-1.0,
      "predicted_timeline": "24h" | "72h" | "7d" | "14d" | "30d",
      "risk_level": "low" | "moderate" | "high" | "critical",
      "likely_mobilizers": ["group1", "group2"],
      "narrative_archetypes": ["archetype1"],
      "counter_narratives": ["suggestion1"]
    }}
  ]
}}"""

        data = await self.provider._generate_json(
            prompt, PREDICTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )

        if not data:
            logger.warning("Prediction LLM returned empty — building minimal prediction")
            return ImpactPrediction(
                bill_title=bill.title,
                overall_mobilization=bill.total_severity * 0.7,
                overall_risk_level=_severity_to_risk(bill.total_severity),
                confidence=0.2,
            )

        return self._parse_prediction(data, bill, evidence)

    async def predict_provision(
        self,
        provision,
        bill_title: str = "",
        evidence_text: str = "",
    ) -> ProvisionImpact:
        """Predict impact for a single provision (drill-down)."""
        prompt = f"""Predict social impact for this specific policy provision:

BILL: {bill_title}
PROVISION: {provision.clause_id} — {provision.description}
SECTOR: {provision.sector}
SEVERITY: {provision.severity:.2f}
AFFECTED: {', '.join(provision.affected_groups)}
COUNTIES: {', '.join(provision.affected_counties)}
MONETARY: {provision.monetary_impact}

{evidence_text}

Return JSON:
{{
  "mobilization_probability": 0.0-1.0,
  "predicted_timeline": "24h" | "72h" | "7d" | "14d" | "30d",
  "risk_level": "low" | "moderate" | "high" | "critical",
  "likely_mobilizers": ["group1", "group2"],
  "narrative_archetypes": ["archetype1", "archetype2"],
  "counter_narratives": ["suggestion1", "suggestion2"]
}}"""

        data = await self.provider._generate_json(
            prompt, PREDICTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )

        return ProvisionImpact(
            clause_id=provision.clause_id,
            description=provision.description,
            mobilization_probability=_clamp(float(data.get("mobilization_probability", provision.severity * 0.7))),
            predicted_timeline=str(data.get("predicted_timeline", "7d")),
            risk_level=str(data.get("risk_level", _severity_to_risk(provision.severity))),
            likely_mobilizers=_ensure_list(data.get("likely_mobilizers", [])),
            narrative_archetypes=_ensure_list(data.get("narrative_archetypes", [])),
            counter_narratives=_ensure_list(data.get("counter_narratives", [])),
        )

    async def answer_question(
        self,
        question: str,
        bill,
        prediction: Optional[ImpactPrediction] = None,
        evidence=None,
    ) -> str:
        """
        Answer a follow-up question about a bill/prediction.
        
        Returns plain text response (for chat display).
        """
        context_parts = [f"BILL: {bill.title}", f"SUMMARY: {bill.summary}"]

        if prediction:
            context_parts.append(
                f"PREDICTION: mobilization={prediction.overall_mobilization:.2f}, "
                f"risk={prediction.overall_risk_level}, "
                f"peak={prediction.predicted_peak_timeline}"
            )
            if prediction.top_risk_counties:
                context_parts.append(
                    f"TOP RISK COUNTIES: {', '.join(prediction.top_risk_counties[:10])}"
                )
            if prediction.dominant_narratives:
                context_parts.append(
                    f"NARRATIVES: {', '.join(prediction.dominant_narratives)}"
                )
            if prediction.historical_match:
                context_parts.append(
                    f"HISTORICAL MATCH: {prediction.historical_match} "
                    f"(similarity={prediction.historical_similarity:.2f}) — "
                    f"{prediction.historical_outcome}"
                )

        if evidence:
            context_parts.append(
                f"\nEVIDENCE:\n{evidence.summary_text(max_items=8)}"
            )

        context = "\n".join(context_parts)

        system = f"""You are a Kenyan policy intelligence analyst chatbot.
You have analyzed a policy bill and have prediction data.
Answer the user's follow-up question using the context below.

Be specific to Kenya. Reference counties, demographics, historical events.
Use data from the evidence when available.
If asked about something outside your context, say so honestly.

{KENYA_POLITICAL_CONTEXT_2025}

CONTEXT:
{context}"""

        return await self.provider._generate_text(
            question, system, task=AnalysisTask.POLICY_IMPACT,
        )

    # ─── Parse LLM Output ───────────────────────────────────────────────

    def _parse_prediction(
        self,
        data: Dict[str, Any],
        bill,
        evidence,
    ) -> ImpactPrediction:
        """Parse LLM prediction JSON into ImpactPrediction."""
        # Parse provision impacts
        provision_impacts = []
        for pi in data.get("provision_impacts", []):
            provision_impacts.append(ProvisionImpact(
                clause_id=str(pi.get("clause_id", "")),
                description=str(pi.get("description", "")),
                mobilization_probability=_clamp(float(pi.get("mobilization_probability", 0.5))),
                predicted_timeline=str(pi.get("predicted_timeline", "7d")),
                risk_level=str(pi.get("risk_level", "moderate")),
                likely_mobilizers=_ensure_list(pi.get("likely_mobilizers", [])),
                narrative_archetypes=_ensure_list(pi.get("narrative_archetypes", [])),
                counter_narratives=_ensure_list(pi.get("counter_narratives", [])),
            ))

        # Parse county risks
        county_risks_raw = data.get("county_risks", {})
        county_risks = {}
        if isinstance(county_risks_raw, dict):
            for county, risk in county_risks_raw.items():
                try:
                    county_risks[str(county)] = _clamp(float(risk))
                except (ValueError, TypeError):
                    pass

        return ImpactPrediction(
            bill_title=bill.title,
            overall_mobilization=_clamp(float(data.get("overall_mobilization", bill.total_severity * 0.7))),
            overall_risk_level=str(data.get("overall_risk_level", _severity_to_risk(bill.total_severity))),
            predicted_phase=str(data.get("predicted_phase", "REACT")),
            predicted_peak_timeline=str(data.get("predicted_peak_timeline", "")),
            county_risks=county_risks,
            top_risk_counties=_ensure_list(data.get("top_risk_counties", [])),
            provision_impacts=provision_impacts,
            historical_match=str(data.get("historical_match", "")),
            historical_similarity=_clamp(float(data.get("historical_similarity", 0.0))),
            historical_outcome=str(data.get("historical_outcome", "")),
            dominant_narratives=_ensure_list(data.get("dominant_narratives", [])),
            likely_hashtags=_ensure_list(data.get("likely_hashtags", [])),
            likely_mobilizers=_ensure_list(data.get("likely_mobilizers", [])),
            counter_narrative_suggestions=_ensure_list(data.get("counter_narrative_suggestions", [])),
            evidence_count=evidence.total_found if evidence else 0,
            confidence=_clamp(float(data.get("confidence", 0.5))),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _ensure_list(val) -> list:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []


def _severity_to_risk(severity: float) -> str:
    if severity >= 0.8:
        return "critical"
    if severity >= 0.6:
        return "high"
    if severity >= 0.4:
        return "moderate"
    return "low"

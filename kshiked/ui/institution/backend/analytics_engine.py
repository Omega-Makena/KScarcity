"""
Executive Analytics Engine — Powers the 5 Analytical Pillars.

Pillar 1: "SO WHAT?"       — generate_inaction_projection()
Pillar 2: "COMPARED TO WHAT?" — get_historical_context()
Pillar 3: "WHERE EXACTLY?"    — build_county_convergence()
Pillar 4: "WHAT SHOULD I DO?" — generate_recommendation()
Pillar 5: "DID IT WORK?"      — compute_outcome_impact()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("analytics_engine")


# ═══════════════════════════════════════════════════════════════════════
#  PILLAR 1 — "SO WHAT?" — Projected consequence of inaction
# ═══════════════════════════════════════════════════════════════════════

_SEVERITY_THRESHOLD = 6.0  # Only run projection when severity exceeds this


def generate_inaction_projection(
    severity: float,
    shock_vector: Optional[Dict[str, Any]] = None,
    incident_type: str = "ANOMALY",
    composite_scores: Optional[Dict[str, float]] = None,
    projection_steps: int = 4,
) -> Optional[str]:
    """
    Run a no-intervention SFC scenario and return a plain-language
    consequence string.  Returns None if severity is below threshold.

    Parameters
    ----------
    severity : float
        The anomaly's severity score (0-10).
    shock_vector : dict, optional
        Pre-shock / post-shock values keyed by metric name.
    incident_type : str
        E.g. "ANOMALY", "THRESHOLD_BREACH", "CONVERGENCE".
    composite_scores : dict, optional
        A/B/C composite intelligence scores.
    projection_steps : int
        Number of SFC steps (≈ weeks/months) to project forward.
    """
    if severity < _SEVERITY_THRESHOLD:
        return None

    try:
        from scarcity.simulation.research_sfc import (
            ResearchSFCConfig,
            ResearchSFCEconomy,
        )

        # Build a shock vector from the anomaly payload
        sfc_shocks: Dict[str, Any] = {}
        magnitude = (severity / 10.0) * 0.08  # Scale: sev 10 → 8% shock

        if shock_vector:
            # Use actual shock data if available
            for metric, vals in shock_vector.items():
                if isinstance(vals, dict):
                    delta = abs(vals.get("peak_shock_value", 0) - vals.get("pre_shock_baseline", 0))
                    norm = max(abs(vals.get("pre_shock_baseline", 1)), 0.01)
                    sfc_shocks["supply_shock"] = delta / norm * 0.5
                else:
                    sfc_shocks["supply_shock"] = magnitude
        else:
            sfc_shocks["supply_shock"] = magnitude

        # Run the no-intervention scenario
        cfg = ResearchSFCConfig()
        cfg.sfc.steps = projection_steps + 5
        engine = ResearchSFCEconomy(cfg)
        engine.initialize(gdp=100.0)

        # Inject shock at step 1
        for key, val in sfc_shocks.items():
            if engine.economy.config.shock_vectors is None:
                engine.economy.config.shock_vectors = {}
            vec = np.zeros(cfg.sfc.steps)
            vec[1] = val
            engine.economy.config.shock_vectors[key] = vec

        # Run
        engine.run(projection_steps)
        final = engine.summary()

        # Extract key outcomes
        gdp_growth = final.get("gdp_growth", 0.0)
        unemployment = final.get("unemployment", 0.0)
        inflation = final.get("inflation", 0.0)
        fin = final.get("financial", {})
        ext = final.get("external", {})
        fsi = engine.financial_stability_index()
        evi = engine.external_vulnerability_index()

        # Generate consequence narrative
        parts = []

        if gdp_growth < 0.01:
            credit_growth_pct = fin.get("credit_spread", 0) * 100
            parts.append(
                f"the model projects GDP growth falling to {gdp_growth*100:.1f}%, "
                f"which historically triggers employment contraction"
            )
        elif gdp_growth < 0.02:
            parts.append(
                f"GDP growth decelerates to {gdp_growth*100:.1f}%, "
                f"approaching stagnation territory"
            )

        if fsi < 0.4:
            npl = fin.get("npl_ratio", 0)
            parts.append(
                f"financial stability drops to {fsi:.0%} "
                f"(NPL ratio rises to {npl*100:.1f}%)"
            )

        if evi > 0.6:
            reserves = ext.get("reserves_months", 0)
            parts.append(
                f"external vulnerability reaches {evi:.0%} "
                f"with reserves at {reserves:.1f} months of import cover"
            )

        if unemployment > 0.12:
            parts.append(
                f"unemployment rises to {unemployment*100:.1f}%"
            )

        if inflation > 0.08:
            parts.append(
                f"inflation accelerates to {inflation*100:.1f}%"
            )

        if not parts:
            # Mild consequences
            return (
                f"If this trend continues for {projection_steps} more periods, "
                f"the model projects moderate stress (GDP growth {gdp_growth*100:.1f}%, "
                f"stability index {fsi:.0%}) but no critical threshold breach."
            )

        consequence = "; ".join(parts)
        return (
            f"If this trend continues for {projection_steps} more weeks, "
            f"{consequence}."
        )

    except Exception as e:
        logger.warning(f"SFC projection failed: {e}")
        return (
            f"Projection unavailable (engine error). "
            f"Based on severity {severity:.1f}/10, manual review is strongly recommended."
        )


# ═══════════════════════════════════════════════════════════════════════
#  PILLAR 2 — "COMPARED TO WHAT?" — Historical context
# ═══════════════════════════════════════════════════════════════════════

def get_historical_context(
    basket_id: int,
    severity: float,
    incident_type: str = "ANOMALY",
    indicator_name: Optional[str] = None,
) -> str:
    """
    Compare an anomaly against the historical archive and institutional
    memory.  Returns a plain-text comparison string.
    """
    from .delta_sync import DeltaSyncManager
    from .project_manager import ProjectManager

    # 1. Query historical syncs for this basket
    historical = DeltaSyncManager.get_historical_syncs(basket_id)
    memories = ProjectManager.get_institutional_memory()

    if not historical and not memories:
        return "No comparable pattern in the archive."

    parts = []

    # 2. Find similar past events (same basket, similar severity)
    similar_events = []
    for h in historical:
        h_payload = h.get("payload", {})
        h_severity = h_payload.get("severity_score", 0)
        h_type = h_payload.get("incident_type", "ANOMALY")

        # Match on same type or close severity (within ±2.0)
        if h_type == incident_type or abs(h_severity - severity) < 2.0:
            similar_events.append(h)

    if similar_events:
        # Severity ranking
        all_severities = [
            e["payload"].get("severity_score", 0) for e in historical
        ]
        rank = sum(1 for s in all_severities if s >= severity)
        total = len(all_severities)

        if rank <= 1:
            # Calculate how many months of history we have
            timestamps = [e.get("timestamp", 0) for e in historical if e.get("timestamp")]
            if timestamps:
                oldest = min(timestamps)
                months = max(1, int((time.time() - oldest) / (30 * 86400)))
                parts.append(
                    f"This is the highest severity for this indicator in {months} month(s)"
                )
        elif rank <= 3:
            parts.append(
                f"This ranks #{rank} out of {total} recorded events for this sector"
            )

        # Find most similar past event
        best_match = max(
            similar_events,
            key=lambda e: -abs(e["payload"].get("severity_score", 0) - severity)
        )
        match_ts = best_match.get("timestamp", 0)
        if match_ts:
            import datetime
            match_date = datetime.datetime.fromtimestamp(match_ts).strftime("%Y-%m-%d")
            match_status = best_match.get("status", "PROCESSED")
            parts.append(
                f"Similar pattern detected on {match_date} — "
                f"that event was {match_status.lower()}"
            )

    # 3. Check institutional memory for related closed projects
    if memories:
        for mem in memories:
            # Check if this memory's severity is close
            if abs(mem.get("severity", 0) - severity) < 2.5:
                res_state = mem.get("resolution_state", "UNKNOWN")
                pol_score = mem.get("policy_effectiveness_score", 0)
                parts.append(
                    f'Related project "{mem["title"]}" was resolved as '
                    f"{res_state} with policy effectiveness {pol_score}/10"
                )
                break  # Only show the most relevant one

    if not parts:
        return "No comparable pattern in the archive."

    return ". ".join(parts) + "."


# ═══════════════════════════════════════════════════════════════════════
#  PILLAR 3 — "WHERE EXACTLY?" — Geographic specificity
# ═══════════════════════════════════════════════════════════════════════

# Kenya county centroids for mapping (subset of key counties)
_COUNTY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "mombasa": (-4.0435, 39.6682), "kwale": (-4.1816, 39.4521),
    "kilifi": (-3.5107, 39.9093), "tana river": (-1.7750, 40.0300),
    "lamu": (-2.2717, 40.9017), "taita taveta": (-3.3961, 38.4854),
    "garissa": (-0.4532, 39.6461), "wajir": (1.7471, 40.0573),
    "mandera": (3.9373, 41.8569), "marsabit": (2.3284, 37.9908),
    "isiolo": (0.3546, 37.5822), "meru": (0.0480, 37.6559),
    "tharaka nithi": (-0.3074, 37.8608), "embu": (-0.5389, 37.4596),
    "kitui": (-1.3668, 38.0106), "machakos": (-1.5177, 37.2634),
    "makueni": (-1.8039, 37.6200), "nyandarua": (-0.1804, 36.5230),
    "nyeri": (-0.4197, 36.9510), "kirinyaga": (-0.5300, 37.2800),
    "murang'a": (-0.7839, 37.0400), "kiambu": (-1.1714, 36.8350),
    "turkana": (3.1122, 35.5897), "west pokot": (1.6219, 35.1195),
    "samburu": (1.0, 36.9), "trans nzoia": (1.0000, 34.9500),
    "uasin gishu": (0.5143, 35.2698), "elgeyo marakwet": (0.7200, 35.5100),
    "nandi": (0.1833, 35.1269), "baringo": (0.4912, 35.9720),
    "laikipia": (0.3606, 36.7819), "nakuru": (-0.3031, 36.0800),
    "narok": (-1.0877, 35.8660), "kajiado": (-2.0981, 36.7820),
    "kericho": (-0.3692, 35.2863), "bomet": (-0.7827, 35.3428),
    "kakamega": (0.2827, 34.7519), "vihiga": (0.0833, 34.7167),
    "bungoma": (0.5695, 34.5584), "busia": (0.4608, 34.1115),
    "siaya": (-0.0617, 34.2422), "kisumu": (-0.0917, 34.7680),
    "homa bay": (-0.5273, 34.4571), "migori": (-1.0634, 34.4731),
    "kisii": (-0.6817, 34.7667), "nyamira": (-0.5633, 34.9350),
    "nairobi": (-1.2921, 36.8219),
}


def build_county_convergence(
    global_risks: List[Dict[str, Any]],
    all_baskets: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate promoted risks into per-county convergence scores.

    Returns a dict: county_name → {score, risk_count, sources, has_data}.
    If no geographic data is found, returns empty dict.
    """
    county_data: Dict[str, Dict[str, Any]] = {}

    if not global_risks:
        return county_data

    for risk in global_risks:
        payload_scores = risk.get("composite_scores", {})
        impact = payload_scores.get("B_Impact", 0)
        detection = payload_scores.get("A_Detection", 0)
        basket_name = all_baskets.get(risk.get("basket_id"), "Unknown")

        # Try to extract geographic metadata from the risk
        # Source sync payloads may contain county/region info
        source_ids = risk.get("source_sync_ids", [])

        # Check if payload has spatial data
        description = risk.get("description", "").lower()
        title = risk.get("title", "").lower()

        # Try to match counties from risk text
        matched_counties = []
        for county_name in _COUNTY_CENTROIDS:
            if county_name in description or county_name in title:
                matched_counties.append(county_name)

        # If no county detected in text, assign to the sector's
        # "headquarters" based on basket/sector name
        if not matched_counties:
            sector_lower = basket_name.lower()
            for county_name in _COUNTY_CENTROIDS:
                if county_name in sector_lower:
                    matched_counties.append(county_name)

        # If still no match, mark as unmapped but counted
        if not matched_counties:
            # Distribute across Nairobi as the default administrative center
            matched_counties = ["nairobi"]

        # Distribute the risk score across matched counties
        per_county_score = (impact + detection) / 2.0 / len(matched_counties)

        for county in matched_counties:
            if county not in county_data:
                county_data[county] = {
                    "score": 0.0,
                    "risk_count": 0,
                    "sources": [],
                    "has_data": True,
                }
            county_data[county]["score"] += per_county_score
            county_data[county]["risk_count"] += 1
            county_data[county]["sources"].append(basket_name)

    # Normalize scores to 0-100 range
    if county_data:
        max_score = max(c["score"] for c in county_data.values())
        if max_score > 0:
            for c in county_data.values():
                c["score"] = (c["score"] / max_score) * 100

    return county_data


def get_county_centroid(county_name: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a county, or None."""
    return _COUNTY_CENTROIDS.get(county_name.lower())


# ═══════════════════════════════════════════════════════════════════════
#  PILLAR 4 — "WHAT SHOULD I DO?" — Coordination Trigger
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoordinationRecommendation:
    """Structured recommendation output."""
    level: str            # WATCH | REVIEW | COORDINATE | MITIGATE
    level_color: str      # CSS color
    who: List[str]        # Which institutions/baskets should meet
    what: List[str]       # Specific indicators aligning
    urgency: str          # Based on persistence
    persistence: int      # Consecutive windows the pattern has held
    summary: str          # One-sentence recommendation


def generate_recommendation(
    risk: Dict[str, Any],
    all_baskets: Dict[int, str],
    global_risks: List[Dict[str, Any]],
    historical_syncs: Optional[List[Dict[str, Any]]] = None,
) -> CoordinationRecommendation:
    """
    Generate a concrete coordination recommendation based on cross-basket
    convergence and persistence.
    """
    scores = risk.get("composite_scores", {})
    severity = scores.get("B_Impact", 0)
    detection = scores.get("A_Detection", 0)
    certainty = scores.get("C_Certainty", 0)
    basket_id = risk.get("basket_id")
    basket_name = all_baskets.get(basket_id, f"Sector {basket_id}")

    # Calculate convergence: how many baskets have similar active risks
    active_basket_ids = set()
    converging_indicators = []
    for r in global_risks:
        r_scores = r.get("composite_scores", {})
        r_impact = r_scores.get("B_Impact", 0)
        if r_impact > 4.0:  # Only count significant risks
            active_basket_ids.add(r.get("basket_id"))
            converging_indicators.append(r.get("title", "Unknown"))

    convergence_count = len(active_basket_ids)

    # Calculate persistence from historical data
    persistence = 0
    if historical_syncs:
        # Count consecutive recent events with similar severity
        recent = sorted(historical_syncs, key=lambda x: x.get("timestamp", 0), reverse=True)
        for h in recent[:10]:
            h_sev = h.get("payload", {}).get("severity_score", 0)
            if h_sev > severity * 0.6:
                persistence += 1
            else:
                break

    # Determine recommendation level
    composite = severity * 0.4 + detection * 0.2 + certainty * 0.2 + convergence_count * 0.5

    if composite > 9.0 or (severity > 8.0 and convergence_count >= 3):
        level = "MITIGATE"
        level_color = "#DC2626"
        urgency = "IMMEDIATE — respond within 24 hours"
    elif composite > 6.5 or (severity > 6.0 and convergence_count >= 2):
        level = "COORDINATE"
        level_color = "#F59E0B"
        urgency = f"HIGH — convene within 48 hours (pattern held for {persistence} windows)"
    elif composite > 4.0 or severity > 5.0:
        level = "REVIEW"
        level_color = "#3B82F6"
        urgency = f"MODERATE — schedule discussion this week (persistence: {persistence})"
    else:
        level = "WATCH"
        level_color = "#10B981"
        urgency = "LOW — continue monitoring, no immediate action required"

    # WHO should meet
    who = [all_baskets.get(bid, f"Sector {bid}") for bid in active_basket_ids]
    if not who:
        who = [basket_name]

    # WHAT to discuss
    what = list(set(converging_indicators))[:5]  # Cap at 5
    if not what:
        what = [risk.get("title", "Unclassified anomaly")]

    # Summary sentence
    if level == "MITIGATE":
        summary = (
            f"Immediate cross-sector response required. "
            f"{convergence_count} sectors reporting aligned threats. "
            f"Convene {', '.join(who[:3])} for emergency coordination."
        )
    elif level == "COORDINATE":
        summary = (
            f"Cross-sector convergence detected across {convergence_count} sector(s). "
            f"Schedule joint review between {', '.join(who[:3])} "
            f"to assess interdependencies."
        )
    elif level == "REVIEW":
        summary = (
            f"Elevated activity in {basket_name}. "
            f"Review the aligning indicators and assess whether coordination is needed."
        )
    else:
        summary = (
            f"Monitor {basket_name} — current signals are within normal parameters. "
            f"Re-evaluate if persistence exceeds 3 consecutive windows."
        )

    return CoordinationRecommendation(
        level=level,
        level_color=level_color,
        who=who,
        what=what,
        urgency=urgency,
        persistence=persistence,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PILLAR 5 — "DID IT WORK?" — Closed-loop outcome tracking
# ═══════════════════════════════════════════════════════════════════════

def compute_outcome_impact(
    project_id: int,
    project_created_at: float,
    project_archived_at: float,
    participant_basket_ids: List[int],
) -> Dict[str, Any]:
    """
    Compute before/after signal comparison when a project is archived.

    Compares average anomaly severity and convergence BEFORE the project
    was created vs. AFTER it was archived.
    """
    from .delta_sync import DeltaSyncManager

    result: Dict[str, Any] = {
        "has_data": False,
        "narrative": "",
        "before_avg_severity": 0.0,
        "after_avg_severity": 0.0,
        "before_count": 0,
        "after_count": 0,
        "delta_pct": 0.0,
    }

    # Collect signals from all participating baskets
    all_before: List[float] = []
    all_after: List[float] = []

    for basket_id in participant_basket_ids:
        historical = DeltaSyncManager.get_historical_syncs(basket_id)

        for event in historical:
            ts = event.get("timestamp", 0)
            sev = event.get("payload", {}).get("severity_score", 0)

            if ts < project_created_at:
                all_before.append(sev)
            elif ts > project_archived_at:
                all_after.append(sev)

    if not all_before:
        result["narrative"] = (
            "Insufficient historical data before this project to compute impact. "
            "The archive does not contain pre-project signal baselines."
        )
        return result

    before_avg = np.mean(all_before) if all_before else 0.0
    after_avg = np.mean(all_after) if all_after else 0.0
    before_count = len(all_before)
    after_count = len(all_after)

    result["has_data"] = True
    result["before_avg_severity"] = round(float(before_avg), 2)
    result["after_avg_severity"] = round(float(after_avg), 2)
    result["before_count"] = before_count
    result["after_count"] = after_count

    if before_avg > 0:
        delta_pct = ((after_avg - before_avg) / before_avg) * 100
        result["delta_pct"] = round(delta_pct, 1)
    else:
        delta_pct = 0.0

    # Generate the narrative
    if after_count == 0:
        result["narrative"] = (
            f"Before this project, the average severity across participating sectors "
            f"was {before_avg:.1f}/10 ({before_count} signals). "
            f"No post-project signals have been recorded yet — too early to assess impact."
        )
    elif delta_pct < -10:
        result["narrative"] = (
            f"During this project, average severity dropped from "
            f"{before_avg:.1f}/10 to {after_avg:.1f}/10 "
            f"({abs(delta_pct):.0f}% improvement, {after_count} post-project signals). "
            f"The intervention appears to have reduced systemic stress."
        )
    elif delta_pct > 10:
        result["narrative"] = (
            f"The indicators that triggered this project are still elevated. "
            f"Average severity rose from {before_avg:.1f}/10 to {after_avg:.1f}/10 "
            f"({delta_pct:.0f}% increase). "
            f"Consider reopening or launching a follow-up project."
        )
    else:
        result["narrative"] = (
            f"Average severity remained stable: "
            f"{before_avg:.1f}/10 before → {after_avg:.1f}/10 after "
            f"(Δ {delta_pct:+.0f}%). "
            f"The situation has neither improved nor deteriorated significantly."
        )

    return result

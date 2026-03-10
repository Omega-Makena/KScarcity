"""
Report Narrator — Translates technical scores and data into
plain-language explanations for decision-makers who are not
data scientists.

Used by all three dashboard levels (Spoke, Admin, Executive).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
#  Composite Intelligence Scores — A / B / C
# ═══════════════════════════════════════════════════════════════════════

def narrate_composite_scores(scores: Dict[str, float]) -> str:
    """
    Turn A_Detection / B_Impact / C_Certainty into a paragraph that
    a Cabinet Secretary or County Governor can read.
    """
    detection = scores.get("A_Detection", 0)
    impact = scores.get("B_Impact", 0)
    certainty = scores.get("C_Certainty", 0)

    parts = []

    # Detection
    if detection >= 8:
        parts.append(
            "The system detected a **strong, unmistakable signal** — the data "
            "deviation is large enough that it is very unlikely to be coincidence"
        )
    elif detection >= 5:
        parts.append(
            "A **moderate signal** was detected — the pattern is notable but "
            "could still reflect normal variation in some circumstances"
        )
    else:
        parts.append(
            "The signal is **weak** — the data shows a small deviation that "
            "may or may not represent a real problem"
        )

    # Impact
    if impact >= 8:
        parts.append(
            "If this pattern is real, its potential **impact is severe** — it could "
            "affect employment, credit markets, or public welfare at a national level"
        )
    elif impact >= 5:
        parts.append(
            "The estimated **impact is moderate** — expect sector-level disruptions "
            "that may require targeted intervention"
        )
    else:
        parts.append(
            "The estimated **impact is limited** — effects are likely contained "
            "within a single sub-sector or institution"
        )

    # Certainty
    if certainty >= 8:
        parts.append(
            "We have **high confidence** in these findings — multiple independent "
            "data sources and analytical methods agree"
        )
    elif certainty >= 5:
        parts.append(
            "Confidence is **moderate** — some analytical methods agree but "
            "the picture is not yet fully clear"
        )
    else:
        parts.append(
            "Confidence is **low** — the data is limited or contradictory. "
            "Treat these findings as preliminary"
        )

    return ". ".join(parts) + "."


# ═══════════════════════════════════════════════════════════════════════
#  Severity Score
# ═══════════════════════════════════════════════════════════════════════

def narrate_severity(score: float) -> str:
    """Plain-language explanation of what a severity score means."""
    if score >= 8.0:
        return (
            f"**Severity {score:.1f}/10 — Critical.** "
            "This event requires immediate attention from senior leadership. "
            "Historically, events at this level have led to measurable economic disruption within weeks."
        )
    elif score >= 6.0:
        return (
            f"**Severity {score:.1f}/10 — High.** "
            "This is above normal operating risk. It warrants scheduled review "
            "within the next few days, and monitoring should be increased."
        )
    elif score >= 4.0:
        return (
            f"**Severity {score:.1f}/10 — Elevated.** "
            "This is mildly above normal. Worth noting but not yet alarming. "
            "Continue routine monitoring and flag if it persists."
        )
    elif score >= 2.0:
        return (
            f"**Severity {score:.1f}/10 — Low.** "
            "This is within the range of normal fluctuations. "
            "No specific action is required at this time."
        )
    else:
        return (
            f"**Severity {score:.1f}/10 — Minimal.** "
            "No meaningful deviation from expected patterns."
        )


# ═══════════════════════════════════════════════════════════════════════
#  Threat Level
# ═══════════════════════════════════════════════════════════════════════

def narrate_threat_level(level: str) -> str:
    """What does each threat level actually mean for decision-makers?"""
    explanations = {
        "CRITICAL": (
            "**CRITICAL** — Multiple severe anomalies are active simultaneously. "
            "The system recommends convening an emergency coordination meeting. "
            "Inaction at this level has historically led to cascading failures."
        ),
        "HIGH": (
            "**HIGH** — Significant anomalies detected that could escalate. "
            "Senior officials should be briefed. Prepare contingency responses "
            "and increase monitoring frequency."
        ),
        "ELEVATED": (
            "🟡 **ELEVATED** — Notable signals warrant increased attention. "
            "Schedule a review meeting this week. Ensure reporting channels are open."
        ),
        "GUARDED": (
            "**GUARDED** — Minor signals present but within acceptable bounds. "
            "Routine monitoring continues. No special action needed."
        ),
        "LOW": (
            "**LOW** — All indicators are within the normal range. "
            "Standard operations continue."
        ),
    }
    return explanations.get(level, f"**{level}** — Threat level assessed by the analysis engine.")


# ═══════════════════════════════════════════════════════════════════════
#  Shock Vector — Plain language
# ═══════════════════════════════════════════════════════════════════════

def narrate_shock_vector(shock_vector: Dict[str, Any]) -> str:
    """
    Turn a shock_vector dict into a plain description of what changed.
    """
    if not shock_vector:
        return "No specific indicator movement data is available for this event."

    parts = []
    for metric, vals in shock_vector.items():
        readable_name = metric.replace("_", " ").title()

        if isinstance(vals, dict):
            pre = vals.get("pre_shock_baseline", 0)
            post = vals.get("peak_shock_value", 0)

            if pre == 0 and post == 0:
                continue

            direction = "increased" if post > pre else "decreased"
            if pre != 0:
                pct_change = abs((post - pre) / pre) * 100
            else:
                pct_change = 0

            if pct_change > 50:
                magnitude = "dramatically"
            elif pct_change > 20:
                magnitude = "significantly"
            elif pct_change > 5:
                magnitude = "moderately"
            else:
                magnitude = "slightly"

            parts.append(
                f"**{readable_name}** {magnitude} {direction} "
                f"(from {pre:.2f} to {post:.2f}, a {pct_change:.0f}% change)"
            )
        elif isinstance(vals, (int, float)):
            if abs(vals) > 0.1:
                parts.append(f"**{readable_name}** shifted by {vals:+.2f}")

    if not parts:
        return "The detected changes were small — within normal operating bounds."

    return "**What changed:** " + ". ".join(parts) + "."


# ═══════════════════════════════════════════════════════════════════════
#  Economic State — Plain language
# ═══════════════════════════════════════════════════════════════════════

def narrate_economic_state(es: Dict[str, Any]) -> str:
    """
    Turn raw SFC economic state numbers into a paragraph a
    non-economist can understand.
    """
    if not es:
        return ""

    gdp_g = es.get("gdp_growth", 0)
    inflation = es.get("inflation", 0)
    unemployment = es.get("unemployment", 0)
    rate = es.get("interest_rate", 0)
    fin_stab = es.get("financial_stability", 0.5)
    output_gap = es.get("output_gap", 0)

    parts = []

    # GDP
    if gdp_g > 0.05:
        parts.append(f"The economy is growing strongly at {gdp_g:.1%} per period")
    elif gdp_g > 0.02:
        parts.append(f"Economic growth is moderate at {gdp_g:.1%}")
    elif gdp_g > 0:
        parts.append(f"Growth is sluggish at {gdp_g:.1%} — near stagnation")
    else:
        parts.append(f"The economy is **contracting** at {gdp_g:.1%} — this is a recessionary signal")

    # Inflation
    if inflation > 0.08:
        parts.append(f"inflation is high at {inflation:.1%}, eroding purchasing power")
    elif inflation > 0.04:
        parts.append(f"inflation is elevated at {inflation:.1%}")
    else:
        parts.append(f"inflation is contained at {inflation:.1%}")

    # Unemployment
    if unemployment > 0.15:
        parts.append(f"unemployment stands at a concerning {unemployment:.1%}")
    elif unemployment > 0.08:
        parts.append(f"unemployment is {unemployment:.1%}")
    else:
        parts.append(f"employment conditions are stable ({unemployment:.1%} unemployment)")

    # Financial stability
    if fin_stab < 0.3:
        parts.append("the financial system is under **severe stress**")
    elif fin_stab < 0.5:
        parts.append("financial stability is below comfortable levels")
    else:
        parts.append("the financial system appears stable")

    summary = ". ".join(parts).capitalize() + "."

    # Output gap context
    if abs(output_gap) > 0.03:
        if output_gap > 0:
            summary += (
                f" The economy is running **above** its sustainable capacity "
                f"(output gap: +{output_gap:.1%}), which typically leads to "
                f"rising prices and potential overheating."
            )
        else:
            summary += (
                f" The economy has **spare capacity** "
                f"(output gap: {output_gap:.1%}), meaning resources "
                f"(workers, factories) are underutilized."
            )

    return summary


# ═══════════════════════════════════════════════════════════════════════
#  Full Risk Narrative — For Admin/Executive views
# ═══════════════════════════════════════════════════════════════════════

def narrate_risk_for_executive(
    title: str,
    description: str,
    composite_scores: Dict[str, float],
    severity: float,
    sector_name: str,
    threat_level: str = "",
) -> str:
    """
    Generate a 2-3 sentence executive-ready paragraph that summarizes
    a risk without ANY jargon.
    """
    impact = composite_scores.get("B_Impact", 0)
    detection = composite_scores.get("A_Detection", 0)
    certainty = composite_scores.get("C_Certainty", 0)

    # Severity verdict
    if severity >= 8:
        verdict = "a **critical-level threat**"
        action = "Immediate executive attention is recommended"
    elif severity >= 6:
        verdict = "a **high-priority concern**"
        action = "This should be reviewed by senior leadership within 48 hours"
    elif severity >= 4:
        verdict = "a **notable development**"
        action = "Continue monitoring and escalate if the pattern persists"
    else:
        verdict = "a **routine observation**"
        action = "No specific action is required at this time"

    # Confidence qualifier
    if certainty >= 7:
        confidence_q = "with high analytical confidence"
    elif certainty >= 4:
        confidence_q = "with moderate analytical confidence"
    else:
        confidence_q = "though analytical confidence is low — treat as preliminary"

    # Build narrative
    narrative = (
        f"The **{sector_name}** sector has reported {verdict} "
        f"related to *{title}*. "
    )

    if description and len(description) > 10:
        # Use first sentence of description if available
        first_sentence = description.split(".")[0].strip()
        if first_sentence:
            narrative += f"{first_sentence}. "

    narrative += (
        f"The automated assessment rates this event at "
        f"severity **{severity:.1f}/10** {confidence_q}. "
        f"{action}."
    )

    return narrative


# ═══════════════════════════════════════════════════════════════════════
#  Threat Index — Plain language explanations
# ═══════════════════════════════════════════════════════════════════════

_THREAT_INDEX_EXPLANATIONS = {
    "polarization": (
        "**What this measures:** How divided public opinion is on key issues. "
        "High polarization means opposing groups are hardening their positions."
    ),
    "legitimacy_erosion": (
        "**What this measures:** Whether trust in institutions (government, banks, courts) "
        "is declining. Falling legitimacy signals governance risk."
    ),
    "mobilization_readiness": (
        "**What this measures:** Whether social media and communication patterns suggest "
        "people are organizing for protests or collective action."
    ),
    "elite_cohesion": (
        "**What this measures:** Whether leadership and elite groups are aligned. "
        "Low cohesion means internal disagreements that could lead to policy paralysis."
    ),
    "information_warfare": (
        "**What this measures:** The volume and intensity of deliberate misinformation "
        "and coordinated narratives designed to manipulate public perception."
    ),
    "security_friction": (
        "**What this measures:** Incidents of violence, policing tension, or security "
        "force deployment. Rising friction signals escalation risk."
    ),
    "economic_cascade": (
        "**What this measures:** The likelihood that a single economic shock (e.g., bank failure, "
        "supply chain disruption) could trigger chain-reaction failures across sectors."
    ),
    "ethnic_tension": (
        "**What this measures:** Inter-community tension indicators. A proxy for "
        "identity-based conflict risk based on communication and incident patterns."
    ),
}


def get_threat_index_explanation(key: str) -> str:
    """Return a plain-language explanation for a threat index."""
    return _THREAT_INDEX_EXPLANATIONS.get(key, "")


# ═══════════════════════════════════════════════════════════════════════
#  Module-by-Module Explanations (for spoke level)
# ═══════════════════════════════════════════════════════════════════════

def narrate_anomaly_detection(peak_score: float, structural_breaks: List[int]) -> str:
    """Plain-language summary of anomaly detection results."""
    if peak_score >= 2.0:
        narrative = (
            f"**The system found a major anomaly** (score: {peak_score:.1f}). "
            "This means something in your data changed sharply and unexpectedly — "
            "it's far outside the normal pattern. "
            "Think of it like a temperature reading that suddenly spikes: "
            "it doesn't tell you *why*, but it tells you something happened."
        )
    elif peak_score >= 1.0:
        narrative = (
            f"**A moderate anomaly detected** (score: {peak_score:.1f}). "
            "Your data shows an unusual shift. It's not dramatic, but it's enough "
            "to warrant attention. Monitor whether the trend continues."
        )
    elif peak_score > 0.3:
        narrative = (
            f"🟡 **Minor deviation detected** (score: {peak_score:.1f}). "
            "The data shows small deviations from the expected pattern. "
            "This is common and usually resolves on its own."
        )
    else:
        narrative = (
            f"**Your data looks normal** (score: {peak_score:.1f}). "
            "No meaningful anomalies were detected."
        )

    if structural_breaks:
        n = len(structural_breaks)
        narrative += (
            f"\n\nAdditionally, **{n} structural break{'s' if n > 1 else ''}** "
            f"{'were' if n > 1 else 'was'} detected. This means the underlying "
            f"pattern of your data changed fundamentally at these points — "
            f"like a new 'normal' was established."
        )

    return narrative


def narrate_trend_analysis(trend_signals: List[Dict[str, Any]]) -> str:
    """Plain-language summary of trend results."""
    if not trend_signals:
        return "Not enough data to detect trends. Upload more rows for better analysis."

    accelerating = []
    decelerating = []
    volatile = []

    for t in trend_signals:
        col = t.get("column", "?")
        direction = t.get("direction", "stable")
        vol = t.get("volatility", "stable")

        if direction == "acceleration":
            accelerating.append(col)
        elif direction == "deceleration":
            decelerating.append(col)
        if "increasing" in vol:
            volatile.append(col)

    parts = []
    if accelerating:
        parts.append(
            f"📈 **{', '.join(accelerating[:3])}** "
            f"{'are' if len(accelerating) > 1 else 'is'} accelerating — "
            f"the rate of change is getting faster, not just moving up"
        )
    if decelerating:
        parts.append(
            f"📉 **{', '.join(decelerating[:3])}** "
            f"{'are' if len(decelerating) > 1 else 'is'} slowing down"
        )
    if volatile:
        parts.append(
            f"⚡ **{', '.join(volatile[:3])}** "
            f"{'are' if len(volatile) > 1 else 'is'} becoming more unpredictable — "
            f"swinging wider from period to period"
        )

    if not parts:
        return "All tracked variables are following stable trends — no acceleration or unusual volatility."

    return "**Trend Summary:** " + ". ".join(parts) + "."


def narrate_propagation_chain(chain: Dict[str, Any]) -> str:
    """Explain a risk propagation chain in plain English."""
    desc = chain.get("description", "")
    trigger = chain.get("trigger", "Unknown")
    impact_est = chain.get("estimated_impact", 0)

    if impact_est > 0.5:
        severity_word = "major"
    elif impact_est > 0.2:
        severity_word = "moderate"
    else:
        severity_word = "minor"

    return (
        f"**Domino effect identified:** A shock starting from *{trigger}* "
        f"could cause a {severity_word} chain reaction. "
        f"In plain terms: {desc}. "
        f"The estimated total ripple effect is **{impact_est:.0%}** of the original shock."
    )

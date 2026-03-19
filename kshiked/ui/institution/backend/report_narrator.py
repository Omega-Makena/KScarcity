"""
Report Narrator — Translates technical scores and data into
plain-language explanations for decision-makers who are not
data scientists.

Used by all three dashboard levels (Spoke, Admin, Executive).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


# ═══════════════════════════════════════════════════════════════════════
# Composite Intelligence Scores — A / B / C
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
      "disrupt core services, operational throughput, or community stability"
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
# Severity Score
# ═══════════════════════════════════════════════════════════════════════

def narrate_severity(score: float) -> str:
  """Plain-language explanation of what a severity score means."""
  if score >= 8.0:
    return (
      f"**Severity {score:.1f}/10 — Critical.** "
      "This event requires immediate attention from senior leadership. "
      "Historically, events at this level can trigger measurable system-wide disruption within weeks."
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
# Threat Level
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
      " **ELEVATED** — Notable signals warrant increased attention. "
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
# Shock Vector — Plain language
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
# Economic State — Plain language
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
# Full Risk Narrative — For Admin/Executive views
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
# Threat Index — Plain language explanations
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
# Module-by-Module Explanations (for spoke level)
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
      f" **Minor deviation detected** (score: {peak_score:.1f}). "
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
      f" **{', '.join(accelerating[:3])}** "
      f"{'are' if len(accelerating) > 1 else 'is'} accelerating — "
      f"the rate of change is getting faster, not just moving up"
    )
  if decelerating:
    parts.append(
      f" **{', '.join(decelerating[:3])}** "
      f"{'are' if len(decelerating) > 1 else 'is'} slowing down"
    )
  if volatile:
    parts.append(
      f" **{', '.join(volatile[:3])}** "
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


def narrate_anomaly_chart_stats(
  anomaly_scores: List[float],
  peak_score: float,
  peak_index: int,
  structural_breaks: List[int],
) -> str:
  """Explain the anomaly line chart using dynamic counts and timing."""
  if not anomaly_scores:
    return "No anomaly score timeline is available yet. Upload more numeric data to enable this view."

  total = len(anomaly_scores)
  high = sum(1 for s in anomaly_scores if s >= 2.0)
  moderate = sum(1 for s in anomaly_scores if 1.0 <= s < 2.0)
  high_pct = (high / total) * 100

  if peak_index <= total * 0.33:
    timing = "early"
  elif peak_index <= total * 0.66:
    timing = "midway"
  else:
    timing = "late"

  break_text = (
    f"The engine also marked {len(structural_breaks)} structural break(s), meaning the baseline pattern changed."
    if structural_breaks
    else "No structural breaks were detected, so the baseline pattern stayed relatively stable."
  )

  return (
    f"Across {total} observations, {high} point(s) were severe and {moderate} were moderate. "
    f"The highest spike ({peak_score:.2f}) happened {timing} in the series around point {peak_index}. "
    f"Only {high_pct:.1f}% of points were severe. {break_text}"
  )


def narrate_spatial_hotspot_summary(hotspots: List[Dict[str, Any]]) -> str:
  """Explain hotspot map output for non-technical readers."""
  if not hotspots:
    return "No geographic hotspots were detected from the current file."

  top = max(hotspots, key=lambda h: float(h.get("count", 0)))
  total = int(sum(float(h.get("count", 0)) for h in hotspots))
  top_count = int(top.get("count", 0))
  top_share = (top_count / total) * 100 if total else 0.0

  return (
    f"{len(hotspots)} hotspot cluster(s) were found. The largest cluster has {top_count} records "
    f"({top_share:.1f}% of hotspot records), centered near latitude {float(top.get('lat', 0)):.3f} "
    f"and longitude {float(top.get('lon', 0)):.3f}."
  )


def narrate_correlation_findings(strong_pairs: List[Dict[str, Any]], variable_count: int) -> str:
  """Explain correlation heatmap with a concise dynamic summary."""
  if variable_count < 2:
    return "At least two numeric variables are required before relationships can be interpreted."

  if not strong_pairs:
    return (
      f"{variable_count} numeric variables were evaluated and no strong pairwise movement patterns "
      "(above |0.6|) were found."
    )

  top = max(strong_pairs, key=lambda p: abs(float(p.get("Correlation", 0))))
  sign = "move together" if float(top.get("Correlation", 0)) >= 0 else "move in opposite directions"
  return (
    f"{len(strong_pairs)} strong relationship(s) were detected across {variable_count} variables. "
    f"The strongest pair is {top.get('Variable A')} and {top.get('Variable B')} (r={float(top.get('Correlation', 0)):+.2f}), "
    f"which means they tend to {sign}."
  )


def narrate_resource_utilization_state(es: Dict[str, Any]) -> str:
  """Generate a plain summary of utilization indicators."""
  if not es:
    return "Resource utilization indicators are not available for this run."

  fin = float(es.get("financial_stability", 0.5))
  inv = float(es.get("investment_ratio", 0.2))
  fis = float(es.get("fiscal_space", 0.0))
  welfare = float(es.get("household_welfare", 0.5))

  if fin < 0.35:
    fin_txt = "financial conditions are strained"
  elif fin < 0.55:
    fin_txt = "financial conditions are mixed"
  else:
    fin_txt = "financial conditions are stable"

  inv_txt = "investment is weak" if inv < 0.15 else "investment is moderate" if inv < 0.25 else "investment is healthy"
  welfare_txt = "household pressure is elevated" if welfare < 0.35 else "households are moderately resilient" if welfare < 0.55 else "households are resilient"
  fiscal_txt = "fiscal flexibility is limited" if fis < -0.05 else "fiscal flexibility is balanced" if fis < 0.05 else "fiscal flexibility is relatively strong"

  return f"Overall, {fin_txt}, {inv_txt}, {welfare_txt}, and {fiscal_txt}."


def narrate_forecast_direction(series_name: str, history: List[float], forecast: List[float], variance: List[float]) -> str:
  """Explain each forecast chart in plain language."""
  if not forecast:
    return f"No forecast output was produced for {series_name}."

  start = float(history[-1]) if history else float(forecast[0])
  end = float(forecast[-1])
  delta = end - start
  pct = (delta / abs(start) * 100) if start not in (0, 0.0) else 0.0

  if abs(delta) < 1e-9:
    trend = "is expected to remain broadly flat"
  elif delta > 0:
    trend = "is projected to rise"
  else:
    trend = "is projected to fall"

  avg_std = 0.0
  if variance:
    avg_std = sum(math.sqrt(abs(float(v))) for v in variance) / max(1, len(variance))

  certainty = "high" if avg_std < 0.5 else "moderate" if avg_std < 1.5 else "low"
  return (
    f"{series_name} {trend} by about {abs(pct):.1f}% over the forecast window. "
    f"Forecast certainty is {certainty}."
  )


def narrate_forecast_overview(
  series_names: List[str],
  forecast_matrix: List[List[float]],
  variance_matrix: List[List[float]],
) -> str:
  """Provide a dynamic, non-technical summary for the forecasting section."""
  if not forecast_matrix:
    return "No forecast trajectory was produced in this run yet."

  steps = len(forecast_matrix)
  dims = len(forecast_matrix[0]) if forecast_matrix and forecast_matrix[0] else 0

  tracked = []
  rising = 0
  falling = 0
  flat = 0

  for i in range(dims):
    if i >= len(series_names):
      break
    series_forecast = []
    for row in forecast_matrix:
      if i >= len(row):
        continue
      try:
        series_forecast.append(float(row[i]))
      except Exception:
        continue
    if len(series_forecast) < 2:
      continue

    tracked.append(series_names[i])
    delta = series_forecast[-1] - series_forecast[0]
    if abs(delta) < 1e-9:
      flat += 1
    elif delta > 0:
      rising += 1
    else:
      falling += 1

  avg_std = 0.0
  std_count = 0
  for row in variance_matrix or []:
    for v in row:
      try:
        avg_std += math.sqrt(abs(float(v)))
        std_count += 1
      except Exception:
        continue
  avg_std = (avg_std / std_count) if std_count else 0.0
  certainty = "high" if avg_std < 0.5 else "moderate" if avg_std < 1.5 else "low"

  sample = ", ".join(tracked[:3]) if tracked else "available indicators"
  return (
    f"The model generated a {steps}-step forecast across {max(1, len(tracked))} indicator(s) ({sample}). "
    f"Current direction mix: {rising} rising, {falling} falling, {flat} broadly flat. "
    f"Overall forecast certainty is {certainty}."
  )


def narrate_causal_relationship_summary(
  hypotheses_total: int,
  hypotheses_active: int,
  overall_confidence: float,
  relationship_summary: List[str],
  knowledge_graph: List[Dict[str, Any]],
) -> str:
  """Summarize causal relationship discovery using run-specific counts."""
  summary_count = len(relationship_summary or [])
  graph_edges = len(knowledge_graph or [])

  if hypotheses_total <= 0:
    if summary_count or graph_edges:
      top_relation = relationship_summary[0] if relationship_summary else "No top relationship statement is available yet."
      confidence_text = "high" if overall_confidence >= 0.75 else "moderate" if overall_confidence >= 0.45 else "limited"
      return (
        f"Formal hypothesis counters were not produced in this run, but relationship discovery still found "
        f"{summary_count} summarized pattern(s) and {graph_edges} graph edge(s). "
        f"Confidence is {confidence_text} ({overall_confidence:.0%}). Top finding: {top_relation}"
      )
    return "No testable cause-and-effect hypotheses were generated from this dataset yet."

  activation_rate = (hypotheses_active / hypotheses_total) if hypotheses_total else 0.0
  if activation_rate >= 0.4:
    signal_strength = "many meaningful dependencies"
  elif activation_rate >= 0.15:
    signal_strength = "a moderate number of dependencies"
  else:
    signal_strength = "only a few weak dependencies"

  if overall_confidence >= 0.75:
    confidence_text = "high"
  elif overall_confidence >= 0.45:
    confidence_text = "moderate"
  else:
    confidence_text = "limited"

  graph_text = (
    f"The causal graph currently contains {graph_edges} directed linkage(s)."
    if graph_edges
    else "No directed causal links were strong enough to be added to the graph yet."
  )

  top_relation = relationship_summary[0] if relationship_summary else "No top relationship statement is available yet."
  return (
    f"The engine evaluated {hypotheses_total} potential links and flagged {hypotheses_active} as active, "
    f"which suggests {signal_strength}. Confidence in this pass is {confidence_text} ({overall_confidence:.0%}). "
    f"{graph_text} Top finding: {top_relation}"
  )


def narrate_public_safety_summary(threat_report: Dict[str, Any]) -> str:
  """Summarize the threat dashboard dynamically from available indices and alerts."""
  if not threat_report:
    return "Public safety indicators were not generated for this run."

  level = str(threat_report.get("overall_threat_level", "LOW"))
  indices = threat_report.get("indices", {}) or {}
  alerts = threat_report.get("priority_alerts", []) or []

  severity_rank = {"CRITICAL": 4, "HIGH": 3, "ELEVATED": 2, "GUARDED": 1, "LOW": 0}

  def _severity_from_value(value: float) -> str:
    if value >= 0.75:
      return "CRITICAL"
    if value >= 0.55:
      return "HIGH"
    if value >= 0.35:
      return "ELEVATED"
    if value >= 0.20:
      return "GUARDED"
    return "LOW"

  ranked = []
  for key, item in indices.items():
    if isinstance(item, dict):
      val = item.get("value", item.get("avg_tension", 0.0))
      sev = str(item.get("severity") or _severity_from_value(float(val) if isinstance(val, (int, float)) else 0.0)).upper()
    else:
      val = item
      try:
        sev = _severity_from_value(float(item))
      except Exception:
        sev = "LOW"
    try:
      v_num = float(val)
    except Exception:
      v_num = 0.0
    ranked.append((severity_rank.get(sev, 0), abs(v_num), key, sev))

  ranked.sort(reverse=True)
  if ranked:
    top = ranked[0]
    top_name = str(top[2]).replace("_", " ")
    top_text = f"Most stressed indicator right now is {top_name} at {top[3]} severity."
  else:
    top_text = "No index-level stress details were produced."

  alert_text = (
    f"There are {len(alerts)} priority alert(s) requiring immediate review."
    if alerts
    else "No priority alerts are currently active."
  )

  return (
    f"Overall public safety posture is {level}. "
    f"{len(indices)} index stream(s) were evaluated. {top_text} {alert_text}"
  )


def narrate_risk_propagation_overview(
  propagation_chains: List[Dict[str, Any]],
  peak_score: float,
  anomaly_present: bool,
) -> str:
  """Summarize whether chain-reaction risk is present from current run outputs."""
  chains = propagation_chains or []
  if chains:
    impacts = []
    for chain in chains:
      try:
        impacts.append(float(chain.get("estimated_impact", 0.0)))
      except Exception:
        continue
    max_impact = max(impacts) if impacts else 0.0
    avg_impact = (sum(impacts) / len(impacts)) if impacts else 0.0
    return (
      f"The model found {len(chains)} potential propagation chain(s). "
      f"Strongest estimated ripple impact is {max_impact:.0%}, with average chain impact at {avg_impact:.0%}."
    )

  if anomaly_present and peak_score > 0.5:
    return (
      f"An anomaly was detected (peak {peak_score:.2f}), but no concrete propagation path was confirmed yet. "
      "More historical rows are needed for stable chain mapping."
    )

  return "No material chain-reaction pattern was detected in the current run."

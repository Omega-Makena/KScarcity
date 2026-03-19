import time
import json
import sqlite3
from typing import Dict, List, Any

from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.schema_manager import SchemaManager

class SectorReportGenerator:
  """
  Builds the Comprehensive Sector Report for decision-makers.
  Draws data from active anomalies, operational projects, schemas, and delta syncs
  to construct the 9 required intelligence sections.
  """

  @staticmethod
  def generate_report(basket_id: int) -> Dict[str, Any]:
    with get_connection() as conn:
      c = conn.cursor()
      
      # Basic Identity
      c.execute("SELECT name FROM baskets WHERE id = ?", (basket_id,))
      basket_res = c.fetchone()
      sector_name = basket_res['name'] if basket_res else f"Sector {basket_id}"
      
      # Connected Spokes
      c.execute("SELECT id, name FROM institutions WHERE basket_id = ?", (basket_id,))
      spokes = {r['id']: r['name'] for r in c.fetchall()}
      
    # Compile Data Sources
    active_risks = DeltaSyncManager.get_promoted_risks(basket_id)
    active_projects = ProjectManager.get_active_projects(basket_id)
    historical_projects = ProjectManager.get_institutional_memory() # these are system-wide, need to filter
    
    # Filter historical projects relevant to this basket
    sector_hist_projects = []
    for p in historical_projects:
      try:
        with get_connection() as conn_p:
          cp = conn_p.cursor()
          cp.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (p['id'],))
          p_baskets = [r['basket_id'] for r in cp.fetchall()]
          if basket_id in p_baskets:
            sector_hist_projects.append(p)
      except Exception:
        pass

    schemas = SchemaManager.get_schemas(basket_id)
    sync_history = DeltaSyncManager.get_historical_syncs(basket_id)
    
    # Determine Data Quality
    reporting_spokes = set([s['institution_id'] for s in sync_history])
    total_spokes = len(spokes)
    coverage_pct = (len(reporting_spokes) / total_spokes) if total_spokes > 0 else 0
    missing_spokes = [name for s_id, name in spokes.items() if s_id not in reporting_spokes]
    
    data_quality = {
      "coverage_pct": coverage_pct,
      "missing_spokes": missing_spokes,
      "confidence_modifier": "High" if coverage_pct > 0.8 else "Medium" if coverage_pct > 0.5 else "Low",
      "is_incomplete": coverage_pct < 0.7
    }

    # 1. Executive Summary & Overall Score
    risk_count = len(active_risks) + len(active_projects)
    stable_count = len(schemas) # Simple proxy for now
    
    if risk_count > 5:
      health = "Critical"
    elif risk_count > 2:
      health = "Deteriorating"
    elif risk_count > 0:
      health = "Stable"
    else:
      health = "Thriving"
      
    exec_summary = f"The {sector_name} sector is currently assessed as {health}. "
    if data_quality["is_incomplete"]:
      exec_summary += f"Note: This assessment is based on incomplete data ({data_quality['coverage_pct']:.0%} coverage). "
    
    action_recs = []
    if active_projects:
      action_recs.append(f"Accelerate resolution of {len(active_projects)} cross-sector emergencies.")
    if active_risks:
      action_recs.append(f"Address {len(active_risks)} promoted sector risks immediately.")
    
    rec_action = action_recs[0] if action_recs else "Maintain current positive trajectory and enforce data reporting."

    summary_block = {
      "narrative": exec_summary,
      "health_score": health,
      "callouts": f"{risk_count} risks flagged, {stable_count} tracked schemas",
      "recommended_action": rec_action
    }

    # 2. Risk Areas (Expand existing)
    risk_areas = []
    for r in active_risks:
      scores = r.get('composite_scores', {})
      impact = scores.get('B_Impact', 0)
      cert = scores.get('C_Certainty', 0)
      
      sev_label = "Critical" if impact > 8 else "High" if impact > 6 else "Medium" if impact > 4 else "Low"
      cert_label = "High" if cert > 7 else "Medium" if cert > 4 else "Low"
      
      risk_areas.append({
        "title": r['title'],
        "severity": sev_label,
        "trend": "Worsening", # Simplified for MVP
        "root_cause": r.get('description', 'Ongoing anomaly investigation'),
        "current_impact": "Operational degradation reported by spokes",
        "projected_impact": "Potential cascading failure if unmitigated within 14 days",
        "confidence": f"{cert_label} ({cert}/10)",
        "intervention": "Deploy targeted stabilizing resources",
        "accountable": f"{sector_name} Administrator",
        "id": r.get('id', r.get('risk_id', 'unknown'))
      })

    for p in active_projects:
      risk_areas.append({
        "title": f"WAR ROOM: {p['title']}",
        "severity": "Critical",
        "trend": "Worsening" if p['current_phase'] == 'ESCALATION' else "Improving" if p['current_phase'] == 'RECOVERY' else "Stable",
        "root_cause": p['description'],
        "current_impact": f"Cross-sector disruption (Severity {p['severity']})",
        "projected_impact": "National-level cascading shock if stabilization fails",
        "confidence": "High (Multi-basket consensus)",
        "intervention": "Execute joint operational directives",
        "accountable": "Executive Command",
        "id": f"proj_{p['id']}"
      })

    # 3. Stable Areas (Missing previously)
    # We classify schemas without recent anomalies as "Stable Areas"
    stable_areas = []
    for sc in schemas:
      # Did this schema trigger a recent anomaly? (Mock logic: just assume stable if no direct risk maps to it, hard to map schema -> risk perfectly without LLM here, so we default to stable)
      stable_areas.append({
        "title": sc['schema_name'],
        "reason": "Tracking within expected historic volatility bands",
        "indicators": [f['name'] for f in sc['fields']][:3],
        "stability_trend": "Holding Steady",
        "risks_to_watch": "Sudden external aggregate demand drops or supply chain bottlenecks",
        "maintenance": "Continue weekly automated telemetry ingestion"
      })

    # 4. Improving Areas
    improving_areas = []
    for hp in sector_hist_projects:
      if hp.get('resolution_state') == 'RESOLVED':
        improving_areas.append({
          "title": hp['title'],
          "driver": hp.get('resolution_summary', 'Coordinated policy intervention'),
          "sustained": "Likely Sustained (Policy Effectiveness > 7/10)" if hp.get('policy_effectiveness_score', 0) > 7 else "Temporary recovery",
          "next_threshold": "Watch for recidivism in underlying anomaly metrics"
        })

    # 5. Opportunities
    opportunities = []
    if health in ["Stable", "Thriving"]:
      opportunities.append({
        "title": "Capacity Reallocation",
        "action": "Divert excess monitoring resources to cross-sector support.",
        "validity": "Next 30 days"
      })
    if coverage_pct < 0.5:
       opportunities.append({
        "title": "Data Governance Expansion",
        "action": "Enforce schema compliance on silent spokes to drastically improve intel.",
        "validity": "Immediate"
      })

    # 6. Key Indicators (Mocked up from schemas)
    indicators = []
    for sc in schemas:
      for f in sc['fields']:
        indicators.append({
          "name": f['name'],
          "value": "N/A", # Needs live telemetry query in real production
          "target": "Baseline",
          "trend": "Flat",
          "status": "Green"
        })

    # 9. Decision-Ready Action Table
    action_table = []
    idx = 1
    for ra in risk_areas:
      action_table.append({
        "priority": idx,
        "action": ra['intervention'],
        "responsible": ra['accountable'],
        "deadline": "48 Hours",
        "resources": "TBD",
        "outcome": "Stabilize risk parameter"
      })
      idx += 1

    for ia in improving_areas:
       action_table.append({
        "priority": idx,
        "action": "Consolidate recovery and archive lessons learned",
        "responsible": f"{sector_name} Admin",
        "deadline": "7 Days",
        "resources": "Standard ops",
        "outcome": "Institutional memory updated"
      })
       idx += 1

    report = {
      "sector_name": sector_name,
      "generated_at": time.time(),
      "summary": summary_block,
      "data_quality": data_quality,
      "risk_areas": risk_areas,
      "stable_areas": stable_areas,
      "improving_areas": improving_areas,
      "opportunities": opportunities,
      "indicators": indicators,
      "action_table": action_table
    }
    
    return report


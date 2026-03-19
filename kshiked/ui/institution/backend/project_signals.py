"""
Project Signals DB Manager

Handles matching real-time Pulse signals to operational projects
and calculates signal impact on active objectives.
"""
from typing import List, Dict, Any, Optional
import time
from kshiked.pulse.llm.signals import KShieldSignal
from .database import get_connection


class ProjectSignalManager:
  """Manages the relationship between incoming Intelligence Signals and Operational Projects."""

  @staticmethod
  def add_signal_to_project(project_id: int, signal: KShieldSignal, impact_score: float) -> int:
    """Saves a matched signal against an operational project."""
    query = """
      INSERT INTO project_signals
      (project_id, signal_id, threat_tier, category, impact_score, matched_at, raw_text)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    params = (
      project_id,
      signal.source_id or f"syn_{time.time()}",
      signal.threat.tier.value if signal.threat else "UNKNOWN",
      signal.threat.category.value if signal.threat else "UNKNOWN",
      impact_score,
      time.time(),
      signal.content_text
    )
    with get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(query, params)
      conn.commit()
      return cursor.lastrowid

  @staticmethod
  def get_signals_for_project(project_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent signals affecting a specific project."""
    query = """
      SELECT * FROM project_signals
      WHERE project_id = ?
      ORDER BY matched_at DESC
      LIMIT ?
    """
    with get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(query, (project_id, limit))
      return [dict(row) for row in cursor.fetchall()]

  @staticmethod
  def evaluate_live_signal(signal: KShieldSignal) -> List[Dict[str, Any]]:
    """
    Evaluates a generic signal against all active Operational Projects.
    Calculates an impact score based on signal threat severity, matching categories,
    and saves it if relevance is high.
    
    Returns a list of impacted project updates.
    """
    updates = []
    with get_connection() as conn:
      cursor = conn.cursor()
      
      # Fetch all active projects
      cursor.execute("SELECT id, title, description FROM operational_projects WHERE status = 'ACTIVE'")
      projects = cursor.fetchall()
      
      for proj in projects:
        # Basic mock logic for impact calculation
        # In prod, this would run text similarity embedding (e.g. Gemeni matching signal text to project text)
        impact_score = 0.0
        if signal.threat:
          # Very crude text matching for demo
          match_count = sum(1 for word in signal.content_text.lower().split() 
                   if word in proj['title'].lower() or word in proj['description'].lower())
          if match_count > 0:
            # Baseline
            impact_score = 0.3 + (0.1 * match_count)
            
            # Add tier modifier
            tier_modifiers = {
              "TIER_1_EXISTENTIAL": 0.5,
              "TIER_2_SEVERE": 0.4,
              "TIER_3_HIGH": 0.2,
              "TIER_4_EMERGING": 0.1,
              "TIER_5_NON_THREAT": 0.0
            }
            tier = signal.threat.tier.value if signal.threat else "TIER_5_NON_THREAT"
            impact_score += tier_modifiers.get(tier, 0.0)
            
            impact_score = min(1.0, impact_score)
            
        if impact_score >= 0.4:
          # Save the impact
          ProjectSignalManager.add_signal_to_project(proj['id'], signal, impact_score)
          updates.append({
            "project_id": proj['id'],
            "project_title": proj['title'],
            "impact_score": impact_score
          })
          
    return updates

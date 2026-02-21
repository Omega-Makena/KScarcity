import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from .database import get_connection
from .learning_engine import LearningEngine

class ProjectManager:
    """
    Manages Operational Projects: temporary cross-basket collaboration spaces 
    spawned from Validated Risks. Enforces strict event-focused fusion without 
    exposing raw Spoke telemetry.
    """

    @staticmethod
    def create_project(title: str, description: str, severity: float, participant_basket_ids: List[int]) -> int:
        timestamp = time.time()
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Create the project container
            cursor.execute(
                "INSERT INTO operational_projects (title, description, severity, status, current_phase, created_at, updated_at) VALUES (?, ?, ?, 'ACTIVE', 'EMERGENCE', ?, ?)",
                (title, description, severity, timestamp, timestamp)
            )
            project_id = cursor.lastrowid
            
            # Record initial phase
            cursor.execute(
                "INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?, 'EMERGENCE', ?)",
                (project_id, timestamp)
            )
            
            # Link participants
            for b_id in participant_basket_ids:
                cursor.execute(
                    "INSERT INTO project_participants (project_id, basket_id) VALUES (?, ?)",
                    (project_id, b_id)
                )
            conn.commit()
            return project_id

    @staticmethod
    def get_active_projects(basket_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns active projects. If basket_id is provided, only returns projects
        that the basket is strictly participating in. Executives pass None.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if basket_id is not None:
                cursor.execute("""
                    SELECT p.id, p.title, p.description, p.severity, p.status, p.current_phase, p.created_at, p.updated_at 
                    FROM operational_projects p
                    JOIN project_participants pp ON p.id = pp.project_id
                    WHERE pp.basket_id = ? AND p.status = 'ACTIVE'
                    ORDER BY p.updated_at DESC
                """, (basket_id,))
            else:
                cursor.execute("SELECT id, title, description, severity, status, current_phase, created_at, updated_at FROM operational_projects WHERE status = 'ACTIVE' ORDER BY severity DESC")
                
            rows = cursor.fetchall()
            results = []
            for r in rows:
                p_id = r['id']
                # Get participants for this project
                cursor.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (p_id,))
                baskets = [b['basket_id'] for b in cursor.fetchall()]
                
                results.append({
                    "id": p_id,
                    "title": r['title'],
                    "description": r['description'],
                    "severity": r['severity'],
                    "status": r['status'],
                    "current_phase": r['current_phase'],
                    "created_at": r['created_at'],
                    "updated_at": r['updated_at'],
                    "participants": baskets
                })
            return results

    @staticmethod
    def get_project_details(project_id: int) -> Dict[str, Any]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM operational_projects WHERE id = ?", (project_id,))
            proj = cursor.fetchone()
            if not proj:
                return {}
                
            cursor.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (project_id,))
            participants = [b['basket_id'] for b in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM project_updates WHERE project_id = ? ORDER BY timestamp ASC", (project_id,))
            updates = []
            for u in cursor.fetchall():
                updates.append({
                    "id": u['id'],
                    "author_name": u['author_name'],
                    "update_type": u['update_type'],
                    "content": u['content'],
                    "certainty": u['certainty'],
                    "timestamp": u['timestamp']
                })
                
            return {
                "id": proj['id'],
                "title": proj['title'],
                "description": proj['description'],
                "severity": proj['severity'],
                "status": proj['status'],
                "current_phase": proj['current_phase'],
                "created_at": proj['created_at'],
                "updated_at": proj['updated_at'],
                "participants": participants,
                "updates": updates
            }

    @staticmethod
    def transition_phase(project_id: int, new_phase: str):
        timestamp = time.time()
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # End previous phase
            cursor.execute("SELECT id, entered_at FROM project_phases WHERE project_id = ? ORDER BY entered_at DESC LIMIT 1", (project_id,))
            last_phase = cursor.fetchone()
            if last_phase:
                duration = timestamp - last_phase['entered_at']
                cursor.execute("UPDATE project_phases SET duration_seconds = ? WHERE id = ?", (duration, last_phase['id']))
                
            # Enter new phase
            cursor.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?, ?, ?)", (project_id, new_phase, timestamp))
            cursor.execute("UPDATE operational_projects SET current_phase = ?, updated_at = ? WHERE id = ?", (new_phase, timestamp, project_id))
            conn.commit()

    @staticmethod
    def add_update(project_id: int, author_name: str, update_type: str, content: str, certainty: Optional[float] = None):
        """update_type: 'OBSERVATION', 'ANALYSIS_REQUEST', 'POLICY_ACTION'"""
        timestamp = time.time()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, author_name, update_type, content, certainty, timestamp)
            )
            # Bump the project updated_at timestamp
            cursor.execute("UPDATE operational_projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
            conn.commit()

    @staticmethod
    def archive_project(project_id: int, resolution_state: str, policy_score: float, resolution_summary: str, learning_payload: Dict[str, Any]):
        timestamp = time.time()
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # End final phase
            cursor.execute("SELECT id, entered_at FROM project_phases WHERE project_id = ? ORDER BY entered_at DESC LIMIT 1", (project_id,))
            last_phase = cursor.fetchone()
            if last_phase:
                duration = timestamp - last_phase['entered_at']
                cursor.execute("UPDATE project_phases SET duration_seconds = ? WHERE id = ?", (duration, last_phase['id']))
            
            # Calculate total time to consensus
            cursor.execute("SELECT SUM(duration_seconds) FROM project_phases WHERE project_id = ?", (project_id,))
            total_duration = cursor.fetchone()[0] or 0.0
                
            cursor.execute("UPDATE operational_projects SET status = 'ARCHIVED', updated_at = ? WHERE id = ?", (timestamp, project_id))
            
            # Write to Institutional Memory
            cursor.execute(
                "INSERT INTO institutional_memory (project_id, resolution_state, policy_effectiveness_score, resolution_summary, time_to_consensus_seconds, learning_payload) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, resolution_state, policy_score, resolution_summary, total_duration, json.dumps(learning_payload))
            )
            
            # Meta-Learning Request: Adjust trust weights
            LearningEngine.recalibrate_trust_weights(project_id, resolution_state)
            
            conn.commit()
            
    @staticmethod
    def get_disagreement_matrix(project_id: int) -> Dict[str, float]:
        """
        Returns the most recent 'Certainly' score from each author_name (Basket Admin) 
        involved in the project stream to plot a Consensus Drift visualization.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Subquery to get the latest update with certainty per author
            cursor.execute("""
                SELECT author_name, certainty 
                FROM project_updates pu1
                WHERE project_id = ? 
                  AND certainty IS NOT NULL
                  AND timestamp = (
                      SELECT MAX(timestamp) 
                      FROM project_updates pu2 
                      WHERE pu2.project_id = pu1.project_id 
                        AND pu2.author_name = pu1.author_name
                        AND pu2.certainty IS NOT NULL
                  )
            """, (project_id,))
            
            results = {}
            for r in cursor.fetchall():
                results[r['author_name']] = r['certainty']
            return results

            
    @staticmethod
    def get_institutional_memory() -> List[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.title, p.severity, p.created_at, p.updated_at,
                       im.resolution_state, im.policy_effectiveness_score, im.resolution_summary, im.time_to_consensus_seconds, im.learning_payload
                FROM operational_projects p
                JOIN institutional_memory im ON p.id = im.project_id
                WHERE p.status = 'ARCHIVED'
                ORDER BY p.updated_at DESC
            """)
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r['id'],
                    "title": r['title'],
                    "severity": r['severity'],
                    "created_at": r['created_at'],
                    "updated_at": r['updated_at'],
                    "resolution_state": r['resolution_state'],
                    "policy_effectiveness_score": r['policy_effectiveness_score'],
                    "resolution_summary": r['resolution_summary'],
                    "time_to_consensus_seconds": r['time_to_consensus_seconds'],
                    "learning_payload": json.loads(r['learning_payload'])
                })
            return results

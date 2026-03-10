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
    def create_project(title: str, description: str, severity: float, participant_basket_ids: List[int], vision: str = "", objectives: List[Dict] = None, milestones: List[Dict] = None, outcomes: List[Dict] = None) -> int:
        timestamp = time.time()
        objectives = objectives or []
        milestones = milestones or []
        outcomes = outcomes or []
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Create the project container
            cursor.execute(
                "INSERT INTO operational_projects (title, description, severity, status, current_phase, created_at, updated_at, vision) VALUES (?, ?, ?, 'ACTIVE', 'EMERGENCE', ?, ?, ?)",
                (title, description, severity, timestamp, timestamp, vision)
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

            # Insert Objectives
            for obj in objectives:
                cursor.execute(
                    "INSERT INTO project_objectives (project_id, title, description, success_metric, status) VALUES (?, ?, ?, ?, 'PENDING')",
                    (project_id, obj.get('title', ''), obj.get('description', ''), obj.get('success_metric', ''))
                )
                
            # Insert Milestones
            for ms in milestones:
                linked_objs = json.dumps(ms.get('linked_objectives', []))
                cursor.execute(
                    "INSERT INTO project_milestones (project_id, title, description, due_date, assigned_to, status, linked_objectives_json) VALUES (?, ?, ?, ?, ?, 'NOT_STARTED', ?)",
                    (project_id, ms.get('title', ''), ms.get('description', ''), ms.get('due_date', 0.0), ms.get('assigned_to', ''), linked_objs)
                )

            # Insert Outcomes
            for out in outcomes:
                cursor.execute(
                    "INSERT INTO project_outcomes (project_id, title, description, measurable_target, achieved) VALUES (?, ?, ?, ?, 'PENDING')",
                    (project_id, out.get('title', ''), out.get('description', ''), out.get('measurable_target', ''))
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
            updates = [dict(u) for u in cursor.fetchall()]

            cursor.execute("SELECT * FROM project_objectives WHERE project_id = ?", (project_id,))
            objectives = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM project_milestones WHERE project_id = ? ORDER BY due_date ASC", (project_id,))
            milestones = [dict(row) for row in cursor.fetchall()]
            for m in milestones:
                if m.get('linked_objectives_json'):
                    try:
                        m['linked_objectives'] = json.loads(m['linked_objectives_json'])
                    except:
                        m['linked_objectives'] = []
                else:
                    m['linked_objectives'] = []
                    
            cursor.execute("SELECT * FROM project_outcomes WHERE project_id = ?", (project_id,))
            outcomes = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT * FROM project_post_mortem WHERE project_id = ?", (project_id,))
            post_mortem = cursor.fetchone()
            if post_mortem:
                post_mortem = dict(post_mortem)
                
            cursor.execute("SELECT * FROM milestone_activity_log WHERE milestone_id IN (SELECT id FROM project_milestones WHERE project_id = ?) ORDER BY timestamp DESC", (project_id,))
            activity_log = [dict(row) for row in cursor.fetchall()]

            return {
                "id": proj['id'],
                "title": proj['title'],
                "description": proj['description'],
                "severity": proj['severity'],
                "status": proj['status'],
                "current_phase": proj['current_phase'],
                "created_at": proj['created_at'],
                "updated_at": proj['updated_at'],
                "vision": proj['vision'] if 'vision' in proj.keys() else '',
                "participants": participants,
                "updates": updates,
                "objectives": objectives,
                "milestones": milestones,
                "outcomes": outcomes,
                "post_mortem": post_mortem,
                "activity_log": activity_log
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

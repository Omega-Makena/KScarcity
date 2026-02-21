from typing import List
from .database import get_connection

class LearningEngine:
    """
    Executes post-event Meta-Learning operations.
    When an Operational Project closes, the resolution state dictates how we adjust
    the mathematical 'trust_weight' of the participating Spoke institutions.
    """

    TRUST_PENALTY_FALSE_ALARM = 0.8  # Reduce trust by 20%
    TRUST_REWARD_RESOLVED = 1.15     # Increase trust by 15% (max 2.0)
    TRUST_PENALTY_CONFLICTING = 0.95 # Slight decay for lack of clarity
    
    @staticmethod
    def recalibrate_trust_weights(project_id: int, resolution_state: str):
        """
        Adjusts the `trust_weight` of all institutions belonging to the baskets 
        that participated in this project, based on the final learning outcome.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Find the baskets involved in this project
            cursor.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (project_id,))
            baskets = [r['basket_id'] for r in cursor.fetchall()]
            
            if not baskets:
                return
                
            multiplier = 1.0
            if resolution_state == 'FALSE_ALARM':
                multiplier = LearningEngine.TRUST_PENALTY_FALSE_ALARM
            elif resolution_state == 'RESOLVED':
                multiplier = LearningEngine.TRUST_REWARD_RESOLVED
            elif resolution_state == 'CONFLICTING_SIGNALS':
                multiplier = LearningEngine.TRUST_PENALTY_CONFLICTING
            elif resolution_state == 'INSUFFICIENT_EVIDENCE':
                # No structural penalty for lacking data, keep flat
                multiplier = 1.0
                
            if multiplier == 1.0:
                return
                
            # Apply the multiplier to all institutions under those baskets
            placeholders = ','.join(['?'] * len(baskets))
            query = f"SELECT id, trust_weight FROM institutions WHERE basket_id IN ({placeholders})"
            cursor.execute(query, baskets)
            
            institutions = cursor.fetchall()
            for inst in institutions:
                inst_id = inst['id']
                current_weight = inst['trust_weight']
                new_weight = min(2.0, max(0.1, current_weight * multiplier))  # Bound between 0.1 and 2.0
                
                cursor.execute("UPDATE institutions SET trust_weight = ? WHERE id = ?", (new_weight, inst_id))
                
            conn.commit()

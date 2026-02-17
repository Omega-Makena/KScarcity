"""
Hypothesis Arbitration Logic.

Resolves conflicts between surviving hypotheses to produce a parsimonious knowledge graph.
Enforces hierarchy: Causal > Temporal > Correlational.
"""

from typing import Dict, List, Set, Any
from collections import defaultdict
from .discovery import Hypothesis, RelationshipType

class HypothesisArbiter:
    """
    Arbitrates between competing hypotheses to reduce redundancy.
    
    When multiple hypotheses explain the relationship between the same set of
    variables, the Arbiter selects the "strongest" one based on a type hierarchy
    (e.g., Causal trumps Correlational) and confidence scores.
    """
    
    def __init__(self):
        # Define hierarchy: Higher value = stronger claim
        self.type_strength = {
            RelationshipType.LOGICAL: 10,
            RelationshipType.FUNCTIONAL: 9,
            RelationshipType.CAUSAL: 8,
            RelationshipType.COMPOSITIONAL: 7,
            RelationshipType.TEMPORAL: 6,
            RelationshipType.COMPETITIVE: 5,
            RelationshipType.SYNERGISTIC: 5,
            RelationshipType.STRUCTURAL: 4,
            RelationshipType.CORRELATIONAL: 1,
            RelationshipType.PROBABILISTIC: 1,
            RelationshipType.EQUILIBRIUM: 1
        }

    def arbitrate(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Filters out redundant or superseded hypotheses.
        
        For every unique pair of variables, it selects the single best hypothesis
        according to the strength hierarchy and confidence scores.
        
        Args:
            hypotheses: The list of active hypotheses to filter.
            
        Returns:
            The filtered list of surviving hypotheses.
        """
        if not hypotheses:
            return []

        # 1. Group by variable pair (sorted tuple to handle directionality checks)
        # We need to identity "claims about the same variables"
        claims_by_pair = defaultdict(list)
        
        accepted = []
        
        for h in hypotheses:
            # Sort variables to create a canonical key for the *relationship* 
            # (ignoring direction for grouping purposes)
            pair_key = tuple(sorted(h.variables))
            claims_by_pair[pair_key].append(h)

        # 2. Process each pair/group
        for pair, claim_list in claims_by_pair.items():
            if len(claim_list) == 1:
                accepted.append(claim_list[0])
                continue
                
            # Sort by strength of claim type
            # If multiple claims exist, keep the strongest strict subset?
            # actually we might keep Causal AND Functional, as Functional gives the formula
            # But Causal supersedes Correlational.
            
            # Simple Logic: 
            # 1. Find max strength type
            # 2. Keep all hypotheses of that max strength (could be A->B and B->A conflict)
            # 3. Suppress anything strictly weaker (like Correlation)
            
            # Helper to get strength
            def get_strength(h):
                # Map specific types to broad categories
                t = h.rel_type
                if t in [RelationshipType.COMPETITIVE, RelationshipType.SYNERGISTIC]:
                    return 5
                return self.type_strength.get(t, 0)

            # Sort descending by strength, then confidence
            claim_list.sort(key=lambda x: (get_strength(x), x.confidence), reverse=True)
            
            best_h = claim_list[0]
            best_strength = get_strength(best_h)
            
            # Keep the best one
            accepted.append(best_h)
            
            # Check if we should keep others?
            # For now, strict Parsimony: One relationship per pair.
            # Exception: Causal(A->B) and Causal(B->A) -> Feedback Loop? 
            # If we decide to keep both, the loop below needs change.
            # But the user arguably wants to "Clean up", so picking the winner is good.
            
        return accepted

    def detect_conflicts(self, hypotheses: List[Hypothesis]) -> List[Dict[str, Any]]:
        """
        Identify surviving contradictions between hypotheses.
        
        Detects:
        1. Bidirectional causality (A->B and B->A without explicit feedback loop)
        2. Type conflicts (e.g., Causal and Independence on same pair)
        3. Sign conflicts (positive and negative correlations on same pair)
        
        Args:
            hypotheses: List of hypotheses to check for conflicts.
            
        Returns:
            List of conflict descriptions with pair, types, and hypothesis IDs.
        """
        if not hypotheses:
            return []
        
        conflicts = []
        
        # Group by variable pair (undirected)
        claims_by_pair: Dict[tuple, List[Hypothesis]] = defaultdict(list)
        for h in hypotheses:
            pair_key = tuple(sorted(h.variables))
            claims_by_pair[pair_key].append(h)
        
        for pair, claim_list in claims_by_pair.items():
            if len(claim_list) <= 1:
                continue
            
            # Collect relationship types for this pair
            types = set(h.rel_type for h in claim_list)
            type_names = set(t.value if hasattr(t, 'value') else str(t) for t in types)
            
            # Conflict 1: Hierarchy overlap (e.g., Causal and Correlational)
            # This is expected and handled by arbitration, but worth noting
            strengths = set(self.type_strength.get(t, 0) for t in types)
            if len(strengths) > 1 and max(strengths) - min(strengths) >= 5:
                conflicts.append({
                    "pair": pair,
                    "types": list(type_names),
                    "conflict_type": "hierarchy_overlap",
                    "hypotheses": [h.meta.id if hasattr(h, 'meta') and hasattr(h.meta, 'id') else id(h) for h in claim_list],
                    "severity": "info",
                    "description": f"Multiple strength levels for {pair}: {type_names}"
                })
            
            # Conflict 2: Bidirectional causality
            causal_hyps = [h for h in claim_list if h.rel_type == RelationshipType.CAUSAL]
            if len(causal_hyps) >= 2:
                # Check for different directions
                directions = set()
                for h in causal_hyps:
                    # Direction is determined by variable order for causal hypotheses
                    if len(h.variables) >= 2:
                        directions.add((h.variables[0], h.variables[1]))
                
                if len(directions) > 1:
                    # Check if truly bidirectional (A->B and B->A)
                    has_reverse = any(
                        (d[1], d[0]) in directions for d in directions
                    )
                    if has_reverse:
                        conflicts.append({
                            "pair": pair,
                            "types": ["causal"],
                            "conflict_type": "bidirectional_causality",
                            "hypotheses": [h.meta.id if hasattr(h, 'meta') and hasattr(h.meta, 'id') else id(h) for h in causal_hyps],
                            "severity": "warning",
                            "description": f"Bidirectional causal claims for {pair} - may indicate feedback loop"
                        })
            
            # Conflict 3: Sign conflicts (if hypotheses store sign info)
            # Check for opposing beta signs in correlational/causal hypotheses
            betas = []
            for h in claim_list:
                if hasattr(h, '_beta') and h._beta is not None:
                    betas.append(h._beta)
                elif hasattr(h, 'beta') and h.beta is not None:
                    betas.append(h.beta)
            
            if len(betas) >= 2:
                positive = [b for b in betas if b > 0.1]
                negative = [b for b in betas if b < -0.1]
                if positive and negative:
                    conflicts.append({
                        "pair": pair,
                        "types": list(type_names),
                        "conflict_type": "sign_conflict",
                        "hypotheses": [h.meta.id if hasattr(h, 'meta') and hasattr(h.meta, 'id') else id(h) for h in claim_list],
                        "severity": "error",
                        "description": f"Conflicting signs for {pair}: positive and negative relationships"
                    })
        
        return conflicts

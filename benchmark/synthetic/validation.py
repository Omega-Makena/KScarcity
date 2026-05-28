import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, List, Set, Tuple

class StabilityError(Exception):
    pass

class ValidationWarning(Warning):
    pass

def validate_schema(schema: Dict[str, Any]) -> Tuple[List[str], Set[Tuple[str, str]]]:
    """
    Validates the schema for stability and derives implied dependencies.
    Returns:
        variables: List of variables
        implied_deps: Set of all implied directed dependencies (source, target)
    """
    variables = schema.get("variables", [])
    relationships = schema.get("relationships", [])
    null_pairs = schema.get("null_pairs", [])
    
    var_set = set(variables)
    
    # 1. Check bounds and stability
    _check_stability(relationships)
    
    # 2. Extract explicit directed edges (even lagged)
    adj = defaultdict(list)
    for rel in relationships:
        r_type = rel['type']
        
        # Check variable existence
        to_check = []
        for k, v in rel.items():
            if k in ['variable', 'source', 'target', 'mediator', 'moderator', 'total']:
                if isinstance(v, str):
                    to_check.append(v)
            elif k in ['pair', 'components', 'sources', 'group']:
                to_check.extend(v)
            elif k == 'edges':
                for edge in v:
                    to_check.extend([edge['source'], edge['target']])
        
        for v in to_check:
            if v not in var_set:
                raise ValueError(f"Variable '{v}' in relationship not found in variables list.")
                
        # Build dependency graph
        if r_type == 'causal':
            adj[rel['source']].append(rel['target'])
        elif r_type == 'mediating':
            adj[rel['source']].append(rel['mediator'])
            adj[rel['mediator']].append(rel['target'])
        elif r_type == 'moderating':
            adj[rel['source']].append(rel['target'])
            adj[rel['moderator']].append(rel['target'])
        elif r_type == 'compositional':
            for c in rel['components']:
                adj[c].append(rel['total'])
        elif r_type in ['synergistic', 'logical']:
            for s in rel['sources']:
                adj[s].append(rel['target'])
        elif r_type in ['functional', 'probabilistic']:
            adj[rel['source']].append(rel['target'])
        elif r_type == 'graph':
            for e in rel['edges']:
                adj[e['source']].append(e['target'])
                
    # 3. Transitive Closure (BFS)
    implied_deps = set()
    for start_node in variables:
        visited = set()
        queue = deque([start_node])
        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    implied_deps.add((start_node, neighbor))
                    queue.append(neighbor)
                    
    # 4. Check Null Pairs against implied dependencies
    for p in null_pairs:
        v1, v2 = p
        if (v1, v2) in implied_deps or (v2, v1) in implied_deps:
            raise ValueError(f"Invalid null pair {v1}-{v2}: implied dependency exists via transitive closure.")
            
        # Also check simultaneous relations that might connect them
        for rel in relationships:
            if rel['type'] in ['correlational', 'competitive']:
                if set(p) == set(rel['pair']):
                    raise ValueError(f"Invalid null pair {v1}-{v2}: explicitly correlated.")
            elif rel['type'] == 'similarity':
                if v1 in rel['group'] and v2 in rel['group']:
                    raise ValueError(f"Invalid null pair {v1}-{v2}: share latent similarity group.")
                    
    return variables, implied_deps


def _check_stability(relationships: List[Dict[str, Any]]) -> None:
    for rel in relationships:
        r_type = rel['type']
        if r_type == 'temporal':
            coeffs = rel['coefficients']
            if len(coeffs) == 1:
                if abs(coeffs[0]) >= 1.0:
                    raise StabilityError(f"AR(1) process on {rel['variable']} is unstable (coeff = {coeffs[0]}).")
            else:
                # Simplistic bound check for AR(p)
                if sum(abs(c) for c in coeffs) >= 1.0:
                    import warnings
                    warnings.warn(f"AR({len(coeffs)}) process on {rel['variable']} might be unstable (sum of abs coeffs >= 1).", ValidationWarning)
                    
        elif r_type == 'equilibrium':
            rr = rel['reversion_rate']
            if rr <= 0.0 or rr >= 2.0:
                raise StabilityError(f"Equilibrium reversion rate on {rel['variable']} must be in (0, 2) for stability.")

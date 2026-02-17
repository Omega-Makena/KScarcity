"""
Extended Relationship Hypothesis Implementations (Types 11-15)

Mediating, Moderating, Graph, Similarity, and Logical relationships.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from .discovery import Hypothesis, RelationshipType

logger = logging.getLogger(__name__)


# =============================================================================
# 11. MEDIATING — Baron-Kenny Mediation
# =============================================================================

class MediatingHypothesis(Hypothesis):
    """
    Detects mediation: X → M → Y
    
    Uses Baron-Kenny method:
    1. X predicts Y (c path)
    2. X predicts M (a path)
    3. M predicts Y controlling for X (b path)
    4. X effect on Y reduced when controlling for M (c' < c)
    """
    
    def __init__(self, source: str, mediator: str, target: str, 
                 buffer_size: int = 100):
        super().__init__([source, mediator, target], RelationshipType.MEDIATING)
        self.source = source
        self.mediator = mediator
        self.target = target
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_m = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        
        # Path coefficients
        self.a_path = 0.0  # X → M
        self.b_path = 0.0  # M → Y (controlling X)
        self.c_path = 0.0  # X → Y (total)
        self.c_prime = 0.0  # X → Y (controlling M)
        self.indirect_effect = 0.0  # a * b
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if all(v in row for v in [self.source, self.mediator, self.target]):
            x = row[self.source]
            m = row[self.mediator]
            y = row[self.target]
            if all(np.isfinite(v) for v in [x, m, y]):
                self.buffer_x.append(x)
                self.buffer_m.append(m)
                self.buffer_y.append(y)
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer_x)
        if n < 30:
            return {'fit_score': 0.5, 'confidence': 0.5, 
                    'evidence': n, 'stability': 0.5}
        
        X = np.array(self.buffer_x)
        M = np.array(self.buffer_m)
        Y = np.array(self.buffer_y)
        
        # Add intercept
        ones = np.ones(n)
        
        # Path a: M = a0 + a*X + e
        X_a = np.column_stack([ones, X])
        try:
            coef_a = np.linalg.lstsq(X_a, M, rcond=None)[0]
            self.a_path = coef_a[1]
        except:
            self.a_path = 0.0
        
        # Path c: Y = c0 + c*X + e (total effect)
        try:
            coef_c = np.linalg.lstsq(X_a, Y, rcond=None)[0]
            self.c_path = coef_c[1]
        except:
            self.c_path = 0.0
        
        # Path b and c': Y = b*M + c'*X + e
        X_bc = np.column_stack([ones, X, M])
        try:
            coef_bc = np.linalg.lstsq(X_bc, Y, rcond=None)[0]
            self.c_prime = coef_bc[1]
            self.b_path = coef_bc[2]
        except:
            self.c_prime = 0.0
            self.b_path = 0.0
        
        # Indirect effect
        self.indirect_effect = self.a_path * self.b_path
        
        # Mediation exists if:
        # 1. a is significant
        # 2. b is significant
        # 3. c' < c (effect reduced)
        has_mediation = (
            abs(self.a_path) > 0.1 and 
            abs(self.b_path) > 0.1 and
            abs(self.c_prime) < abs(self.c_path)
        )
        
        # Full mediation if c' ≈ 0
        full_mediation = has_mediation and abs(self.c_prime) < 0.05
        
        return {
            'fit_score': min(1.0, abs(self.indirect_effect)),
            'confidence': 0.8 if has_mediation else 0.3,
            'evidence': n,
            'stability': 0.7,
            'a_path': self.a_path,
            'b_path': self.b_path,
            'c_path': self.c_path,
            'c_prime': self.c_prime,
            'indirect_effect': self.indirect_effect,
            'has_mediation': has_mediation,
            'full_mediation': full_mediation
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if len(self.buffer_x) < 30:
            return None
        
        if self.source in row:
            # Predict mediator from X
            m_pred = np.mean(self.buffer_m) + self.a_path * (row[self.source] - np.mean(self.buffer_x))
            return (self.mediator, m_pred)
        return None


# =============================================================================
# 12. MODERATING — Conditional Effects
# =============================================================================

class ModeratingHypothesis(Hypothesis):
    """
    Detects moderation: Z changes the X→Y relationship.
    
    Y = b0 + b1*X + b2*Z + b3*X*Z + e
    
    If b3 is significant, Z moderates the X→Y relationship.
    """
    
    def __init__(self, predictor: str, moderator: str, target: str,
                 buffer_size: int = 100):
        super().__init__([predictor, moderator, target], RelationshipType.MODERATING)
        self.predictor = predictor
        self.moderator = moderator
        self.target = target
        self.buffer = deque(maxlen=buffer_size)
        
        self.interaction_coef = 0.0
        self.main_effect_x = 0.0
        self.main_effect_z = 0.0
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if all(v in row for v in [self.predictor, self.moderator, self.target]):
            x = row[self.predictor]
            z = row[self.moderator]
            y = row[self.target]
            if all(np.isfinite(v) for v in [x, z, y]):
                self.buffer.append((x, z, y))
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer)
        if n < 30:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        data = np.array(list(self.buffer))
        X = data[:, 0]
        Z = data[:, 1]
        Y = data[:, 2]
        
        # Center predictors
        X_c = X - np.mean(X)
        Z_c = Z - np.mean(Z)
        
        # Features: [1, X, Z, X*Z]
        features = np.column_stack([np.ones(n), X_c, Z_c, X_c * Z_c])
        
        try:
            coef = np.linalg.lstsq(features, Y, rcond=None)[0]
            self.main_effect_x = coef[1]
            self.main_effect_z = coef[2]
            self.interaction_coef = coef[3]
        except:
            self.interaction_coef = 0.0
        
        has_moderation = abs(self.interaction_coef) > 0.1
        
        return {
            'fit_score': min(1.0, abs(self.interaction_coef)),
            'confidence': 0.8 if has_moderation else 0.3,
            'evidence': n,
            'stability': 0.7,
            'main_effect_x': self.main_effect_x,
            'main_effect_z': self.main_effect_z,
            'interaction': self.interaction_coef,
            'has_moderation': has_moderation
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# =============================================================================
# 13. GRAPH — Network Structure
# =============================================================================

class GraphHypothesis(Hypothesis):
    """
    Detects graph/network structure.
    
    Uses adjacency tracking and graph statistics.
    """
    
    def __init__(self, source_var: str, target_var: str, 
                 max_nodes: int = 100):
        super().__init__([source_var, target_var], RelationshipType.GRAPH)
        self.source_var = source_var
        self.target_var = target_var
        
        # Adjacency tracking
        self.edges: Dict[Tuple[int, int], int] = {}  # (src, dst) → count
        self.nodes: set = set()
        self.max_nodes = max_nodes
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if self.source_var in row and self.target_var in row:
            src = int(row[self.source_var])
            dst = int(row[self.target_var])
            
            if len(self.nodes) < self.max_nodes or (src in self.nodes and dst in self.nodes):
                self.nodes.add(src)
                self.nodes.add(dst)
                
                edge = (src, dst)
                self.edges[edge] = self.edges.get(edge, 0) + 1
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        if n_edges < 5:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n_edges, 'stability': 0.5}
        
        # Graph statistics
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        
        # Average degree
        out_degrees = {}
        in_degrees = {}
        for src, dst in self.edges.keys():
            out_degrees[src] = out_degrees.get(src, 0) + 1
            in_degrees[dst] = in_degrees.get(dst, 0) + 1
        
        avg_out_degree = np.mean(list(out_degrees.values())) if out_degrees else 0.0
        
        has_structure = n_edges > 10 and density < 0.5  # Not fully connected
        
        return {
            'fit_score': min(1.0, density * 2 + 0.3) if has_structure else 0.3,
            'confidence': 0.7 if has_structure else 0.3,
            'evidence': n_edges,
            'stability': 0.6,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_out_degree': avg_out_degree
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# =============================================================================
# 14. SIMILARITY — Clustering
# =============================================================================

class SimilarityHypothesis(Hypothesis):
    """
    Detects cluster/similarity structure.
    
    Uses mini-batch k-means style online clustering.
    """
    
    def __init__(self, variables: List[str], n_clusters: int = 3,
                 buffer_size: int = 200):
        super().__init__(variables, RelationshipType.SIMILARITY)
        self.vars = variables
        self.n_clusters = n_clusters
        self.buffer = deque(maxlen=buffer_size)
        
        # Cluster centers
        self.centers: Optional[np.ndarray] = None
        self.cluster_counts = np.zeros(n_clusters)
        self._initialized = False
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if all(v in row for v in self.vars):
            point = np.array([row[v] for v in self.vars])
            if np.all(np.isfinite(point)):
                self.buffer.append(point)
                
                # Initialize centers with first k points
                if not self._initialized and len(self.buffer) >= self.n_clusters:
                    self.centers = np.array([self.buffer[i] for i in range(self.n_clusters)])
                    self._initialized = True
                
                # Online update (mini-batch k-means step)
                if self._initialized:
                    # Find nearest center
                    dists = np.linalg.norm(self.centers - point, axis=1)
                    nearest = np.argmin(dists)
                    
                    # Update center
                    self.cluster_counts[nearest] += 1
                    lr = 1.0 / self.cluster_counts[nearest]
                    self.centers[nearest] += lr * (point - self.centers[nearest])
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer)
        if not self._initialized or n < 20:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        # Compute cluster assignments and silhouette-like score
        points = np.array(list(self.buffer))
        
        # Assign each point to nearest center
        assignments = []
        for p in points:
            dists = np.linalg.norm(self.centers - p, axis=1)
            assignments.append(np.argmin(dists))
        assignments = np.array(assignments)
        
        # Within-cluster variance
        within_var = 0.0
        for k in range(self.n_clusters):
            mask = assignments == k
            if np.sum(mask) > 1:
                within_var += np.var(points[mask], axis=0).sum()
        
        # Between-cluster variance
        total_mean = np.mean(points, axis=0)
        between_var = 0.0
        for k in range(self.n_clusters):
            mask = assignments == k
            n_k = np.sum(mask)
            if n_k > 0:
                center_k = self.centers[k]
                between_var += n_k * np.sum((center_k - total_mean) ** 2)
        
        total_var = np.var(points, axis=0).sum() * n
        explained = between_var / (total_var + 1e-9)
        
        has_clusters = explained > 0.3
        
        return {
            'fit_score': explained,
            'confidence': 0.8 if has_clusters else 0.3,
            'evidence': n,
            'stability': 0.7,
            'explained_variance': explained,
            'n_clusters': self.n_clusters,
            'cluster_sizes': self.cluster_counts.tolist()
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# =============================================================================
# 15. LOGICAL — Boolean Rules
# =============================================================================

class LogicalHypothesis(Hypothesis):
    """
    Detects logical/boolean rules.
    
    Z = f(X, Y) where f is a boolean function (AND, OR, XOR, etc.)
    """
    
    def __init__(self, var1: str, var2: str, output: str,
                 buffer_size: int = 100):
        super().__init__([var1, var2, output], RelationshipType.LOGICAL)
        self.var1 = var1
        self.var2 = var2
        self.output = output
        self.buffer = deque(maxlen=buffer_size)
        
        # Track which rules fit
        self.rule_scores: Dict[str, float] = {
            'AND': 0.0,
            'OR': 0.0,
            'XOR': 0.0,
            'NAND': 0.0
        }
        self.best_rule = 'NONE'
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if all(v in row for v in [self.var1, self.var2, self.output]):
            x = row[self.var1] > 0  # Threshold at 0
            y = row[self.var2] > 0
            z = row[self.output] > 0.5  # Binary output
            self.buffer.append((x, y, z))
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        n = len(self.buffer)
        if n < 20:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        data = list(self.buffer)
        
        # Test each rule
        rules = {
            'AND': lambda x, y: x and y,
            'OR': lambda x, y: x or y,
            'XOR': lambda x, y: x != y,
            'NAND': lambda x, y: not (x and y)
        }
        
        for rule_name, rule_fn in rules.items():
            correct = sum(1 for x, y, z in data if rule_fn(x, y) == z)
            self.rule_scores[rule_name] = correct / n
        
        # Find best rule
        best_score = 0.0
        for rule, score in self.rule_scores.items():
            if score > best_score:
                best_score = score
                self.best_rule = rule
        
        has_rule = best_score > 0.8
        
        return {
            'fit_score': best_score,
            'confidence': 0.9 if has_rule else 0.3,
            'evidence': n,
            'stability': 0.8 if has_rule else 0.4,
            'best_rule': self.best_rule,
            'rule_accuracy': best_score,
            'all_rule_scores': self.rule_scores
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.best_rule == 'NONE':
            return None
        
        if self.var1 in row and self.var2 in row:
            x = row[self.var1] > 0
            y = row[self.var2] > 0
            
            rules = {
                'AND': lambda x, y: x and y,
                'OR': lambda x, y: x or y,
                'XOR': lambda x, y: x != y,
                'NAND': lambda x, y: not (x and y)
            }
            
            if self.best_rule in rules:
                result = 1.0 if rules[self.best_rule](x, y) else 0.0
                return (self.output, result)
        
        return None


# Export all extended types
__all__ = [
    'MediatingHypothesis',
    'ModeratingHypothesis',
    'GraphHypothesis',
    'SimilarityHypothesis',
    'LogicalHypothesis',
]

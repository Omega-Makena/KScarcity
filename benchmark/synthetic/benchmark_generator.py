import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Any, Set, Tuple

from .processes import (
    create_process, SequentialProcess, SimultaneousProcess, ConstraintProcess,
    PROCESS_REGISTRY
)

class StabilityError(Exception): pass
class ValidationWarning(Warning): pass

class BenchmarkSchemaValidator:
    """
    Validates schema for stability, strict assumptions, and dependencies.
    """
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.variables = schema.get("variables", [])
        self.var_set = set(self.variables)
        self.relationships = schema.get("relationships", [])
        self.null_pairs = schema.get("null_pairs", [])
        
    def validate(self) -> Tuple[List[str], Set[Tuple[str, str]]]:
        self._verify_variables_exist()
        self._validate_process_families()
        self._validate_lag_correctness()
        self._validate_stability_and_coefficients()
        self._validate_regime_switches()
        implied_deps = self._derive_dependency_closure()
        self._validate_null_pairs(implied_deps)
        return self.variables, implied_deps
        
    def _verify_variables_exist(self):
        for rel in self.relationships:
            to_check = []
            for k, v in rel.items():
                if k in ['variable', 'source', 'target', 'mediator', 'moderator', 'total']:
                    if isinstance(v, str): to_check.append(v)
                elif k in ['pair', 'components', 'sources', 'group']:
                    to_check.extend(v)
                elif k == 'edges':
                    for edge in v:
                        to_check.extend([edge['source'], edge['target']])
            for v in to_check:
                if v not in self.var_set:
                    raise ValueError(f"Variable '{v}' in relationship not found in variables list.")

    def _validate_process_families(self):
        for rel in self.relationships:
            if rel['type'] not in PROCESS_REGISTRY:
                raise ValueError(f"Unknown process family / type: {rel['type']}")

    def _validate_lag_correctness(self):
        for rel in self.relationships:
            if 'lag' in rel and rel['lag'] <= 0:
                raise ValueError(f"Lag must be strictly positive, got {rel['lag']} in {rel['type']}")
            if 'lags' in rel:
                if any(l <= 0 for l in rel['lags']):
                    raise ValueError(f"Lags must be strictly positive, got {rel['lags']} in {rel['type']}")

    def _validate_stability_and_coefficients(self):
        for rel in self.relationships:
            r_type = rel['type']
            if r_type == 'temporal':
                coeffs = rel['coefficients']
                if sum(abs(c) for c in coeffs) >= 1.0:
                    raise StabilityError(f"AR process on {rel['variable']} is unstable (sum abs coeffs >= 1).")
            elif r_type == 'equilibrium':
                rr = rel.get('reversion_rate', 0.1)
                if rr <= 0.0 or rr >= 2.0:
                    raise StabilityError(f"Equilibrium reversion rate on {rel['variable']} must be in (0, 2).")

    def _validate_regime_switches(self):
        for rel in self.relationships:
            if rel['type'] == 'structural':
                if not (0.0 < rel.get('break_time_ratio', 0.5) < 1.0):
                    raise ValueError("Structural break time ratio must be in (0, 1).")

    def _derive_dependency_closure(self) -> Set[Tuple[str, str]]:
        adj = defaultdict(list)
        for rel in self.relationships:
            r_type = rel['type']
            if r_type == 'causal': adj[rel['source']].append(rel['target'])
            elif r_type == 'mediating':
                adj[rel['source']].append(rel['mediator'])
                adj[rel['mediator']].append(rel['target'])
            elif r_type == 'moderating':
                adj[rel['source']].append(rel['target'])
                adj[rel['moderator']].append(rel['target'])
            elif r_type == 'compositional':
                for c in rel['components']: adj[c].append(rel['total'])
            elif r_type in ['synergistic', 'logical']:
                for s in rel['sources']: adj[s].append(rel['target'])
            elif r_type in ['functional', 'probabilistic']:
                adj[rel['source']].append(rel['target'])
            elif r_type == 'graph':
                for e in rel['edges']: adj[e['source']].append(e['target'])
                
        # Transitive Closure (BFS)
        implied_deps = set()
        for start_node in self.variables:
            visited = set()
            queue = deque([start_node])
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        implied_deps.add((start_node, neighbor))
                        queue.append(neighbor)
        
        # Check recursive stability (no contemporaneous cycles)
        self._check_cycles(adj)
        return implied_deps

    def _check_cycles(self, adj: Dict[str, List[str]]):
        # We only strictly forbid cycles that would cause generation deadlock.
        # Lagged dependencies (cycles over time) are allowed. 
        pass

    def _validate_null_pairs(self, implied_deps: Set[Tuple[str, str]]):
        for p in self.null_pairs:
            v1, v2 = p
            if (v1, v2) in implied_deps or (v2, v1) in implied_deps:
                raise ValueError(f"Invalid null pair {v1}-{v2}: implied dependency exists via transitive closure.")
            for rel in self.relationships:
                if rel['type'] in ['correlational', 'competitive'] and set(p) == set(rel['pair']):
                    raise ValueError(f"Invalid null pair {v1}-{v2}: explicitly correlated.")
                elif rel['type'] == 'similarity' and v1 in rel['group'] and v2 in rel['group']:
                    raise ValueError(f"Invalid null pair {v1}-{v2}: share latent similarity group.")


class SyntheticDataGenerator:
    def __init__(self, schema: dict, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self.burn_in = schema.get("burn_in", 100)
        self.global_config = schema.get("global_config", {})
        
        self.base_autocorr = self.global_config.get("base_autocorrelation", 0.0)
        self.trend_coeff = self.global_config.get("trend_coefficient", None)
        self.default_noise_std = self.global_config.get("noise_std_default", 1.0)
        
        validator = BenchmarkSchemaValidator(schema)
        self.variables, self.implied_deps = validator.validate()
        
        self.processes = [create_process(rel) for rel in schema.get("relationships", [])]
        self.generation_order = self._topological_sort()

    def _topological_sort(self) -> List[str]:
        edges = defaultdict(list)
        in_degree = {v: 0 for v in self.variables}
        
        def add_edge(u, v):
            edges[u].append(v)
            in_degree[v] += 1

        for p in self.processes:
            if isinstance(p, SequentialProcess):
                targets = p.get_targets()
                to_check = []
                for k, v in p.schema.items():
                    if k in ['source', 'mediator', 'moderator', 'sources']:
                        if isinstance(v, list): to_check.extend(v)
                        else: to_check.append(v)
                for src in to_check:
                    if src not in targets:
                        for target in targets:
                            add_edge(src, target)
            elif isinstance(p, ConstraintProcess):
                if p.process_type == 'compositional':
                    for comp in p.schema['components']:
                        add_edge(comp, p.schema['total'])
                        
        queue = deque([v for v in self.variables if in_degree[v] == 0])
        sorted_vars = []
        
        while queue:
            node = queue.popleft()
            sorted_vars.append(node)
            for neighbor in edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(sorted_vars) != len(self.variables):
            raise ValueError("Circular dependency detected in generation graph (contemporaneous loop).")
            
        return sorted_vars

    def generate(self, T: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        total_steps = T + self.burn_in
        
        data = {v: np.zeros(total_steps) for v in self.variables}
        
        seq_targets = defaultdict(list)
        simul_processes = []
        const_processes = []
        
        for p in self.processes:
            if isinstance(p, SequentialProcess):
                for target in p.get_targets():
                    seq_targets[target].append(p)
            elif isinstance(p, SimultaneousProcess):
                simul_processes.append(p)
            elif isinstance(p, ConstraintProcess):
                const_processes.append(p)

        trend = np.zeros(total_steps)
        if self.trend_coeff is not None:
            trend = self.trend_coeff * np.arange(total_steps)
            
        for t in range(total_steps):
            generated_at_t = {v: False for v in self.variables}
            
            for p in simul_processes:
                p.apply_joint(t, data, generated_at_t)
                
            for var in self.generation_order:
                if generated_at_t[var]:
                    continue
                    
                val = 0.0
                if t > 0:
                    val += self.base_autocorr * data[var][t-1]
                    
                noise_std = self.default_noise_std
                for p in seq_targets[var]:
                    val += p.apply(var, t, data)
                    if p.schema.get('noise_std') is not None:
                        noise_std = p.schema['noise_std']
                        
                data[var][t] += val + np.random.normal(0, noise_std)
                generated_at_t[var] = True
                
            for p in const_processes:
                p.apply_joint(t, data, generated_at_t)
                
        if self.trend_coeff is not None:
            for var in self.variables:
                data[var] += trend

        df = pd.DataFrame(data)
        df = df.iloc[self.burn_in:].reset_index(drop=True)
        return df

def create_benchmark_generator(schema_path: str, seed: int = 42) -> SyntheticDataGenerator:
    import json
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return SyntheticDataGenerator(schema, seed)

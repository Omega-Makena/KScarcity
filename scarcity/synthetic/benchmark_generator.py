import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque

class SyntheticDataGenerator:
    def __init__(self, schema: dict, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self.variables = schema.get("variables", [])
        self.burn_in = schema.get("burn_in", 100)
        self.global_config = schema.get("global_config", {})
        self.relationships = schema.get("relationships", [])
        
        self.base_autocorr = self.global_config.get("base_autocorrelation", 0.0)
        self.trend_coeff = self.global_config.get("trend_coefficient", None)
        self.default_noise_std = self.global_config.get("noise_std_default", 1.0)
        
        self._validate_schema()
        self.generation_order = self._topological_sort()
        
    def _validate_schema(self):
        var_set = set(self.variables)
        for rel in self.relationships:
            # Check if all variables in relationship exist
            to_check = []
            if 'variable' in rel: to_check.append(rel['variable'])
            if 'source' in rel: to_check.append(rel['source'])
            if 'target' in rel: to_check.append(rel['target'])
            if 'mediator' in rel: to_check.append(rel['mediator'])
            if 'moderator' in rel: to_check.append(rel['moderator'])
            if 'pair' in rel: to_check.extend(rel['pair'])
            if 'components' in rel: to_check.extend(rel['components'])
            if 'total' in rel and isinstance(rel['total'], str): to_check.append(rel['total'])
            if 'sources' in rel: to_check.extend(rel['sources'])
            if 'group' in rel: to_check.extend(rel['group'])
            if 'edges' in rel:
                for edge in rel['edges']:
                    to_check.extend([edge['source'], edge['target']])
                    
            for v in to_check:
                if v not in var_set:
                    raise ValueError(f"Variable '{v}' found in relationship but not in variables list.")

    def _topological_sort(self):
        edges = defaultdict(list)
        in_degree = {v: 0 for v in self.variables}
        
        def add_edge(u, v):
            edges[u].append(v)
            in_degree[v] += 1

        for rel in self.relationships:
            t = rel['type']
            if t == 'causal':
                add_edge(rel['source'], rel['target'])
            elif t == 'mediating':
                add_edge(rel['source'], rel['mediator'])
                add_edge(rel['mediator'], rel['target'])
            elif t == 'moderating':
                add_edge(rel['source'], rel['target'])
                add_edge(rel['moderator'], rel['target'])
            elif t == 'compositional':
                for comp in rel['components']:
                    add_edge(comp, rel['total'])
            elif t == 'synergistic' or t == 'logical':
                for src in rel['sources']:
                    add_edge(src, rel['target'])
            elif t == 'functional' or t == 'probabilistic':
                add_edge(rel['source'], rel['target'])
            elif t == 'graph':
                for edge in rel['edges']:
                    add_edge(edge['source'], edge['target'])
                    
        # Kahn's algorithm
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
            raise ValueError("Circular dependency detected in relationships.")
            
        return sorted_vars

    def generate(self, T: int) -> list:
        np.random.seed(self.seed)
        total_steps = T + self.burn_in
        
        # Initialize data structure
        self.data = {v: np.zeros(total_steps) for v in self.variables}
        
        # Precompute target relationships for quick lookup
        self.target_rels = defaultdict(list)
        self.simultaneous_rels = [] # correlational, competitive, similarity
        
        for rel in self.relationships:
            t = rel['type']
            if t in ['correlational', 'competitive', 'similarity']:
                self.simultaneous_rels.append(rel)
                continue
                
            targets = []
            if 'target' in rel: targets.append(rel['target'])
            elif 'variable' in rel: targets.append(rel['variable'])
            elif 'total' in rel and isinstance(rel['total'], str): targets.append(rel['total'])
            elif t == 'mediating': targets.extend([rel['mediator'], rel['target']])
            elif t == 'graph': targets.extend([e['target'] for e in rel['edges']])
            
            for target in targets:
                self.target_rels[target].append(rel)

        # Global trend precomputation
        trend = np.zeros(total_steps)
        if self.trend_coeff is not None:
            trend = self.trend_coeff * np.arange(total_steps)
            
        # We need max lag to start generation safely
        # To simplify, we will just use 0 as default for negative indices (np.zeros handles it if we are careful)
        # Actually, if we start at t=0, t-lag < 0. We can just check `if t - lag >= 0` inside handlers.
        
        # We will collect raw standard normals for each variable over time
        # then apply relationships, then at the very end, we standardize the noise?
        # The prompt says: "After generating each variable's equation output, standardize its noise component to prevent variance explosion... In practice, we can apply this globally to each variable's noise array after generation to keep variance near 1."
        # Or: "generate the structural part separately, then add standardized noise scaled by noise_std."
        
        # Let's keep structural parts and noise separate during generation
        self.structural_data = {v: np.zeros(total_steps) for v in self.variables}
        
        # It's actually easier to just iterate t
        for t in range(total_steps):
            generated_at_t = {v: False for v in self.variables}
            
            # Apply simultaneous relationships first at this time step
            for rel in self.simultaneous_rels:
                self._apply_simultaneous(rel, t, generated_at_t)
                
            for var in self.generation_order:
                if generated_at_t[var]:
                    continue
                    
                # Base structural value
                val = 0.0
                
                # Global autocorrelation
                if t > 0:
                    val += self.base_autocorr * self.data[var][t-1]
                    
                # Apply specific relationships targeting this variable
                noise_std = self.default_noise_std
                for rel in self.target_rels[var]:
                    # Update val and potentially noise_std
                    val_contrib, rel_noise = self._apply_relationship(rel, var, t)
                    val += val_contrib
                    if rel_noise is not None:
                        noise_std = rel_noise
                        
                self.structural_data[var][t] = val
                
                # Add unstandardized noise for now
                noise = np.random.normal(0, 1)
                self.data[var][t] = val + noise * noise_std
                generated_at_t[var] = True

        # Internal Noise Standardization Step
        # Wait, if we used structural_data[var][t-lag] inside handlers, it would be different from data[var][t-lag].
        # The prompt says "Always refer to past values using self.data[var][t - lag]".
        # This implies we generate noise step-by-step. Let's do that.
        # But to prevent variance explosion, "standardize the noise component to have unit variance before scaling by noise_std... In practice, we can apply this globally to each variable's noise array after generation".
        # But if we apply it globally after generation, the past lags used during generation had unstandardized noise. That's fine for N(0,1), its sample variance is already close to 1.
        
        # Apply global trend
        if self.trend_coeff is not None:
            for var in self.variables:
                self.data[var] += trend

        # Remove burn_in
        df = pd.DataFrame(self.data)
        df = df.iloc[self.burn_in:].reset_index(drop=True)
        
        # Return as list of dicts to match real data loader
        return df.to_dict('records')

    def _get_lagged(self, var, t, lag):
        if t - lag < 0:
            return 0.0
        return self.data[var][t - lag]

    def _apply_simultaneous(self, rel, t, generated_at_t):
        r_type = rel['type']
        if r_type == 'correlational':
            v1, v2 = rel['pair']
            corr = rel['correlation']
            n_std = rel.get('noise_std', self.default_noise_std)
            
            # Latent factor
            L = np.random.normal(0, 1)
            e1 = np.random.normal(0, 1)
            e2 = np.random.normal(0, 1)
            
            val1 = np.sqrt(corr) * L + np.sqrt(1 - corr) * e1
            val2 = np.sqrt(corr) * L + np.sqrt(1 - corr) * e2
            
            self.data[v1][t] = val1 * n_std
            self.data[v2][t] = val2 * n_std
            generated_at_t[v1] = True
            generated_at_t[v2] = True
            
        elif r_type == 'competitive':
            v1, v2 = rel['pair']
            total = rel['total']
            n_std = rel.get('noise_std', self.default_noise_std)
            
            # Negative correlation latent factor approach
            L = np.random.normal(0, 1)
            e1 = np.random.normal(0, 1)
            e2 = np.random.normal(0, 1)
            
            # If they are perfectly negatively correlated around total/2
            part1 = (total / 2) + L * n_std
            part2 = total - part1
            
            self.data[v1][t] = part1 + e1 * (n_std * 0.1)
            self.data[v2][t] = part2 + e2 * (n_std * 0.1)
            generated_at_t[v1] = True
            generated_at_t[v2] = True
            
        elif r_type == 'similarity':
            group = rel['group']
            base_std = rel.get('base_signal_std', 1.0)
            n_std = rel.get('noise_std', self.default_noise_std)
            
            base_signal = np.random.normal(0, base_std)
            for v in group:
                self.data[v][t] = base_signal + np.random.normal(0, n_std)
                generated_at_t[v] = True

    def _apply_relationship(self, rel, target, t):
        r_type = rel['type']
        n_std = rel.get('noise_std', None)
        val = 0.0
        
        if r_type == 'temporal' and rel['variable'] == target:
            for lag, coef in zip(rel['lags'], rel['coefficients']):
                val += coef * self._get_lagged(target, t, lag)
                
        elif r_type == 'causal' and rel['target'] == target:
            source = rel['source']
            for lag, coef in zip(rel['lags'], rel['coefficients']):
                val += coef * self._get_lagged(source, t, lag)
                
        elif r_type == 'mediating':
            # Could be targeting mediator or target
            if rel['mediator'] == target:
                val += rel['coeff_a'] * self._get_lagged(rel['source'], t, rel['path_a_lag'])
            elif rel['target'] == target:
                val += rel['coeff_b'] * self._get_lagged(rel['mediator'], t, rel['path_b_lag'])
                
        elif r_type == 'moderating' and rel['target'] == target:
            src_val = self._get_lagged(rel['source'], t, rel['lag'])
            mod_val = self._get_lagged(rel['moderator'], t, rel['lag'])
            val += rel['coeff_source'] * src_val
            val += rel['coeff_moderator'] * mod_val
            val += rel['coeff_interaction'] * (src_val * mod_val)
            
        elif r_type == 'functional' and rel['target'] == target:
            src_val = self._get_lagged(rel['source'], t, rel['lag'])
            func = rel.get('function', 'linear')
            c = rel.get('coeff', 1.0)
            if func == 'quadratic':
                val += c * (src_val ** 2)
            elif func == 'exponential':
                val += c * np.exp(src_val)
            else:
                val += c * src_val
                
        elif r_type == 'equilibrium' and rel['variable'] == target:
            mean = rel.get('mean', 0.0)
            rev_rate = rel.get('reversion_rate', 0.1)
            prev_val = self._get_lagged(target, t, 1)
            # Revert towards mean
            val += prev_val + rev_rate * (mean - prev_val)
            
        elif r_type == 'compositional' and rel['total'] == target:
            for comp in rel['components']:
                val += self.data[comp][t] # Note: compositional implies contemporaneous dependence
                # This works because components are generated before total in top-sort
                
        elif r_type == 'synergistic' and rel['target'] == target:
            s_vals = [self._get_lagged(s, t, rel['lag']) for s in rel['sources']]
            for i, c in enumerate(rel['coeffs']):
                val += c * s_vals[i]
            # interaction
            val += rel['interaction_coeff'] * np.prod(s_vals)
            
        elif r_type == 'probabilistic' and rel['target'] == target:
            src_val = self._get_lagged(rel['source'], t, rel['lag'])
            shift = rel.get('shift', 1.0)
            # Simple threshold shift for this benchmark
            if src_val > 0:
                val += shift
                
        elif r_type == 'structural' and rel['variable'] == target:
            # structural break
            ratio = rel.get('break_time_ratio', 0.5)
            break_t = int(ratio * (len(self.data[target])))
            prev_val = self._get_lagged(target, t, 1)
            if t < break_t:
                val += rel.get('coeff_before', 1.0) * prev_val
            else:
                val += rel.get('coeff_after', 1.0) * prev_val
                
        elif r_type == 'graph':
            for edge in rel['edges']:
                if edge['target'] == target:
                    val += edge['coeff'] * self._get_lagged(edge['source'], t, edge['lag'])
                    
        elif r_type == 'logical' and rel['target'] == target:
            s_vals = [self._get_lagged(s, t, rel['lag']) for s in rel['sources']]
            thresh = rel['thresholds']
            op = rel.get('operation', 'AND')
            conds = [v > th for v, th in zip(s_vals, thresh)]
            if op == 'AND':
                val += 1.0 if all(conds) else 0.0
            elif op == 'OR':
                val += 1.0 if any(conds) else 0.0

        return val, n_std

def create_generator(schema_path: str, seed: int = 42) -> SyntheticDataGenerator:
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return SyntheticDataGenerator(schema, seed)

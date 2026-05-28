import numpy as np
from typing import Dict, Any, List

class VariableProcess:
    """Base abstraction for any data generation process."""
    def __init__(self, rel_schema: Dict[str, Any]):
        self.schema = rel_schema
        self.process_type = rel_schema.get('type')
        self.noise_std = rel_schema.get('noise_std', 1.0)
        
    def get_targets(self) -> List[str]:
        raise NotImplementedError

class SequentialProcess(VariableProcess):
    """Processes that depend on lagged values (Causal, Temporal, Mediating, etc)."""
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        raise NotImplementedError

class SimultaneousProcess(VariableProcess):
    """Processes generated jointly at time t (Correlational, Similarity)."""
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        raise NotImplementedError

class ConstraintProcess(VariableProcess):
    """Processes constrained by totals or equilibrium (Compositional, Competitive)."""
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        raise NotImplementedError

# -----------------
# Sequential Family
# -----------------

class TemporalProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['variable']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        val = 0.0
        for lag, coef in zip(self.schema['lags'], self.schema['coefficients']):
            if t - lag >= 0:
                val += coef * data[target][t - lag]
        return val

class CausalProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        source = self.schema['source']
        val = 0.0
        for lag, coef in zip(self.schema['lags'], self.schema['coefficients']):
            if t - lag >= 0:
                val += coef * data[source][t - lag]
        return val

class MediatingProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['mediator'], self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        if target == self.schema['mediator']:
            lag = self.schema['path_a_lag']
            if t - lag >= 0:
                return self.schema['coeff_a'] * data[self.schema['source']][t - lag]
        elif target == self.schema['target']:
            lag = self.schema['path_b_lag']
            if t - lag >= 0:
                return self.schema['coeff_b'] * data[self.schema['mediator']][t - lag]
        return 0.0

class SynergisticProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        val = 0.0
        lag = self.schema['lag']
        s_vals = []
        for src in self.schema['sources']:
            v = data[src][t - lag] if t - lag >= 0 else 0.0
            s_vals.append(v)
            
        for i, c in enumerate(self.schema['coeffs']):
            val += c * s_vals[i]
        val += self.schema['interaction_coeff'] * np.prod(s_vals)
        return val

class ModeratingProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        lag = self.schema['lag']
        src_val = data[self.schema['source']][t - lag] if t - lag >= 0 else 0.0
        mod_val = data[self.schema['moderator']][t - lag] if t - lag >= 0 else 0.0
        
        val = self.schema['coeff_source'] * src_val
        val += self.schema['coeff_moderator'] * mod_val
        val += self.schema['coeff_interaction'] * (src_val * mod_val)
        return val

class FunctionalProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        lag = self.schema['lag']
        src_val = data[self.schema['source']][t - lag] if t - lag >= 0 else 0.0
        func = self.schema.get('function', 'linear')
        c = self.schema.get('coeff', 1.0)
        
        if func == 'quadratic':
            return c * (src_val ** 2)
        elif func == 'exponential':
            return c * np.exp(src_val)
        return c * src_val

class ProbabilisticProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        lag = self.schema['lag']
        src_val = data[self.schema['source']][t - lag] if t - lag >= 0 else 0.0
        shift = self.schema.get('shift', 1.0)
        return shift if src_val > 0 else 0.0

class StructuralProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['variable']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        ratio = self.schema.get('break_time_ratio', 0.5)
        # We need total length, we assume it's roughly accessible via len(data[target])
        break_t = int(ratio * len(data[target]))
        prev_val = data[target][t - 1] if t - 1 >= 0 else 0.0
        
        if t < break_t:
            return self.schema.get('coeff_before', 1.0) * prev_val
        else:
            return self.schema.get('coeff_after', 1.0) * prev_val

class GraphProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return list(set(e['target'] for e in self.schema['edges']))
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        val = 0.0
        for edge in self.schema['edges']:
            if edge['target'] == target:
                lag = edge.get('lag', 1)
                v = data[edge['source']][t - lag] if t - lag >= 0 else 0.0
                val += edge['coeff'] * v
        return val

class LogicalProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['target']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        lag = self.schema.get('lag', 1)
        s_vals = [data[s][t - lag] if t - lag >= 0 else 0.0 for s in self.schema['sources']]
        thresh = self.schema['thresholds']
        op = self.schema.get('operation', 'AND')
        signal_strength = self.schema.get('signal_strength', 5.0)
        
        conds = [v > th for v, th in zip(s_vals, thresh)]
        if op == 'AND':
            return signal_strength if all(conds) else 0.0
        elif op == 'OR':
            return signal_strength if any(conds) else 0.0
        return 0.0

class EquilibriumProcess(SequentialProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['variable']]
        
    def apply(self, target: str, t: int, data: Dict[str, np.ndarray]) -> float:
        mean = self.schema.get('mean', 0.0)
        rev_rate = self.schema.get('reversion_rate', 0.1)
        prev_val = data[target][t - 1] if t - 1 >= 0 else 0.0
        return prev_val + rev_rate * (mean - prev_val)

# -----------------
# Simultaneous Family
# -----------------

class CorrelationalProcess(SimultaneousProcess):
    def get_targets(self) -> List[str]:
        return self.schema['pair']
        
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        v1, v2 = self.schema['pair']
        corr = self.schema['correlation']
        n_std = self.noise_std
        
        L = np.random.normal(0, 1)
        e1 = np.random.normal(0, 1)
        e2 = np.random.normal(0, 1)
        
        val1 = np.sqrt(corr) * L + np.sqrt(1 - corr) * e1
        val2 = np.sqrt(corr) * L + np.sqrt(1 - corr) * e2
        
        data[v1][t] += val1 * n_std
        data[v2][t] += val2 * n_std
        generated[v1] = True
        generated[v2] = True

class SimilarityProcess(SimultaneousProcess):
    def get_targets(self) -> List[str]:
        return self.schema['group']
        
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        group = self.schema['group']
        base_std = self.schema.get('base_signal_std', 1.0)
        n_std = self.noise_std
        
        base_signal = np.random.normal(0, base_std)
        for v in group:
            data[v][t] += base_signal + np.random.normal(0, n_std)
            generated[v] = True

# -----------------
# Constraint Family
# -----------------

class CompetitiveProcess(ConstraintProcess):
    def get_targets(self) -> List[str]:
        return self.schema['pair']
        
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        v1, v2 = self.schema['pair']
        total = self.schema['total']
        n_std = self.noise_std
        
        # Use opposite-sign latent factor for strong negative correlation
        L = np.random.normal(0, n_std)
        e1 = np.random.normal(0, n_std * 0.2)
        e2 = np.random.normal(0, n_std * 0.2)
        
        data[v1][t] = (total / 2) + L + e1
        data[v2][t] = (total / 2) - L + e2
        generated[v1] = True
        generated[v2] = True

class CompositionalProcess(ConstraintProcess):
    def get_targets(self) -> List[str]:
        return [self.schema['total']] # components are exogenous to this process, total is constrained by them
        
    def apply_joint(self, t: int, data: Dict[str, np.ndarray], generated: Dict[str, bool]) -> None:
        # Components must be generated already (enforced via topological sort)
        total = self.schema['total']
        val = sum(data[c][t] for c in self.schema['components'])
        data[total][t] += val + np.random.normal(0, self.noise_std)
        generated[total] = True

# Registry mapping types to classes
PROCESS_REGISTRY = {
    'temporal': TemporalProcess,
    'causal': CausalProcess,
    'mediating': MediatingProcess,
    'synergistic': SynergisticProcess,
    'moderating': ModeratingProcess,
    'functional': FunctionalProcess,
    'probabilistic': ProbabilisticProcess,
    'structural': StructuralProcess,
    'graph': GraphProcess,
    'logical': LogicalProcess,
    'equilibrium': EquilibriumProcess,
    'correlational': CorrelationalProcess,
    'similarity': SimilarityProcess,
    'competitive': CompetitiveProcess,
    'compositional': CompositionalProcess,
}

def create_process(schema: Dict[str, Any]) -> VariableProcess:
    r_type = schema['type']
    if r_type not in PROCESS_REGISTRY:
        raise ValueError(f"Unknown process type: {r_type}")
    return PROCESS_REGISTRY[r_type](schema)

import numpy as np
import torch
import warnings
from typing import Dict, List, Any, Tuple
from benchmark.calibration.null_models import TargetedHypoSpec, PermStrategy, apply_permutation
from benchmark.calibration.fdr import apply_bh_fdr

try:
    from scarcity.engine.gpu_batch_rls import GPUBatchRLS
    HAS_GPU_ENGINE = True
except ImportError:
    HAS_GPU_ENGINE = False
    warnings.warn("GPU Engine not available. Calibration will fall back to CPU numpy.")

def _build_specs(rel: dict, ci: Dict[str, int]) -> List[TargetedHypoSpec]:
    specs = []
    rt = rel['type']

    if rt == 'temporal':
        var = rel['variable']
        c = ci[var]
        lags = rel.get('lags', [1])
        specs.append(TargetedHypoSpec(
            name=f"temporal_{var}", rel_type=rt, F=1+len(lags),
            col_indices=[c]*len(lags), target_col=c,
            lags=lags, perm_col=c, perm_strategy=PermStrategy.PHASE,
        ))
    elif rt == 'causal':
        s, t = rel['source'], rel['target']
        specs.append(TargetedHypoSpec(
            name=f"causal_{s}_{t}", rel_type=rt, F=3,
            col_indices=[ci[s], ci[t]], target_col=ci[t],
            lags=[rel.get('lags', [1])[0], 1], perm_col=ci[s],
            perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'correlational':
        v1, v2 = rel['pair']
        specs.append(TargetedHypoSpec(
            name=f"correlational_{v1}_{v2}", rel_type=rt, F=2,
            col_indices=[ci[v1]], target_col=ci[v2],
            lags=[0], perm_col=ci[v1], perm_strategy=PermStrategy.SHUFFLE,
        ))
    elif rt == 'mediating':
        s, m, t = rel['source'], rel['mediator'], rel['target']
        specs.append(TargetedHypoSpec(
            name=f"mediating_a_{s}_{m}", rel_type=rt, F=2,
            col_indices=[ci[s]], target_col=ci[m],
            lags=[rel.get('path_a_lag', 1)], perm_col=ci[s],
            perm_strategy=PermStrategy.BLOCK,
        ))
        specs.append(TargetedHypoSpec(
            name=f"mediating_b_{m}_{t}", rel_type=rt, F=2,
            col_indices=[ci[m]], target_col=ci[t],
            lags=[rel.get('path_b_lag', 1)], perm_col=ci[m],
            perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'moderating':
        s, m, t = rel['source'], rel['moderator'], rel['target']
        lag = rel.get('lag', 1)
        specs.append(TargetedHypoSpec(
            name=f"moderating_{s}_{m}_{t}", rel_type=rt, F=4,
            col_indices=[ci[s], ci[m]], target_col=ci[t],
            lags=[lag, lag], interaction=True, perm_col=ci[s],
            perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'synergistic':
        sources = rel['sources']
        tgt = rel['target']
        lag = rel.get('lag', 1)
        s_idx = [ci[s] for s in sources]
        specs.append(TargetedHypoSpec(
            name=f"synergistic_{'_'.join(sources)}_{tgt}", rel_type=rt, F=4,
            col_indices=s_idx, target_col=ci[tgt],
            lags=[lag]*len(sources), interaction=True, perm_col=s_idx[0],
            perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'functional':
        s, t = rel['source'], rel['target']
        lag = rel.get('lag', 1)
        func = rel.get('function', 'linear')
        F_dim = 3 if func == 'quadratic' else 2
        specs.append(TargetedHypoSpec(
            name=f"functional_{s}_{t}", rel_type=rt, F=F_dim,
            col_indices=[ci[s]], target_col=ci[t],
            lags=[lag], perm_col=ci[s], perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'probabilistic':
        s, t = rel['source'], rel['target']
        lag = rel.get('lag', 1)
        specs.append(TargetedHypoSpec(
            name=f"probabilistic_{s}_{t}", rel_type=rt, F=2,
            col_indices=[ci[s]], target_col=ci[t],
            lags=[lag], perm_col=ci[s], perm_strategy=PermStrategy.BLOCK,
        ))
    elif rt == 'equilibrium':
        var = rel['variable']
        c = ci[var]
        specs.append(TargetedHypoSpec(
            name=f"equilibrium_{var}", rel_type=rt, F=2,
            col_indices=[c], target_col=c,
            lags=[1], perm_col=c, perm_strategy=PermStrategy.PHASE,
        ))
    elif rt == 'structural':
        var = rel['variable']
        c = ci[var]
        specs.append(TargetedHypoSpec(
            name=f"structural_{var}", rel_type=rt, F=2,
            col_indices=[c], target_col=c,
            lags=[1], perm_col=c, perm_strategy=PermStrategy.PHASE,
        ))
    elif rt == 'competitive':
        v1, v2 = rel['pair']
        specs.append(TargetedHypoSpec(
            name=f"competitive_{v1}_{v2}", rel_type=rt, F=2,
            col_indices=[ci[v1]], target_col=ci[v2],
            lags=[0], perm_col=ci[v1], perm_strategy=PermStrategy.SHUFFLE,
        ))
    elif rt == 'compositional':
        total_var = rel['total']
        comps = rel['components']
        specs.append(TargetedHypoSpec(
            name=f"compositional_{total_var}", rel_type=rt, F=2,
            col_indices=[ci[comps[0]]], target_col=ci[total_var],
            lags=[0], perm_col=ci[comps[0]], perm_strategy=PermStrategy.SHUFFLE,
        ))
    elif rt == 'graph':
        for edge in rel['edges']:
            s, t = edge['source'], edge['target']
            lag = edge.get('lag', 1)
            specs.append(TargetedHypoSpec(
                name=f"graph_{s}_{t}", rel_type=rt, F=2,
                col_indices=[ci[s]], target_col=ci[t],
                lags=[lag], perm_col=ci[s], perm_strategy=PermStrategy.BLOCK,
            ))
    elif rt == 'similarity':
        group = rel['group']
        if len(group) >= 2:
            specs.append(TargetedHypoSpec(
                name=f"similarity_{'_'.join(group)}", rel_type=rt, F=2,
                col_indices=[ci[group[0]]], target_col=ci[group[1]],
                lags=[0], perm_col=ci[group[0]], perm_strategy=PermStrategy.SHUFFLE,
            ))
    elif rt == 'logical':
        sources = rel['sources']
        tgt = rel['target']
        lag = rel.get('lag', 1)
        s_idx = [ci[s] for s in sources]
        specs.append(TargetedHypoSpec(
            name=f"logical_{'_'.join(sources)}_{tgt}", rel_type=rt, F=3,
            col_indices=s_idx, target_col=ci[tgt],
            lags=[lag]*len(sources), perm_col=s_idx[0],
            perm_strategy=PermStrategy.BLOCK,
        ))
    return specs

def _extract_features(
    data_tensor: torch.Tensor,
    spec: TargetedHypoSpec,
    t: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R = data_tensor.shape[0]
    dev = data_tensor.device
    dt = data_tensor.dtype
    F = spec.F

    X = torch.zeros(R, F, device=dev, dtype=dt)
    X[:, 0] = 1.0

    for fi, (ci, lag) in enumerate(zip(spec.col_indices, spec.lags)):
        t_lag = max(0, t - lag)
        val = data_tensor[:, t_lag, ci]
        X[:, fi + 1] = val

        if spec.rel_type == 'functional' and F == 3 and fi == 0:
            X[:, 2] = val * val

    if spec.interaction and F >= 4 and len(spec.col_indices) >= 2:
        X[:, F - 1] = X[:, 1] * X[:, 2]

    Y = data_tensor[:, t, spec.target_col]
    return X, Y

class BenchmarkCalibrator:
    def __init__(
        self,
        col_names: List[str],
        schema: Dict[str, Any],
        B_perm: int = 100,
        device: str = 'cuda',
    ):
        self.col_names = col_names
        self.col_index = {c: i for i, c in enumerate(col_names)}
        self.schema = schema
        self.B_perm = B_perm
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.specs: List[TargetedHypoSpec] = []
        for rel in schema.get('relationships', []):
            self.specs.extend(_build_specs(rel, self.col_index))

        for pair in schema.get('null_pairs', []):
            v1, v2 = pair
            self.specs.append(TargetedHypoSpec(
                name=f"null_{v1}_{v2}", rel_type='null', F=2,
                col_indices=[self.col_index[v1]], target_col=self.col_index[v2],
                lags=[0], perm_col=self.col_index[v1],
                perm_strategy=PermStrategy.SHUFFLE,
            ))

        self._groups: Dict[int, List[int]] = {}
        for idx, s in enumerate(self.specs):
            self._groups.setdefault(s.F, []).append(idx)

    def calibrate(self, data: np.ndarray) -> Dict[str, Any]:
        T_steps, N_vars = data.shape
        R = 1 + self.B_perm

        data_gpu = torch.tensor(data, dtype=torch.float64, device=self.device)
        data_expanded = data_gpu.unsqueeze(0).repeat(R, 1, 1)

        apply_permutation(data_expanded, self.specs, T_steps)

        results = {}

        for F_val, spec_indices in self._groups.items():
            group_specs = [self.specs[i] for i in spec_indices]
            N_g = len(group_specs)
            M = R * N_g

            if not HAS_GPU_ENGINE:
                for spec in group_specs:
                    results[spec.name] = self._cpu_fallback(data, spec)
                continue

            rls = GPUBatchRLS(M=M, F=F_val, device=self.device, dtype=torch.float64)

            for t in range(T_steps):
                X_all = torch.zeros(M, F_val, device=self.device, dtype=torch.float64)
                Y_all = torch.zeros(M, device=self.device, dtype=torch.float64)

                for gi, spec in enumerate(group_specs):
                    X_s, Y_s = _extract_features(data_expanded, spec, t)
                    start = gi * R
                    end = start + R
                    X_all[start:end] = X_s
                    Y_all[start:end] = Y_s

                rls.update(X_all, Y_all)

            conf = rls.confidence.reshape(N_g, R)
            fit = rls.fit_score.reshape(N_g, R)

            for gi, spec in enumerate(group_specs):
                c_obs = conf[gi, 0].item()
                f_obs = fit[gi, 0].item()
                c_nulls = conf[gi, 1:]
                f_nulls = fit[gi, 1:]

                p_conf = (1 + (c_nulls >= c_obs).float().sum().item()) / (1 + self.B_perm)
                p_fit = (1 + (f_nulls >= f_obs).float().sum().item()) / (1 + self.B_perm)
                p_val = min(p_conf, p_fit)

                results[spec.name] = {
                    "conf_obs": c_obs, "fit_obs": f_obs,
                    "p_value": p_val, "p_conf": p_conf, "p_fit": p_fit,
                    "rel_type": spec.rel_type, "perm_strategy": spec.perm_strategy,
                    "null_conf_mean": c_nulls.mean().item(), "null_fit_mean": f_nulls.mean().item(),
                }

        apply_bh_fdr(results, q=0.05)
        return results

    def _cpu_fallback(self, data: np.ndarray, spec: TargetedHypoSpec) -> Dict[str, Any]:
        T = data.shape[0]
        lag = spec.lags[0] if spec.lags else 0
        c = spec.col_indices[0] if spec.col_indices else 0
        ti = spec.target_col

        if lag > 0 and lag < T:
            x, y = data[:-lag, c], data[lag:, ti]
        else:
            x, y = data[:, c], data[:, ti]

        corr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0.0
        n = len(x)
        if abs(corr) < 1.0 and n > 2:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            from scipy import stats
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        else:
            p_val = 0.0

        return {
            "conf_obs": abs(corr), "fit_obs": corr**2, "p_value": p_val,
            "p_conf": p_val, "p_fit": p_val,
            "rel_type": spec.rel_type, "perm_strategy": spec.perm_strategy,
            "null_conf_mean": 0, "null_fit_mean": 0, "significant": p_val < 0.05,
        }

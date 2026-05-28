import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

class PermStrategy:
    BLOCK = "block"         # block permutation (preserves local structure)
    SHUFFLE = "shuffle"     # iid random shuffle
    PHASE = "phase"         # phase randomization (preserves spectrum)

@dataclass
class TargetedHypoSpec:
    name: str
    rel_type: str
    F: int
    col_indices: list
    target_col: int
    lags: list
    interaction: bool = False
    perm_col: int = -1
    perm_strategy: str = PermStrategy.BLOCK

def apply_permutation(
    data_expanded: torch.Tensor,  # (R, T, N_vars)
    specs: List[TargetedHypoSpec],
    T_steps: int,
    block_size: int = 50,
) -> None:
    """Apply type-appropriate permutation in-place for runs 1..R-1."""
    R = data_expanded.shape[0]
    device = data_expanded.device

    # Collect which columns need which treatment
    col_strategies: Dict[int, str] = {}
    for spec in specs:
        existing = col_strategies.get(spec.perm_col)
        if existing is None:
            col_strategies[spec.perm_col] = spec.perm_strategy

    for r in range(1, R):
        for col, strategy in col_strategies.items():
            series = data_expanded[r, :, col].clone()

            if strategy == PermStrategy.SHUFFLE:
                # Full random shuffle
                perm_idx = torch.randperm(T_steps, device=device)
                data_expanded[r, :, col] = series[perm_idx]

            elif strategy == PermStrategy.BLOCK:
                # Block permutation
                n_blocks = max(1, T_steps // block_size)
                block_starts = list(range(0, T_steps, block_size))
                perm_order = np.random.permutation(len(block_starts))
                new_series = torch.empty_like(series)
                write_pos = 0
                for bi in perm_order:
                    start = block_starts[bi]
                    end = min(start + block_size, T_steps)
                    chunk_len = end - start
                    new_series[write_pos:write_pos+chunk_len] = series[start:end]
                    write_pos += chunk_len
                data_expanded[r, :write_pos, col] = new_series[:write_pos]

            elif strategy == PermStrategy.PHASE:
                # AR surrogate
                series_np = series.cpu().numpy()
                mu = series_np.mean()
                std = series_np.std()
                if std < 1e-12:
                    continue
                null_ar = np.random.uniform(0.1, 0.9)
                innovation_std = std * np.sqrt(1 - null_ar**2)
                surrogate = np.empty(T_steps)
                surrogate[0] = mu
                for tt in range(1, T_steps):
                    surrogate[tt] = mu + null_ar * (surrogate[tt-1] - mu) + np.random.normal(0, innovation_std)
                data_expanded[r, :, col] = torch.tensor(surrogate, dtype=data_expanded.dtype, device=device)

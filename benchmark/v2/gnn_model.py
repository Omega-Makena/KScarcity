"""
Typed-edge Temporal GNN for Scarcity v2 benchmark.

Tests whether graph topology (message-passing along discovered edges) adds
value beyond type-aware features alone. Cleanly separates three questions:

  Q1: Does the type-aware feature (X×Z, A+B, ECM, etc.) help? → compare xgb_blind vs xgb_typed
  Q2: Does graph topology help beyond features?               → compare xgb_typed vs tgcn_typed
  Q3: Does edge TYPE matter for message passing?              → compare tgcn_untyped vs tgcn_typed

Architecture: TypedEdgeGNN
  Each relationship type has its own learnable message transform.
  Node updates via GRU over time.
  Output: scalar forecast for target node at h steps ahead.

Requires:
  pip install torch-geometric torch-geometric-temporal

If unavailable, all methods gracefully return NaN.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_PYG_AVAILABLE: Optional[bool] = None
_TORCH_AVAILABLE: Optional[bool] = None


def _check_pyg() -> bool:
    global _PYG_AVAILABLE
    if _PYG_AVAILABLE is not None:
        return _PYG_AVAILABLE
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        _PYG_AVAILABLE = True
    except ImportError:
        _PYG_AVAILABLE = False
    return _PYG_AVAILABLE


def _check_torch() -> bool:
    """Check for plain PyTorch only — no torch-geometric required."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        import torch  # noqa: F401
        _TORCH_AVAILABLE = True
    except ImportError:
        _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def is_available() -> bool:
    return _check_pyg()


def is_available_torch_only() -> bool:
    """True when PyTorch is installed, even without torch-geometric."""
    return _check_torch()


# ---------------------------------------------------------------------------
# Relationship type → integer index (for embedding lookup)
# ---------------------------------------------------------------------------

REL_TYPES = [
    'causal', 'functional', 'temporal', 'equilibrium', 'structural',
    'probabilistic', 'graph', 'correlational', 'competitive', 'similarity',
    'mediating', 'moderating', 'synergistic', 'logical', 'compositional',
]
REL_TO_IDX = {r: i for i, r in enumerate(REL_TYPES)}
N_REL_TYPES = len(REL_TYPES)


def rel_to_idx(rel_type: str) -> int:
    return REL_TO_IDX.get(str(rel_type).lower(), N_REL_TYPES - 1)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_model(n_vars: int, hidden: int = 32, n_rel: int = N_REL_TYPES):
    """
    Build a typed-edge temporal GNN.

    For each time step t:
      1. Node embedding: linear(raw_value) → hidden
      2. Typed message passing: for each edge (src→tgt, type):
           message = W_type * emb[src]
           aggregate at tgt: sum of messages
      3. GRU update: gru(aggregated_message, hidden_state) → new_hidden_state
      4. Output head: linear(hidden_state[target_node]) → scalar

    Returns a torch.nn.Module.
    """
    if not _check_pyg():
        raise ImportError("torch-geometric not installed")

    import torch
    import torch.nn as nn

    class TypedEdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_embed = nn.Linear(1, hidden)
            # One weight matrix per relationship type
            self.msg_transforms = nn.ModuleList([
                nn.Linear(hidden, hidden, bias=False) for _ in range(n_rel)
            ])
            self.gru = nn.GRUCell(hidden, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(
            self,
            x_seq: 'torch.Tensor',      # (T, N) raw values
            edge_index: 'torch.Tensor',  # (2, E) int64
            edge_type: 'torch.Tensor',   # (E,) int64
        ) -> 'torch.Tensor':             # (N,) output at final step
            T, N = x_seq.shape
            h = torch.zeros(N, hidden, device=x_seq.device)

            for t in range(T):
                # Node embeddings
                node_vals = x_seq[t].unsqueeze(1)   # (N, 1)
                emb = torch.relu(self.node_embed(node_vals))  # (N, hidden)

                # Typed message passing
                agg = torch.zeros(N, hidden, device=x_seq.device)
                if edge_index.shape[1] > 0:
                    src_idx = edge_index[0]   # (E,)
                    tgt_idx = edge_index[1]   # (E,)
                    src_emb = emb[src_idx]    # (E, hidden)

                    # Scatter typed messages
                    for rt in range(n_rel):
                        mask = edge_type == rt
                        if mask.any():
                            msg = self.msg_transforms[rt](src_emb[mask])  # (k, hidden)
                            tgt_k = tgt_idx[mask]
                            agg.scatter_add_(0, tgt_k.unsqueeze(1).expand_as(msg), msg)

                # GRU update
                h = self.gru(agg, h)

            return self.out(h).squeeze(1)  # (N,)

    return TypedEdgeGNN()


# ---------------------------------------------------------------------------
# Rolling-origin prediction interface
# ---------------------------------------------------------------------------

def predict_tgcn(
    train_df,
    target: str,
    h: int,
    typed_edges: List[Dict[str, Any]],
    var_names: List[str],
    hidden: int = 32,
    epochs: int = 80,
    lr: float = 1e-2,
    min_pairs: int = 4,
) -> float:
    """
    Train a TypedEdgeGNN on train_df and return a scalar forecast h steps ahead.

    Args:
        train_df:     pd.DataFrame with year index, columns = variables
        target:       Target variable name
        h:            Forecast horizon
        typed_edges:  Edge list with 'source', 'target', 'type', 'variables'
        var_names:    Ordered list of all variable names (node ordering)
        hidden:       GRU hidden dimension
        epochs:       Training epochs
        lr:           Learning rate
        min_pairs:    Minimum training pairs

    Returns:
        Scalar forecast or np.nan if model cannot be trained.
    """
    if not _check_pyg():
        return np.nan
    if target not in var_names:
        return np.nan

    try:
        import torch
        import torch.nn as nn

        N = len(var_names)
        col_idx = {v: i for i, v in enumerate(var_names)}
        tgt_idx = col_idx[target]

        # Build node feature tensor: (T, N) — impute NaN with col mean
        arr = np.zeros((len(train_df), N), dtype=np.float32)
        for j, v in enumerate(var_names):
            if v in train_df.columns:
                col = train_df[v].values.astype(np.float32)
                mean = np.nanmean(col)
                col = np.where(np.isnan(col), mean if not np.isnan(mean) else 0.0, col)
                arr[:, j] = col

        # Normalise per column
        col_mean = arr.mean(axis=0)
        col_std = arr.std(axis=0) + 1e-8
        arr = (arr - col_mean) / col_std

        T = len(arr)
        n_pairs = T - h
        if n_pairs < min_pairs:
            return np.nan

        # Build edge tensors from typed_edges
        src_list, tgt_list, type_list = [], [], []
        for e in typed_edges:
            s = e.get('source', '')
            t = e.get('target', '')
            if s in col_idx and t in col_idx:
                src_list.append(col_idx[s])
                tgt_list.append(col_idx[t])
                type_list.append(rel_to_idx(e.get('type', 'causal')))

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_type = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        # Training pairs: X = sequence[:t+1], y = target at t+h
        x_seq_full = torch.tensor(arr, dtype=torch.float32)

        model = _build_model(N, hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        # Build direct training targets: for each t in [0, T-h-1], y = arr[t+h, tgt_idx]
        # Use truncated sequences: feed steps 0..t to predict t+h
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for t in range(min_pairs, n_pairs):
                x_t = x_seq_full[:t + 1]   # (t+1, N)
                y_t = torch.tensor(arr[t + h, tgt_idx], dtype=torch.float32)
                out = model(x_t, edge_index, edge_type)
                pred = out[tgt_idx]
                loss = (pred - y_t) ** 2
                total_loss += loss
                count += 1
            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Predict: feed full training sequence, extract target node output
        model.eval()
        with torch.no_grad():
            out = model(x_seq_full, edge_index, edge_type)
            pred_norm = out[tgt_idx].item()

        # Denormalise
        pred = pred_norm * col_std[tgt_idx] + col_mean[tgt_idx]
        return float(pred)

    except Exception as exc:
        warnings.warn(f"TGCN prediction failed: {exc}")
        return np.nan


def predict_tgcn_all_targets(
    train_df,
    targets: List[str],
    h: int,
    typed_edges: List[Dict[str, Any]],
    var_names: List[str],
    hidden: int = 16,
    epochs: int = 15,
    lr: float = 1e-2,
    min_pairs: int = 4,
) -> Dict[str, float]:
    """
    Train ONE TypedEdgeGNN with multi-task loss across all targets for a given h.
    Returns a dict mapping target_name → forecast scalar.

    Calling once per (cutoff × h) instead of per (cutoff × target × h) reduces
    TGCN inference calls from 34×10×4 = 1360 to 34×4 = 136, making full benchmarks
    practical (~15–20 min per country at epochs=15 vs 720 min at epochs=80).
    """
    if not _check_pyg():
        return {t: np.nan for t in targets}
    if not targets:
        return {}

    try:
        import torch
        import torch.nn as nn

        N = len(var_names)
        col_idx = {v: i for i, v in enumerate(var_names)}
        tgt_indices = [col_idx[t] for t in targets if t in col_idx]
        if not tgt_indices:
            return {t: np.nan for t in targets}

        # Build node feature tensor (T, N)
        arr = np.zeros((len(train_df), N), dtype=np.float32)
        for j, v in enumerate(var_names):
            if v in train_df.columns:
                col = train_df[v].values.astype(np.float32)
                mean = np.nanmean(col)
                col = np.where(np.isnan(col), mean if not np.isnan(mean) else 0.0, col)
                arr[:, j] = col

        col_mean = arr.mean(axis=0)
        col_std = arr.std(axis=0) + 1e-8
        arr = (arr - col_mean) / col_std

        T = len(arr)
        n_pairs = T - h
        if n_pairs < min_pairs:
            return {t: np.nan for t in targets}

        # Build edge tensors
        src_list, tgt_list, type_list = [], [], []
        for e in typed_edges:
            s, t = e.get('source', ''), e.get('target', '')
            if s in col_idx and t in col_idx:
                src_list.append(col_idx[s])
                tgt_list.append(col_idx[t])
                type_list.append(rel_to_idx(e.get('type', 'causal')))

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_type = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        x_seq_full = torch.tensor(arr, dtype=torch.float32)
        model = _build_model(N, hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        # Multi-task training: sum MSE across all target nodes
        model.train()
        for _ in range(epochs):
            total_loss = 0.0
            count = 0
            for t_idx in range(min_pairs, n_pairs):
                x_t = x_seq_full[:t_idx + 1]
                out = model(x_t, edge_index, edge_type)  # (N,)
                for node_idx in tgt_indices:
                    y_true = torch.tensor(arr[t_idx + h, node_idx], dtype=torch.float32)
                    total_loss = total_loss + (out[node_idx] - y_true) ** 2
                    count += 1
            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Predict from full sequence
        model.eval()
        with torch.no_grad():
            out = model(x_seq_full, edge_index, edge_type)  # (N,)

        result = {}
        for target in targets:
            if target not in col_idx:
                result[target] = np.nan
                continue
            node_idx = col_idx[target]
            pred_norm = out[node_idx].item()
            result[target] = float(pred_norm * col_std[node_idx] + col_mean[node_idx])

        return result

    except Exception as exc:
        warnings.warn(f"TGCN multi-target prediction failed: {exc}")
        return {t: np.nan for t in targets}


# ---------------------------------------------------------------------------
# Pure-PyTorch path — same architecture, no torch-geometric required
#
# The TypedEdgeGNN forward pass uses only standard PyTorch operations
# (scatter_add_, GRUCell, Linear).  The torch-geometric requirement above
# was a guard, not a runtime dependency.  This section exposes the same
# model and predictors gated on torch-only availability so the benchmark
# can run gnn_scarcity on any machine that has PyTorch installed.
# ---------------------------------------------------------------------------

def _build_model_torch_only(n_vars: int, hidden: int = 32, n_rel: int = N_REL_TYPES):
    """TypedEdgeGNN built with plain PyTorch (no torch-geometric)."""
    if not _check_torch():
        raise ImportError("PyTorch not installed")

    import torch
    import torch.nn as nn

    class TypedEdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_embed = nn.Linear(1, hidden)
            self.msg_transforms = nn.ModuleList([
                nn.Linear(hidden, hidden, bias=False) for _ in range(n_rel)
            ])
            self.gru = nn.GRUCell(hidden, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x_seq, edge_index, edge_type):
            T, N = x_seq.shape
            h_state = torch.zeros(N, hidden, device=x_seq.device)

            for t in range(T):
                node_vals = x_seq[t].unsqueeze(1)
                emb = torch.relu(self.node_embed(node_vals))

                agg = torch.zeros(N, hidden, device=x_seq.device)
                if edge_index.shape[1] > 0:
                    src_idx = edge_index[0]
                    tgt_idx = edge_index[1]
                    src_emb = emb[src_idx]
                    for rt in range(n_rel):
                        mask = edge_type == rt
                        if mask.any():
                            msg = self.msg_transforms[rt](src_emb[mask])
                            tgt_k = tgt_idx[mask]
                            agg.scatter_add_(0, tgt_k.unsqueeze(1).expand_as(msg), msg)

                h_state = self.gru(agg, h_state)

            return self.out(h_state).squeeze(1)

    return TypedEdgeGNN()


def predict_tgcn_all_targets_torch_only(
    train_df,
    targets: List[str],
    h: int,
    typed_edges: List[Dict[str, Any]],
    var_names: List[str],
    hidden: int = 32,
    epochs: int = 30,
    lr: float = 1e-2,
    min_pairs: int = 4,
) -> Dict[str, float]:
    """
    Train a TypedEdgeGNN (pure PyTorch, no torch-geometric) with multi-task
    loss across all targets for a given horizon h.

    Uses the full discovered graph (all edges, not per-target selection) so
    message-passing propagates information across the entire variable graph.
    Returns dict of target_name → forecast scalar.
    """
    if not _check_torch():
        return {t: np.nan for t in targets}
    if not targets:
        return {}

    try:
        import torch

        N = len(var_names)
        col_idx = {v: i for i, v in enumerate(var_names)}
        tgt_indices = [col_idx[t] for t in targets if t in col_idx]
        if not tgt_indices:
            return {t: np.nan for t in targets}

        # Build (T, N) node feature tensor with imputation
        arr = np.zeros((len(train_df), N), dtype=np.float32)
        for j, v in enumerate(var_names):
            if v in train_df.columns:
                col = train_df[v].values.astype(np.float32)
                mean = np.nanmean(col)
                col = np.where(np.isnan(col), mean if not np.isnan(mean) else 0.0, col)
                arr[:, j] = col

        col_mean = arr.mean(axis=0)
        col_std  = arr.std(axis=0) + 1e-8
        arr = (arr - col_mean) / col_std

        T = len(arr)
        n_pairs = T - h
        if n_pairs < min_pairs:
            return {t: np.nan for t in targets}

        # Build edge tensors from the full graph
        src_list, tgt_list, type_list = [], [], []
        for e in typed_edges:
            s, t_var = e.get('source', ''), e.get('target', '')
            if s in col_idx and t_var in col_idx:
                src_list.append(col_idx[s])
                tgt_list.append(col_idx[t_var])
                type_list.append(rel_to_idx(e.get('type', 'causal')))

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_type  = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type  = torch.zeros(0, dtype=torch.long)

        x_seq_full = torch.tensor(arr, dtype=torch.float32)
        model      = _build_model_torch_only(N, hidden=hidden)
        optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        # Multi-task training: MSE loss across all target nodes
        model.train()
        for _ in range(epochs):
            total_loss = torch.tensor(0.0)
            count = 0
            for t_step in range(min_pairs, n_pairs):
                x_t = x_seq_full[:t_step + 1]
                out = model(x_t, edge_index, edge_type)
                for node_idx in tgt_indices:
                    y_true = torch.tensor(arr[t_step + h, node_idx])
                    total_loss = total_loss + (out[node_idx] - y_true) ** 2
                    count += 1
            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Inference on full sequence
        model.eval()
        with torch.no_grad():
            out = model(x_seq_full, edge_index, edge_type)

        result = {}
        for target in targets:
            if target not in col_idx:
                result[target] = np.nan
                continue
            node_idx = col_idx[target]
            pred_norm = out[node_idx].item()
            result[target] = float(pred_norm * col_std[node_idx] + col_mean[node_idx])
        return result

    except Exception as exc:
        warnings.warn(f"GNN+Scarcity (torch-only) prediction failed: {exc}")
        return {t: np.nan for t in targets}

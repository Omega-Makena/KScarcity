"""
Type-aware feature translation for Scarcity's 15 relationship types.

The old benchmark collapsed every relationship type to a simple lag feature:
    parent_variable_value_at_t → predict target at t+h

That is wrong for 14 of 15 types. Each type implies a different mathematical
structure and therefore a different feature transformation:

  CAUSAL / FUNCTIONAL / TEMPORAL / STRUCTURAL / PROBABILISTIC / GRAPH
      A → B  (directional)
      Feature: lag(A)  — A's current value predicts B's future value

  CORRELATIONAL / SIMILARITY
      A ↔ B  (symmetric co-movement)
      Feature: lag(A) for B, lag(B) for A

  COMPETITIVE
      A ↔ B  (substitution / rivalry)
      Features: lag(A−B), lag(A/(A+B))  — relative position matters, not levels

  EQUILIBRIUM
      A ↔ B  (long-run balance, mean-reverting spread)
      Feature: error-correction term  lag(A − β·B)  where β is the OLS ratio
               estimated cumulatively from training data (no look-ahead)

  MEDIATING  [X, M, Y]
      X → M → Y  (X effects Y through mediator M)
      Features: lag(X), lag(M), lag(X·M)  — mediation requires the interaction

  MODERATING  [X, Z, Y]
      X → Y, moderated by Z  (Z changes the strength of X's effect on Y)
      Features: lag(X), lag(Z), lag(X·Z)  — the interaction IS the signal

  SYNERGISTIC  [X, Z, Y]
      X and Z together → Y  (joint effect exceeds sum of parts)
      Features: lag(X), lag(Z), lag(X·Z), lag(X+Z)

  LOGICAL  [A, B, C]
      A ∧ B → C  (both conditions must hold)
      Features: lag(A), lag(B), lag(A·B), lag(I(A>0)·I(B>0))

  COMPOSITIONAL  [A, B, C]
      A + B ≈ C  (accounting identity or component relationship)
      Features: lag(A), lag(B), lag(A+B), lag(A+B−C)  — residual from identity
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Relationship type classification (lowercase string values from engine)
# ---------------------------------------------------------------------------

_DIRECTIONAL = frozenset({
    'causal', 'functional', 'temporal', 'structural', 'probabilistic', 'graph',
})
_SYMMETRIC = frozenset({'correlational', 'similarity'})
_COMPETITIVE = frozenset({'competitive'})
_EQUILIBRIUM = frozenset({'equilibrium'})
_MEDIATING = frozenset({'mediating'})
_MODERATING = frozenset({'moderating'})
_SYNERGISTIC = frozenset({'synergistic'})
_LOGICAL = frozenset({'logical'})
_COMPOSITIONAL = frozenset({'compositional'})

_EPS = 1e-8
_MIN_PAIRS = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_typed_edges(
    edges: List[Dict[str, Any]],
    target: str,
    max_parents: int = 6,
) -> List[Dict[str, Any]]:
    """
    Choose the best edges for a target, preserving type diversity.

    Selection strategy:
      1. One champion per relationship type (ensures diverse feature types)
      2. Fill remaining slots by confidence until max_parents reached

    Returns list of edge dicts, each guaranteed to have a 'variables' key.
    """
    target_edges = [e for e in edges if e.get('target') == target]
    if not target_edges:
        return []

    # Best edge per (source, type) pair
    champ: Dict[Tuple[str, str], Dict] = {}
    for e in target_edges:
        key = (e.get('source', ''), str(e.get('type', '')))
        if key not in champ or e.get('confidence', 0) > champ[key].get('confidence', 0):
            champ[key] = e

    # One champion per relationship type (type diversity)
    type_best: Dict[str, Dict] = {}
    for e in champ.values():
        rt = str(e.get('type', ''))
        if rt not in type_best or e.get('confidence', 0) > type_best[rt].get('confidence', 0):
            type_best[rt] = e

    selected = list(type_best.values())
    used_keys = {(e.get('source', ''), str(e.get('type', ''))) for e in selected}

    # Fill with highest-confidence edges not yet selected
    remaining = sorted(
        [e for e in champ.values()
         if (e.get('source', ''), str(e.get('type', ''))) not in used_keys],
        key=lambda x: -x.get('confidence', 0),
    )
    for e in remaining:
        if len(selected) >= max_parents:
            break
        selected.append(e)

    return selected[:max_parents]


def build_type_aware_matrix(
    df: pd.DataFrame,
    target: str,
    typed_edges: List[Dict[str, Any]],
    h: int,
    min_pairs: int = _MIN_PAIRS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Build (X, y, X_last, feature_names) for direct h-step prediction.

    X[t] contains type-specific features from typed_edges.
    y[t] = target value at t+h.
    X_last = feature vector at final training row (used for prediction).

    Each relationship type contributes different features (see module docstring).

    Returns (None, None, None, []) if insufficient training data.
    """
    if not typed_edges or target not in df.columns:
        return None, None, None, []

    n = len(df)
    if n < h + min_pairs:
        return None, None, None, []

    # Collect (feature_name, values_array) pairs
    feat_names: List[str] = []
    feat_arrays: List[np.ndarray] = []

    seen_names: set = set()

    for edge in typed_edges:
        src = edge.get('source', '')
        rel = str(edge.get('type', 'causal')).lower()
        variables = edge.get('variables', [src, target])

        new_feats = _edge_features(df, target, src, rel, variables, n)
        for fname, arr in new_feats:
            if fname not in seen_names:
                seen_names.add(fname)
                feat_names.append(fname)
                feat_arrays.append(arr)

    if not feat_arrays:
        return None, None, None, []

    # Stack into matrix: rows = time steps, cols = features
    X_full = np.column_stack(feat_arrays)   # (n, F)
    y_full = df[target].values.astype(float)

    # Direct pairs: X[t] predicts y[t+h]
    n_pairs = n - h
    X = X_full[:n_pairs]
    y = y_full[h:]

    if len(y) < min_pairs:
        return None, None, None, []

    # Column means for NaN imputation (no look-ahead: use training rows only)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    # Impute NaN in X
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_means[j]

    # Drop rows where y is NaN
    valid = ~np.isnan(y)
    if valid.sum() < min_pairs:
        return None, None, None, []

    # X_last: features at the final row (prediction point)
    X_last = X_full[-1].copy()
    X_last_nan = np.isnan(X_last)
    X_last[X_last_nan] = col_means[X_last_nan]

    return X[valid], y[valid], X_last, feat_names


def summarise_edge_types(typed_edges: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count edges by relationship type across a full edge list."""
    counts: Dict[str, int] = {}
    for e in typed_edges:
        rt = str(e.get('type', 'unknown'))
        counts[rt] = counts.get(rt, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Internal: edge → feature arrays
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    if name in df.columns:
        return df[name].values.astype(float).copy()
    return np.full(n, np.nan)


def _edge_features(
    df: pd.DataFrame,
    target: str,
    src: str,
    rel: str,
    variables: List[str],
    n: int,
) -> List[Tuple[str, np.ndarray]]:
    """
    Translate one typed edge into (feature_name, array[n]) pairs.
    """
    feats: List[Tuple[str, np.ndarray]] = []

    if rel in _DIRECTIONAL or rel in _SYMMETRIC:
        s = _col(df, src, n)
        feats.append((src, s))

    elif rel in _COMPETITIVE:
        s = _col(df, src, n)
        t = _col(df, target, n)
        feats.append((f'{src}_minus_{target}', s - t))
        total = s + t
        share = np.where(np.abs(total) > _EPS, s / total, np.nan)
        feats.append((f'{src}_share_of_{target}', share))

    elif rel in _EQUILIBRIUM:
        s = _col(df, src, n)
        t = _col(df, target, n)
        # Cumulative OLS beta (no look-ahead): β = Σ(s·t) / Σ(t²)
        ecm = np.full(n, np.nan)
        sum_tt, sum_st = 0.0, 0.0
        for i in range(n):
            si, ti = s[i], t[i]
            if not (np.isnan(si) or np.isnan(ti)):
                sum_tt += ti * ti
                sum_st += si * ti
            if sum_tt > _EPS:
                beta = sum_st / sum_tt
                ecm[i] = si - beta * ti
        feats.append((f'{src}_{target}_ecm', ecm))
        feats.append((src, s))

    elif rel in _MEDIATING and len(variables) >= 3:
        x_nm, m_nm = variables[0], variables[1]
        x = _col(df, x_nm, n)
        m = _col(df, m_nm, n)
        feats.append((x_nm, x))
        feats.append((m_nm, m))
        feats.append((f'{x_nm}_x_{m_nm}', x * m))

    elif rel in _MODERATING and len(variables) >= 3:
        x_nm, z_nm = variables[0], variables[1]
        x = _col(df, x_nm, n)
        z = _col(df, z_nm, n)
        feats.append((x_nm, x))
        feats.append((z_nm, z))
        feats.append((f'{x_nm}_x_{z_nm}', x * z))

    elif rel in _SYNERGISTIC and len(variables) >= 3:
        x_nm, z_nm = variables[0], variables[1]
        x = _col(df, x_nm, n)
        z = _col(df, z_nm, n)
        feats.append((x_nm, x))
        feats.append((z_nm, z))
        feats.append((f'{x_nm}_x_{z_nm}', x * z))
        feats.append((f'{x_nm}_plus_{z_nm}', x + z))

    elif rel in _LOGICAL and len(variables) >= 3:
        a_nm, b_nm = variables[0], variables[1]
        a = _col(df, a_nm, n)
        b = _col(df, b_nm, n)
        feats.append((a_nm, a))
        feats.append((b_nm, b))
        feats.append((f'{a_nm}_x_{b_nm}', a * b))
        a_pos = (a > 0).astype(float)
        b_pos = (b > 0).astype(float)
        feats.append((f'{a_nm}_and_{b_nm}_pos', a_pos * b_pos))

    elif rel in _COMPOSITIONAL and len(variables) >= 3:
        a_nm, b_nm = variables[0], variables[1]
        a = _col(df, a_nm, n)
        b = _col(df, b_nm, n)
        tgt_vals = _col(df, target, n)
        feats.append((a_nm, a))
        feats.append((b_nm, b))
        feats.append((f'{a_nm}_plus_{b_nm}', a + b))
        feats.append((f'{a_nm}_plus_{b_nm}_minus_{target}', a + b - tgt_vals))

    else:
        # Fallback: treat as directional lag
        s = _col(df, src, n)
        feats.append((src, s))

    return feats

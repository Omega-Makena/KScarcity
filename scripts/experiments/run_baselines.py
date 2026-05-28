"""Phase 4 — Causal discovery baseline runners.

PC, FCI, GES, DirectLiNGAM (all via causal-learn), NOTEARS (numpy implementation),
and a naive Pearson correlation threshold baseline.

All return the same standardised edge-list format as run_kscarcity_discovery().
"""
from __future__ import annotations

import time
import traceback
from typing import Any

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.experiments.synthetic_data import generate_ground_truth


# ---------------------------------------------------------------------------
# Graph parsing helpers (causal-learn adjacency matrix format)
# ---------------------------------------------------------------------------
# causal-learn graph[i,j] / graph[j,i] encoding:
#   Directed i→j:   graph[i,j]==-1, graph[j,i]==1
#   Undirected i-j: graph[i,j]==1,  graph[j,i]==1
#   Bidirected i<->j (FCI): graph[i,j]==1, graph[j,i]==1, edge_mark differs
#   No edge:        graph[i,j]==0
#
# We use GraphUtils.getEdges when available, and fall back to raw matrix parsing.

def _parse_cl_graph(G, col_names: list[str], default_type: str = 'causal') -> list[dict]:
    """Parse a causal-learn GeneralGraph (CPDAG) into standardised edge list.

    causal-learn CPDAG encoding:
      Directed i→j:   g[i,j]==-1, g[j,i]==1   (arrowhead at j, tail at i)
      Undirected i-j: g[i,j]==-1, g[j,i]==-1   (both arrowheads in CPDAG)
      No edge:        g[i,j]==0,  g[j,i]==0
    """
    edges: list[dict] = []
    n = len(col_names)
    try:
        g = G.graph  # numpy matrix (n x n)
    except AttributeError:
        return edges

    for i in range(n):
        for j in range(i + 1, n):
            gij = int(g[i, j])
            gji = int(g[j, i])
            if gij == 0 and gji == 0:
                continue  # no edge

            # Directed i→j: g[i,j]==-1, g[j,i]==1
            if gij == -1 and gji == 1:
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'causal', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Directed j→i: g[j,i]==-1, g[i,j]==1
            elif gij == 1 and gji == -1:
                edges.append({
                    'vars': [col_names[j], col_names[i]],
                    'source': col_names[j], 'target': col_names[i],
                    'type': 'causal', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Undirected: g[i,j]==-1, g[j,i]==-1
            elif gij == -1 and gji == -1:
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'correlational', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Bidirected: g[i,j]==1, g[j,i]==1 (rare in PC but possible in FCI)
            elif gij == 1 and gji == 1:
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'correlational', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
    return edges


def _parse_fci_pag(G, col_names: list[str]) -> list[dict]:
    """Parse a causal-learn PAG from FCI.

    FCI PAG endpoint marks:
      1 = tail  (-),  2 = circle (o),  3 = arrowhead (>)
      g[i,j] is the mark AT j on the edge from i.

    Common patterns:
      Directed i→j:   g[i,j]==3, g[j,i]==1  (arrow at j, tail at i)
      Bidirected i<->j: g[i,j]==3, g[j,i]==3
      Undirected i-j: g[i,j]==1, g[j,i]==1  (rare; also use causal-learn's -1/-1)
      Partially: o-> : g[i,j]==3, g[j,i]==2

    causal-learn may also encode as -1/-1 for undirected (same as PC).
    """
    edges: list[dict] = []
    n = len(col_names)
    try:
        g = G.graph
    except AttributeError:
        return edges

    for i in range(n):
        for j in range(i + 1, n):
            gij = int(g[i, j])
            gji = int(g[j, i])
            if gij == 0 and gji == 0:
                continue

            # Directed i→j (arrowhead at j)
            if (gij == -1 and gji == 1) or (gij == 3 and gji == 1):
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'causal', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Directed j→i
            elif (gij == 1 and gji == -1) or (gij == 1 and gji == 3):
                edges.append({
                    'vars': [col_names[j], col_names[i]],
                    'source': col_names[j], 'target': col_names[i],
                    'type': 'causal', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Bidirected i<->j or undirected (latent confounder)
            elif gij in (-1, 1, 3) and gji in (-1, 1, 3):
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'correlational', 'confidence': 1.0,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
            # Circle marks — partially determined
            elif gij == 2 or gji == 2:
                edges.append({
                    'vars': [col_names[i], col_names[j]],
                    'source': col_names[i], 'target': col_names[j],
                    'type': 'correlational', 'confidence': 0.5,
                    'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                })
    return edges


# ---------------------------------------------------------------------------
# PC
# ---------------------------------------------------------------------------

def run_pc(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Run PC algorithm via causal-learn."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        import io, contextlib
        X = df.values.astype(float)
        n = X.shape[0]
        if n < 10:
            return []
        with contextlib.redirect_stderr(io.StringIO()):
            cg = pc(X, alpha=alpha, indep_test='fisherz', show_progress=False)
        return _parse_cl_graph(cg.G, list(df.columns))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# FCI
# ---------------------------------------------------------------------------

def run_fci(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Run FCI algorithm via causal-learn."""
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz
        import io, contextlib
        X = df.values.astype(float)
        if X.shape[0] < 10:
            return []
        with contextlib.redirect_stderr(io.StringIO()):
            G, _ = fci(X, fisherz, alpha, verbose=False)
        return _parse_fci_pag(G, list(df.columns))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# GES
# ---------------------------------------------------------------------------

def run_ges(df: pd.DataFrame) -> list[dict]:
    """Run GES via causal-learn."""
    try:
        from causallearn.search.ScoreBased.GES import ges
        X = df.values.astype(float)
        if X.shape[0] < 5:
            return []
        res = ges(X, score_func='local_score_BIC')
        return _parse_cl_graph(res['G'], list(df.columns))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# NOTEARS (numpy implementation — Zheng et al. 2018)
# ---------------------------------------------------------------------------

def _notears_linear(X: np.ndarray, lambda1: float = 0.1,
                    max_iter: int = 100, h_tol: float = 1e-8,
                    rho_max: float = 1e+16) -> np.ndarray:
    """Minimal NOTEARS-linear implementation (Zheng et al. 2018).

    Minimises 0.5/n * ||X - X*W||^2 + lambda*||W||_1
    subject to the acyclicity constraint tr(e^(W*W)) - d == 0.
    """
    from scipy.optimize import minimize
    n, d = X.shape
    W0 = np.zeros(d * d)

    def _h(w_flat: np.ndarray) -> float:
        W = w_flat.reshape(d, d)
        # Acyclicity: tr(e^{W*W}) - d  (element-wise squared, matrix exp)
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        return float(np.trace(E)) - d

    def _dh(w_flat: np.ndarray) -> np.ndarray:
        W = w_flat.reshape(d, d)
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        grad = (E.T * 2 * W)
        return grad.flatten()

    def _loss(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        W = w_flat.reshape(d, d)
        R = X - X @ W
        loss = 0.5 / n * np.sum(R ** 2) + lambda1 * np.sum(np.abs(W))
        grad = -1.0 / n * X.T @ R + lambda1 * np.sign(W)
        return loss, grad.flatten()

    rho, alpha_aug = 1.0, 0.0
    w = W0.copy()
    h_prev = float('inf')

    for _ in range(max_iter):
        def _aug(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
            h_val = _h(w_flat)
            dh_val = _dh(w_flat)
            loss, gloss = _loss(w_flat)
            f = loss + 0.5 * rho * h_val ** 2 + alpha_aug * h_val
            g = gloss + rho * h_val * dh_val + alpha_aug * dh_val
            return f, g

        res = minimize(_aug, w, method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'ftol': 1e-8})
        w = res.x
        h_val = _h(w)
        if abs(h_val) <= h_tol:
            break
        alpha_aug += rho * h_val
        rho = min(rho * 10, rho_max)
        h_prev = h_val

    W = w.reshape(d, d)
    W[np.abs(W) < 0.3] = 0.0
    return W


def run_notears(df: pd.DataFrame, lambda1: float = 0.1) -> list[dict]:
    """Run NOTEARS-linear on a DataFrame."""
    try:
        X = df.values.astype(float)
        if X.shape[0] < 5 or X.shape[1] < 2:
            return []
        # Standardise to help convergence
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        W = _notears_linear(X, lambda1=lambda1)
        col_names = list(df.columns)
        d = len(col_names)
        edges: list[dict] = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if abs(W[i, j]) > 0:
                    edges.append({
                        'vars': [col_names[j], col_names[i]],   # W[i,j]: j->i
                        'source': col_names[j], 'target': col_names[i],
                        'type': 'causal',
                        'confidence': float(min(abs(W[i, j]), 1.0)),
                        'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                    })
        return edges
    except Exception:
        return []


# ---------------------------------------------------------------------------
# DirectLiNGAM (via causal-learn)
# ---------------------------------------------------------------------------

def run_direct_lingam(df: pd.DataFrame) -> list[dict]:
    """Run DirectLiNGAM via causal-learn."""
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        X = df.values.astype(float)
        if X.shape[0] < 10:
            return []
        model = DirectLiNGAM()
        model.fit(X)
        W = model.adjacency_matrix_
        col_names = list(df.columns)
        d = len(col_names)
        edges: list[dict] = []
        threshold = 0.1
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if abs(W[i, j]) > threshold:
                    edges.append({
                        'vars': [col_names[j], col_names[i]],
                        'source': col_names[j], 'target': col_names[i],
                        'type': 'causal',
                        'confidence': float(min(abs(W[i, j]) / (abs(W[i, j]) + 1), 1.0)),
                        'fit_score': 0.0, 'evidence': 0, 'status': 'active',
                    })
        return edges
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pearson correlation threshold (naive baseline)
# ---------------------------------------------------------------------------

def run_correlation_threshold(df: pd.DataFrame, threshold: float = 0.3) -> list[dict]:
    """Naive baseline: Pearson |r| > threshold -> edge."""
    try:
        corr = df.corr().values
        col_names = list(df.columns)
        d = len(col_names)
        edges: list[dict] = []
        for i in range(d):
            for j in range(i + 1, d):
                r = float(corr[i, j])
                if abs(r) > threshold:
                    edges.append({
                        'vars': [col_names[i], col_names[j]],
                        'source': col_names[i], 'target': col_names[j],
                        'type': 'correlational',
                        'confidence': float(abs(r)),
                        'fit_score': float(r ** 2),
                        'evidence': len(df),
                        'status': 'active',
                    })
        return edges
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

ALL_BASELINES: dict[str, Any] = {
    'PC': run_pc,
    'FCI': run_fci,
    'GES': run_ges,
    'NOTEARS': run_notears,
    'DirectLiNGAM': run_direct_lingam,
    'CorrThreshold': run_correlation_threshold,
}


def run_all_baselines_n_sweep(
    n_values: list[int],
    n_seeds: int = 10,
) -> dict[str, dict[int, list[list[dict]]]]:
    """Run ALL baselines across all N values and seeds.

    Returns:
        Dict: baseline_name -> {N -> [list_of_edges_per_seed]}
    """
    results: dict[str, dict[int, list[list[dict]]]] = {
        name: {} for name in ALL_BASELINES
    }

    for name, runner in ALL_BASELINES.items():
        for n in n_values:
            results[name][n] = []
            for seed in range(n_seeds):
                t0 = time.perf_counter()
                df = generate_ground_truth(N=n, seed=seed)
                try:
                    edges = runner(df)
                except Exception:
                    edges = []
                elapsed = time.perf_counter() - t0
                print(f"  {name:15s} N={n:4d} seed={seed+1:2d}/{n_seeds} "
                      f"... {len(edges):4d} edges ({elapsed:.2f}s)")
                results[name][n].append(edges)

    return results


# ---------------------------------------------------------------------------
# Phase 4 self-test at N=100
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 4 — Baseline runners at N=100")
    print("=" * 60)

    df = generate_ground_truth(N=100, seed=42)

    for name, runner in ALL_BASELINES.items():
        t0 = time.perf_counter()
        edges = runner(df)
        elapsed = time.perf_counter() - t0
        confident = [e for e in edges if e.get('confidence', 1.0) >= 0.25]
        print(f"\n{name}: {len(edges)} edges ({len(confident)} confident) in {elapsed:.2f}s")
        for e in sorted(confident, key=lambda x: -x.get('confidence', 1.0))[:5]:
            print(f"  {e['vars']} | {e['type']:15s} | conf={e.get('confidence',1.0):.3f}")

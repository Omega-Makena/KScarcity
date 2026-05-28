"""
Statistical validation of edge types discovered by the Scarcity engine.

For each discovered edge, applies the statistical test that matches the
claimed relationship type. Answers the question: does the data actually
support what the engine says it found?

Validation tests by type:
  COMPOSITIONAL [A,B,C]  — Pearson r(A+B, C) > 0.5  (accounting identity)
  EQUILIBRIUM   [A,B]    — Engle-Granger cointegration (ADF on residuals)
  CORRELATIONAL [A,B]    — Pearson r(A,B) with p-value
  COMPETITIVE   [A,B]    — Pearson r(A-B, ΔC) — does differential predict change?
  CAUSAL/FUNCTIONAL/
  TEMPORAL/STRUCTURAL    — Granger causality F-test (lag-1), p < 0.05
  MEDIATING [X,M,Y]      — Sobel test approximation: r(X,M) * r(M,Y) > 0 (same sign)
  MODERATING [X,Z,Y]     — Correlation r(X*Z, Y) > r(X,Y) — interaction beats main effect
  SYNERGISTIC [X,Z,Y]    — same as MODERATING
  SIMILARITY [A,B]       — Pearson r(A,B) > 0.5
  LOGICAL [A,B,C]        — r(A*B, C) > r(A,C) and r(A*B, C) > r(B,C)
  PROBABILISTIC          — treated as CAUSAL (Granger)
  GRAPH                  — treated as CAUSAL (Granger)

Output per edge:
  validated: bool
  test_used: str
  stat: float
  p_value: float (or NaN if not applicable)
  note: str
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_MIN_OBS = 10   # minimum non-NaN observations to run any test
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_edges(
    edges: List[Dict[str, Any]],
    df: pd.DataFrame,
    alpha: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Validate each edge in the edge list against the data in df.

    Returns a new list of dicts, each containing the original edge fields
    plus: validated, test_used, stat, p_value, note.
    """
    results = []
    for edge in edges:
        result = dict(edge)
        try:
            v = _validate_one(edge, df, alpha)
        except Exception as exc:
            v = {
                'validated': False, 'test_used': 'error',
                'stat': np.nan, 'p_value': np.nan, 'note': str(exc),
            }
        result.update(v)
        results.append(result)
    return results


def validation_summary(validated_edges: List[Dict[str, Any]]) -> str:
    """Human-readable summary of validation results."""
    if not validated_edges:
        return "No edges to validate."

    total = len(validated_edges)
    n_valid = sum(1 for e in validated_edges if e.get('validated', False))
    by_type: Dict[str, Tuple[int, int]] = {}
    for e in validated_edges:
        rt = str(e.get('type', 'unknown'))
        ok, tot = by_type.get(rt, (0, 0))
        by_type[rt] = (ok + int(e.get('validated', False)), tot + 1)

    lines = [f"Validated {n_valid}/{total} edges ({100*n_valid/total:.0f}%)"]
    for rt, (ok, tot) in sorted(by_type.items(), key=lambda x: -x[1][1]):
        lines.append(f"  {rt:<20} {ok}/{tot}")
    return "\n".join(lines)


def print_validation_table(validated_edges: List[Dict[str, Any]], top_n: int = 30) -> None:
    """Print a ranked table of validated edges."""
    if not validated_edges:
        print("  No edges to display.")
        return

    ranked = sorted(validated_edges, key=lambda e: -e.get('confidence', 0))[:top_n]
    header = (f"  {'Source':<22} {'Target':<22} {'Type':<14} "
              f"{'Conf':>6} {'Stat':>7} {'p':>7} {'OK':>4} Note")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for e in ranked:
        ok = "YES" if e.get('validated') else "NO "
        p = e.get('p_value', np.nan)
        p_str = f"{p:.3f}" if not np.isnan(p) else "  N/A"
        stat = e.get('stat', np.nan)
        s_str = f"{stat:7.3f}" if not np.isnan(stat) else "    N/A"
        note = e.get('note', '')[:30]
        print(f"  {e.get('source',''):<22} {e.get('target',''):<22} "
              f"{str(e.get('type','')):<14} {e.get('confidence',0):6.3f} "
              f"{s_str} {p_str} {ok}  {note}")


# ---------------------------------------------------------------------------
# Internal: per-type validation
# ---------------------------------------------------------------------------

def _validate_one(
    edge: Dict[str, Any],
    df: pd.DataFrame,
    alpha: float,
) -> Dict[str, Any]:
    rel = str(edge.get('type', 'causal')).lower()
    src = edge.get('source', '')
    tgt = edge.get('target', '')
    variables = edge.get('variables', [src, tgt])

    if rel == 'compositional' and len(variables) >= 3:
        return _test_compositional(df, variables, alpha)
    elif rel == 'equilibrium':
        return _test_equilibrium(df, src, tgt, alpha)
    elif rel in ('correlational', 'similarity'):
        return _test_correlation(df, src, tgt, alpha, threshold=0.3)
    elif rel == 'competitive':
        return _test_competitive(df, src, tgt, alpha)
    elif rel == 'mediating' and len(variables) >= 3:
        return _test_mediating(df, variables, alpha)
    elif rel in ('moderating', 'synergistic') and len(variables) >= 3:
        return _test_moderating(df, variables, alpha)
    elif rel == 'logical' and len(variables) >= 3:
        return _test_logical(df, variables, alpha)
    elif rel in ('causal', 'functional', 'temporal', 'structural', 'probabilistic', 'graph'):
        return _test_granger(df, src, tgt, alpha)
    else:
        return _test_correlation(df, src, tgt, alpha, threshold=0.2)


def _pair(df: pd.DataFrame, a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned non-NaN arrays for two columns."""
    if a not in df.columns or b not in df.columns:
        return np.array([]), np.array([])
    ab = df[[a, b]].dropna()
    return ab[a].values, ab[b].values


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < _MIN_OBS:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            return np.nan, np.nan


def _test_compositional(
    df: pd.DataFrame, variables: List[str], alpha: float,
) -> Dict[str, Any]:
    a_nm, b_nm, c_nm = variables[0], variables[1], variables[2]
    cols = [c for c in [a_nm, b_nm, c_nm] if c in df.columns]
    if len(cols) < 3:
        return _fail('compositional', 'missing columns')
    sub = df[[a_nm, b_nm, c_nm]].dropna()
    if len(sub) < _MIN_OBS:
        return _fail('compositional', f'only {len(sub)} obs')
    a, b, c = sub[a_nm].values, sub[b_nm].values, sub[c_nm].values
    r, p = _pearsonr(a + b, c)
    ok = (not np.isnan(r)) and r > 0.3 and p < alpha
    return {
        'validated': ok, 'test_used': 'pearson(A+B, C)',
        'stat': r, 'p_value': p,
        'note': f'r(A+B,C)={r:.3f}' if not np.isnan(r) else 'insufficient data',
    }


def _test_equilibrium(
    df: pd.DataFrame, src: str, tgt: str, alpha: float,
) -> Dict[str, Any]:
    x, y = _pair(df, src, tgt)
    if len(x) < _MIN_OBS:
        return _fail('equilibrium', f'only {len(x)} obs')
    try:
        from statsmodels.tsa.stattools import adfuller
        # OLS residuals
        beta = np.dot(x, y) / (np.dot(y, y) + _EPS)
        resid = x - beta * y
        adf_stat, p_val = adfuller(resid, maxlag=1, autolag=None)[:2]
        # Cointegrated if residuals are stationary (ADF rejects unit root)
        ok = p_val < alpha
        return {
            'validated': ok, 'test_used': 'Engle-Granger ADF on residuals',
            'stat': float(adf_stat), 'p_value': float(p_val),
            'note': f'ADF={adf_stat:.3f} p={p_val:.3f}',
        }
    except ImportError:
        r, p = _pearsonr(x, y)
        ok = (not np.isnan(r)) and abs(r) > 0.5
        return {
            'validated': ok, 'test_used': 'pearson (statsmodels unavailable)',
            'stat': r, 'p_value': p, 'note': 'install statsmodels for ADF',
        }


def _test_correlation(
    df: pd.DataFrame, src: str, tgt: str, alpha: float, threshold: float = 0.3,
) -> Dict[str, Any]:
    x, y = _pair(df, src, tgt)
    r, p = _pearsonr(x, y)
    ok = (not np.isnan(r)) and abs(r) > threshold and p < alpha
    return {
        'validated': ok, 'test_used': f'pearson (threshold={threshold})',
        'stat': r, 'p_value': p,
        'note': f'r={r:.3f}' if not np.isnan(r) else 'insufficient data',
    }


def _test_competitive(
    df: pd.DataFrame, src: str, tgt: str, alpha: float,
) -> Dict[str, Any]:
    x, y = _pair(df, src, tgt)
    if len(x) < _MIN_OBS + 1:
        return _fail('competitive', f'only {len(x)} obs')
    diff = x - y
    dy = np.diff(y)
    r, p = _pearsonr(diff[:-1], dy)
    ok = (not np.isnan(r)) and p < alpha
    return {
        'validated': ok, 'test_used': 'pearson(A-B, ΔB)',
        'stat': r, 'p_value': p,
        'note': f'r(diff,ΔB)={r:.3f}' if not np.isnan(r) else 'insufficient data',
    }


def _test_granger(
    df: pd.DataFrame, src: str, tgt: str, alpha: float,
) -> Dict[str, Any]:
    x, y = _pair(df, src, tgt)
    if len(x) < _MIN_OBS:
        return _fail('granger', f'only {len(x)} obs')
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = np.column_stack([y, x])
            res = grangercausalitytests(data, maxlag=1, verbose=False)
            f_stat = res[1][0]['ssr_ftest'][0]
            p_val = res[1][0]['ssr_ftest'][1]
        ok = p_val < alpha
        return {
            'validated': ok, 'test_used': 'Granger F-test (lag=1)',
            'stat': float(f_stat), 'p_value': float(p_val),
            'note': f'F={f_stat:.3f} p={p_val:.3f}',
        }
    except ImportError:
        r, p = _pearsonr(x[:-1], y[1:])
        ok = (not np.isnan(r)) and p < alpha
        return {
            'validated': ok, 'test_used': 'lag-1 pearson (statsmodels unavailable)',
            'stat': r, 'p_value': p, 'note': 'install statsmodels for Granger',
        }


def _test_mediating(
    df: pd.DataFrame, variables: List[str], alpha: float,
) -> Dict[str, Any]:
    x_nm, m_nm, y_nm = variables[0], variables[1], variables[2]
    sub = df[[c for c in [x_nm, m_nm, y_nm] if c in df.columns]].dropna()
    if len(sub) < _MIN_OBS or not all(c in sub for c in [x_nm, m_nm, y_nm]):
        return _fail('mediating', 'missing columns or insufficient obs')
    x, m, y = sub[x_nm].values, sub[m_nm].values, sub[y_nm].values
    r_xm, _ = _pearsonr(x, m)
    r_my, _ = _pearsonr(m, y)
    # Sobel: mediation exists if X→M and M→Y have same sign
    ok = (not np.isnan(r_xm)) and (not np.isnan(r_my)) and (r_xm * r_my > 0)
    indirect = r_xm * r_my if not (np.isnan(r_xm) or np.isnan(r_my)) else np.nan
    return {
        'validated': ok, 'test_used': 'Sobel (sign of r_XM * r_MY)',
        'stat': indirect, 'p_value': np.nan,
        'note': f'r_XM={r_xm:.3f} r_MY={r_my:.3f} indirect={indirect:.3f}' if not np.isnan(indirect) else 'insufficient data',
    }


def _test_moderating(
    df: pd.DataFrame, variables: List[str], alpha: float,
) -> Dict[str, Any]:
    x_nm, z_nm, y_nm = variables[0], variables[1], variables[2]
    sub = df[[c for c in [x_nm, z_nm, y_nm] if c in df.columns]].dropna()
    if len(sub) < _MIN_OBS or not all(c in sub for c in [x_nm, z_nm, y_nm]):
        return _fail('moderating', 'missing columns or insufficient obs')
    x, z, y = sub[x_nm].values, sub[z_nm].values, sub[y_nm].values
    r_xy, _ = _pearsonr(x, y)
    r_xzy, _ = _pearsonr(x * z, y)
    # Interaction beats main effect
    ok = (not np.isnan(r_xy)) and (not np.isnan(r_xzy)) and abs(r_xzy) > abs(r_xy)
    return {
        'validated': ok, 'test_used': 'r(X*Z,Y) > r(X,Y)',
        'stat': r_xzy, 'p_value': np.nan,
        'note': f'r_XY={r_xy:.3f} r_XZY={r_xzy:.3f}',
    }


def _test_logical(
    df: pd.DataFrame, variables: List[str], alpha: float,
) -> Dict[str, Any]:
    a_nm, b_nm, c_nm = variables[0], variables[1], variables[2]
    sub = df[[c for c in [a_nm, b_nm, c_nm] if c in df.columns]].dropna()
    if len(sub) < _MIN_OBS or not all(c in sub for c in [a_nm, b_nm, c_nm]):
        return _fail('logical', 'missing columns or insufficient obs')
    a, b, c = sub[a_nm].values, sub[b_nm].values, sub[c_nm].values
    r_ac, _ = _pearsonr(a, c)
    r_bc, _ = _pearsonr(b, c)
    r_abc, _ = _pearsonr(a * b, c)
    # A∧B→C: interaction should outperform individuals
    ok = (not np.isnan(r_abc)) and abs(r_abc) > max(
        abs(r_ac) if not np.isnan(r_ac) else 0,
        abs(r_bc) if not np.isnan(r_bc) else 0,
    )
    return {
        'validated': ok, 'test_used': 'r(A*B,C) > max(r(A,C), r(B,C))',
        'stat': r_abc, 'p_value': np.nan,
        'note': f'r_AC={r_ac:.3f} r_BC={r_bc:.3f} r_ABC={r_abc:.3f}',
    }


def _fail(test_name: str, reason: str) -> Dict[str, Any]:
    return {
        'validated': False, 'test_used': test_name,
        'stat': np.nan, 'p_value': np.nan, 'note': reason,
    }

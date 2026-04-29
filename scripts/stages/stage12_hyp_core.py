"""stage12_hyp_core.py — Stages 12.1–12.10: Core hypothesis type benchmarks.

Each stage generates signal-rich synthetic data, feeds it through the
hypothesis class, and asserts fit_score > threshold AND > 1.5x null fit_score.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

import numpy as np

from scripts.stages.utils import fail_result, make_result

# ---------------------------------------------------------------------------
# Shared: null generator
# ---------------------------------------------------------------------------

def _gen_null_keyed(keys: List[str], n: int = 80, seed: int = 99) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{k: float(rng.standard_normal()) for k in keys} for _ in range(n)]


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _gen_causal(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    X = np.zeros(n); Y = np.zeros(n)
    X[0] = rng.standard_normal()
    for i in range(1, n):
        X[i] = 0.7 * X[i - 1] + rng.standard_normal() * 0.5
    for i in range(1, n):
        Y[i] = 0.6 * X[i - 1] + rng.standard_normal() * 0.4
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_correlational(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal(n) * 2.0
    X = 0.8 * F + rng.standard_normal(n) * 0.5
    Y = 0.7 * F + rng.standard_normal(n) * 0.5
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_temporal(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    V = np.zeros(n)
    V[0] = rng.standard_normal()
    for i in range(1, n):
        V[i] = 0.85 * V[i - 1] + rng.standard_normal() * 0.3
    return [{"V": float(V[i])} for i in range(n)]


def _gen_functional(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    Y = 2.5 * X - 0.3 * X ** 2 + rng.standard_normal(n) * 0.3
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_equilibrium(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    theta, mu, sigma = 0.4, 5.0, 0.5
    V = np.zeros(n)
    V[0] = mu
    for i in range(1, n):
        V[i] = V[i - 1] + theta * (mu - V[i - 1]) + sigma * rng.standard_normal()
    return [{"V": float(V[i])} for i in range(n)]


def _gen_compositional(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        A = float(rng.gamma(2.0, 1.0))
        B = float(rng.gamma(3.0, 1.0))
        C = float(rng.gamma(1.5, 1.0))
        T = A + B + C + float(rng.standard_normal() * 0.001)
        rows.append({"A": A, "B": B, "C": C, "T": T})
    return rows


def _gen_compositional_null(n: int = 80, seed: int = 99) -> List[Dict[str, float]]:
    """Null: parts and total are independent."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(n) * 2 + 5
    B = rng.standard_normal(n) * 2 + 5
    C = rng.standard_normal(n) * 2 + 5
    T = rng.standard_normal(n) * 5 + 10  # NOT equal to A+B+C
    return [{"A": float(A[i]), "B": float(B[i]), "C": float(C[i]), "T": float(T[i])} for i in range(n)]


def _gen_competitive(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    K = 10.0
    X = rng.uniform(1, K - 1, n)
    Y = K - X + rng.standard_normal(n) * 0.3
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_synergistic(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    Y = 1.0 + 0.5 * X1 + 0.3 * X2 + 1.2 * X1 * X2 + rng.standard_normal(n) * 0.3
    return [{"X1": float(X1[i]), "X2": float(X2[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_probabilistic(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        C = float(rng.standard_normal())
        Y = float(rng.normal(3.0, 0.5) if C > 0 else rng.normal(1.0, 0.5))
        rows.append({"C": C, "Y": Y})
    return rows


def _gen_structural(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    means = [1.0, 3.0, 6.0, 10.0]
    n_per = n // 4
    for g, mu in enumerate(means):
        for _ in range(n_per):
            rows.append({"G": float(g), "O": float(rng.normal(mu, 0.5))})
    rng.shuffle(rows)
    return rows


def _gen_structural_null(n: int = 80, seed: int = 99) -> List[Dict[str, float]]:
    """Null: all 4 groups have the same distribution."""
    rng = np.random.default_rng(seed)
    rows = []
    n_per = n // 4
    for g in range(4):
        for _ in range(n_per):
            rows.append({"G": float(g), "O": float(rng.standard_normal())})
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Core runner helper
# ---------------------------------------------------------------------------

def _safe_evaluate(hyp, last_row: Dict | None = None) -> Dict[str, Any]:
    """Call evaluate() with or without row argument depending on signature."""
    try:
        return hyp.evaluate()
    except TypeError:
        return hyp.evaluate(last_row or {})


def _run_hyp_stage(
    stage_id: str,
    name: str,
    hyp_cls,
    hyp_kwargs: Dict,
    signal_rows: List[Dict],
    null_rows: List[Dict],
    threshold: float,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        # Signal run
        hyp_s = hyp_cls(**hyp_kwargs)
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1] if signal_rows else {})
        fs_signal = float(eval_s.get("fit_score", 0.0))

        # Null run
        hyp_n = hyp_cls(**hyp_kwargs)
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1] if null_rows else {})
        fs_null = float(eval_n.get("fit_score", 0.0))

        wall = time.time() - t0
        signal_ok = fs_signal >= threshold
        # Allow near-zero null (< 0.05) to pass without the ratio check
        separation_ok = (fs_signal > fs_null * 1.5) or (fs_null < 0.05)

        if signal_ok and separation_ok:
            status = "PASS"
        elif signal_ok:
            status = "WARN"
        else:
            status = "FAIL"

        return make_result(
            stage=stage_id, name=name, status=status,
            target=f"fit_score_signal >= {threshold} and > null * 1.5",
            result={
                "fit_score_signal": round(fs_signal, 4),
                "fit_score_null": round(fs_null, 4),
                "evidence": eval_s.get("evidence", 0),
                "ready": eval_s.get("ready", False),
                "threshold": threshold,
            },
            wallclock_s=wall,
        )
    except Exception as e:
        return fail_result(
            stage=stage_id, name=name,
            target=f"fit_score_signal >= {threshold}",
            error=f"{e}\n{traceback.format_exc()[-1200:]}",
            wallclock_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Stage 12.1 — CausalHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_1(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import CausalHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.1", "CausalHypothesis",
        CausalHypothesis, {"source": "X", "target": "Y", "lag": 2, "buffer_size": 150},
        _gen_causal(n, 42), _gen_null_keyed(["X", "Y"], n, 99), 0.4,
    )


# ---------------------------------------------------------------------------
# Stage 12.2 — CorrelationalHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_2(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import CorrelationalHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.2", "CorrelationalHypothesis",
        CorrelationalHypothesis, {"var1": "X", "var2": "Y", "buffer_size": 150},
        _gen_correlational(n, 42), _gen_null_keyed(["X", "Y"], n, 99), 0.5,
    )


# ---------------------------------------------------------------------------
# Stage 12.3 — TemporalHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_3(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import TemporalHypothesis
    # Needs ≥80 rows: the residual deque maxlen=60 must overflow past the initial
    # AR convergence period so early large residuals are dropped before evaluate().
    n = 80 if fast else 100
    return _run_hyp_stage(
        "12.3", "TemporalHypothesis",
        TemporalHypothesis, {"variable": "V", "lag": 3, "buffer_size": 150},
        _gen_temporal(n, 42), _gen_null_keyed(["V"], n, 99), 0.3,
    )


# ---------------------------------------------------------------------------
# Stage 12.4 — FunctionalHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_4(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import FunctionalHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.4", "FunctionalHypothesis",
        FunctionalHypothesis, {"source": "X", "target": "Y", "degree": 2, "buffer_size": 150},
        _gen_functional(n, 42), _gen_null_keyed(["X", "Y"], n, 99), 0.6,
    )


# ---------------------------------------------------------------------------
# Stage 12.5 — EquilibriumHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_5(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import EquilibriumHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.5", "EquilibriumHypothesis",
        EquilibriumHypothesis, {"variable": "V", "buffer_size": 150},
        _gen_equilibrium(n, 42), _gen_null_keyed(["V"], n, 99), 0.3,
    )


# ---------------------------------------------------------------------------
# Stage 12.6 — CompositionalHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_6(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import CompositionalHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.6", "CompositionalHypothesis",
        CompositionalHypothesis, {"parts": ["A", "B", "C"], "total": "T", "buffer_size": 100},
        _gen_compositional(n, 42), _gen_compositional_null(n, 99), 0.9,
    )


# ---------------------------------------------------------------------------
# Stage 12.7 — CompetitiveHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_7(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import CompetitiveHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.7", "CompetitiveHypothesis",
        CompetitiveHypothesis, {"var1": "X", "var2": "Y", "buffer_size": 150},
        _gen_competitive(n, 42), _gen_null_keyed(["X", "Y"], n, 99), 0.5,
    )


# ---------------------------------------------------------------------------
# Stage 12.8 — SynergisticHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_8(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import SynergisticHypothesis
    n = 60 if fast else 100
    return _run_hyp_stage(
        "12.8", "SynergisticHypothesis",
        SynergisticHypothesis, {"var1": "X1", "var2": "X2", "target": "Y", "buffer_size": 150},
        _gen_synergistic(n, 42), _gen_null_keyed(["X1", "X2", "Y"], n, 99), 0.3,
    )


# ---------------------------------------------------------------------------
# Stage 12.9 — ProbabilisticHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_9(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import ProbabilisticHypothesis
    n = 60 if fast else 100
    null_rows = [{"C": float(np.random.default_rng(99 + i).standard_normal()),
                  "Y": float(np.random.default_rng(199 + i).standard_normal())}
                 for i in range(n)]
    return _run_hyp_stage(
        "12.9", "ProbabilisticHypothesis",
        ProbabilisticHypothesis, {"condition": "C", "target": "Y", "buffer_size": 200},
        _gen_probabilistic(n, 42), null_rows, 0.3,
    )


# ---------------------------------------------------------------------------
# Stage 12.10 — StructuralHypothesis
# ---------------------------------------------------------------------------

def run_stage_12_10(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships import StructuralHypothesis
    n = 60 if fast else 80  # multiple of 4
    n = (n // 4) * 4
    return _run_hyp_stage(
        "12.10", "StructuralHypothesis",
        StructuralHypothesis, {"group": "G", "outcome": "O", "buffer_size": 200},
        _gen_structural(n, 42), _gen_structural_null(n, 99), 0.4,
    )

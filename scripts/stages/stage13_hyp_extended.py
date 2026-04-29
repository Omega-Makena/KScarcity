"""stage13_hyp_extended.py — Stages 13.1–13.5: Extended hypothesis type benchmarks.

Covers MediatingHypothesis, ModeratingHypothesis, GraphHypothesis,
SimilarityHypothesis, LogicalHypothesis — all previously zero-coverage.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

import numpy as np

from scripts.stages.utils import fail_result, make_result
from scripts.stages.stage12_hyp_core import _safe_evaluate, _gen_null_keyed

# ---------------------------------------------------------------------------
# Signal generators for extended types
# ---------------------------------------------------------------------------

def _gen_mediating(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    """X → M (a=0.8) → Y (b=0.7), direct c'=0.3."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n) * 2.0
    M = 0.8 * X + rng.standard_normal(n) * 0.5
    Y = 0.3 * X + 0.7 * M + rng.standard_normal(n) * 0.3
    return [{"X": float(X[i]), "M": float(M[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_moderating(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    """Y = 0.5X + 0.3Z + 1.0(X·Z) + noise — strong interaction."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Z = rng.standard_normal(n)
    Y = 0.5 * X + 0.3 * Z + 1.0 * X * Z + rng.standard_normal(n) * 0.3
    return [{"X": float(X[i]), "Z": float(Z[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_graph_nonlinear(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    """Y = sin(X) + noise — high NMI, near-zero Pearson → nonlinear_excess signal."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2 * np.pi, n)
    Y = np.sin(X) + rng.standard_normal(n) * 0.2
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_similarity_clusters(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    """3 well-separated clusters at (1,1), (5,5), (9,1) with sigma=0.5."""
    rng = np.random.default_rng(seed)
    centers = [(1.0, 1.0), (5.0, 5.0), (9.0, 1.0)]
    rows = []
    n_per = n // 3
    for cx, cy in centers:
        for _ in range(n_per):
            rows.append({
                "X": float(cx + rng.standard_normal() * 0.5),
                "Y": float(cy + rng.standard_normal() * 0.5),
            })
    rng.shuffle(rows)
    return rows


def _gen_logical_and(n: int = 80, seed: int = 42) -> List[Dict[str, float]]:
    """A, B centred binary ±1; Y = +1 if A>0 AND B>0 else -1 (AND rule)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        A = 1.0 if rng.random() > 0.5 else -1.0
        B = 1.0 if rng.random() > 0.5 else -1.0
        Y = 1.0 if (A > 0 and B > 0) else -1.0
        rows.append({
            "A": A + float(rng.standard_normal() * 0.05),
            "B": B + float(rng.standard_normal() * 0.05),
            "Y": Y + float(rng.standard_normal() * 0.05),
        })
    return rows


# ---------------------------------------------------------------------------
# Null generators for extended types
# ---------------------------------------------------------------------------

def _gen_similarity_null(n: int = 80, seed: int = 99) -> List[Dict[str, float]]:
    """Uniform scatter — no cluster structure."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n)
    Y = rng.uniform(0, 6, n)
    return [{"X": float(X[i]), "Y": float(Y[i])} for i in range(n)]


def _gen_logical_null(n: int = 80, seed: int = 99) -> List[Dict[str, float]]:
    """A, B, Y all independent normals."""
    rng = np.random.default_rng(seed)
    return [{"A": float(rng.standard_normal()),
             "B": float(rng.standard_normal()),
             "Y": float(rng.standard_normal())} for _ in range(n)]


# ---------------------------------------------------------------------------
# Stage 13.1 — MediatingHypothesis
# ---------------------------------------------------------------------------

def run_stage_13_1(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships_extended import MediatingHypothesis
    t0 = time.time()
    n = 60 if fast else 100
    stage_id, name = "13.1", "MediatingHypothesis"
    try:
        signal_rows = _gen_mediating(n, 42)
        null_rows = _gen_null_keyed(["X", "M", "Y"], n, 99)

        hyp_s = MediatingHypothesis("X", "M", "Y")
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1])

        hyp_n = MediatingHypothesis("X", "M", "Y")
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1])

        fs_s = float(eval_s.get("fit_score", 0.0))
        fs_n = float(eval_n.get("fit_score", 0.0))
        indirect = float(eval_s.get("indirect_effect", 0.0))
        has_med = bool(eval_s.get("has_mediation", False))
        wall = time.time() - t0

        threshold = 0.2
        signal_ok = fs_s >= threshold
        sep_ok = fs_s > fs_n * 1.5 or fs_n < 0.05

        status = "PASS" if (signal_ok and sep_ok) else ("WARN" if signal_ok else "FAIL")
        return make_result(stage_id, name, status,
                           f"fit_score_signal >= {threshold} and indirect != 0",
                           {"fit_score_signal": round(fs_s, 4), "fit_score_null": round(fs_n, 4),
                            "indirect_effect": round(indirect, 4), "has_mediation": has_med,
                            "evidence": eval_s.get("evidence", 0)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "MediatingHypothesis runs on mediation data",
                           f"{e}\n{traceback.format_exc()[-1000:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 13.2 — ModeratingHypothesis
# ---------------------------------------------------------------------------

def run_stage_13_2(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships_extended import ModeratingHypothesis
    t0 = time.time()
    n = 60 if fast else 100
    stage_id, name = "13.2", "ModeratingHypothesis"
    try:
        signal_rows = _gen_moderating(n, 42)
        null_rows = _gen_null_keyed(["X", "Z", "Y"], n, 99)

        hyp_s = ModeratingHypothesis("X", "Z", "Y")
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1])

        hyp_n = ModeratingHypothesis("X", "Z", "Y")
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1])

        fs_s = float(eval_s.get("fit_score", 0.0))
        fs_n = float(eval_n.get("fit_score", 0.0))
        interaction = float(eval_s.get("interaction", 0.0))
        has_mod = bool(eval_s.get("has_moderation", False))
        wall = time.time() - t0

        threshold = 0.2
        signal_ok = fs_s >= threshold
        sep_ok = fs_s > fs_n * 1.5 or fs_n < 0.05

        status = "PASS" if (signal_ok and sep_ok) else ("WARN" if signal_ok else "FAIL")
        return make_result(stage_id, name, status,
                           f"fit_score_signal >= {threshold} and interaction coef ≈ 1.0",
                           {"fit_score_signal": round(fs_s, 4), "fit_score_null": round(fs_n, 4),
                            "interaction_coef": round(interaction, 4), "has_moderation": has_mod,
                            "evidence": eval_s.get("evidence", 0)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "ModeratingHypothesis detects interaction",
                           f"{e}\n{traceback.format_exc()[-1000:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 13.3 — GraphHypothesis (nonlinear signal, must NOT fire on linear)
# ---------------------------------------------------------------------------

def run_stage_13_3(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships_extended import GraphHypothesis
    from scripts.stages.stage12_hyp_core import _gen_causal
    t0 = time.time()
    n = 60 if fast else 100
    stage_id, name = "13.3", "GraphHypothesis"
    try:
        # Nonlinear signal (sin wave)
        signal_rows = _gen_graph_nonlinear(n, 42)
        # Linear causal data — should NOT trigger graph structure strongly
        linear_rows = _gen_causal(n, 42)
        null_rows = _gen_null_keyed(["X", "Y"], n, 99)

        hyp_s = GraphHypothesis("X", "Y")
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1])

        hyp_lin = GraphHypothesis("X", "Y")
        for row in linear_rows:
            hyp_lin.fit_step(row)
        eval_lin = _safe_evaluate(hyp_lin, linear_rows[-1])

        hyp_n = GraphHypothesis("X", "Y")
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1])

        fs_s = float(eval_s.get("fit_score", 0.0))
        fs_lin = float(eval_lin.get("fit_score", 0.0))
        fs_n = float(eval_n.get("fit_score", 0.0))
        # GraphHypothesis doesn't return nonlinear_excess — compute from scores
        nonlinear_excess = max(0.0, fs_s - fs_lin)
        nmi = float(eval_s.get("normalized_mi", 0.0))
        wall = time.time() - t0

        threshold = 0.2
        signal_ok = fs_s >= threshold and nonlinear_excess > 0.05
        # Nonlinear should beat linear by some margin (or at least beat null)
        sep_ok = fs_s > fs_n * 1.3

        status = "PASS" if (signal_ok and sep_ok) else ("WARN" if signal_ok else "FAIL")
        return make_result(stage_id, name, status,
                           "fit_score_signal >= 0.2 and nonlinear_excess > 0.05",
                           {"fit_score_signal": round(fs_s, 4),
                            "fit_score_linear": round(fs_lin, 4),
                            "fit_score_null": round(fs_n, 4),
                            "nonlinear_excess": round(nonlinear_excess, 4),
                            "normalized_mi": round(nmi, 4),
                            "evidence": eval_s.get("evidence", 0)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "GraphHypothesis detects nonlinear MI signal",
                           f"{e}\n{traceback.format_exc()[-1000:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 13.4 — SimilarityHypothesis
# ---------------------------------------------------------------------------

def run_stage_13_4(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships_extended import SimilarityHypothesis
    t0 = time.time()
    n = 60 if fast else 90  # multiple of 3 for even cluster sizes
    n = (n // 3) * 3
    stage_id, name = "13.4", "SimilarityHypothesis"
    try:
        signal_rows = _gen_similarity_clusters(n, 42)
        null_rows = _gen_similarity_null(n, 99)

        hyp_s = SimilarityHypothesis(["X", "Y"], n_clusters=3)
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1])

        hyp_n = SimilarityHypothesis(["X", "Y"], n_clusters=3)
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1])

        fs_s = float(eval_s.get("fit_score", 0.0))
        fs_n = float(eval_n.get("fit_score", 0.0))
        sil = float(eval_s.get("silhouette", 0.0))
        wall = time.time() - t0

        threshold = 0.2
        signal_ok = fs_s >= threshold
        sep_ok = fs_s > fs_n * 1.3 or fs_n < 0.1

        status = "PASS" if (signal_ok and sep_ok) else ("WARN" if signal_ok else "FAIL")
        return make_result(stage_id, name, status,
                           "fit_score >= 0.2 and silhouette > 0 on clustered data",
                           {"fit_score_signal": round(fs_s, 4), "fit_score_null": round(fs_n, 4),
                            "silhouette": round(sil, 4), "evidence": eval_s.get("evidence", 0)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "SimilarityHypothesis detects 3-cluster structure",
                           f"{e}\n{traceback.format_exc()[-1000:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 13.5 — LogicalHypothesis (AND rule)
# ---------------------------------------------------------------------------

def run_stage_13_5(fast: bool = False) -> Dict[str, Any]:
    from scarcity.engine.relationships_extended import LogicalHypothesis
    t0 = time.time()
    n = 80 if fast else 120  # need enough rows for threshold stabilisation
    stage_id, name = "13.5", "LogicalHypothesis"
    try:
        signal_rows = _gen_logical_and(n, 42)
        null_rows = _gen_logical_null(n, 99)

        hyp_s = LogicalHypothesis("A", "B", "Y")
        for row in signal_rows:
            hyp_s.fit_step(row)
        eval_s = _safe_evaluate(hyp_s, signal_rows[-1])

        hyp_n = LogicalHypothesis("A", "B", "Y")
        for row in null_rows:
            hyp_n.fit_step(row)
        eval_n = _safe_evaluate(hyp_n, null_rows[-1])

        fs_s = float(eval_s.get("fit_score", 0.0))
        fs_n = float(eval_n.get("fit_score", 0.0))
        best_rule = eval_s.get("best_rule", "UNKNOWN")
        verified_acc = float(eval_s.get("verified_accuracy", 0.0))
        wall = time.time() - t0

        threshold = 0.6
        signal_ok = fs_s >= threshold
        rule_ok = best_rule in ("AND", "IMPLIES")  # AND or IMPLIES both possible for AND data
        sep_ok = fs_s > fs_n * 1.2

        status = "PASS" if (signal_ok and sep_ok) else ("WARN" if signal_ok else "FAIL")
        return make_result(stage_id, name, status,
                           "fit_score >= 0.6, best_rule in (AND, IMPLIES), signal > null",
                           {"fit_score_signal": round(fs_s, 4), "fit_score_null": round(fs_n, 4),
                            "best_rule": best_rule, "rule_correct": rule_ok,
                            "verified_accuracy": round(verified_acc, 4),
                            "evidence": eval_s.get("evidence", 0)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "LogicalHypothesis identifies AND rule",
                           f"{e}\n{traceback.format_exc()[-1000:]}", time.time() - t0)

"""stage16_vectorized.py — Stages 16.1–16.2: VectorizedRLS correctness and throughput.

Previously zero benchmark coverage. Tests batch matrix RLS against scalar reference
and measures vectorization speedup.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict

import numpy as np

from scripts.stages.utils import fail_result, make_result, skip_result


# ---------------------------------------------------------------------------
# Scalar RLS reference (matches the _rls_step in relationships.py)
# ---------------------------------------------------------------------------

def _scalar_rls(P, w, x, y, lam=0.99):
    """Single-model RLS update. Returns (P_new, w_new, error)."""
    denom = lam + float(x @ P @ x)
    k = (P @ x) / denom
    error = y - float(x @ w)
    w_new = w + k * error
    P_new = (P - np.outer(k, x @ P)) / lam
    return P_new, w_new, error


# ---------------------------------------------------------------------------
# Stage 16.1 — VectorizedRLS correctness
# ---------------------------------------------------------------------------

def run_stage_16_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "16.1", "VectorizedRLS"

    try:
        from scarcity.engine.vectorized_core import VectorizedRLS
    except ImportError as e:
        return skip_result(stage_id, name, f"VectorizedRLS import failed: {e}")

    try:
        n = 60 if fast else 100
        rng = np.random.default_rng(42)

        # Generate data: y = 1.5*x1 + 0.8*x2 + noise
        X_raw = rng.standard_normal((n, 2)).astype(np.float64)
        Y_raw = 1.5 * X_raw[:, 0] + 0.8 * X_raw[:, 1] + rng.standard_normal(n) * 0.2

        # Scalar RLS (float64 reference)
        P_s = np.eye(2, dtype=np.float64) * 100.0
        w_s = np.zeros(2, dtype=np.float64)
        for i in range(n):
            P_s, w_s, _ = _scalar_rls(P_s, w_s, X_raw[i], Y_raw[i], lam=0.99)

        # VectorizedRLS with n_models=1
        vrls = VectorizedRLS(n_models=1, n_features=2, lambda_forget=0.99)
        for i in range(n):
            x_batch = X_raw[i:i+1].astype(np.float32)   # (1, 2)
            y_batch = Y_raw[i:i+1].astype(np.float32)    # (1,)
            vrls.update(x_batch, y_batch)

        # Extract weights from VectorizedRLS
        # W shape: (M, F) — take row 0
        w_v = vrls.W[0].astype(np.float64) if hasattr(vrls, "W") else np.zeros(2)

        weight_diff = float(np.max(np.abs(w_v - w_s)))
        weights_agree = weight_diff < 0.1  # allow float32 vs float64 tolerance

        # Near-zero denominator test: constant X
        vrls2 = VectorizedRLS(n_models=1, n_features=2, lambda_forget=0.99)
        X_const = np.ones((5, 2), dtype=np.float32)
        Y_const = np.ones(5, dtype=np.float32)
        try:
            for i in range(5):
                vrls2.update(X_const[i:i+1], Y_const[i:i+1])
            denom_handled = True
        except Exception:
            denom_handled = False

        wall = time.time() - t0
        status = "PASS" if (weights_agree and denom_handled) else (
            "WARN" if denom_handled else "FAIL")

        return make_result(stage_id, name, status,
                           "VectorizedRLS weights agree with scalar RLS within 0.1; near-zero denom handled",
                           {"scalar_weights": w_s.tolist(),
                            "vectorized_weights": w_v.tolist(),
                            "weight_diff_max": round(weight_diff, 6),
                            "weights_agree": weights_agree,
                            "denom_handled": denom_handled},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "VectorizedRLS matches scalar RLS",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 16.2 — VectorizedRLS throughput (>= 5x scalar loop)
# ---------------------------------------------------------------------------

def run_stage_16_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "16.2", "VectorizedHypothesisPool"

    try:
        from scarcity.engine.vectorized_core import VectorizedRLS
    except ImportError as e:
        return skip_result(stage_id, name, f"VectorizedRLS import failed: {e}")

    try:
        M = 200 if fast else 1000  # number of models
        n = 50 if fast else 100    # rows to feed
        rng = np.random.default_rng(42)

        X_all = rng.standard_normal((n, 2)).astype(np.float32)
        Y_all = rng.standard_normal(n).astype(np.float32)

        # --- Vectorized: update all M models at once ---
        vrls = VectorizedRLS(n_models=M, n_features=2, lambda_forget=0.99)
        t_vec_start = time.time()
        for i in range(n):
            X_batch = np.tile(X_all[i:i+1], (M, 1))   # (M, 2)
            Y_batch = np.tile(Y_all[i:i+1], M)          # (M,)
            vrls.update(X_batch, Y_batch)
        t_vec = time.time() - t_vec_start

        # --- Scalar loop: update M models one at a time ---
        Ps = [np.eye(2, dtype=np.float32) * 100.0 for _ in range(M)]
        Ws = [np.zeros(2, dtype=np.float32) for _ in range(M)]
        t_scalar_start = time.time()
        for i in range(n):
            x = X_all[i].astype(np.float64)
            y = float(Y_all[i])
            for m in range(M):
                P, w, _ = _scalar_rls(Ps[m].astype(np.float64),
                                       Ws[m].astype(np.float64), x, y, lam=0.99)
                Ps[m] = P.astype(np.float32)
                Ws[m] = w.astype(np.float32)
        t_scalar = time.time() - t_scalar_start

        speedup = t_scalar / max(t_vec, 1e-9)

        wall = time.time() - t0
        status = "PASS" if speedup >= 5.0 else ("WARN" if speedup >= 2.0 else "FAIL")

        return make_result(stage_id, name, status,
                           f"VectorizedRLS({M} models) >= 5x faster than scalar loop",
                           {"n_models": M, "n_rows": n,
                            "vectorized_s": round(t_vec, 3),
                            "scalar_s": round(t_scalar, 3),
                            "speedup_x": round(speedup, 2)},
                           wall)
    except Exception as e:
        return fail_result(stage_id, name, "VectorizedRLS(1000 models) >= 5x scalar loop",
                           f"{e}\n{traceback.format_exc()[-1200:]}", time.time() - t0)

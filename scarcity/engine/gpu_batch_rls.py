"""
GPU-accelerated batch Recursive Least Squares engine.

Runs M independent RLS models simultaneously using PyTorch bmm/einsum.
Mirrors the scalar _rls_step() in relationships.py but over an entire
batch of models in one GPU kernel.

Shape conventions (matching vectorized_core.py but on CUDA):
  M : total models  (= n_runs × n_hypotheses_in_group)
  F : feature dim   (2 for linear, 3 for lag-2 / triplet, 4 for interaction)
  W : (M, F)        — weight vectors
  P : (M, F, F)     — covariance matrices  (initialised to 10·I)

Tracks the four metrics needed by MetaController:
  fit_score   — R² from online Welford SSE / SST
  confidence  — Bayesian accumulator alpha / (alpha + beta), λ=0.99
  stability   — 1 − CV(residuals), EMA-based
  evidence    — integer step count
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level CUDA check
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Core batched RLS step (free function, usable independently)
# ---------------------------------------------------------------------------

def rls_step_batch(
    P: torch.Tensor,   # (M, F, F)
    W: torch.Tensor,   # (M, F)
    X: torch.Tensor,   # (M, F)
    Y: torch.Tensor,   # (M,)
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One batched RLS step.  Returns (W_new, P_new, residual).

    Implements the recursive formula:
        Px    = P @ x
        denom = lambda + x^T Px
        K     = Px / denom
        W_new = W + K * (y - x^T W)
        P_new = (P - outer(K, Px)) / lambda
    """
    # Px = P @ x  →  (M, F)
    Px = torch.einsum("mij,mj->mi", P, X)
    # denom = lambda + x^T Px  →  (M,)
    denom = lam + torch.einsum("mi,mi->m", X, Px)
    # Kalman gain  →  (M, F)
    K = Px / denom.unsqueeze(-1).clamp(min=1e-12)
    # prediction  →  (M,)
    y_hat = torch.einsum("mi,mi->m", W, X)
    # residual  →  (M,)
    res = Y - y_hat
    # weight update
    W_new = W + K * res.unsqueeze(-1)
    # covariance update: P_new = (P - K ⊗ Px) / lam
    KPx = torch.einsum("mi,mj->mij", K, Px)
    P_new = (P - KPx) / lam
    return W_new, P_new, res


# ---------------------------------------------------------------------------
# GPUBatchRLS
# ---------------------------------------------------------------------------

class GPUBatchRLS:
    """
    Stateful batched RLS engine that maintains all M models' state on GPU.

    Usage
    -----
    rls = GPUBatchRLS(M=51_000, F=2)
    for t in range(T):
        X = ...  # (M, 2)
        Y = ...  # (M,)
        rls.update(X, Y)

    scores = rls.fit_score   # (M,) R² in [0, 1]
    confs  = rls.confidence  # (M,) Bayesian confidence in [0, 1]
    """

    # Bayesian accumulator forgetting factor — matches Hypothesis.update()
    _LAMBDA_CONF: float = 0.99
    # EMA alpha for stability tracking
    _EMA_ALPHA: float = 0.05

    def __init__(
        self,
        M: int,
        F: int,
        lam: float = 0.99,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        alpha0: float = 0.1,
        beta0: float = 1.0,
    ) -> None:
        self.M = M
        self.F = F
        self.lam = lam
        self.device = device
        self.dtype = dtype

        kw = {"device": device, "dtype": dtype}

        # RLS state
        self.W = torch.zeros(M, F, **kw)
        eye = torch.eye(F, **kw)
        self.P = (eye * 10.0).unsqueeze(0).expand(M, -1, -1).clone()

        # R² tracking (online Welford)
        self.n      = torch.zeros(M, device=device, dtype=torch.int64)
        self.mean_y = torch.zeros(M, **kw)
        self.sse    = torch.zeros(M, **kw)   # sum of squared errors
        self.sst    = torch.zeros(M, **kw)   # sum of squared totals (variance * n)

        # Bayesian confidence — mirrors Hypothesis.__init__ skeptical prior
        self.alpha = torch.full((M,), alpha0, **kw)
        self.beta  = torch.full((M,), beta0,  **kw)

        # Stability: EMA of |residual| and residual²
        self._res_abs_ema = torch.zeros(M, **kw)
        self._res_sq_ema  = torch.zeros(M, **kw)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        X: torch.Tensor,  # (M, F)
        Y: torch.Tensor,  # (M,)
    ) -> None:
        """
        Update all M models with new observations.

        Inputs are cast to self.dtype automatically.
        NaN in X (any feature) or Y causes that model's step to be skipped via
        masking (its state is left unchanged).
        """
        X = X.to(dtype=self.dtype, device=self.device)
        Y = Y.to(dtype=self.dtype, device=self.device)
        valid = torch.isfinite(X).all(-1) & torch.isfinite(Y)   # (M,)

        if valid.all():
            self._full_update(X, Y)
        else:
            # Only update valid models — avoid corrupting state with NaN
            idx = valid.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                self._partial_update(idx, X[idx], Y[idx])

    def _full_update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        W_new, P_new, res = rls_step_batch(self.P, self.W, X, Y, self.lam)
        self.W = W_new
        self.P = P_new
        self._accumulate(slice(None), Y, res)

    def _partial_update(
        self,
        idx: torch.Tensor,
        X_sub: torch.Tensor,
        Y_sub: torch.Tensor,
    ) -> None:
        W_new, P_new, res = rls_step_batch(
            self.P[idx], self.W[idx], X_sub, Y_sub, self.lam
        )
        self.W[idx] = W_new
        self.P[idx] = P_new
        self._accumulate(idx, Y_sub, res)

    def _accumulate(self, idx, Y: torch.Tensor, res: torch.Tensor) -> None:
        """Update n, Welford SSE/SST, Bayesian accumulators, stability EMA."""
        n_old   = self.n[idx].to(self.dtype)
        n_new   = n_old + 1.0
        mean_y  = self.mean_y[idx]

        # Welford one-pass update for variance of Y
        delta   = Y - mean_y
        mean_y_new = mean_y + delta / n_new
        delta2  = Y - mean_y_new
        sst_new = self.sst[idx] + delta * delta2
        sse_new = self.sse[idx] + res * res

        # R² → signal for Bayesian accumulator
        r2 = (1.0 - sse_new / (sst_new + 1e-12)).clamp(0.0, 1.0)
        signal = ((r2 - 0.5) * 2.0).clamp(0.0, 1.0)

        lam = self._LAMBDA_CONF
        alpha_new = lam * self.alpha[idx] + signal
        beta_new  = lam * self.beta[idx]  + (1.0 - signal)

        ema = self._EMA_ALPHA
        res_abs_new = (1 - ema) * self._res_abs_ema[idx] + ema * res.abs()
        res_sq_new  = (1 - ema) * self._res_sq_ema[idx]  + ema * res * res

        # Write-back
        self.n[idx]          = n_new.to(torch.int64)
        self.mean_y[idx]     = mean_y_new
        self.sst[idx]        = sst_new
        self.sse[idx]        = sse_new
        self.alpha[idx]      = alpha_new
        self.beta[idx]       = beta_new
        self._res_abs_ema[idx] = res_abs_new
        self._res_sq_ema[idx]  = res_sq_new

    # ------------------------------------------------------------------
    # Derived metrics (properties matching Hypothesis attributes)
    # ------------------------------------------------------------------

    @property
    def fit_score(self) -> torch.Tensor:
        """R² goodness-of-fit per model. Shape: (M,). Range [0, 1]."""
        return (1.0 - self.sse / (self.sst + 1e-12)).clamp(0.0, 1.0)

    @property
    def confidence(self) -> torch.Tensor:
        """Bayesian confidence. Shape: (M,). Range [0, 1]."""
        return self.alpha / (self.alpha + self.beta + 1e-12)

    @property
    def stability(self) -> torch.Tensor:
        """1 - CV(residuals). Shape: (M,). Range [0, 1]."""
        std_res = torch.sqrt(self._res_sq_ema.clamp(0.0))
        cv = std_res / (self._res_abs_ema + 1e-8)
        return (1.0 - cv).clamp(0.0, 1.0)

    @property
    def evidence(self) -> torch.Tensor:
        """Step count per model. Shape: (M,). dtype int64."""
        return self.n

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-initialize all state tensors to starting values."""
        self.__init__(self.M, self.F, self.lam, self.device, self.dtype)

    def to_cpu(self) -> "GPUBatchRLS":
        """Return a copy of this RLS pool moved to CPU (for inspection)."""
        cpu = GPUBatchRLS(self.M, self.F, self.lam, "cpu", self.dtype)
        for attr in ("W", "P", "n", "mean_y", "sse", "sst",
                     "alpha", "beta", "_res_abs_ema", "_res_sq_ema"):
            setattr(cpu, attr, getattr(self, attr).cpu())
        return cpu

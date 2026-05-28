"""
GPU Hypothesis Pool — hypothesis index, feature extraction, and lifecycle.

Mirrors OnlineDiscoveryEngine.initialize_v2() but stores all state as GPU
tensors rather than Python objects.  One instance represents ALL hypotheses
for ONE data schema (variable names / column layout).

Key design
----------
Hypotheses are grouped by (permuted_column_index, feature_dim_F).  Within a
group every model shares the same feature shape, so one GPUBatchRLS handles
all of them.  Different groups share no GPU state.

Permutation grouping: for null-distribution Monte-Carlo we permute exactly
one column per hypothesis.  All hypotheses with the same perm_col share the
same set of B_perm scrambled datasets, letting us batch
(1 + B_perm) × N_hyp_in_group models in one GPU call.

Supported hypothesis types (15 total, matches initialize_v2):
  Pairwise (8):    causal, correlational, functional, competitive,
                   compositional, probabilistic, structural, graph
  Univariate (2):  temporal, equilibrium
  Triplet (4):     synergistic, mediating, moderating, logical
  Collective (1):  similarity (runs on CPU — K-means, not RLS)

Feature vectors
  F=2: correlational, functional, competitive, compositional,
        probabilistic, structural, equilibrium
        X = [1,  a_t]                    Y = b_t
  F=3: causal, temporal, mediating, logical, graph
        causal    X = [1, a_{t-1}, b_{t-1}]  Y = b_t
        temporal  X = [1, v_{t-2}, v_{t-1}]  Y = v_t
        mediating X = [1, a_t,     b_t   ]   Y = c_t
        logical   X = [1, a_t,     b_t   ]   Y = c_t
        graph     X = [cos(a_t), sin(a_t), 1] Y = b_t
  F=4: synergistic, moderating
        X = [1, a_t, b_t, a_t*b_t]          Y = c_t
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .gpu_batch_rls import GPUBatchRLS

# ---------------------------------------------------------------------------
# Permutation strategy constants
# ---------------------------------------------------------------------------

PERM_SHUFFLE = "shuffle"   # independent random permutation of perm_col
PERM_SHIFT   = "shift"     # circular shift (preserves autocorrelation)
PERM_PHASE   = "phase"     # phase-randomise FFT (temporal/equilibrium)


# ---------------------------------------------------------------------------
# HypoSpec — descriptor for one hypothesis
# ---------------------------------------------------------------------------

@dataclass
class HypoSpec:
    """All information needed to extract features and permute data for one hypothesis."""
    rel_type:    str           # e.g. 'causal'
    variables:   List[str]     # variable names (source first, then predictors, then target)
    F:           int           # feature vector dimension (-1 for non-RLS types)
    perm_col:    str           # column to permute for null distribution
    perm_type:   str           # PERM_SHUFFLE / PERM_SHIFT / PERM_PHASE
    col_a:       int = -1      # first predictor column index
    col_b:       int = -1      # second predictor column index (-1 if unused)
    col_y:       int = -1      # target column index
    lag_a:       int = 0       # lag for col_a (0 = current step)
    lag_b:       int = 0       # lag for col_b (0 = current step)
    interaction: bool = False  # include col_a * col_b as 4th feature (F=4 types)

    # Per-run lifecycle state tracking — not part of hash/eq
    _lifecycle_state: str = field(default="tentative", repr=False, compare=False)


# ---------------------------------------------------------------------------
# GPUHypothesisPool
# ---------------------------------------------------------------------------

class GPUHypothesisPool:
    """
    Builds and manages the GPU hypothesis index for one data schema.

    Parameters
    ----------
    col_names   : list of variable names (same order as DataFrame columns)
    buffer_size : passed through for parity with engine API (unused by GPU)
    device      : 'cuda' or 'cpu'
    dtype       : torch.float64 matches the engine's scalar RLS precision
    """

    # Lifecycle thresholds matching MetaController defaults
    MIN_EVIDENCE: int   = 20
    CONF_THRESH:  float = 0.70
    STAB_THRESH:  float = 0.60
    KILL_THRESH:  float = 0.10

    def __init__(
        self,
        col_names:   List[str],
        buffer_size: int = 150,
        device:      str = "cuda",
        dtype:       torch.dtype = torch.float64,
    ) -> None:
        self.col_names  = col_names
        self.col_index: Dict[str, int] = {c: i for i, c in enumerate(col_names)}
        self.N_vars     = len(col_names)
        self.buffer_size = buffer_size
        self.device     = device
        self.dtype      = dtype

        self.specs: List[HypoSpec] = []
        self._build_specs()

        # {(perm_col_idx, F): [list of spec indices]}
        self._groups: Dict[Tuple[int, int], List[int]] = {}
        self._index_groups()

    # ------------------------------------------------------------------
    # Spec construction — mirrors OnlineDiscoveryEngine.initialize_v2()
    # ------------------------------------------------------------------

    def _build_specs(self) -> None:
        cols = self.col_names
        K    = len(cols)
        ci   = self.col_index

        # ── 1. Univariate ────────────────────────────────────────────────
        for v in cols:
            # Temporal AR(2): X=[1, v_{t-2}, v_{t-1}], Y=v_t
            # PERM_SHUFFLE destroys autocorrelation → null R²≈0.
            # PERM_PHASE would preserve linear autocorrelation, giving
            # T_obs ≈ T_perm and zero power for the linear AR(2) statistic.
            self.specs.append(HypoSpec(
                rel_type="temporal", variables=[v], F=3,
                perm_col=v, perm_type=PERM_SHUFFLE,
                col_a=ci[v], lag_a=2,
                col_b=ci[v], lag_b=1,
                col_y=ci[v],
            ))
            # Equilibrium (mean-reverting AR(1)): X=[1, v_{t-1}], Y=v_t
            self.specs.append(HypoSpec(
                rel_type="equilibrium", variables=[v], F=2,
                perm_col=v, perm_type=PERM_SHUFFLE,
                col_a=ci[v], lag_a=1,
                col_y=ci[v],
            ))

        # ── 2. Pairwise ordered pairs ────────────────────────────────────
        ordered_pairs = [
            (cols[i], cols[j])
            for i in range(K)
            for j in range(K)
            if i != j
        ]

        for a, b in ordered_pairs:
            # Correlational: X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="correlational", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Functional (linear): X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="functional", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Causal (Granger approx): X=[1, a_{t-1}, b_{t-1}], Y=b_t
            # Permute the CAUSE (a) with circular shift — matches engine
            self.specs.append(HypoSpec(
                rel_type="causal", variables=[a, b], F=3,
                perm_col=a, perm_type=PERM_SHIFT,
                col_a=ci[a], lag_a=1, col_b=ci[b], lag_b=1,
                col_y=ci[b],
            ))
            # Competitive: X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="competitive", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Compositional: X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="compositional", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Probabilistic: X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="probabilistic", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Structural: X=[1, a_t], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="structural", variables=[a, b], F=2,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))
            # Graph (nonlinear): X=[cos(a_t), sin(a_t), 1], Y=b_t
            self.specs.append(HypoSpec(
                rel_type="graph", variables=[a, b], F=3,
                perm_col=b, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_y=ci[b],
            ))

        # ── 3. Triplets (cap at 100 matching engine) ─────────────────────
        triplets = list(itertools.combinations(cols, 3))[:100]

        for a, b, c in triplets:
            # Synergistic: X=[1, a_t, b_t, a_t*b_t], Y=c_t
            self.specs.append(HypoSpec(
                rel_type="synergistic", variables=[a, b, c], F=4,
                perm_col=c, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_b=ci[b], lag_b=0,
                col_y=ci[c], interaction=True,
            ))
            # Mediating: X=[1, a_t, b_t], Y=c_t
            self.specs.append(HypoSpec(
                rel_type="mediating", variables=[a, b, c], F=3,
                perm_col=c, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_b=ci[b], lag_b=0,
                col_y=ci[c],
            ))
            # Moderating: X=[1, a_t, b_t, a_t*b_t], Y=c_t
            self.specs.append(HypoSpec(
                rel_type="moderating", variables=[a, b, c], F=4,
                perm_col=c, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_b=ci[b], lag_b=0,
                col_y=ci[c], interaction=True,
            ))
            # Logical (boolean approx as linear): X=[1, a_t, b_t], Y=c_t
            self.specs.append(HypoSpec(
                rel_type="logical", variables=[a, b, c], F=3,
                perm_col=c, perm_type=PERM_SHUFFLE,
                col_a=ci[a], lag_a=0, col_b=ci[b], lag_b=0,
                col_y=ci[c],
            ))

        # ── 4. Similarity (collective, CPU-only) ──────────────────────────
        if K >= 3:
            subset = cols[:min(5, K)]
            self.specs.append(HypoSpec(
                rel_type="similarity", variables=subset, F=-1,
                perm_col="__all__", perm_type=PERM_SHUFFLE,
                col_a=-1, col_y=-1,
            ))

    def _index_groups(self) -> None:
        """Group spec indices by (perm_col_idx, F) for batched GPU execution."""
        ci = self.col_index
        for idx, s in enumerate(self.specs):
            if s.F <= 0:
                continue  # similarity handled separately on CPU
            perm_idx = ci.get(s.perm_col, -1)
            key = (perm_idx, s.F)
            self._groups.setdefault(key, []).append(idx)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def n_gpu_hypotheses(self) -> int:
        """Number of hypotheses handled by GPU (excludes similarity)."""
        return sum(len(v) for v in self._groups.values())

    @property
    def n_total(self) -> int:
        return len(self.specs)

    def groups(self) -> Dict[Tuple[int, int], List[HypoSpec]]:
        """Return {(perm_col_idx, F): [HypoSpec, ...]} for GPU processing."""
        return {
            key: [self.specs[i] for i in idxs]
            for key, idxs in self._groups.items()
        }

    # ------------------------------------------------------------------
    # Feature extraction (GPU)
    # ------------------------------------------------------------------

    def extract_features_gpu(
        self,
        data:      torch.Tensor,     # (R, T, N_vars)
        spec_list: List[HypoSpec],
        t:         int,              # current timestep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build X (R, N_g, F) and Y (R, N_g) GPU tensors for timestep t.

        All specs in spec_list must have the same F.
        """
        R   = data.shape[0]
        N_g = len(spec_list)
        F   = spec_list[0].F
        dev = data.device
        dt  = data.dtype

        X = torch.zeros(R, N_g, F, device=dev, dtype=dt)
        Y = torch.zeros(R, N_g, device=dev, dtype=dt)

        for gi, s in enumerate(spec_list):

            if s.rel_type == "graph":
                a_vals = data[:, t, s.col_a]          # (R,)
                X[:, gi, 0] = torch.cos(a_vals)
                X[:, gi, 1] = torch.sin(a_vals)
                X[:, gi, 2] = 1.0
                Y[:, gi]    = data[:, t, s.col_y]

            elif s.rel_type == "temporal":
                # X = [1, v_{t-lag_a}, v_{t-lag_b}]
                t_a = max(0, t - s.lag_a)
                t_b = max(0, t - s.lag_b)
                X[:, gi, 0] = 1.0
                X[:, gi, 1] = data[:, t_a, s.col_a]
                X[:, gi, 2] = data[:, t_b, s.col_b]
                Y[:, gi]    = data[:, t, s.col_y]

            elif s.rel_type == "causal":
                # X = [1, a_{t-lag_a}, b_{t-lag_b}]
                t_a = max(0, t - s.lag_a)
                t_b = max(0, t - s.lag_b)
                X[:, gi, 0] = 1.0
                X[:, gi, 1] = data[:, t_a, s.col_a]
                X[:, gi, 2] = data[:, t_b, s.col_b]
                Y[:, gi]    = data[:, t, s.col_y]

            elif s.interaction:
                # F=4: X = [1, a_t, b_t, a_t*b_t]
                a_v = data[:, t, s.col_a]
                b_v = data[:, t, s.col_b]
                X[:, gi, 0] = 1.0
                X[:, gi, 1] = a_v
                X[:, gi, 2] = b_v
                X[:, gi, 3] = a_v * b_v
                Y[:, gi]    = data[:, t, s.col_y]

            elif s.col_b >= 0 and s.F == 3:
                # F=3 triplet without interaction: X = [1, a_t, b_t]
                X[:, gi, 0] = 1.0
                X[:, gi, 1] = data[:, t, s.col_a]
                X[:, gi, 2] = data[:, t, s.col_b]
                Y[:, gi]    = data[:, t, s.col_y]

            else:
                # F=2 with optional lag: X = [1, a_{t-lag_a}]
                t_a = max(0, t - s.lag_a)
                X[:, gi, 0] = 1.0
                X[:, gi, 1] = data[:, t_a, s.col_a]
                Y[:, gi]    = data[:, t, s.col_y]

        return X, Y


# ---------------------------------------------------------------------------
# LifecycleEmulator — mirrors MetaController on numpy arrays
# ---------------------------------------------------------------------------

class LifecycleEmulator:
    """
    Tracks TENTATIVE / ACTIVE / DECAYING / DEAD state per (run, hypothesis).

    Called every lifecycle_interval steps with GPU-derived metrics transferred
    to CPU numpy.  Mirrors the exact MetaController.manage_lifecycle() logic.
    """

    _STATE_TENTATIVE = 0
    _STATE_ACTIVE    = 1
    _STATE_DECAYING  = 2
    _STATE_DEAD      = 3

    def __init__(
        self,
        N_hyp:        int,
        R:            int,
        min_evidence: int   = GPUHypothesisPool.MIN_EVIDENCE,
        conf_thresh:  float = GPUHypothesisPool.CONF_THRESH,
        stab_thresh:  float = GPUHypothesisPool.STAB_THRESH,
        kill_thresh:  float = GPUHypothesisPool.KILL_THRESH,
        small_dataset: bool = False,
    ) -> None:
        if small_dataset:
            # Mirror MetaController.small_dataset() thresholds exactly
            min_evidence = 10
            conf_thresh  = 0.55
            stab_thresh  = 0.50
            kill_thresh  = 0.0   # disabled — pool capacity is the only pruning
        self.N = N_hyp
        self.R = R
        self.min_ev   = min_evidence
        self.conf_th  = conf_thresh
        self.stab_th  = stab_thresh
        self.kill_th  = kill_thresh

        # Integer state codes: 0=tentative, 1=active, 2=decaying, 3=dead
        self.state = np.zeros((R, N_hyp), dtype=np.int8)

    def update(
        self,
        conf: np.ndarray,   # (R, N_hyp)
        stab: np.ndarray,   # (R, N_hyp)
        evid: np.ndarray,   # (R, N_hyp)  int
    ) -> None:
        """Apply MetaController state machine to all (run, hyp) pairs."""
        T  = self._STATE_TENTATIVE
        A  = self._STATE_ACTIVE
        D  = self._STATE_DECAYING
        DX = self._STATE_DEAD

        ev_ok = evid > self.min_ev

        is_tent  = self.state == T
        is_act   = self.state == A
        is_decay = self.state == D

        # TENTATIVE → ACTIVE
        self.state[is_tent & ev_ok & (conf > self.conf_th) & (stab > self.stab_th)] = A
        # TENTATIVE → DEAD (early kill before re-checking is_tent)
        self.state[(self.state == T) & ev_ok & (conf < self.kill_th)] = DX

        # ACTIVE → DECAYING
        self.state[is_act & ((conf < self.conf_th - 0.1) | (stab < self.stab_th - 0.1))] = D
        # DECAYING → ACTIVE (recovered)
        self.state[is_decay & (conf > self.conf_th) & (stab > self.stab_th)] = A
        # DECAYING → DEAD
        self.state[is_decay & (conf < self.kill_th)] = DX

    def dead_mask(self) -> np.ndarray:
        """Boolean (R, N_hyp): True where DEAD."""
        return self.state == self._STATE_DEAD

    def summary(self, run: int = 0) -> Dict[str, int]:
        """State count dictionary for one run index."""
        s = self.state[run]
        return {
            "tentative": int((s == self._STATE_TENTATIVE).sum()),
            "active":    int((s == self._STATE_ACTIVE).sum()),
            "decaying":  int((s == self._STATE_DECAYING).sum()),
            "dead":      int((s == self._STATE_DEAD).sum()),
        }

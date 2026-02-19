"""
Exogenous Shock System for Kshield.
Phase 4: Adversarial Stress Test (Stochastic Processes)
Phase 5: Research-Grade Extensions — Markov-switching, jump-diffusion,
         fat-tailed (Student-t), and correlated multi-channel shocks.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import random


class ShockType(Enum):
    """Types of economic shocks."""
    IMPULSE = "impulse"
    OU_PROCESS = "ou_process"
    BROWNIAN = "brownian"
    MARKOV_SWITCHING = "markov_switching"
    JUMP_DIFFUSION = "jump_diffusion"
    STUDENT_T = "student_t"

@dataclass
class Shock:
    """Base Shock Class"""
    name: str
    target_metric: str
    active: bool = True

    def get_delta(self, current_val: float) -> float:
        return 0.0

    def step(self):
        pass

@dataclass
class ImpulseShock(Shock):
    """Classic 'Hit' shock"""
    magnitude: float = 0.0
    decay: float = 0.9
    _current_val: float = 0.0
    
    def __post_init__(self):
        self._current_val = self.magnitude

    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        return self._current_val

    def step(self):
        self._current_val *= self.decay
        if abs(self._current_val) < 1e-4:
            self.active = False

@dataclass
class OUProcessShock(Shock):
    """
    Ornstein-Uhlenbeck Process.
    Mean-reverting stochastic process. Use for realistic volatility.
    dx = theta * (mu - x) * dt + sigma * dW
    """
    theta: float = 0.15 # Speed of mean reversion
    mu: float = 0.0     # Long-Run Mean (offset, usually 0)
    sigma: float = 0.2  # Volatility
    dt: float = 1.0     # Time step
    _x: float = 0.0
    
    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        # Return the NOISE offset to add to the value
        # We assume this shock adds specific noise to the metric
        return self._x

    def step(self):
        # Euler-Maruyama method
        dw = np.random.normal(0, np.sqrt(self.dt))
        dx = self.theta * (self.mu - self._x) * self.dt + self.sigma * dw
        self._x += dx

@dataclass
class BrownianShock(Shock):
    """
    Geometric Brownian Motion (Random Walk).
    Good for asset prices (Drift + Volatility).
    """
    drift: float = 0.0
    volatility: float = 0.1
    dt: float = 1.0
    _accumulated: float = 0.0

    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        return self._accumulated

    def step(self):
        epsilon = np.random.normal(0, 1)
        # Delta S = S * (mu dt + sigma epsilon sqrt(dt))
        # Simplified additive walk for generic metric:
        change = self.drift * self.dt + self.volatility * epsilon * np.sqrt(self.dt)
        self._accumulated += change


# =========================================================================
# Phase 5: Research-Grade Stochastic Processes
# =========================================================================

@dataclass
class MarkovSwitchingShock(Shock):
    """
    Markov Regime-Switching shock process.

    Maintains a discrete regime state that transitions according to a
    stochastic transition matrix.  Each regime has its own OU parameters
    (mean, volatility), implementing Hamilton (1989)-style dynamics.

    Transition matrix rows must sum to 1.  Default: 2-regime (calm/crisis).
    """
    n_regimes: int = 2
    # Transition matrix P[i,j] = Pr(regime j | regime i).  Default: sticky regimes.
    transition_matrix: Optional[np.ndarray] = None
    # Per-regime OU parameters: list of (theta, mu, sigma)
    regime_params: Optional[List[Tuple[float, float, float]]] = None
    dt: float = 1.0
    _regime: int = 0
    _x: float = 0.0

    def __post_init__(self):
        if self.transition_matrix is None:
            # Default: 95% stay, 5% switch
            self.transition_matrix = np.array([
                [0.95, 0.05],
                [0.10, 0.90],
            ])
        if self.regime_params is None:
            # Regime 0 (calm): low vol, Regime 1 (crisis): high vol, negative drift
            self.regime_params = [
                (0.15, 0.0, 0.05),   # calm: θ=0.15, μ=0, σ=0.05
                (0.05, -0.02, 0.25), # crisis: θ=0.05, μ=-0.02, σ=0.25
            ]

    @property
    def regime(self) -> int:
        return self._regime

    def get_delta(self, current_val: float) -> float:
        if not self.active:
            return 0.0
        return self._x

    def step(self):
        # 1. Regime transition (draw from row of transition matrix)
        probs = self.transition_matrix[self._regime]
        self._regime = int(np.random.choice(self.n_regimes, p=probs))

        # 2. OU dynamics under current regime
        theta, mu, sigma = self.regime_params[self._regime]
        dw = np.random.normal(0, np.sqrt(self.dt))
        dx = theta * (mu - self._x) * self.dt + sigma * dw
        self._x += dx

    def regime_filtered_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """
        Hamilton filter: compute filtered regime probabilities from observations.

        Uses the transition matrix and per-regime Gaussian likelihoods to
        produce a (T x n_regimes) array of P(regime_t | y_1,...,y_t).

        Args:
            observations: 1D array of observed shock realizations.

        Returns:
            (T, n_regimes) array of filtered regime probabilities.
        """
        T = len(observations)
        K = self.n_regimes
        P = self.transition_matrix
        filtered = np.zeros((T, K))

        # Initial regime probability: uniform
        prob = np.ones(K) / K

        for t in range(T):
            y = observations[t]
            # Predicted probabilities
            pred = prob @ P

            # Likelihoods under each regime (Gaussian)
            likelihoods = np.zeros(K)
            for k in range(K):
                _, mu, sigma = self.regime_params[k]
                s = max(sigma, 1e-6)
                likelihoods[k] = np.exp(-0.5 * ((y - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

            # Update
            joint = pred * likelihoods
            total = joint.sum()
            if total > 0:
                filtered[t] = joint / total
            else:
                filtered[t] = pred
            prob = filtered[t]

        return filtered


@dataclass
class JumpDiffusionShock(Shock):
    """
    Merton Jump-Diffusion process.

    Combines continuous GBM dynamics with a compound Poisson jump component.
    Jumps arrive at rate λ (per time step) with log-normal jump sizes.
    Captures sudden market dislocations (currency crashes, commodity spikes).

    dX = drift·dt + σ·dW + J·dN
    where dN ~ Poisson(λ·dt), J ~ Normal(jump_mean, jump_std)
    """
    drift: float = 0.0
    volatility: float = 0.1
    jump_intensity: float = 0.1   # λ: expected jumps per time step
    jump_mean: float = -0.05      # Average jump size (negative = crash)
    jump_std: float = 0.10        # Jump size dispersion
    dt: float = 1.0
    _x: float = 0.0
    _jump_count: int = 0          # Total jumps observed

    def get_delta(self, current_val: float) -> float:
        if not self.active:
            return 0.0
        return self._x

    def step(self):
        # Continuous (GBM) component
        dw = np.random.normal(0, np.sqrt(self.dt))
        dx = self.drift * self.dt + self.volatility * dw

        # Jump component: Poisson arrivals
        n_jumps = np.random.poisson(self.jump_intensity * self.dt)
        if n_jumps > 0:
            jumps = np.random.normal(self.jump_mean, self.jump_std, size=n_jumps)
            dx += jumps.sum()
            self._jump_count += n_jumps

        self._x += dx


@dataclass
class StudentTShock(Shock):
    """
    Fat-tailed stochastic process using Student-t innovations.

    Replaces Gaussian noise with Student-t distributed innovations,
    producing heavier tails (more extreme events) while preserving
    the OU mean-reversion structure.

    When df → ∞, converges to standard OU.  df ∈ [3,6] gives
    empirically realistic fat tails for financial/macro variables.
    """
    theta: float = 0.15   # Mean reversion speed
    mu: float = 0.0       # Long-run mean
    scale: float = 0.2    # Scale parameter (analogous to σ)
    df: float = 4.0       # Degrees of freedom (lower = fatter tails)
    dt: float = 1.0
    _x: float = 0.0

    def get_delta(self, current_val: float) -> float:
        if not self.active:
            return 0.0
        return self._x

    def step(self):
        # Student-t innovation (scaled to have unit variance for comparability)
        if self.df > 2:
            # Variance of Student-t is df/(df-2); normalize to unit variance
            t_innovation = np.random.standard_t(self.df) / np.sqrt(self.df / (self.df - 2))
        else:
            t_innovation = np.random.standard_t(max(self.df, 2.01))

        dw = t_innovation * np.sqrt(self.dt)
        dx = self.theta * (self.mu - self._x) * self.dt + self.scale * dw
        self._x += dx


@dataclass
class CorrelatedShockBundle:
    """
    Generates correlated shocks across multiple channels simultaneously.

    Uses a Cholesky decomposition of the correlation matrix to produce
    correlated innovations, then routes them through individual shock
    processes.  This is essential for realistic multi-channel scenarios
    where, e.g., supply and FX shocks co-move.

    The bundle is NOT a Shock subclass — it wraps multiple Shock objects
    and coordinates their joint step().
    """
    shocks: List[Shock] = field(default_factory=list)
    # Correlation matrix (n_shocks × n_shocks).  Default: identity (independent)
    correlation_matrix: Optional[np.ndarray] = None
    _cholesky: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.shocks)
        if self.correlation_matrix is None and n > 0:
            self.correlation_matrix = np.eye(n)
        if n > 0:
            self._compute_cholesky()

    def _compute_cholesky(self):
        """Compute Cholesky factor for correlated sampling."""
        try:
            self._cholesky = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Fallback: nearest positive-definite via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(self.correlation_matrix)
            eigvals = np.maximum(eigvals, 1e-6)
            fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self._cholesky = np.linalg.cholesky(fixed)

    def step(self) -> Dict[str, float]:
        """
        Advance all shocks with correlated innovations.

        Returns:
            Dict mapping shock.name -> delta for this step.
        """
        n = len(self.shocks)
        if n == 0:
            return {}

        # Generate correlated standard normals
        z = np.random.normal(0, 1, size=n)
        correlated_z = self._cholesky @ z

        deltas = {}
        for i, shock in enumerate(self.shocks):
            if not shock.active:
                deltas[shock.name] = 0.0
                continue

            # Inject correlated innovation into the shock's step
            # We override the randomness by temporarily seeding the per-shock dynamics
            # For OU-family shocks, we directly compute the step with our correlated noise
            if isinstance(shock, OUProcessShock):
                dw = correlated_z[i] * np.sqrt(shock.dt)
                dx = shock.theta * (shock.mu - shock._x) * shock.dt + shock.sigma * dw
                shock._x += dx
            elif isinstance(shock, BrownianShock):
                dw = correlated_z[i] * np.sqrt(shock.dt)
                dx = shock.drift * shock.dt + shock.volatility * dw
                shock._accumulated += dx
            elif isinstance(shock, StudentTShock):
                # Use correlated normal, then transform to Student-t scale
                dw = correlated_z[i] * np.sqrt(shock.dt)
                dx = shock.theta * (shock.mu - shock._x) * shock.dt + shock.scale * dw
                shock._x += dx
            elif isinstance(shock, MarkovSwitchingShock):
                # Regime transition uses its own randomness, OU part uses correlated
                probs = shock.transition_matrix[shock._regime]
                shock._regime = int(np.random.choice(shock.n_regimes, p=probs))
                theta, mu, sigma = shock.regime_params[shock._regime]
                dw = correlated_z[i] * np.sqrt(shock.dt)
                dx = theta * (mu - shock._x) * shock.dt + sigma * dw
                shock._x += dx
            elif isinstance(shock, JumpDiffusionShock):
                # Continuous part uses correlated noise, jumps are independent
                dw = correlated_z[i] * np.sqrt(shock.dt)
                dx = shock.drift * shock.dt + shock.volatility * dw
                n_jumps = np.random.poisson(shock.jump_intensity * shock.dt)
                if n_jumps > 0:
                    jumps = np.random.normal(shock.jump_mean, shock.jump_std, size=n_jumps)
                    dx += jumps.sum()
                    shock._jump_count += n_jumps
                shock._x += dx
            else:
                # Generic fallback: let the shock handle its own step
                shock.step()

            deltas[shock.name] = shock.get_delta(0.0)

        return deltas

class ShockManager:
    """
    Manages active stochastic processes, including correlated bundles.
    """
    def __init__(self):
        self.active_shocks: List[Shock] = []
        self.bundles: List[CorrelatedShockBundle] = []
        
    def add_shock(self, shock: Shock):
        self.active_shocks.append(shock)

    def add_bundle(self, bundle: CorrelatedShockBundle):
        """Add a correlated shock bundle.  Bundle shocks are also added to active_shocks."""
        self.bundles.append(bundle)
        for shock in bundle.shocks:
            if shock not in self.active_shocks:
                self.active_shocks.append(shock)
        
    def apply_shocks(self, env_state) -> List[str]:
        """
        Applies shocks to environment state (Additively).
        Correlated bundles are stepped first (they handle their own step()),
        then remaining independent shocks are stepped normally.
        """
        applied = []
        node_map = {name: i for i, name in enumerate(env_state.node_ids)}

        # Track which shocks are handled by bundles
        bundled_shocks = set()
        for bundle in self.bundles:
            deltas = bundle.step()
            for shock in bundle.shocks:
                bundled_shocks.add(id(shock))
            for name, delta in deltas.items():
                # Find matching shock to get target_metric
                for s in bundle.shocks:
                    if s.name == name and s.target_metric in node_map:
                        idx = node_map[s.target_metric]
                        if abs(delta) > 1e-6:
                            env_state.values[idx] += delta
        
        for shock in self.active_shocks:
            if not shock.active or id(shock) in bundled_shocks:
                continue
            
            if shock.target_metric in node_map:
                idx = node_map[shock.target_metric]
                val = env_state.values[idx]
                
                delta = shock.get_delta(val)
                
                if abs(delta) > 1e-6:
                    env_state.values[idx] += delta
            
            shock.step()
            
        return applied

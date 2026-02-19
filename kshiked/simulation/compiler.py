"""
Shock Compiler for KShield Simulation.

Transforms stochastic shock definitions (Impulse, OU Process, Brownian,
Markov-Switching, Jump-Diffusion, Student-t) into deterministic vectors
compatible with Scarcity's SFC Engine.

Phase 5: Extended to support regime-switching, fat-tailed, jump processes,
and correlated multi-channel shock generation.  Produces per-channel vectors
plus joint regime/jump metadata for downstream analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Import Shock Definitions
from kshiked.core.shocks import (
    Shock, ImpulseShock, OUProcessShock, BrownianShock,
    MarkovSwitchingShock, JumpDiffusionShock, StudentTShock,
    CorrelatedShockBundle,
)
from scarcity.simulation.sfc import SFCConfig

logger = logging.getLogger("kshield.compiler")

class ShockCompiler:
    """
    Compiles high-level Shock objects into time-series vectors.
    Extended to support all Phase 5 shock types and correlated bundles.
    """
    
    def __init__(self, steps: int = 50, dt: float = 1.0, seed: int = 42):
        self.steps = steps
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def compile(self, shocks: List[Shock]) -> Dict[str, np.ndarray]:
        """
        Generate vectors for all active shocks.
        
        Returns:
            Dict[str, np.ndarray]: Map of 'shock_channel' -> 'time_series_vector'
            compatible with SFCConfig.shock_vectors.
        """
        # Initialize zero vectors for supported channels
        channels = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
        vectors = {k: np.zeros(self.steps, dtype=np.float32) for k in channels}
        
        for shock in shocks:
            if not shock.active:
                continue
                
            channel_key = self._map_channel(shock.target_metric)
                
            # Generate Time Series
            series = self._generate_series(shock)
            
            # Accumulate (additive)
            vectors[channel_key] += series
            
        return vectors

    def compile_with_metadata(
        self, shocks: List[Shock]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Extended compile that also returns regime and jump metadata.

        Returns:
            Tuple of (vectors, metadata) where metadata contains:
            - 'regime_paths': Dict[shock_name → np.ndarray of regime indices]
            - 'jump_times': Dict[shock_name → List[int] of jump time indices]
            - 'confidence_bands': Dict[channel → (lower_2.5%, upper_97.5%)]
        """
        channels = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
        vectors = {k: np.zeros(self.steps, dtype=np.float32) for k in channels}
        metadata: Dict[str, Any] = {
            "regime_paths": {},
            "jump_times": {},
            "confidence_bands": {},
        }

        for shock in shocks:
            if not shock.active:
                continue

            channel_key = self._map_channel(shock.target_metric)

            if isinstance(shock, MarkovSwitchingShock):
                series, regimes = self._generate_markov_series(shock)
                metadata["regime_paths"][shock.name] = regimes
            elif isinstance(shock, JumpDiffusionShock):
                series, jump_ts = self._generate_jump_series(shock)
                metadata["jump_times"][shock.name] = jump_ts
            else:
                series = self._generate_series(shock)

            vectors[channel_key] += series

        # Compute confidence bands via Monte Carlo (100 replications)
        for ch in channels:
            if np.any(vectors[ch] != 0):
                metadata["confidence_bands"][ch] = self._confidence_bands(
                    shocks, ch, n_replications=100
                )

        return vectors, metadata

    def compile_correlated(
        self, bundle: CorrelatedShockBundle
    ) -> Dict[str, np.ndarray]:
        """
        Compile a CorrelatedShockBundle, preserving cross-channel correlation.

        Returns:
            Dict[str, np.ndarray] of channel vectors with properly correlated shocks.
        """
        channels = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
        vectors = {k: np.zeros(self.steps, dtype=np.float32) for k in channels}

        n = len(bundle.shocks)
        if n == 0:
            return vectors

        for t in range(self.steps):
            deltas = bundle.step()
            for shock in bundle.shocks:
                if not shock.active:
                    continue
                ch = self._map_channel(shock.target_metric)
                vectors[ch][t] += deltas.get(shock.name, 0.0)

        return vectors

    def _map_channel(self, target_metric: str) -> str:
        """Map a target metric name to one of the 4 canonical SFC shock channels."""
        target = target_metric.lower()
        if "demand" in target or "consumption" in target or "investment" in target:
            return "demand_shock"
        elif "supply" in target or "productivity" in target:
            return "supply_shock"
        elif "fiscal" in target or "spending" in target or "tax" in target:
            return "fiscal_shock"
        elif "fx" in target or "rate" in target or "currency" in target:
            return "fx_shock"
        else:
            logger.warning(
                f"Shock target '{target_metric}' not mapped to canonical channel. "
                "Defaulting to demand."
            )
            return "demand_shock"

    def _generate_series(self, shock: Shock) -> np.ndarray:
        """Type-specific generation logic."""
        series = np.zeros(self.steps)
        
        if isinstance(shock, ImpulseShock):
            val = shock.magnitude
            for t in range(self.steps):
                series[t] = val
                val *= shock.decay
                if abs(val) < 1e-6: val = 0.0
                
        elif isinstance(shock, OUProcessShock):
            x = 0.0
            for t in range(self.steps):
                series[t] = x
                dw = self.rng.normal(0, np.sqrt(shock.dt))
                dx = shock.theta * (shock.mu - x) * shock.dt + shock.sigma * dw
                x += dx
                
        elif isinstance(shock, BrownianShock):
            x = 0.0
            for t in range(self.steps):
                series[t] = x
                dw = self.rng.normal(0, np.sqrt(shock.dt))
                dx = shock.drift * shock.dt + shock.volatility * dw
                x += dx

        elif isinstance(shock, MarkovSwitchingShock):
            series, _ = self._generate_markov_series(shock)

        elif isinstance(shock, JumpDiffusionShock):
            series, _ = self._generate_jump_series(shock)

        elif isinstance(shock, StudentTShock):
            x = 0.0
            for t in range(self.steps):
                series[t] = x
                if shock.df > 2:
                    innov = self.rng.standard_t(shock.df) / np.sqrt(shock.df / (shock.df - 2))
                else:
                    innov = self.rng.standard_t(max(shock.df, 2.01))
                dw = innov * np.sqrt(shock.dt)
                dx = shock.theta * (shock.mu - x) * shock.dt + shock.scale * dw
                x += dx
                
        return series

    def _generate_markov_series(
        self, shock: MarkovSwitchingShock
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate series from Markov-switching shock, returning (values, regimes)."""
        series = np.zeros(self.steps)
        regimes = np.zeros(self.steps, dtype=np.int32)
        x = 0.0
        regime = shock._regime

        for t in range(self.steps):
            # Transition
            probs = shock.transition_matrix[regime]
            regime = int(self.rng.choice(shock.n_regimes, p=probs))
            regimes[t] = regime

            # OU under current regime
            theta, mu, sigma = shock.regime_params[regime]
            dw = self.rng.normal(0, np.sqrt(shock.dt))
            dx = theta * (mu - x) * shock.dt + sigma * dw
            x += dx
            series[t] = x

        return series, regimes

    def _generate_jump_series(
        self, shock: JumpDiffusionShock
    ) -> Tuple[np.ndarray, List[int]]:
        """Generate series from jump-diffusion, returning (values, jump_times)."""
        series = np.zeros(self.steps)
        jump_times = []
        x = 0.0

        for t in range(self.steps):
            dw = self.rng.normal(0, np.sqrt(shock.dt))
            dx = shock.drift * shock.dt + shock.volatility * dw

            n_jumps = self.rng.poisson(shock.jump_intensity * shock.dt)
            if n_jumps > 0:
                jumps = self.rng.normal(shock.jump_mean, shock.jump_std, size=n_jumps)
                dx += jumps.sum()
                jump_times.append(t)

            x += dx
            series[t] = x

        return series, jump_times

    def _confidence_bands(
        self,
        shocks: List[Shock],
        channel: str,
        n_replications: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo confidence bands (2.5%, 97.5%) for a channel."""
        mc_runs = np.zeros((n_replications, self.steps), dtype=np.float32)

        relevant = [
            s for s in shocks
            if s.active and self._map_channel(s.target_metric) == channel
        ]
        if not relevant:
            return np.zeros(self.steps), np.zeros(self.steps)

        for r in range(n_replications):
            rng_r = np.random.default_rng(self.seed + r + 1)
            for shock in relevant:
                series = np.zeros(self.steps)
                x = 0.0
                if isinstance(shock, OUProcessShock):
                    for t in range(self.steps):
                        dw = rng_r.normal(0, np.sqrt(shock.dt))
                        dx = shock.theta * (shock.mu - x) * shock.dt + shock.sigma * dw
                        x += dx
                        series[t] = x
                elif isinstance(shock, BrownianShock):
                    for t in range(self.steps):
                        dw = rng_r.normal(0, np.sqrt(shock.dt))
                        dx = shock.drift * shock.dt + shock.volatility * dw
                        x += dx
                        series[t] = x
                elif isinstance(shock, StudentTShock):
                    for t in range(self.steps):
                        if shock.df > 2:
                            innov = rng_r.standard_t(shock.df) / np.sqrt(shock.df / (shock.df - 2))
                        else:
                            innov = rng_r.standard_t(max(shock.df, 2.01))
                        dw = innov * np.sqrt(shock.dt)
                        dx = shock.theta * (shock.mu - x) * shock.dt + shock.scale * dw
                        x += dx
                        series[t] = x
                elif isinstance(shock, MarkovSwitchingShock):
                    regime = 0
                    for t in range(self.steps):
                        probs = shock.transition_matrix[regime]
                        regime = int(rng_r.choice(shock.n_regimes, p=probs))
                        theta, mu, sigma = shock.regime_params[regime]
                        dw = rng_r.normal(0, np.sqrt(shock.dt))
                        dx = theta * (mu - x) * shock.dt + sigma * dw
                        x += dx
                        series[t] = x
                elif isinstance(shock, JumpDiffusionShock):
                    for t in range(self.steps):
                        dw = rng_r.normal(0, np.sqrt(shock.dt))
                        dx = shock.drift * shock.dt + shock.volatility * dw
                        n_jumps = rng_r.poisson(shock.jump_intensity * shock.dt)
                        if n_jumps > 0:
                            jumps = rng_r.normal(shock.jump_mean, shock.jump_std, size=n_jumps)
                            dx += jumps.sum()
                        x += dx
                        series[t] = x
                elif isinstance(shock, ImpulseShock):
                    val = shock.magnitude
                    for t in range(self.steps):
                        series[t] = val
                        val *= shock.decay
                mc_runs[r] += series

        lower = np.percentile(mc_runs, 2.5, axis=0)
        upper = np.percentile(mc_runs, 97.5, axis=0)
        return lower, upper

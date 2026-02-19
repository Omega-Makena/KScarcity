"""
Counterfactual / what-if scenario manager.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from .environment import SimulationEnvironment, EnvironmentState
from .dynamics import DynamicsEngine, DynamicsConfig


@dataclass
class WhatIfConfig:
    """
    Configuration for scenario execution.

    Attributes:
        horizon_steps: Number of forward-simulation steps to run.
        bootstrap_runs: Number of Monte Carlo runs for estimating confidence intervals.
        noise_sigma: Standard deviation of noise injected during bootstrap runs.
    """
    horizon_steps: int = 12
    bootstrap_runs: int = 8
    noise_sigma: float = 0.02


class WhatIfManager:
    """
    Manages the execution of counterfactual scenarios.
    
    Allows for "forking" the simulation state to test hypotheses (e.g., "What if
    inflation increases by 2%?"). Simulates parallel trajectories using the
    current causal model (DynamicsEngine) and computes the delta against the baseline.
    """

    def __init__(
        self,
        environment: SimulationEnvironment,
        dynamics_config: DynamicsConfig,
        config: WhatIfConfig,
    ):
        """
        Args:
            environment: The source environment to fork state from.
            dynamics_config: Configuration for the physics/causal engine used in projection.
            config: Scenario configuration (horizon, etc.).
        """
        self.env = environment
        self.base_dynamics = DynamicsEngine(environment, dynamics_config)
        self.config = config
        self._rng = np.random.default_rng(environment.config.seed + 13)

    def run_scenario(
        self,
        scenario_id: str,
        node_shocks: Optional[Dict[str, float]] = None,
        edge_shocks: Optional[Dict[Tuple[str, str], float]] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Executes a defined scenario.

        Args:
            scenario_id: Identifier for tracking.
            node_shocks: Map of {node_id: delta_value} to apply at t=0.
            edge_shocks: Map of {(src, dst): delta_weight} to apply to the adjacency matrix.
            horizon: Override default horizon.

        Returns:
            Dictionary containing:
            - Trajectory deltas (time series of impact).
            - Confidence intervals (from bootstrapping).
            - Top impacted nodes list.
        """
        horizon = horizon or self.config.horizon_steps
        baseline_state = self.env.clone_state()
        perturbed_state = self._apply_shocks(baseline_state, node_shocks, edge_shocks)

        baseline_traj = self._simulate_trajectory(baseline_state, horizon)
        perturbed_traj = self._simulate_trajectory(perturbed_state, horizon)

        deltas = [
            {
                node_id: perturbed_traj[t][node_id] - baseline_traj[t][node_id]
                for node_id in baseline_traj[t]
            }
            for t in range(horizon + 1)
        ]

        ci = self._bootstrap_ci(baseline_state, node_shocks, edge_shocks, horizon, baseline_traj)

        top_impacts = self._top_impacts(deltas[-1])

        return {
            "scenario_id": scenario_id,
            "horizon": horizon,
            "delta": deltas,
            "confidence_interval": ci,
            "top_impacts": top_impacts,
        }

    def _simulate_trajectory(self, start_state: EnvironmentState, horizon: int) -> List[Dict[str, float]]:
        """
        Projects a system state forward in time using the dynamics engine.
        
        Isolates side effects by restoring the environment state after execution.
        """
        original_state = self.env.clone_state()
        self.env.set_state(copy.deepcopy(start_state))
        dynamics = DynamicsEngine(self.env, self.base_dynamics.config)
        trajectory = [dict(zip(start_state.node_ids, start_state.values.tolist()))]
        for _ in range(horizon):
            trajectory.append(dynamics.step())
        self.env.set_state(original_state)
        return trajectory

    def _apply_shocks(
        self,
        state: EnvironmentState,
        node_shocks: Optional[Dict[str, float]],
        edge_shocks: Optional[Dict[Tuple[str, str], float]],
    ) -> EnvironmentState:
        """
        Applies instantaneous shocks to the environment state.
        """
        shocked = EnvironmentState(
            values=state.values.copy(),
            node_ids=state.node_ids.copy(),
            adjacency=state.adjacency.copy(),
            stability=state.stability.copy(),
            timestamp=state.timestamp,
        )
        if node_shocks:
            for node_id, delta in node_shocks.items():
                if node_id in shocked.node_ids:
                    idx = shocked.node_ids.index(node_id)
                    shocked.values[idx] += delta
        if edge_shocks:
            for (src, dst), delta in edge_shocks.items():
                if src in shocked.node_ids and dst in shocked.node_ids:
                    i = shocked.node_ids.index(src)
                    j = shocked.node_ids.index(dst)
                    shocked.adjacency[i, j] += delta
        return shocked

    def _bootstrap_ci(
        self,
        start_state: EnvironmentState,
        node_shocks: Optional[Dict[str, float]],
        edge_shocks: Optional[Dict[Tuple[str, str], float]],
        horizon: int,
        baseline_traj: List[Dict[str, float]],
    ) -> Tuple[float, float]:
        """
        Estimates confidence intervals via Monte Carlo bootstrapping with noise injection.
        """
        if self.config.bootstrap_runs <= 1:
            return (0.0, 0.0)

        impacts = []
        for _ in range(self.config.bootstrap_runs):
            noise_shocks = {
                node_id: self._rng.normal(0.0, self.config.noise_sigma)
                for node_id in (node_shocks or {})
            }
            perturbed = self._apply_shocks(start_state, noise_shocks, edge_shocks)
            traj = self._simulate_trajectory(perturbed, horizon)
            impacts.append(
                sum(
                    abs(traj[-1][node] - baseline_traj[-1][node])
                    for node in baseline_traj[-1]
                )
            )
        if not impacts:
            return (0.0, 0.0)
        impacts = np.asarray(impacts, dtype=np.float32)
        mean = float(np.mean(impacts))
        std = float(np.std(impacts))
        return (mean - std, mean + std)

    def _top_impacts(self, final_delta: Dict[str, float], k: int = 5) -> List[Dict[str, float]]:
        """Identifies the K nodes with the largest absolute change."""
        sorted_nodes = sorted(final_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [{"id": node, "delta": float(delta)} for node, delta in sorted_nodes[:k]]

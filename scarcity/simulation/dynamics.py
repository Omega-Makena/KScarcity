"""
Causal propagation dynamics for the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np  # type: ignore

from .environment import SimulationEnvironment


@dataclass
class DynamicsConfig:
    global_damping: float = 0.9
    delta_t: float = 1.0
    stability_floor: float = 0.05


class DynamicsEngine:
    def __init__(self, environment: SimulationEnvironment, config: DynamicsConfig):
        self.env = environment
        self.config = config

    def step(self) -> Dict[str, float]:
        state = self.env.state()
        old_values = state.values.copy()
        adjacency = state.adjacency
        stability = np.maximum(state.stability, self.config.stability_floor)
        weights = adjacency * stability * self.config.global_damping

        incoming = weights.T @ old_values
        outgoing = weights.sum(axis=0) * old_values
        delta = self.config.delta_t * (incoming - outgoing)

        new_values = old_values + delta.astype(np.float32)
        new_values = self.env.apply_noise(new_values)
        new_values = self.env.enforce_energy_cap(old_values, new_values)

        self.env.update_values(new_values)
        return dict(zip(state.node_ids, new_values.tolist()))


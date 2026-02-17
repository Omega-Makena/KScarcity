"""
Simulation environment state container.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np  # type: ignore

from .agents import AgentRegistry


@dataclass
class EnvironmentConfig:
    damping: float = 0.9
    noise_sigma: float = 0.01
    energy_cap: float = 5.0
    seed: int = 42


@dataclass
class EnvironmentState:
    values: np.ndarray
    node_ids: List[str]
    adjacency: np.ndarray
    stability: np.ndarray
    timestamp: int = 0


class SimulationEnvironment:
    def __init__(self, registry: AgentRegistry, config: EnvironmentConfig):
        self.registry = registry
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._state = self._build_initial_state()

    def _build_initial_state(self) -> EnvironmentState:
        adjacency, stability, node_ids = self.registry.adjacency_matrix()
        if not node_ids:
             # Handle empty registry gracefuly
             return EnvironmentState(
                 values=np.array([]),
                 node_ids=[],
                 adjacency=np.array([]),
                 stability=np.array([])
             )
        values = np.array(
            [self.registry.nodes()[node_id].value for node_id in node_ids],
            dtype=np.float32,
        )
        return EnvironmentState(
            values=values,
            node_ids=node_ids,
            adjacency=adjacency,
            stability=stability,
        )

    def state(self) -> EnvironmentState:
        return self._state

    def clone_state(self) -> EnvironmentState:
        return copy.deepcopy(self._state)

    def set_state(self, state: EnvironmentState) -> None:
        self._state = state

    def update_values(self, new_values: np.ndarray) -> None:
        self._state.values = new_values.astype(np.float32)
        self._state.timestamp += 1

    def apply_noise(self, values: np.ndarray) -> np.ndarray:
        if self.config.noise_sigma <= 0:
            return values
        noise = self.rng.normal(0.0, self.config.noise_sigma, size=values.shape)
        return values + noise.astype(np.float32)

    def enforce_energy_cap(self, old_values: np.ndarray, new_values: np.ndarray) -> np.ndarray:
        delta = new_values - old_values
        norm = np.linalg.norm(delta)
        if norm > self.config.energy_cap > 0:
            delta *= self.config.energy_cap / norm
        return old_values + delta

    def reset(self) -> None:
        self._state = self._build_initial_state()

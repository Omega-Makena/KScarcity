"""
Configuration for the Federated Learning Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FLOrchestratorConfig:
    """Configuration for the event-driven FL orchestrator."""

    # Training
    model_name: str = "logistic"
    learning_rate: float = 0.12
    min_nodes_per_round: int = 2
    max_wait_seconds: float = 300.0
    auto_ingest: bool = True
    lookback_hours: int = 24
    source_path: str = "data/synthetic_kenya_policy/tweets.csv"

    # WebSocket
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    auth_token: Optional[str] = None

    # Aggregation
    aggregation_method: str = "trimmed_mean"
    trim_alpha: float = 0.1

    # Orchestrator behavior
    round_timeout: float = 600.0
    max_rounds: int = 0  # 0 = unlimited
    auto_start: bool = True

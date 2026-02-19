"""
Application configuration loaded from environment variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI service."""

    project_name: str = Field(default="Scarce Demo Backend", description="Human readable name.")
    api_v1_prefix: str = Field(default="/api/v1", description="Prefix for version 1 of the API.")
    api_v2_prefix: str = Field(default="/api/v2", description="Prefix for version 2 of the API (scarcity-backed).")
    allow_origins: list[str] = Field(
        default_factory=lambda: [
            # Next.js dev server
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            # Storybook / Vite dev servers often run on 3001/5173/8080
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        description="List of origins allowed for CORS.",
    )
    simulation_seed: int = Field(default=42, description="Seed used for deterministic simulation.")
    simulation_tick_seconds: float = Field(default=1.0, description="Base tick interval in seconds.")
    
    # Scarcity Core Configuration
    scarcity_enabled: bool = Field(default=True, description="Enable scarcity core components.")
    scarcity_mpie_enabled: bool = Field(default=True, description="Enable MPIE orchestrator.")
    scarcity_drg_enabled: bool = Field(default=True, description="Enable Dynamic Resource Governor.")
    scarcity_federation_enabled: bool = Field(default=False, description="Enable Federation layer (TODO).")
    scarcity_meta_enabled: bool = Field(default=True, description="Enable Meta Learning agent.")
    scarcity_simulation_enabled: bool = Field(default=False, description="Enable Simulation engine (TODO).")

    # Federation (v1) configuration
    scarcity_federation_aggregation_method: str = Field(
        default="trimmed_mean",
        description="Aggregation method (fedavg, weighted, adaptive, median, trimmed_mean, krum, multi_krum, bulyan).",
    )
    scarcity_federation_aggregation_trim_alpha: float = Field(
        default=0.1, description="Trim fraction for trimmed mean / bulyan."
    )
    scarcity_federation_aggregation_multi_krum_m: int = Field(
        default=5, description="Number of selected updates for multi-krum/bulyan."
    )
    scarcity_federation_aggregation_adaptive_metric_is_loss: bool = Field(
        default=True, description="Adaptive aggregation treats metric as loss."
    )
    scarcity_federation_privacy_secure_aggregation: bool = Field(
        default=True, description="Enable secure aggregation masking for exports."
    )
    scarcity_federation_privacy_noise_sigma: float = Field(
        default=0.0, description="Gaussian/Laplace noise sigma (overrides epsilon/delta)."
    )
    scarcity_federation_privacy_epsilon: float = Field(
        default=0.0, description="DP epsilon (used if sigma is not provided)."
    )
    scarcity_federation_privacy_delta: float = Field(
        default=0.0, description="DP delta (used with epsilon for Gaussian noise)."
    )
    scarcity_federation_privacy_sensitivity: float = Field(
        default=1.0, description="DP sensitivity for epsilon/delta or Laplace noise."
    )
    scarcity_federation_privacy_noise_type: str = Field(
        default="gaussian", description="DP noise type: gaussian or laplace."
    )
    scarcity_federation_privacy_seed_length: int = Field(
        default=16, description="Seed length (bytes) for secure masking."
    )
    scarcity_federation_validator_trust_min: float = Field(
        default=0.2, description="Minimum trust score for federated packets."
    )
    scarcity_federation_validator_max_edges: int = Field(
        default=2048, description="Maximum edges allowed in a packet."
    )
    scarcity_federation_validator_max_concepts: int = Field(
        default=256, description="Maximum concepts allowed in a causal packet."
    )
    scarcity_federation_transport_protocol: str = Field(
        default="loopback", description="Transport protocol for federation client."
    )
    scarcity_federation_transport_endpoint: str | None = Field(
        default=None, description="Transport endpoint for federation client."
    )
    scarcity_federation_transport_reconnect_backoff: float = Field(
        default=5.0, description="Reconnect backoff for federation transport."
    )
    scarcity_federation_coordinator_heartbeat_timeout: float = Field(
        default=60.0, description="Peer heartbeat timeout for federation coordinator."
    )
    scarcity_federation_coordinator_fairness_quota_kb_min: int = Field(
        default=512, description="Minimum fairness quota in KB for coordinator."
    )
    scarcity_federation_coordinator_mode: str = Field(
        default="mesh", description="Coordinator mode (mesh, star, etc.)."
    )

    # Federation (v2) configuration
    scarcity_federation_v2_strategy: str = Field(
        default="fedavg", description="Aggregation strategy for federation coordinator v2."
    )
    scarcity_federation_v2_enable_privacy: bool = Field(
        default=False, description="Enable differential privacy for federation coordinator v2."
    )
    
    # Resource Limits
    scarcity_mpie_max_candidates: int = Field(default=200, description="Max candidate paths for MPIE.")
    scarcity_mpie_resamples: int = Field(default=1000, description="Bootstrap resamples for evaluator.")
    scarcity_drg_control_interval: float = Field(default=0.5, description="DRG control loop interval in seconds.")
    scarcity_drg_cpu_threshold: float = Field(default=90.0, description="CPU utilization threshold percentage.")
    scarcity_drg_memory_threshold: float = Field(default=85.0, description="Memory utilization threshold percentage.")

    model_config = SettingsConfigDict(
        env_prefix="SCARCE_",
        extra="ignore",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    # Ensure paths that depend on runtime are resolved lazily.
    _ = Path.cwd()
    return settings


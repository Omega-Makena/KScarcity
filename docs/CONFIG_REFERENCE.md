# Configuration Reference

> All backend settings loaded from environment variables with the `SCARCE_` prefix.  
> Source: `backend/app/core/config.py`  
> Env files: `.env`, `.env.local`

---

## Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_PROJECT_NAME` | `Scarce Demo Backend` | Human-readable project name |
| `SCARCE_API_V1_PREFIX` | `/api/v1` | URL prefix for v1 endpoints (deprecated) |
| `SCARCE_API_V2_PREFIX` | `/api/v2` | URL prefix for v2 endpoints (current) |
| `SCARCE_ALLOW_ORIGINS` | `localhost:3000,3001,5173,8080` | CORS allowed origins |
| `SCARCE_SIMULATION_SEED` | `42` | Deterministic simulation seed |
| `SCARCE_SIMULATION_TICK_SECONDS` | `1.0` | Base tick interval (seconds) |

---

## Scarcity Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_ENABLED` | `True` | Master switch for scarcity core |
| `SCARCE_SCARCITY_MPIE_ENABLED` | `True` | MPIE orchestrator |
| `SCARCE_SCARCITY_DRG_ENABLED` | `True` | Dynamic Resource Governor |
| `SCARCE_SCARCITY_FEDERATION_ENABLED` | `False` | Federation layer (TODO) |
| `SCARCE_SCARCITY_META_ENABLED` | `True` | Meta Learning agent |
| `SCARCE_SCARCITY_SIMULATION_ENABLED` | `False` | Simulation engine (TODO) |

---

## Federation v1 — Aggregation

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_AGGREGATION_METHOD` | `trimmed_mean` | Method: fedavg, weighted, adaptive, median, trimmed_mean, krum, multi_krum, bulyan |
| `SCARCE_SCARCITY_FEDERATION_AGGREGATION_TRIM_ALPHA` | `0.1` | Trim fraction for trimmed_mean / bulyan |
| `SCARCE_SCARCITY_FEDERATION_AGGREGATION_MULTI_KRUM_M` | `5` | Selected updates for multi-krum/bulyan |
| `SCARCE_SCARCITY_FEDERATION_AGGREGATION_ADAPTIVE_METRIC_IS_LOSS` | `True` | Treat metric as loss in adaptive aggregation |

---

## Federation v1 — Privacy

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_SECURE_AGGREGATION` | `True` | Secure aggregation masking |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_NOISE_SIGMA` | `0.0` | Gaussian/Laplace noise sigma |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_EPSILON` | `0.0` | DP epsilon |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_DELTA` | `0.0` | DP delta |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_SENSITIVITY` | `1.0` | DP sensitivity |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_NOISE_TYPE` | `gaussian` | Noise type: gaussian or laplace |
| `SCARCE_SCARCITY_FEDERATION_PRIVACY_SEED_LENGTH` | `16` | Seed length (bytes) for secure masking |

---

## Federation v1 — Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_VALIDATOR_TRUST_MIN` | `0.2` | Minimum trust score for federated packets |
| `SCARCE_SCARCITY_FEDERATION_VALIDATOR_MAX_EDGES` | `2048` | Max edges in a packet |
| `SCARCE_SCARCITY_FEDERATION_VALIDATOR_MAX_CONCEPTS` | `256` | Max concepts in a causal packet |

---

## Federation v1 — Transport

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_TRANSPORT_PROTOCOL` | `loopback` | Transport protocol |
| `SCARCE_SCARCITY_FEDERATION_TRANSPORT_ENDPOINT` | `None` | Transport endpoint URL |
| `SCARCE_SCARCITY_FEDERATION_TRANSPORT_RECONNECT_BACKOFF` | `5.0` | Reconnect backoff (seconds) |

---

## Federation v1 — Coordinator

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_COORDINATOR_HEARTBEAT_TIMEOUT` | `60.0` | Peer heartbeat timeout (seconds) |
| `SCARCE_SCARCITY_FEDERATION_COORDINATOR_FAIRNESS_QUOTA_KB_MIN` | `512` | Minimum fairness quota (KB) |
| `SCARCE_SCARCITY_FEDERATION_COORDINATOR_MODE` | `mesh` | Coordinator mode (mesh, star, etc.) |

---

## Federation v2

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_FEDERATION_V2_STRATEGY` | `fedavg` | Aggregation strategy for v2 coordinator |
| `SCARCE_SCARCITY_FEDERATION_V2_ENABLE_PRIVACY` | `False` | Enable DP for v2 federation |

---

## Resource Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `SCARCE_SCARCITY_MPIE_MAX_CANDIDATES` | `200` | Max candidate paths for MPIE |
| `SCARCE_SCARCITY_MPIE_RESAMPLES` | `1000` | Bootstrap resamples for evaluator |
| `SCARCE_SCARCITY_DRG_CONTROL_INTERVAL` | `0.5` | DRG control loop interval (seconds) |
| `SCARCE_SCARCITY_DRG_CPU_THRESHOLD` | `90.0` | CPU utilization threshold (%) |
| `SCARCE_SCARCITY_DRG_MEMORY_THRESHOLD` | `85.0` | Memory utilization threshold (%) |

---

## Usage

Settings are loaded automatically from environment variables or `.env` files:

```python
from app.core.config import get_settings

settings = get_settings()  # cached singleton
print(settings.api_v2_prefix)  # "/api/v2"
```

Override any setting via environment:

```bash
# Windows
set SCARCE_SCARCITY_FEDERATION_ENABLED=True

# Linux/Mac
export SCARCE_SCARCITY_FEDERATION_ENABLED=True
```

---

*Source: `backend/app/core/config.py` · Last updated: 2026-02-11*

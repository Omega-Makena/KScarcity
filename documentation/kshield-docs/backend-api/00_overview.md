# KShield Backend API — Overview

The **backend/app/api** module provides the FastAPI REST endpoints for KShield.

---

## API Versions

### v1 — Legacy (Deprecated)

- Uses mock data
- Domain management, data upload, risk, MPIE, onboarding
- **Status**: Deprecated — migrate to v2

### v2 — Current (Scarcity-Backed)

- Real-time data from Runtime Bus, MPIE, DRG, Meta Learning
- Domain management with multi-domain support
- Federation (v1 stub + v2 multi-domain)
- Simulation control (stub)

---

## Route Structure

```
/api/v1/ (deprecated — mock data)
├── status/           # System status
├── metrics/          # Metrics
├── domains/          # Domain CRUD
├── risk/             # Risk analysis
├── controls/         # Control actions
├── ws/               # WebSocket streams
├── data/             # Data upload
├── onboarding/       # Domain onboarding wizard
├── mpie/             # MPIE inference
└── datasets/         # Dataset management

/api/v2/ (current)
├── /health           # Health check
├── metrics/          # Metrics
├── domains/          # Domain CRUD + data viz
├── demo/             # Demo mode
├── runtime/          # Runtime Bus status
├── mpie/             # MPIE orchestration
├── drg/              # Dynamic Resource Governor
├── federation/       # Federation (stub)
├── federation-v2/    # Multi-domain federation
├── meta/             # Meta Learning
└── simulation/       # Simulation (stub)
```

---

## Key Endpoints

### Domains

```http
POST   /api/v1/domains           # Create domain
GET    /api/v1/domains           # List domains
GET    /api/v1/domains/{id}      # Get domain
DELETE /api/v1/domains/{id}      # Delete domain
POST   /api/v1/domains/{id}/pause   # Pause
POST   /api/v1/domains/{id}/resume  # Resume
```

### Data Ingestion

```http
POST   /api/v1/data/upload/{domain_id}  # Upload CSV
POST   /api/v1/data/stream/{domain_id}  # Stream data
```

### Federation

```http
POST   /api/v1/federation/round     # Trigger round
GET    /api/v1/federation/status    # Status
GET    /api/v1/federation/history   # Round history
```

### Simulation

```http
POST   /api/v1/simulation/start     # Start sim
POST   /api/v1/simulation/stop      # Stop sim
GET    /api/v1/simulation/state     # Get state
POST   /api/v1/simulation/shock     # Apply shock
```

---

## Example Usage

### Create Domain

```bash
curl -X POST http://localhost:8000/api/v1/domains \
  -H "Content-Type: application/json" \
  -d '{"name": "Healthcare", "distribution_type": "normal"}'
```

### Upload Data

```bash
curl -X POST http://localhost:8000/api/v1/data/upload/1 \
  -F "file=@data.csv"
```

### Trigger Federation

```bash
curl -X POST http://localhost:8000/api/v1/federation/round
```

---

## File Guide

| File | Purpose |
|------|---------|
| `routes.py` | Router registration |
| `v1/domains.py` | Domain endpoints (deprecated) |
| `v1/data.py` | Data ingestion (deprecated) |
| `v1/status.py` | System status |
| `v1/metrics.py` | Metrics |
| `v1/mpie.py` | MPIE inference |
| `v1/onboarding.py` | Onboarding wizard |
| `v1/risk.py` | Risk analysis |
| `v1/datasets.py` | Dataset management |
| `v1/controls.py` | Control actions |
| `v1/streams.py` | WebSocket streams |
| `v2/health.py` | Health check |
| `v2/runtime.py` | Runtime Bus |
| `v2/mpie.py` | MPIE orchestration |
| `v2/drg.py` | Dynamic Resource Governor |
| `v2/domains.py` | Domain CRUD (v2) |
| `v2/domain_data.py` | Domain data visualization |
| `v2/federation.py` | Federation (stub) |
| `v2/federation_v2.py` | Multi-domain federation |
| `v2/meta.py` | Meta Learning |
| `v2/simulation.py` | Simulation (stub) |
| `v2/metrics.py` | Metrics (v2) |
| `v2/demo.py` | Demo mode |

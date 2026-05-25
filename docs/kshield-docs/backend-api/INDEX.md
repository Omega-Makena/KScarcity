# Backend API — Documentation Index

Complete documentation for the `backend/app/api` module — REST endpoints.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — API routes |

---

## Key Endpoints

### Domains
- `POST /api/v1/domains` — Create
- `GET /api/v1/domains` — List
- `DELETE /api/v1/domains/{id}` — Delete

### Data
- `POST /api/v1/data/upload/{domain_id}` — Upload CSV

### Federation
- `POST /api/v1/federation/round` — Trigger round

### Simulation
- `POST /api/v1/simulation/start` — Start
- `POST /api/v1/simulation/shock` — Apply shock

---

## Versions

- **v1**: Core federation features
- **v2**: Enhanced analytics

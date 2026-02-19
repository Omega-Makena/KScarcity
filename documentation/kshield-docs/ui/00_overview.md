# Dashboard (UI) — Overview

> `kshiked.ui` — SENTINEL Command Center (Streamlit, routed single-port app)

---

## Purpose

The dashboard is a **single Streamlit app on one port** that routes between SENTINEL views and the K-SHIELD module without restarting the app.

---

## Current Navigation Model

Top-level views are routed through `kshiked/ui/sentinel/router.py` and synced to URL query params (`?view=...`).

| View Key | UI Label | Purpose |
|---|---|---|
| `HOME` | Home | Landing cards and entry navigation |
| `LIVE_MAP` | Live Threat Map | Geographic threat overlays |
| `EXECUTIVE` | Executive Overview | Decision summary and KPIs |
| `SIGNALS` | Signal Intelligence | Social signal monitoring |
| `CAUSAL` | Causal Analysis | Causal analysis workspace |
| `KSHIELD` | K-SHIELD | K-SHIELD card module |
| `SIMULATION` | Simulation (Legacy) | Legacy WhatIf workbench |
| `ESCALATION` | Escalation Pathways | Escalation and response pathways |
| `FEDERATION` | Federation / Federated Databases | Node registry, sync controls, metrics, audit log |
| `OPERATIONS` | Operations | Operational alerts and summaries |
| `SYSTEM_GUIDE` | System Guide | Embedded docs/help |
| `DOCS` | Document Intelligence | News + dossier intelligence |
| `POLICY_CHAT` | Policy Intelligence | Policy chatbot and evidence traces |

---

## K-SHIELD Card Module

K-SHIELD is rendered from `kshiked/ui/kshield/page.py` and keeps its own internal view state.

| Card | Main implementation |
|---|---|
| Causal Relationships | `kshiked/ui/kshield/causal.py` |
| Policy Terrain | `kshiked/ui/kshield/terrain.py` |
| Simulations | `kshiked/ui/kshield/simulation.py` |
| Policy Impact (existing card) | `kshiked/ui/kshield/impact/components/layout.py` + `kshiked/ui/kshield/impact/components/live_policy.py` |

The Policy Impact card now includes live synthetic criticality overlays (baseline vs counterfactual), filters, and data-freshness indicators.

---

## Federation + Route Visibility

The federation page (`kshiked/ui/sentinel/federation.py`) displays:

- stable route path (`/?view=FEDERATION`)
- full URL (host + port + path)
- node registration controls
- sync triggers
- sync metrics and exchange audit log

---

## Running the Dashboard

```bash
streamlit run kshiked/ui/sentinel_dashboard.py
```

Default URL: `http://localhost:8501`

Deep-link examples:

- `http://localhost:8501/?view=FEDERATION`
- `http://localhost:8501/?view=KSHIELD`
- `http://localhost:8501/?view=POLICY_CHAT`

---

*Source: `kshiked/ui/` · Last updated: 2026-02-19*

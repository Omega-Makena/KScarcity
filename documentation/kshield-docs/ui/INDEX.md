# Dashboard (UI) — Index

> `kshiked.ui` — SENTINEL Command Center Dashboard v3.0

## Architecture Summary

The dashboard is a **single-port Streamlit app** with 12 sidebar views and a modular K-SHIELD sub-module that hosts four analysis cards, each with its own workspace.

### Top-Level Navigation (12 views)

| View | Key | Description |
|------|-----|-------------|
| Home | `HOME` | 4 K-module cards — K-SHIELD, K-PULSE, K-COLLAB, K-EDUCATION |
| Live Threat Map | `LIVE_MAP` | Real-time geographic threat overlay |
| Executive Overview | `EXECUTIVE` | Traffic-light summary + gauges |
| Signal Intelligence | `SIGNALS` | 15 signals, heatmap, trend charts |
| Causal Analysis | `CAUSAL` | Legacy causal discovery tab |
| **K-SHIELD** | `KSHIELD` | Intelligence module — see below |
| Simulation (Legacy) | `SIMULATION` | WhatIf Workbench |
| Escalation Pathways | `ESCALATION` | Decision intelligence + pathways |
| Federation | `FEDERATION` | Multi-agency status + gossip |
| Operations | `OPERATIONS` | County drilldown, alert table |
| System Guide | `SYSTEM_GUIDE` | Built-in interactive docs |
| Document Intelligence | `DOCS` | PDF / document analysis |

### K-SHIELD Sub-Module (4 cards)

| Card | Module | Description |
|------|--------|-------------|
| Causal Relationships | `kshield/causal.py` | DoWhy causal inference workspace |
| Policy Terrain | `kshield/terrain.py` | Multi-dimensional policy landscape |
| **Simulations** | `kshield/simulation.py` | SFC simulation — **11 analysis tabs** (all dynamic, 3D-capable) |
| Policy Impact | `kshield/impact.py` | Impact assessment per county |

### Simulation Tabs (11)

| # | Tab | 3D |
|---|-----|----|
| 1 | Scenario Runner | — |
| 2 | Sensitivity Matrix | — |
| 3 | 3D State Cube | Yes |
| 4 | Compare Runs | — |
| 5 | Phase Explorer | Yes |
| 6 | Impulse Response | Yes |
| 7 | Flow Dynamics | Yes |
| 8 | Monte Carlo | Yes |
| 9 | Stress Matrix | Yes |
| 10 | Parameter Surface | Yes |
| 11 | Diagnostics | — |

## Files

| File | Description |
|------|-------------|
| [00_overview.md](00_overview.md) | Legacy overview (pre-K-SHIELD architecture) |

## Quick Links

- Source: `kshiked/ui/`
- Entrypoint: `streamlit run kshiked/ui/sentinel_dashboard.py`
- Theme: `kshiked/ui/theme.py`
- Data: `kshiked/ui/data_connector.py`
- Routing: See [DASHBOARD_ROUTING.md](../../DASHBOARD_ROUTING.md) for full navigation flow

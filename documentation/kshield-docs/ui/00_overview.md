# Dashboard (UI) — Overview

> `kshiked.ui` — SENTINEL Command Center Dashboard v2.0 (Streamlit)

---

## Purpose

The dashboard is a **9-tab Streamlit application** that provides real-time visualization of SENTINEL intelligence. It serves as the primary human interface for analysts and decision-makers.

---

## Tab Architecture

| # | Tab | Description | Key Visuals |
|---|-----|-------------|-------------|
| 1 | **Live Threat Map** | Kaspersky-inspired real-time view | 3D Globe (Globe.gl), threat counter, top counties |
| 2 | **Executive Overview** | Traffic-light status summary | Traffic light, escalation gauge, top threats, unknown-unknowns |
| 3 | **Signal Intelligence** | 15 SIGINT signal analysis | Signal gauges, cascade Sankey, co-occurrence heatmap, silence detector |
| 4 | **Causal Network** | Discovered relationships | 3D WebGL causal graph (full-economy base map), relationship table, Granger tests |
| 5 | **Simulation** | Scenario analysis platform | Scenario builder, 4D state cube, policy sensitivity, economic terrain |
| 6 | **Escalation** | Threat escalation pathways | Escalation tree, decision countdown |
| 7 | **Governance** | Economic governance dashboard | Policy controls, SFC model state |
| 8 | **Regional Map** | Kenya county-level threat map | Choropleth, county drill-down |
| 9 | **Documents** | Document intelligence | Uploaded document analysis |

---

## File Guide

| File | Size | Purpose |
|------|------|---------|
| `sentinel_dashboard.py` | 65 KB | Main dashboard — 9 tabs, 40+ render functions |
| `theme.py` | 19 KB | `DARK_THEME`, `LIGHT_THEME`, threat-level colours, CSS generator, Plotly theme |
| `data_connector.py` | 37 KB | `DashboardData` dataclass + `get_dashboard_data()` — interfaces with backend or generates demo data |
| `globe_viz.py` | 16 KB | 3D Globe.gl component — Kenya boundaries, threat arcs, CesiumJS |
| `kenya_threatmap.py` | 20 KB | Kenya county choropleth with Folium |
| `kenya_data_loader.py` | 11 KB | Loads GeoJSON county boundaries and economic indicators |
| `pulse_data_loader.py` | 5 KB | Loads Pulse sensor state for dashboard display |
| `causal_viz.py` | 4 KB | 3D causal network WebGL renderer |
| `full_economy_graph.py` | 4 KB | 40+ variable base economy map (nodes and edges) |
| `flux_viz.py` | 5 KB | Economic money-flow 3D animations |
| `animated_header.py` | 3 KB | WebGL animated header with particle effects |
| `document_intel.py` | 9 KB | Document upload and NLP analysis interface |

---

## Running the Dashboard

```bash
# From the project root
streamlit run kshiked/ui/sentinel_dashboard.py
```

The dashboard will be available at `http://localhost:8501`.

---

## Theme System

The theme is defined in `theme.py` and provides:

- **Dark mode** (default) and **light mode** themes
- Threat-level colour mappings: `LOW` (green), `MODERATE` (amber), `HIGH` (orange), `CRITICAL` (red)
- CSS generator for Streamlit custom styling
- Plotly chart theme integration

```python
from theme import DARK_THEME, generate_css, get_plotly_theme

st.markdown(f"<style>{generate_css(DARK_THEME)}</style>", unsafe_allow_html=True)
fig.update_layout(template=get_plotly_theme(DARK_THEME))
```

---

## Data Connector

`data_connector.py` provides the `DashboardData` dataclass that feeds all 9 tabs:

```python
from data_connector import get_dashboard_data, DashboardData

data: DashboardData = get_dashboard_data()
# data.threat_level, data.signals, data.hypotheses, data.simulation_state, ...
```

In production, this connects to the FastAPI backend. In demo mode, it generates synthetic data.

---

*Source: `kshiked/ui/` · Last updated: 2026-02-11*

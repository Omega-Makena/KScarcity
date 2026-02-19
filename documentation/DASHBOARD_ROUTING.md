# Dashboard Routing — Navigation Reference

> How users navigate from the SENTINEL home screen to any analysis view, all on a single port.

---

## Entry Point

```bash
streamlit run kshiked/ui/sentinel_dashboard.py --server.port 8501
```

Everything runs on **http://localhost:8501** — one Streamlit process, one port.

---

## Navigation Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SENTINEL HOME                             │
│                                                               │
│   ┌─────────────┐  ┌─────────────┐                          │
│   │  K-SHIELD   │  │  K-PULSE    │                          │
│   │  (KSHIELD)  │  │  (SIGNALS)  │                          │
│   └──────┬──────┘  └─────────────┘                          │
│   ┌──────┴──────┐  ┌─────────────┐                          │
│   │  K-COLLAB   │  │ K-EDUCATION │                          │
│   │ (FEDERATION)│  │   (DOCS)    │                          │
│   └─────────────┘  └─────────────┘                          │
└──────────────────────────────────────────────────────────────┘

              ┌─── K-SHIELD (click card or sidebar) ───┐
              │                                         │
              ▼                                         │
┌──────────────────────────────────────────────┐       │
│           K-SHIELD LANDING PAGE               │       │
│                                               │       │
│  ┌──────────────┐  ┌──────────────┐          │       │
│  │   CAUSAL     │  │   TERRAIN    │          │       │
│  │ Relationships│  │   Policy     │          │       │
│  └──────┬───────┘  └──────┬───────┘          │       │
│  ┌──────┴───────┐  ┌──────┴───────┐          │       │
│  │ SIMULATIONS  │  │   IMPACT     │          │       │
│  │  (11 tabs)   │  │   Policy     │          │       │
│  └──────┬───────┘  └──────────────┘          │       │
│         │                                     │       │
│  ← Back to K-SHIELD                          │       │
└──────────────────────────────────────────────┘       │
                                                        │
              ▼                                         │
┌──────────────────────────────────────────────┐       │
│         SIMULATION WORKSPACE                  │       │
│                                               │       │
│  Nav: [Workspace] [Guide & Tutorial]          │       │
│  Data: [World Bank] [Upload CSV] [Shared]     │       │
│  Config: Scenario + Policy + Dimensions       │       │
│  [RUN SIMULATION]                             │       │
│                                               │       │
│  Tabs:                                        │       │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐       │
│  │1  │2  │3  │4  │5  │6  │7  │8  │9  │10 │11 │       │
│  │Run│Sen│Cub│Cmp│Phs│IRF│Flw│MC │Str│Par│Dia│       │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘       │
│                                               │       │
│  ← Back to K-SHIELD                          │       │
└──────────────────────────────────────────────┘       │
```

---

## State Management

All navigation is driven by two `st.session_state` keys:

| Key | Values | Scope |
|-----|--------|-------|
| `current_view` | `HOME`, `KSHIELD`, `SIGNALS`, `FEDERATION`, `DOCS`, `LIVE_MAP`, `EXECUTIVE`, `CAUSAL`, `SIMULATION`, `ESCALATION`, `OPERATIONS`, `SYSTEM_GUIDE`, `POLICY_CHAT` | Top-level sentinel dashboard |
| `kshield_view` | `LANDING`, `CAUSAL`, `TERRAIN`, `SIMULATION`, `IMPACT` | Within K-SHIELD module |

### Navigation Flow

1. **HOME** → User clicks K-SHIELD card → sets `current_view = "KSHIELD"` → `st.rerun()`
2. **Sidebar** radio is synced: `st.session_state["sb_nav_radio"] = "K-SHIELD"` before render
3. **Router** dispatches: `view == "KSHIELD"` → `from kshield.page import render` → `render(theme, data)`
4. **K-SHIELD page.py** checks `kshield_view`:
   - `"LANDING"` → 4 sub-cards
   - `"SIMULATION"` → `from kshield.simulation import render_simulation` → 11 tabs
   - `"CAUSAL"` → `from kshield.causal import render_causal`
   - etc.
5. Each K-SHIELD sub-page has a **"← Back to K-SHIELD"** button that resets `kshield_view = "LANDING"`

### Sidebar Sync

The sidebar radio widget caches its value under `st.session_state["sb_nav_radio"]`. When a card button sets `current_view` programmatically and triggers `st.rerun()`, the code explicitly syncs:

```python
if st.session_state.get("sb_nav_radio") != current_name:
    st.session_state["sb_nav_radio"] = current_name
```

This prevents the cached radio value from overriding the card-button navigation.

### URL Deep Links

The router also syncs view state with URL query params:

- Read: `?view=<VIEW_KEY>` on load
- Write: sets `?view=<CURRENT_VIEW>` on navigation

Examples:

- `/?view=FEDERATION`
- `/?view=KSHIELD`
- `/?view=POLICY_CHAT`

---

## Sidebar Navigation Options

| Label | View Key | Target |
|-------|----------|--------|
| Home | `HOME` | 4-card home screen |
| Live Threat Map | `LIVE_MAP` | Real-time threat map |
| Executive Overview | `EXECUTIVE` | Traffic light + gauges |
| Signal Intelligence | `SIGNALS` | 15 signals, heatmap |
| Causal Analysis | `CAUSAL` | Legacy causal tab |
| **K-SHIELD** | `KSHIELD` | K-SHIELD module (landing → sub-pages) |
| Simulation (Legacy) | `SIMULATION` | WhatIf Workbench (pre-K-SHIELD) |
| Escalation Pathways | `ESCALATION` | Decision intelligence |
| Federation / Federated Databases | `FEDERATION` | Node registration, sync rounds, metrics, audit logs |
| Operations | `OPERATIONS` | County table, alerts |
| System Guide | `SYSTEM_GUIDE` | Built-in documentation |
| Document Intelligence | `DOCS` | PDF/document analysis |
| Policy Intelligence | `POLICY_CHAT` | Policy chatbot + evidence traces |

---

## Auth Gate

K-SHIELD sub-pages pass through `check_access("K-SHIELD", theme)` which reads `kshiked/config/access_codes.json`. If no hash is configured (empty string), access is auto-granted (development mode).

---

## File Map

| File | Role |
|------|------|
| `kshiked/ui/sentinel_dashboard.py` | Main Streamlit entrypoint |
| `kshiked/ui/sentinel/router.py` | Top-level router, sidebar navigation, query-param deep-link sync |
| `kshiked/ui/sentinel/federation.py` | Federation / Federated Databases view |
| `kshiked/ui/sentinel/policy_chat.py` | Policy Intelligence chat view |
| `kshiked/ui/kshield/page.py` | K-SHIELD module — auth, landing, sub-page routing |
| `kshiked/ui/kshield/simulation.py` | Simulation card — 11 analysis tabs |
| `kshiked/ui/kshield/causal.py` | Causal Relationships card |
| `kshiked/ui/kshield/terrain.py` | Policy Terrain card |
| `kshiked/ui/kshield/impact/components/live_policy.py` | Live Policy Impact overlays (baseline vs counterfactual, freshness) |
| `kshiked/ui/common/auth.py` | Access code gate |
| `kshiked/ui/common/landing.py` | Reusable landing page renderer |
| `kshiked/ui/common/nav.py` | Back-button helpers |
| `kshiked/config/access_codes.json` | Auth hashes per module |

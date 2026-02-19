"""Shared imports, constants, and data utilities for simulation submodules."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import streamlit as st

if TYPE_CHECKING:
    import pandas as _pd
    DataFrame = _pd.DataFrame
else:
    DataFrame = Any

logger = logging.getLogger("sentinel.kshield.simulation")

try:
    import pandas as pd
    import numpy as np
    HAS_DATA_STACK = True
except ImportError:
    pd = None
    np = None
    HAS_DATA_STACK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

try:
    import streamlit.components.v1 as components
    HAS_COMPONENTS = True
except ImportError:
    HAS_COMPONENTS = False

# ── Shared dataset keys (same as causal.py / terrain.py) ─────────────────────
SHARED_DF_KEY = "kshield_shared_df"
SHARED_SOURCE_KEY = "kshield_shared_source"
SHARED_OWNER_KEY = "kshield_shared_owner"

# ── Consistent colour palette ────────────────────────────────────────────────
PALETTE = [
    "#00ff88", "#00aaff", "#f5d547", "#ff3366", "#8b5cf6",
    "#14b8a6", "#f97316", "#ec4899", "#a3e635", "#06b6d4",
]


# ── Shared dataset helpers ───────────────────────────────────────────────────

def set_shared_dataset(df: DataFrame, source: str, owner: str) -> None:
    if not HAS_DATA_STACK or df is None or df.empty:
        return
    st.session_state[SHARED_DF_KEY] = df.copy(deep=True)
    st.session_state[SHARED_SOURCE_KEY] = source
    st.session_state[SHARED_OWNER_KEY] = owner


def get_shared_dataset() -> Tuple[Optional[DataFrame], str]:
    if not HAS_DATA_STACK:
        return None, ""
    candidate = st.session_state.get(SHARED_DF_KEY)
    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
        source = str(st.session_state.get(SHARED_SOURCE_KEY, "Unknown source"))
        owner = str(st.session_state.get(SHARED_OWNER_KEY, "Unknown card"))
        return candidate, f"{source} via {owner}"
    return None, ""


# ── Data loading ─────────────────────────────────────────────────────────────

def find_csv() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parents[4] / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
        Path(os.getcwd()) / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@st.cache_data(ttl=3600, show_spinner="Loading World Bank data ...")
def load_world_bank_data() -> DataFrame:
    csv_path = find_csv()
    if csv_path is None:
        return pd.DataFrame()
    raw = pd.read_csv(csv_path, skiprows=4, encoding="utf-8-sig")
    return pivot_world_bank(raw)


def pivot_world_bank(raw: DataFrame) -> DataFrame:
    if "Indicator Name" not in raw.columns:
        return pd.DataFrame()
    year_cols = [c for c in raw.columns if c.strip().isdigit()
                 and 1900 <= int(c.strip()) <= 2100]
    if not year_cols:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=["Indicator Name"], value_vars=year_cols,
        var_name="Year", value_name="Value",
    )
    melted["Year"] = melted["Year"].astype(int)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    pivoted = melted.pivot_table(index="Year", columns="Indicator Name", values="Value")
    pivoted = pivoted.sort_index()
    good_cols = pivoted.columns[pivoted.notna().sum() >= 15]
    pivoted = pivoted[good_cols]
    pivoted = pivoted.interpolate(method="linear", limit=3)
    return pivoted


def validate_and_load_upload(uploaded_file) -> Tuple[Optional[DataFrame], Optional[str]]:
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as e:
        return None, f"Could not parse CSV: {e}"
    if "Indicator Name" in raw.columns:
        year_cols = [c for c in raw.columns if c.strip().isdigit()
                     and 1900 <= int(c.strip()) <= 2100]
        if year_cols:
            df = pivot_world_bank(raw)
            if df.empty:
                return None, "World Bank format detected but no usable indicator data."
            return df, None
    if len(raw.columns) < 2:
        return None, "CSV must have at least 2 columns."
    df = raw.copy()
    idx_col = df.columns[0]
    try:
        idx_vals = pd.to_numeric(df[idx_col], errors="coerce")
        if idx_vals.notna().sum() > 0.5 * len(idx_vals):
            df[idx_col] = idx_vals
            df = df.set_index(idx_col)
        else:
            try:
                df[idx_col] = pd.to_datetime(df[idx_col])
                df = df.set_index(idx_col)
                df.index = df.index.year
            except Exception:
                return None, f"First column '{idx_col}' must be numeric years or dates."
    except Exception:
        return None, f"Could not parse index column '{idx_col}'."
    df = df.sort_index()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric data columns."
    df = df[numeric_cols]
    good_cols = df.columns[df.notna().sum() >= 10]
    if len(good_cols) < 2:
        return None, "Not enough data. Each column needs at least 10 non-empty values."
    df = df[good_cols].interpolate(method="linear", limit=3)
    return df, None


# ── Plotly layout helper ─────────────────────────────────────────────────────

def base_layout(theme, height=400, **extra):
    layout = dict(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color=theme.text_muted, size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=theme.border_default,
                   tickfont=dict(color=theme.text_muted)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=theme.border_default,
                   tickfont=dict(color=theme.text_muted)),
        legend=dict(font=dict(color=theme.text_muted, size=10)),
        hovermode="x unified",
    )
    layout.update(extra)
    return layout


# ── Dynamic dimension discovery ──────────────────────────────────────────────

def discover_dimensions(trajectory):
    categories = {}
    vector_keys = [
        "outcomes", "channels", "flows", "sector_balances",
        "policy_vector", "shock_vector",
    ]
    for cat in vector_keys:
        keys = sorted({
            k for f in trajectory
            for k in (f.get(cat, {}) if isinstance(f.get(cat), dict) else {})
        })
        if keys:
            categories[cat] = keys
    return categories


def dim_label(raw: str) -> str:
    return raw.replace("_", " ").title()


def extract_dim(dim_str: str, frame: dict) -> float:
    cat, key = dim_str.split("::", 1)
    container = frame.get(cat, {})
    if isinstance(container, dict):
        return float(container.get(key, 0))
    return 0.0


def flat_dim_options(dims: dict) -> list:
    options = []
    for cat, keys in dims.items():
        for k in keys:
            options.append(f"{cat}::{k}")
    return options

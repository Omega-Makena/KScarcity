"""
K-SHIELD: Causal Relationships — Auto-Discovery Engine

Self-contained causal analysis pipeline that:
1. Auto-discovers ALL indicators from World Bank CSV (no hardcoding)
2. Accepts user-uploaded CSV files (with validation)
3. Runs pairwise Granger causality tests
4. Computes cross-correlation at multiple lags
5. Renders rich interactive Plotly visualizations
6. Provides plain-English interpretations of every result
"""

from __future__ import annotations

import os
import io
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Heavy ML/stats packages loaded LAZILY on first use ──────────────────────
# This avoids 3-8s import overhead when just opening the Causal card.
STRUCTURAL_IMPORT_ERRORS: Dict[str, str] = {}
_LAZY_CACHE: Dict[str, Any] = {}  # module-level cache for lazy imports

def _lazy_statsmodels():
    if "statsmodels" not in _LAZY_CACHE:
        try:
            from statsmodels.tsa.stattools import grangercausalitytests, adfuller
            _LAZY_CACHE["statsmodels"] = (grangercausalitytests, adfuller)
        except ImportError:
            _LAZY_CACHE["statsmodels"] = None
    return _LAZY_CACHE["statsmodels"]

def _lazy_dowhy():
    if "dowhy" not in _LAZY_CACHE:
        try:
            from dowhy import CausalModel
            _LAZY_CACHE["dowhy"] = CausalModel
        except Exception as exc:
            _LAZY_CACHE["dowhy"] = None
            STRUCTURAL_IMPORT_ERRORS["dowhy"] = str(exc)
    return _LAZY_CACHE["dowhy"]

def _lazy_sklearn():
    if "sklearn" not in _LAZY_CACHE:
        try:
            from sklearn.linear_model import LinearRegression, LassoCV
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            _LAZY_CACHE["sklearn"] = {
                "LinearRegression": LinearRegression, "LassoCV": LassoCV,
                "RandomForestRegressor": RandomForestRegressor,
                "GradientBoostingRegressor": GradientBoostingRegressor,
            }
        except Exception as exc:
            _LAZY_CACHE["sklearn"] = None
            STRUCTURAL_IMPORT_ERRORS["scikit-learn"] = str(exc)
    return _LAZY_CACHE["sklearn"]

def _lazy_econml():
    if "econml" not in _LAZY_CACHE:
        try:
            import econml as _econml
            _LAZY_CACHE["econml"] = _econml
        except Exception as exc:
            _LAZY_CACHE["econml"] = None
            STRUCTURAL_IMPORT_ERRORS["econml"] = str(exc)
    return _LAZY_CACHE["econml"]

def _lazy_networkx():
    if "networkx" not in _LAZY_CACHE:
        try:
            import networkx as _nx
            _LAZY_CACHE["networkx"] = _nx
        except Exception as exc:
            _LAZY_CACHE["networkx"] = None
            STRUCTURAL_IMPORT_ERRORS["networkx"] = str(exc)
    return _LAZY_CACHE["networkx"]

# Compatibility shims — check availability without importing
@property
def _has_statsmodels():
    return _lazy_statsmodels() is not None

HAS_STATSMODELS = property(lambda self: _lazy_statsmodels() is not None)

# Use functions for runtime checks instead of module-level booleans
def _check_statsmodels(): return _lazy_statsmodels() is not None
def _check_dowhy(): return _lazy_dowhy() is not None
def _check_sklearn(): return _lazy_sklearn() is not None
def _check_econml(): return _lazy_econml() is not None
def _check_networkx(): return _lazy_networkx() is not None
def _check_structural(): return _check_dowhy() and _check_sklearn()

# Keep backward-compat module-level flags as lazy properties via __getattr__
# These are evaluated on first access, not at import time
HAS_STATSMODELS = None  # sentinel — replaced by _check functions
HAS_DOWHY = None
HAS_SKLEARN = None
HAS_ECONML = None
HAS_NETWORKX = None
HAS_STRUCTURAL = None

# ── Resolve lazy flags on first real access ──────────────────────────────────
def _init_lazy_flags():
    """Populate HAS_* globals lazily — call before any HAS_* check."""
    global HAS_STATSMODELS, HAS_DOWHY, HAS_SKLEARN, HAS_ECONML, HAS_NETWORKX, HAS_STRUCTURAL
    if HAS_STATSMODELS is None:
        HAS_STATSMODELS = _check_statsmodels()
        HAS_DOWHY = _check_dowhy()
        HAS_SKLEARN = _check_sklearn()
        HAS_ECONML = _check_econml()
        HAS_NETWORKX = _check_networkx()
        HAS_STRUCTURAL = _check_structural()

SHARED_DF_KEY = "kshield_shared_df"
SHARED_SOURCE_KEY = "kshield_shared_source"
SHARED_OWNER_KEY = "kshield_shared_owner"


def _set_shared_dataset(df: pd.DataFrame, source: str, owner: str) -> None:
    """Publish a DataFrame for reuse across K-SHIELD cards in current session."""
    if not HAS_PANDAS or df is None or df.empty:
        return
    st.session_state[SHARED_DF_KEY] = df.copy(deep=True)
    st.session_state[SHARED_SOURCE_KEY] = source
    st.session_state[SHARED_OWNER_KEY] = owner


def _get_shared_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """Read shared K-SHIELD DataFrame from session state."""
    if not HAS_PANDAS:
        return None, ""
    candidate = st.session_state.get(SHARED_DF_KEY)
    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
        source = str(st.session_state.get(SHARED_SOURCE_KEY, "Unknown source"))
        owner = str(st.session_state.get(SHARED_OWNER_KEY, "Unknown card"))
        return candidate, f"{source} via {owner}"
    return None, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — AUTO-DISCOVERY (no hardcoded indicators)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Loading World Bank data ...")
def load_world_bank_data() -> pd.DataFrame:
    """Auto-discover and load ALL indicators from the World Bank CSV."""
    csv_path = _find_csv()
    if csv_path is None:
        return pd.DataFrame()

    raw = pd.read_csv(csv_path, skiprows=4, encoding="utf-8-sig")
    return _pivot_world_bank(raw)


def _pivot_world_bank(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot a World Bank CSV into year x indicator DataFrame."""
    if "Indicator Name" not in raw.columns:
        return pd.DataFrame()

    # Detect year columns automatically
    year_cols = [c for c in raw.columns if c.strip().isdigit()
                 and 1900 <= int(c.strip()) <= 2100]
    if not year_cols:
        return pd.DataFrame()

    melted = raw.melt(
        id_vars=["Indicator Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    melted["Year"] = melted["Year"].astype(int)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

    pivoted = melted.pivot_table(index="Year", columns="Indicator Name", values="Value")
    pivoted = pivoted.sort_index()

    # Keep only columns with >= 15 non-null values (enough for analysis)
    good_cols = pivoted.columns[pivoted.notna().sum() >= 15]
    pivoted = pivoted[good_cols]

    # Interpolate small gaps (max 3 year gaps)
    pivoted = pivoted.interpolate(method="linear", limit=3)

    return pivoted


def _find_csv() -> Optional[str]:
    """Find the World Bank CSV file."""
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
        Path(os.getcwd()) / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _validate_and_load_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Validate and load a user-uploaded CSV. Returns (df, error_msg)."""
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as e:
        return None, f"Could not parse CSV: {e}"

    # Check 1: World Bank format (has Indicator Name + year columns)
    if "Indicator Name" in raw.columns:
        year_cols = [c for c in raw.columns if c.strip().isdigit()
                     and 1900 <= int(c.strip()) <= 2100]
        if year_cols:
            df = _pivot_world_bank(raw)
            if df.empty:
                return None, "World Bank format detected but no usable indicator data found."
            return df, None

    # Check 2: Simple format — first column is time index, rest are numeric series
    if len(raw.columns) < 2:
        return None, "CSV must have at least 2 columns (1 index + 1 data column)."

    # Try to use first column as index
    df = raw.copy()
    idx_col = df.columns[0]

    # Check if index looks like years or dates
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
                return None, (
                    f"First column '{idx_col}' must be numeric years or dates. "
                    "Example: 2000, 2001, 2002 ..."
                )
    except Exception:
        return None, f"Could not parse index column '{idx_col}'."

    df = df.sort_index()

    # Only keep numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, (
            f"Found only {len(numeric_cols)} numeric column(s). "
            "Need at least 2 numeric data columns for analysis."
        )

    df = df[numeric_cols]

    # Check for sufficient data
    good_cols = df.columns[df.notna().sum() >= 10]
    if len(good_cols) < 2:
        return None, "Not enough data. Each column needs at least 10 non-empty values."

    df = df[good_cols].interpolate(method="linear", limit=3)
    return df, None


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Running Granger causality tests ...")
def run_granger_analysis(
    _df: pd.DataFrame,
    selected_cols: List[str],
    max_lag: int = 5,
) -> pd.DataFrame:
    """Run pairwise Granger causality tests for selected columns."""
    _init_lazy_flags()
    if not HAS_STATSMODELS:
        return pd.DataFrame()

    _granger, _adf = _lazy_statsmodels()
    results = []
    pairs = list(itertools.permutations(selected_cols, 2))

    for cause, effect in pairs:
        subset = _df[[cause, effect]].dropna()
        if len(subset) < max_lag + 10:
            continue

        try:
            test = _granger(
                subset[[effect, cause]], maxlag=max_lag, verbose=False
            )
            best_lag = None
            best_pval = 1.0
            best_fstat = 0.0
            for lag in range(1, max_lag + 1):
                pval = test[lag][0]["ssr_ftest"][1]
                fstat = test[lag][0]["ssr_ftest"][0]
                if pval < best_pval:
                    best_pval = pval
                    best_lag = lag
                    best_fstat = fstat

            results.append({
                "Cause": cause,
                "Effect": effect,
                "Best Lag (yrs)": best_lag,
                "F-statistic": round(best_fstat, 2),
                "p-value": round(best_pval, 4),
                "Significant": best_pval < 0.05,
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("p-value")


@st.cache_data(ttl=3600, show_spinner="Computing cross-correlations ...")
def compute_cross_correlations(
    _df: pd.DataFrame,
    selected_cols: List[str],
    max_lag: int = 10,
) -> Dict[Tuple[str, str], List[float]]:
    """Compute cross-correlation functions at multiple lags."""
    result: Dict[Tuple[str, str], List[float]] = {}
    pairs = list(itertools.combinations(selected_cols, 2))

    for a, b in pairs:
        subset = _df[[a, b]].dropna()
        if len(subset) < max_lag + 5:
            continue

        x = (subset[a] - subset[a].mean()) / (subset[a].std() + 1e-8)
        y = (subset[b] - subset[b].mean()) / (subset[b].std() + 1e-8)

        ccf = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = x.iloc[:len(x) - lag].reset_index(drop=True).corr(
                    y.iloc[lag:].reset_index(drop=True)
                )
            else:
                corr = x.iloc[-lag:].reset_index(drop=True).corr(
                    y.iloc[:len(y) + lag].reset_index(drop=True)
                )
            ccf.append(corr if not np.isnan(corr) else 0.0)

        result[(a, b)] = ccf

    return result


@st.cache_data(ttl=3600, show_spinner="Running stationarity tests ...")
def run_stationarity_tests(
    _df: pd.DataFrame, selected_cols: List[str]
) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller for each series."""
    _init_lazy_flags()
    if not HAS_STATSMODELS:
        return pd.DataFrame()

    _granger, _adf = _lazy_statsmodels()
    results = []
    for col in selected_cols:
        series = _df[col].dropna()
        if len(series) < 10:
            continue
        try:
            stat, pval, used_lag, nobs, crit, _ = _adf(series, autolag="AIC")
            results.append({
                "Variable": col,
                "ADF Stat": round(stat, 3),
                "p-value": round(pval, 4),
                "Lags Used": used_lag,
                "Obs": nobs,
                "Stationary": pval < 0.05,
                "1%": round(crit["1%"], 3),
                "5%": round(crit["5%"], 3),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY LAYOUT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _base_layout(theme, height=400, **extra):
    """Standard Plotly layout matching SENTINEL theme."""
    layout = dict(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Space Mono, monospace",
            color=theme.text_muted,
            size=11,
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor=theme.border_default,
            tickfont=dict(color=theme.text_muted),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor=theme.border_default,
            tickfont=dict(color=theme.text_muted),
        ),
        legend=dict(font=dict(color=theme.text_muted, size=10)),
        hovermode="x unified",
    )
    layout.update(extra)
    return layout


# Consistent color palette matching SENTINEL accent colors
PALETTE = [
    "#00ff88", "#00aaff", "#f5d547", "#ff3366", "#8b5cf6",
    "#14b8a6", "#f97316", "#ec4899", "#a3e635", "#06b6d4",
]


def _render_causal_guide_page(theme):
    """Dedicated tutorial page with internal navigation."""
    st.markdown(
        '<div class="section-header">CAUSAL GUIDE &mdash; FULL TUTORIAL</div>',
        unsafe_allow_html=True,
    )

    nav_col, content_col = st.columns([1, 3])
    sections = [
        "1) Why this page exists",
        "2) What each tab means",
        "3) Glossary (plain English)",
        "4) How to run estimands",
        "5) How to interpret results",
        "6) Common mistakes",
        "7) What is Granger causality?",
        "8) What happens when you click RUN",
        "9) Fallback policy and failure handling",
    ]

    with nav_col:
        st.markdown("**Guide Navigation**")
        section = st.radio(
            "Jump to",
            sections,
            key="causal_guide_section",
            label_visibility="collapsed",
        )

    with content_col:
        if section == "1) Why this page exists":
            st.markdown(
                """
                This page helps answer a policy question:
                **If we change one indicator (treatment), what likely happens to another (outcome)?**

                Why not only use correlation:
                - Correlation says variables move together.
                - Causal estimation tries to estimate directional effect under assumptions.
                """
            )
        elif section == "2) What each tab means":
            st.markdown(
                """
                - `Time Series`: trend lines over time.
                - `Correlation Heatmap (Color Grid)`: matrix of pairwise relationship strength.
                - `Granger Causality`: does past X improve prediction of future Y?
                - `Causal Network`: graph of significant directed links.
                - `Cross-Correlations`: lead/lag timing between two series.
                - `Stationarity`: whether time-series assumptions are stable enough.
                - `Scatter Matrix`: pairwise point plots for relationship shape.
                - `Cause-Effect Estimation (Advanced)`: formal estimand execution in Scarcity.
                """
            )
        elif section == "3) Glossary (plain English)":
            st.markdown(
                """
                - `Heatmap`: a table where color shows magnitude.
                - `Estimand`: the exact effect question being estimated.
                - `ATE`: average effect for everyone.
                - `ATT`: average effect for treated group.
                - `ATC`: average effect for untreated/control group.
                - `CATE`: effect for subgroup (requires `econml`).
                - `ITE`: individual-level effect estimate (requires `econml`).
                - `LATE`: effect for instrument-moved group (needs instrument).
                - `Mediation`: splits effect into direct vs indirect path via mediator.
                """
            )
        elif section == "4) How to run estimands":
            st.markdown(
                """
                Recommended run order:
                1. Select treatment and outcome.
                2. Add confounders you trust.
                3. Start with `ATE`, `ATT`, `ATC`.
                4. Add `CATE` only if subgroup logic matters and `econml` is installed.
                5. Add `LATE` only with a valid instrument.
                6. Add mediation only with a valid mediator variable.
                7. Run, then inspect success/failure table and error stages.
                """
            )
        elif section == "5) How to interpret results":
            st.markdown(
                """
                Read in this order:
                1. `Direction`: increase/decrease/near-zero.
                2. `CI Excludes 0`: stronger evidence of non-null effect.
                3. `Agreement`: do estimands mostly have same sign?
                4. `Strongest effect`: which estimand has largest magnitude?

                Decision confidence is stronger when:
                - sign agreement is high
                - many CIs exclude zero
                - failures are low
                """
            )
        elif section == "6) Common mistakes":
            st.markdown(
                """
                - Running `CATE/ITE` without `econml`.
                - Selecting `LATE` without an instrument.
                - Selecting mediation without mediator.
                - Treating disagreement across estimands as a final conclusion.
                - Ignoring data quality warnings (missingness, non-stationarity, short overlap).
                """
            )
        elif section == "7) What is Granger causality?":
            st.markdown(
                """
                **Plain-language definition**
                Granger causality asks:
                *Do past values of X help predict future values of Y, beyond Y’s own past?*

                **What it does well**
                - Finds lead-lag predictive structure in time series.
                - Good for early-warning relationships.

                **What it does NOT guarantee**
                - It is not absolute proof of real-world cause.
                - Hidden drivers can still create false directional signals.

                **How to read it here**
                - Lower p-value = stronger evidence of predictive direction.
                - Best lag = approximate delay between X move and Y response.
                - We use these links to propose a candidate causal graph for deeper estimation.
                """
            )
        elif section == "8) What happens when you click RUN":
            st.markdown(
                """
                The system executes a pipeline:

                1. **Validate inputs**: treatment, outcome, confounders, optional IV/mediator.
                2. **Build selected effect questions (estimands)**.
                3. **Skip invalid effect types** with explicit reasons (for example, missing instrument).
                4. **Run Scarcity engine** on each estimand (in parallel or sequential mode).
                5. **Collect outputs**: estimate, confidence interval, backend/method, diagnostics.
                6. **Render interpretation**:
                   - effect direction,
                   - sign agreement across estimands,
                   - CI robustness (exclude zero or not),
                   - strongest estimated effect.
                7. **Show failures by stage** if anything breaks (identification, estimation, worker execution, etc.).
                """
            )
        elif section == "9) Fallback policy and failure handling":
            st.markdown(
                """
                There are two layers of failure handling:

                **A) Estimand fail policy (user setting)**
                - `Continue`: if one effect type fails, keep running the others.
                - `Fail fast`: stop immediately on first failure.

                **B) Runtime fallback (automatic)**
                - If parallel workers fail (common on some environments),
                  the app automatically retries in **sequential mode**.
                - This keeps the run usable for non-technical users.

                **Why this exists**
                Parallel processing can fail due to environment constraints
                (process spawn/pickling/worker pool issues), not because your model is wrong.

                **What you should do when fallback appears**
                - Trust the sequential results as the stable baseline.
                - Use parallel mode later only for speed if your environment supports it.
                """
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render_causal(theme, data=None):
    """Render the full causal analysis page."""
    _init_lazy_flags()  # resolve HAS_* flags lazily on first render
    st.markdown(
        '<div class="section-header">'
        "CAUSAL RELATIONSHIPS &mdash; ECONOMIC INDICATOR ANALYSIS</div>",
        unsafe_allow_html=True,
    )

    if not HAS_PANDAS or not HAS_PLOTLY:
        st.error("Required packages missing: pandas / plotly.")
        return

    page_mode = st.radio(
        "Causal Navigation",
        ["Analysis Workspace", "Guide & Tutorial"],
        horizontal=True,
        key="causal_page_mode",
    )
    if page_mode == "Guide & Tutorial":
        _render_causal_guide_page(theme)
        return

    # ── Data source selector ──────────────────────────────────────────────────
    st.markdown("---")
    source = st.radio(
        "Data Source",
        ["World Bank (Kenya)", "Upload your own CSV", "Shared K-SHIELD Dataset"],
        horizontal=True,
        key="causal_data_source",
    )

    df = pd.DataFrame()
    shared_df, shared_meta = _get_shared_dataset()

    if source == "World Bank (Kenya)":
        df = load_world_bank_data()
        if df.empty:
            st.error(
                "World Bank CSV not found in data/simulation/. "
                "Place API_KEN_DS2_*.csv there, or upload your own file."
            )
            return
        st.markdown(
            f'<p style="color:{theme.text_muted}; font-size:0.8rem; '
            f'font-family:Space Mono,monospace;">'
            f"Auto-discovered <b>{len(df.columns)}</b> indicators, "
            f"<b>{len(df)}</b> years "
            f"({df.index.min()}&ndash;{df.index.max()}) "
            f"from World Bank Development Indicators</p>",
            unsafe_allow_html=True,
        )
        _set_shared_dataset(df, "World Bank (Kenya)", "CAUSAL")
    elif source == "Shared K-SHIELD Dataset":
        if shared_df is None or shared_df.empty:
            st.warning("No shared K-SHIELD dataset found yet. Load World Bank or upload a CSV first.")
            return
        df = shared_df
        st.caption(
            f"Using shared dataset: {shared_meta}. "
            f"{len(df.columns)} series across {len(df)} time periods."
        )
    else:
        _render_upload_section(theme)
        if "causal_uploaded_df" in st.session_state:
            df = st.session_state["causal_uploaded_df"]
        if df.empty:
            return
        _set_shared_dataset(df, "Uploaded CSV", "CAUSAL")

    # ── Indicator selection (auto-discovered) ─────────────────────────────────
    st.markdown("---")
    available = sorted(df.columns.tolist())

    # No hardcoded defaults — pick the first few alphabetically
    default_count = min(6, len(available))
    default_selection = available[:default_count]

    selected = st.multiselect(
        f"Select indicators to analyse ({len(available)} available, pick 4-10 for best results)",
        options=available,
        default=default_selection,
    )

    if len(selected) < 2:
        st.warning("Select at least 2 indicators to begin analysis.")
        return

    analysis_df = df[selected].dropna(how="all")
    n_obs = len(analysis_df.dropna())

    st.caption(f"Using {n_obs} complete observations across {len(selected)} variables")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Time Series",
        "Correlation Heatmap (Color Grid)",
        "Granger Causality",
        "Causal Network",
        "Cross-Correlations",
        "Stationarity",
        "Scatter Matrix",
        "Cause-Effect Estimation (Advanced)",
    ])

    with tabs[0]:
        _render_time_series(analysis_df, selected, theme)
    with tabs[1]:
        _render_correlation_heatmap(analysis_df, selected, theme)
    with tabs[2]:
        _render_granger_section(analysis_df, selected, theme)
    with tabs[3]:
        _render_causal_network(analysis_df, selected, theme)
    with tabs[4]:
        _render_cross_corr(analysis_df, selected, theme)
    with tabs[5]:
        _render_stationarity(analysis_df, selected, theme)
    with tabs[6]:
        _render_scatter_matrix(analysis_df, selected, theme)
    with tabs[7]:
        _render_structural_inference(analysis_df, selected, theme)


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _render_upload_section(theme):
    """Render CSV upload area with validation feedback."""
    st.markdown(
        f'<div style="background:{theme.bg_card}; backdrop-filter:blur(12px); '
        f"border:1px solid {theme.border_default}; border-radius:12px; "
        f'padding:1.2rem; margin-bottom:1rem;">'
        f'<div style="color:{theme.text_secondary}; font-weight:600; '
        f'margin-bottom:0.5rem; font-family:Space Mono,monospace; '
        f'text-transform:uppercase; letter-spacing:0.5px; font-size:0.85rem;">'
        f"UPLOAD DATA</div>"
        f'<div style="color:{theme.text_muted}; font-size:0.8rem; '
        f'font-family:Space Mono,monospace;">'
        f"Accepted formats:<br>"
        f"&bull; <b>World Bank CSV</b> (with Indicator Name column + year columns)<br>"
        f"&bull; <b>Simple CSV</b> (first column = year/date, remaining columns = numeric series)"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a clean CSV file",
        type=["csv"],
        key="causal_csv_upload",
    )

    if uploaded is not None:
        df, error = _validate_and_load_upload(uploaded)
        if error:
            st.error(f"Validation failed: {error}")
            if "causal_uploaded_df" in st.session_state:
                del st.session_state["causal_uploaded_df"]
        else:
            st.session_state["causal_uploaded_df"] = df
            _set_shared_dataset(df, "Uploaded CSV", "CAUSAL")
            st.markdown(
                f'<p style="color:{theme.accent_success}; font-size:0.8rem; '
                f'font-family:Space Mono,monospace;">'
                f"Loaded <b>{len(df.columns)}</b> series, "
                f"<b>{len(df)}</b> time periods "
                f"({df.index.min()}&ndash;{df.index.max()})</p>",
                unsafe_allow_html=True,
            )
            # Show a preview
            with st.expander("Data preview (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_time_series(df: pd.DataFrame, cols: List[str], theme):
    """Interactive stacked time series."""
    st.markdown(
        '<div class="section-header">INDICATORS OVER TIME</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Each line is z-score normalised so different scales are comparable. "
        "Hover for raw values. Toggle series via the legend.</p>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for i, col in enumerate(cols):
        series = df[col].dropna()
        if len(series) < 2:
            continue
        norm = (series - series.mean()) / (series.std() + 1e-8)
        fig.add_trace(go.Scatter(
            x=series.index,
            y=norm,
            name=col,
            mode="lines+markers",
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
            marker=dict(size=3),
            hovertemplate=(
                f"<b>{col}</b><br>"
                "Year: %{x}<br>"
                "Raw: %{customdata:.2f}<extra></extra>"
            ),
            customdata=series.values,
        ))

    fig.update_layout(**_base_layout(
        theme,
        height=450,
        xaxis_title="Year",
        yaxis_title="Normalised Value (z-score)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=9, color=theme.text_muted)),
    ))
    st.plotly_chart(fig, use_container_width=True)

    _interpretation_box(
        "How to Read",
        "All indicators are z-score normalised (mean=0, std=1). "
        "Lines that track together = correlated. "
        "One line leading another by 1-3 years may indicate causation.\n\n"
        "**Patterns:** co-movement = positive correlation, "
        "mirroring = negative correlation, "
        "leading = potential Granger causality.",
        theme,
    )

    # Individual sparklines
    with st.expander("Individual indicator charts (raw values)", expanded=False):
        per_row = 3
        for row_start in range(0, len(cols), per_row):
            row_cols = st.columns(per_row)
            for j, c in enumerate(cols[row_start : row_start + per_row]):
                with row_cols[j]:
                    series = df[c].dropna()
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=series.index,
                        y=series,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(
                            color=PALETTE[(row_start + j) % len(PALETTE)],
                            width=1.5,
                        ),
                    ))
                    fig2.update_layout(**_base_layout(
                        theme,
                        height=180,
                        margin=dict(l=10, r=10, t=30, b=10),
                        title=dict(
                            text=c[:40],
                            font=dict(size=9, color=theme.text_secondary),
                        ),
                        showlegend=False,
                    ))
                    fig2.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig2, use_container_width=True)


def _render_correlation_heatmap(df: pd.DataFrame, cols: List[str], theme):
    """Correlation heatmap."""
    st.markdown(
        '<div class="section-header">CORRELATION MATRIX</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Pearson correlations. Red = positive, Blue = negative. "
        "Values near +/-1.0 = strong linear relationship.</p>",
        unsafe_allow_html=True,
    )

    corr = df[cols].corr()

    # Cluster for readability
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist = squareform(1 - corr.abs().values, checks=False)
        Z = linkage(dist, method="average")
        order = leaves_list(Z)
        ordered_cols = [corr.columns[i] for i in order]
        corr = corr.loc[ordered_cols, ordered_cols]
    except Exception:
        pass

    text_matrix = [
        [f"{corr.iloc[i, j]:.2f}" for j in range(len(corr.columns))]
        for i in range(len(corr.index))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title=dict(text="r", font=dict(color=theme.text_muted))),
        )
    )
    fig.update_layout(**_base_layout(
        theme,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=45, tickfont=dict(color=theme.text_muted)),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Top correlations
    pairs_list = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            pairs_list.append({
                "Var A": corr.index[i],
                "Var B": corr.columns[j],
                "Correlation": round(corr.iloc[i, j], 3),
                "Strength": _strength_label(corr.iloc[i, j]),
            })
    pairs_df = pd.DataFrame(pairs_list).sort_values(
        "Correlation", key=abs, ascending=False
    )

    st.markdown(
        '<div class="section-header" style="font-size:0.85rem;">'
        "STRONGEST CORRELATIONS</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)

    _interpretation_box("Interpretation", _interpret_correlations(pairs_df), theme)


def _render_granger_section(df: pd.DataFrame, cols: List[str], theme):
    """Granger causality."""
    st.markdown(
        '<div class="section-header">GRANGER CAUSALITY TESTS</div>',
        unsafe_allow_html=True,
    )

    if not HAS_STATSMODELS:
        st.error("Install statsmodels: pip install statsmodels")
        return

    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Tests whether past values of X improve predictions of Y beyond "
        "Y's own past. <b>p &lt; 0.05</b> = significant predictive causality.</p>",
        unsafe_allow_html=True,
    )

    max_lag = st.slider("Maximum lag (years)", 1, 8, 5, key="granger_lag")
    granger_df = run_granger_analysis(df, cols, max_lag)

    if granger_df.empty:
        st.warning(
            "Insufficient data for Granger tests. "
            "Select indicators with more overlapping years."
        )
        return

    sig = granger_df[granger_df["Significant"]]
    non_sig = granger_df[~granger_df["Significant"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs Tested", len(granger_df))
    c2.metric(
        "Significant (p<0.05)",
        len(sig),
        delta=f"{len(sig) / max(len(granger_df), 1) * 100:.0f}%",
    )
    c3.metric("Non-significant", len(non_sig))

    if not sig.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{r['Cause']} -> {r['Effect']}" for _, r in sig.iterrows()],
            y=sig["F-statistic"],
            marker_color=theme.accent_success,
            text=[
                f"p={r['p-value']:.3f}, lag={r['Best Lag (yrs)']}"
                for _, r in sig.iterrows()
            ],
            textposition="outside",
            textfont=dict(
                size=9,
                color=theme.text_muted,
                family="Space Mono, monospace",
            ),
        ))
        fig.update_layout(**_base_layout(
            theme,
            height=max(350, len(sig) * 35),
            margin=dict(l=20, r=20, t=30, b=100),
            xaxis_title="Causal Pair",
            yaxis_title="F-statistic",
            xaxis=dict(tickangle=45, tickfont=dict(color=theme.text_muted)),
            title=dict(
                text="Significant Granger-Causal Relationships",
                font=dict(color=theme.text_secondary, size=13),
            ),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full Granger results table"):
        st.dataframe(granger_df, use_container_width=True, hide_index=True)

    _interpretation_box(
        "Granger Causality Interpretation",
        _interpret_granger(sig, non_sig),
        theme,
    )


def _render_causal_network(df: pd.DataFrame, cols: List[str], theme):
    """Causal network graph."""
    st.markdown(
        '<div class="section-header">CAUSAL NETWORK GRAPH</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Nodes = indicators. Arrows = Granger-causal links (p&lt;0.05). "
        "Arrow thickness = evidence strength. Node size = connections.</p>",
        unsafe_allow_html=True,
    )

    granger_df = run_granger_analysis(df, cols, 5)
    sig = (
        granger_df[granger_df["Significant"]]
        if not granger_df.empty
        else pd.DataFrame()
    )

    if sig.empty:
        st.info(
            "No significant Granger-causal links found. "
            "Try adding more indicators or adjusting lag."
        )
        return

    nodes_set = set()
    edges = []
    for _, r in sig.iterrows():
        nodes_set.add(r["Cause"])
        nodes_set.add(r["Effect"])
        edges.append(
            (r["Cause"], r["Effect"], r["F-statistic"], r["Best Lag (yrs)"])
        )

    node_list = sorted(nodes_set)
    node_idx = {n: i for i, n in enumerate(node_list)}
    conn_count = {n: 0 for n in node_list}
    for src, tgt, *_ in edges:
        conn_count[src] += 1
        conn_count[tgt] += 1

    n = len(node_list)
    angles = [2 * np.pi * i / n for i in range(n)]
    x_pos = [np.cos(a) for a in angles]
    y_pos = [np.sin(a) for a in angles]

    fig = go.Figure()

    max_f = max(e[2] for e in edges) if edges else 1
    for src, tgt, fstat, lag in edges:
        i, j = node_idx[src], node_idx[tgt]
        width = 1 + (fstat / max_f) * 4
        opacity = 0.3 + 0.5 * fstat / max_f
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[j], None],
            y=[y_pos[i], y_pos[j], None],
            mode="lines",
            line=dict(width=width, color=f"rgba(0, 255, 136, {opacity})"),
            hoverinfo="text",
            text=f"{src} -> {tgt}<br>F={fstat:.1f}, Lag={lag}yr",
            showlegend=False,
        ))
        fig.add_annotation(
            ax=x_pos[i], ay=y_pos[i],
            x=x_pos[j], y=y_pos[j],
            axref="x", ayref="y", xref="x", yref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=1.5,
            arrowcolor="rgba(0,255,136,0.5)",
        )

    sizes = [15 + conn_count[n_item] * 8 for n_item in node_list]
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=[PALETTE[i % len(PALETTE)] for i in range(n)],
            line=dict(width=2, color="rgba(255,255,255,0.3)"),
        ),
        text=[nm[:25] for nm in node_list],
        textposition="top center",
        textfont=dict(
            size=9,
            color=theme.text_primary,
            family="Space Mono, monospace",
        ),
        hovertemplate="%{text}<br>Connections: %{customdata}<extra></extra>",
        customdata=[conn_count[n_item] for n_item in node_list],
        showlegend=False,
    ))

    fig.update_layout(**_base_layout(
        theme,
        height=550,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_cross_corr(df: pd.DataFrame, cols: List[str], theme):
    """Cross-correlation analysis."""
    st.markdown(
        '<div class="section-header">CROSS-CORRELATION ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "How two series correlate at different time lags. "
        "Peak at <b>lag &gt; 0</b> = first variable leads the second.</p>",
        unsafe_allow_html=True,
    )

    max_lag = st.slider("Max lag (years)", 3, 15, 8, key="ccf_lag")
    ccf_results = compute_cross_correlations(df, cols, max_lag)

    if not ccf_results:
        st.warning("Insufficient data for cross-correlation analysis.")
        return

    pair_labels = [f"{a}  /  {b}" for (a, b) in ccf_results.keys()]
    selected_pair = st.selectbox("Select pair", pair_labels)

    if selected_pair:
        pair_key = list(ccf_results.keys())[pair_labels.index(selected_pair)]
        ccf = ccf_results[pair_key]
        lags = list(range(-max_lag, max_lag + 1))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lags,
            y=ccf,
            marker_color=[
                theme.accent_success if abs(c) > 0.5
                else theme.accent_warning if abs(c) > 0.3
                else theme.text_muted
                for c in ccf
            ],
            hovertemplate="Lag: %{x} yrs<br>Correlation: %{y:.3f}<extra></extra>",
        ))

        n_eff = len(df[[pair_key[0], pair_key[1]]].dropna())
        sig_level = 1.96 / np.sqrt(n_eff) if n_eff > 0 else 0.3
        fig.add_hline(
            y=sig_level,
            line_dash="dash",
            line_color="rgba(255,100,100,0.5)",
            annotation_text=f"95% sig ({sig_level:.2f})",
            annotation_font=dict(color=theme.text_muted, size=9),
        )
        fig.add_hline(
            y=-sig_level,
            line_dash="dash",
            line_color="rgba(255,100,100,0.5)",
        )

        fig.update_layout(**_base_layout(
            theme,
            height=350,
            xaxis_title=(
                f"Lag (years) -- Positive = {pair_key[0][:20]} leads "
                f"{pair_key[1][:20]}"
            ),
            yaxis_title="Correlation",
            title=dict(
                text=f"Cross-Correlation: {pair_key[0][:25]} / {pair_key[1][:25]}",
                font=dict(color=theme.text_secondary, size=12),
            ),
        ))
        st.plotly_chart(fig, use_container_width=True)

        peak_idx = int(np.argmax(np.abs(ccf)))
        peak_lag = lags[peak_idx]
        peak_val = ccf[peak_idx]
        lead = pair_key[0] if peak_lag > 0 else pair_key[1]
        follows = pair_key[1] if peak_lag > 0 else pair_key[0]

        strength = (
            "Strong" if abs(peak_val) > 0.5
            else "Moderate" if abs(peak_val) > 0.3
            else "Weak"
        )
        direction = "positive" if peak_val > 0 else "negative"

        if peak_lag != 0:
            detail = (
                f"This suggests that **{lead}** leads **{follows}** by about "
                f"**{abs(peak_lag)} year(s)**. Changes in {lead} today may "
                f"predict changes in {follows} {abs(peak_lag)} year(s) later."
            )
        else:
            detail = (
                "The strongest correlation is at lag 0 (contemporaneous), "
                "suggesting these variables move together in the same year."
            )

        _interpretation_box(
            "Interpretation",
            f"**Peak correlation: r = {peak_val:.3f} at lag = {peak_lag} years**\n\n"
            f"{strength} {direction} correlation.\n\n{detail}",
            theme,
        )


def _render_stationarity(df: pd.DataFrame, cols: List[str], theme):
    """ADF stationarity tests."""
    st.markdown(
        '<div class="section-header">'
        "STATIONARITY TESTS (AUGMENTED DICKEY-FULLER)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Stationarity is a prerequisite for Granger causality. "
        "<b>p &lt; 0.05</b> = stationary (no unit root). "
        "Non-stationary series should be differenced.</p>",
        unsafe_allow_html=True,
    )

    if not HAS_STATSMODELS:
        st.error("Install statsmodels for stationarity tests.")
        return

    adf_df = run_stationarity_tests(df, cols)
    if adf_df.empty:
        return

    st.dataframe(adf_df, use_container_width=True, hide_index=True)

    stationary = adf_df[adf_df["Stationary"]]["Variable"].tolist()
    non_stationary = adf_df[~adf_df["Stationary"]]["Variable"].tolist()

    warning = (
        "Non-stationary series may produce spurious Granger results. "
        "Consider first-differencing for more reliable causality tests."
        if non_stationary
        else "All selected series are stationary -- Granger tests are reliable."
    )

    _interpretation_box(
        "Stationarity Interpretation",
        f"**Stationary** ({len(stationary)}): "
        f"{', '.join(stationary) or 'None'}\n\n"
        f"**Non-stationary** ({len(non_stationary)}): "
        f"{', '.join(non_stationary) or 'None'}\n\n{warning}",
        theme,
    )


def _render_scatter_matrix(df: pd.DataFrame, cols: List[str], theme):
    """Scatter matrix."""
    st.markdown(
        '<div class="section-header">SCATTER MATRIX (PAIR PLOT)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Each cell shows the pairwise relationship. "
        "Diagonals show distributions.</p>",
        unsafe_allow_html=True,
    )

    max_options = list(range(2, min(len(cols), 8) + 1))
    default_cols = min(len(cols), 6)

    if len(max_options) <= 1:
        max_cols = max_options[0] if max_options else len(cols)
        st.caption(f"Showing all {max_cols} selected indicators.")
    else:
        max_cols = st.select_slider(
            "Max indicators to display",
            options=max_options,
            value=default_cols,
            key="scatter_max",
        )
    plot_cols = cols[:max_cols]

    fig = px.scatter_matrix(
        df[plot_cols].dropna(),
        dimensions=plot_cols,
        color_discrete_sequence=[theme.accent_primary],
        opacity=0.6,
    )
    fig.update_layout(**_base_layout(
        theme,
        height=150 * len(plot_cols),
        margin=dict(l=40, r=20, t=20, b=20),
        font=dict(
            family="Space Mono, monospace", color=theme.text_muted, size=8
        ),
    ))
    fig.update_traces(diagonal_visible=True, marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_scalar_estimate(value) -> float:
    """Collapse scalar/list/array estimates into a single display value."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.nanmean(value.astype(float)))
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                return 0.0
            return float(np.nanmean(arr))
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_ci_bounds(ci) -> Tuple[Optional[float], Optional[float]]:
    """Extract scalar lower/upper bounds from CI payload."""
    if ci is None:
        return None, None
    if not isinstance(ci, (list, tuple)) or len(ci) < 2:
        return None, None
    low = _to_scalar_estimate(ci[0])
    high = _to_scalar_estimate(ci[1])
    if low > high:
        low, high = high, low
    return low, high


def _effect_direction(value: float, tol: float = 1e-9) -> str:
    """Map scalar effect to direction label."""
    if value > tol:
        return "Increase"
    if value < -tol:
        return "Decrease"
    return "Near-zero"


def _strength_label(r: float) -> str:
    ar = abs(r)
    if ar >= 0.8:
        return "Very Strong"
    if ar >= 0.6:
        return "Strong"
    if ar >= 0.4:
        return "Moderate"
    if ar >= 0.2:
        return "Weak"
    return "Negligible"


def _interpret_correlations(pairs_df: pd.DataFrame) -> str:
    if pairs_df.empty:
        return "No correlations computed."

    lines = ["**Top findings:**\n"]
    for _, row in pairs_df.head(5).iterrows():
        r = row["Correlation"]
        direction = "positively" if r > 0 else "negatively"
        lines.append(
            f"- **{row['Var A']}** and **{row['Var B']}** are "
            f"{row['Strength'].lower()} {direction} correlated "
            f"(r = {r:.3f})."
        )

    neg = pairs_df[pairs_df["Correlation"] < -0.5]
    if not neg.empty:
        lines.append("\n**Notable inverse relationships:**")
        for _, row in neg.head(3).iterrows():
            lines.append(
                f"- {row['Var A']} / {row['Var B']} "
                f"(r={row['Correlation']:.3f}): "
                "as one increases, the other decreases."
            )

    lines.append(
        "\nRemember: **correlation does not equal causation**. "
        "Check the Granger tab for directional evidence."
    )
    return "\n".join(lines)


def _interpret_granger(sig: pd.DataFrame, non_sig: pd.DataFrame) -> str:
    if sig.empty:
        return (
            "No statistically significant Granger-causal relationships found. "
            "Possible reasons: (1) indicators are truly independent, "
            "(2) nonlinear relationship, (3) insufficient data points."
        )

    lines = [f"**{len(sig)}** significant causal relationships found:\n"]
    for _, r in sig.head(8).iterrows():
        lines.append(
            f"- **{r['Cause']}** -> **{r['Effect']}** "
            f"(F={r['F-statistic']:.1f}, p={r['p-value']:.4f}, "
            f"lag={r['Best Lag (yrs)']}yr). "
            f"Past values of {r['Cause']} significantly improve "
            f"predictions of {r['Effect']}."
        )

    if len(sig) >= 2:
        causes = set(sig["Cause"])
        effects = set(sig["Effect"])
        both = causes & effects
        if both:
            lines.append(
                f"\n**Bidirectional causality** detected for: "
                f"{', '.join(both)}. "
                "This suggests feedback loops."
            )

    lines.append(
        "\n**Policy insight:** Variables that Granger-cause others are "
        "potential leading indicators and policy levers."
    )
    return "\n".join(lines)


def _interpretation_box(title: str, content: str, theme):
    """Styled interpretation box matching SENTINEL glass card theme."""
    st.markdown(
        f"""<div style="
            background: {theme.bg_card};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {theme.border_default};
            border-left: 3px solid {theme.accent_primary};
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin: 1rem 0;
            font-size: 0.85rem;
            font-family: Space Mono, monospace;
            color: {theme.text_secondary};
        ">
        <div style="font-weight:600; margin-bottom:0.5rem;
                     color:{theme.accent_primary};
                     font-size:0.9rem; text-transform:uppercase;
                     letter-spacing:0.5px;">
            {title}
        </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(content)


def _render_structural_tutorial(theme) -> None:
    """Explain what structural inference does, and why this module exists."""
    with st.expander("Tutorial: What Is Happening Here, And Why", expanded=True):
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:{theme.text_secondary};">
            <b>Why this exists</b><br>
            Correlation and trend charts are useful, but they do not answer:
            <i>if we change X, what happens to Y?</i>
            This structural module exists to estimate directional causal effects for policy use.
            <br><br>
            <b>What analysts are inspecting here</b><br>
            - Is the estimated effect direction stable across estimands?<br>
            - Do confidence intervals exclude zero (non-null evidence)?<br>
            - Do subgroup effects (CATE/ITE) differ from population effects (ATE/ATT/ATC)?<br>
            - Are failures assumption-related (missing IV/mediator, poor overlap, weak data)?
            <br><br>
            <b>What this tab does (pipeline)</b><br>
            1. Choose a causal question: treatment (cause) and outcome (effect).<br>
            2. Add assumptions: confounders, optional instrument, optional mediator.<br>
            3. Run multiple estimands in Scarcity (ATE/ATT/ATC/CATE/LATE/Mediation).<br>
            4. Execute in parallel (thread/process) for speed and robustness checks.<br>
            5. Compare results: direction, magnitude, confidence intervals, agreement.
            <br><br>
            <b>Why multiple estimands</b><br>
            Different estimands answer different policy questions:
            population-wide effect (ATE), treated-group effect (ATT), subgroup effect (CATE), etc.
            Agreement across estimands increases confidence; disagreement signals model risk.
            <br><br>
            <b>How to read outputs</b><br>
            - <b>Estimate sign</b>: positive means treatment tends to increase outcome, negative means decrease.<br>
            - <b>CI excludes 0</b>: stronger evidence the effect is not null.<br>
            - <b>Sign agreement</b>: if estimands disagree on direction, treat conclusions as tentative.
            <br><br>
            <b>Important caveat</b><br>
            Causal output quality depends on assumptions and data quality. This is decision support,
            not automatic truth. Use domain review before operational decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL INFERENCE (DoWhy + EconML)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_structural_inference(df: pd.DataFrame, cols: List[str], theme):
    """Structural causal analysis powered by Scarcity's multi-spec engine."""
    st.markdown('<div class="section-header">STRUCTURAL CAUSAL MODELING</div>',
                unsafe_allow_html=True)

    if not HAS_STRUCTURAL:
        missing_core = []
        if not HAS_DOWHY:
            missing_core.append("dowhy")
        if not HAS_SKLEARN:
            missing_core.append("scikit-learn")
        st.error(
            "Missing core libraries: "
            + ", ".join(missing_core)
            + ". Install them to use structural inference."
        )
        with st.expander("Dependency diagnostics", expanded=False):
            for dep, err in STRUCTURAL_IMPORT_ERRORS.items():
                st.code(f"{dep}: {err}")
        return

    try:
        from scarcity.causal.engine import run_causal
        from scarcity.causal.specs import (
            EstimandSpec,
            EstimandType,
            FailPolicy,
            ParallelismMode,
            RuntimeSpec,
            TimeSeriesPolicy,
        )
    except Exception as exc:
        st.error(f"Scarcity causal engine unavailable: {exc}")
        return

    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.8rem;">'
        'Runs Scarcity production causal inference with <b>multiple estimands</b> '
        'and configurable <b>parallel execution</b>. '
        'This replaces single-estimate ad-hoc DoWhy execution in this tab.</p>',
        unsafe_allow_html=True,
    )
    _render_structural_tutorial(theme)

    graph_dot = _build_graph_from_granger(df, cols)
    with st.expander("Auto-discovered DAG (from Granger, cycle-pruned)", expanded=False):
        st.graphviz_chart(graph_dot)
        st.caption(
            "Displayed for analyst context. Scarcity estimation below runs from selected "
            "spec fields and runtime settings."
        )

    c1, c2 = st.columns(2)
    with c1:
        treatment = st.selectbox("Treatment (Cause)", cols, key="scarcity_T")
    with c2:
        outcome_options = [c for c in cols if c != treatment]
        outcome = st.selectbox("Outcome (Effect)", outcome_options, key="scarcity_Y")

    if not treatment or not outcome:
        return

    available_covars = [c for c in cols if c not in (treatment, outcome)]
    confounders = st.multiselect(
        "Confounders",
        available_covars,
        default=available_covars[: min(3, len(available_covars))],
        key="scarcity_confounders",
    )
    effect_modifiers = st.multiselect(
        "Effect Modifiers (for CATE/ITE)",
        [c for c in available_covars if c not in confounders],
        default=[],
        key="scarcity_effect_modifiers",
    )

    iv_candidates = [None] + [c for c in available_covars if c not in effect_modifiers]
    mediator_candidates = [None] + [c for c in available_covars if c not in effect_modifiers]
    c3, c4 = st.columns(2)
    with c3:
        instrument = st.selectbox("Instrument (optional, for LATE)", iv_candidates, key="scarcity_iv")
    with c4:
        mediator = st.selectbox("Mediator (optional, for mediation)", mediator_candidates, key="scarcity_mediator")

    st.markdown("**Choose Effect Types (Estimands) — Plain-English Labels**")
    estimand_help = [
        ("ATE", "Average effect for everyone"),
        ("ATT", "Average effect for treated group"),
        ("ATC", "Average effect for untreated/control group"),
        ("CATE", "Effect by subgroup (requires econml)"),
        ("ITE", "Individual-level effect estimate (requires econml)"),
        ("LATE", "Effect for instrument-moved group (needs Instrument)"),
        ("MEDIATION_NDE", "Direct effect not through mediator (needs Mediator)"),
        ("MEDIATION_NIE", "Indirect effect through mediator (needs Mediator)"),
    ]
    if not HAS_ECONML:
        st.info("econml is not installed: CATE and ITE are hidden.")
        estimand_help = [item for item in estimand_help if item[0] not in ("CATE", "ITE")]

    estimand_labels = [f"{code} — {desc}" for code, desc in estimand_help]
    default_codes = ["ATE", "ATT", "ATC"]
    if HAS_ECONML and effect_modifiers:
        default_codes.append("CATE")
    default_labels = [label for label in estimand_labels if label.split(" — ", 1)[0] in default_codes]

    selected_labels = st.multiselect(
        f"Effect types ({len(estimand_labels)} available)",
        estimand_labels,
        default=default_labels,
        key="scarcity_estimands",
    )
    selected_estimands = [label.split(" — ", 1)[0] for label in selected_labels]

    if not selected_estimands:
        st.warning("Select at least one estimand.")
        return
    st.caption(
        "Selected effect types: " + ", ".join(selected_estimands)
    )

    st.markdown("**Runtime Configuration**")
    c5, c6, c7 = st.columns(3)
    with c5:
        parallelism_options = {
            "Safe Single Mode (most stable)": ParallelismMode.NONE.value,
            "Thread Mode (balanced)": ParallelismMode.THREAD.value,
            "Process Mode (fastest, may fail on some systems)": ParallelismMode.PROCESS.value,
        }
        parallelism_label = st.selectbox(
            "Execution Mode",
            list(parallelism_options.keys()),
            index=1,
            key="scarcity_parallelism",
        )
        parallelism = parallelism_options[parallelism_label]
    with c6:
        n_jobs = st.slider("Workers (n_jobs)", 1, 8, 2, key="scarcity_n_jobs")
    with c7:
        chunk_size = st.slider("Chunk size", 1, 8, 1, key="scarcity_chunk_size")

    c8, c9 = st.columns(2)
    with c8:
        fail_policy_options = {
            "Continue remaining effect types if one fails": FailPolicy.CONTINUE.value,
            "Stop immediately when one effect type fails": FailPolicy.FAIL_FAST.value,
        }
        fail_policy_label = st.selectbox(
            "Failure Handling",
            list(fail_policy_options.keys()),
            index=0,
            key="scarcity_fail_policy",
        )
        fail_policy = fail_policy_options[fail_policy_label]
    with c9:
        ts_policy_options = {
            "No strict time checks": TimeSeriesPolicy.NONE.value,
            "Warn on time-series issues": TimeSeriesPolicy.WARN.value,
            "Strictly enforce time checks": TimeSeriesPolicy.STRICT.value,
        }
        ts_policy_label = st.selectbox(
            "Time-Series Checks",
            list(ts_policy_options.keys()),
            index=0,
            key="scarcity_ts_policy",
        )
        ts_policy = ts_policy_options[ts_policy_label]

    with st.expander("What these runtime settings mean", expanded=False):
        st.markdown(
            """
            - **Execution Mode**:
              - Safe Single Mode: slowest, most reliable.
              - Thread Mode: faster, usually stable.
              - Process Mode: fastest, but can fail in some environments.
            - **Workers**: how many parallel workers to use.
            - **Chunk size**: how many effect types each worker takes at once.
            - **Failure Handling**:
              - Continue: run others even if one fails.
              - Stop immediately: halt on first failure.
            - **Time-Series Checks**:
              - No strict checks: permissive.
              - Warn: report issues but still try.
              - Strict: block invalid temporal setup.
            """
        )

    run_clicked = st.button("Run Scarcity Structural Inference", key="run_scarcity_structural")
    if not run_clicked:
        st.caption("Configure estimands and runtime, then click run.")
        return

    skipped = []
    specs = []
    for estimand_name in selected_estimands:
        estimand_type = EstimandType(estimand_name)
        if estimand_type in (EstimandType.CATE, EstimandType.ITE) and not HAS_ECONML:
            skipped.append(f"{estimand_name} (requires econml)")
            continue
        if estimand_type == EstimandType.LATE and not instrument:
            skipped.append("LATE (missing instrument)")
            continue
        if estimand_type in (EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE) and not mediator:
            skipped.append(f"{estimand_name} (missing mediator)")
            continue
        specs.append(
            EstimandSpec(
                treatment=treatment,
                outcome=outcome,
                confounders=list(confounders),
                effect_modifiers=list(effect_modifiers),
                instrument=instrument,
                mediator=mediator,
                type=estimand_type,
            )
        )

    if skipped:
        st.warning("Skipped estimands: " + ", ".join(skipped))
    if not specs:
        st.error("No valid estimands to run after validation checks.")
        return

    runtime = RuntimeSpec(
        parallelism=ParallelismMode(parallelism),
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        fail_policy=FailPolicy(fail_policy),
        time_series_policy=TimeSeriesPolicy(ts_policy),
        estimator_method=None,
    )

    with st.spinner(f"Running Scarcity on {len(specs)} estimand(s)..."):
        result = run_causal(df, specs, runtime)
    
    # Fallback for environments where process/thread workers fail (common in some Streamlit runtimes).
    fallback_used = False
    if (
        not result.results
        and result.errors
        and parallelism != ParallelismMode.NONE.value
    ):
        worker_like_failures = []
        for err in result.errors:
            stage = str(getattr(err, "stage", ""))
            msg = str(getattr(err, "message", "")).lower()
            ex_type = str(getattr(err, "exception_type", "")).lower()
            if (
                stage == "worker_execution"
                or "pickle" in msg
                or "spawn" in msg
                or "brokenprocesspool" in ex_type
                or "processpoolexecutor" in msg
            ):
                worker_like_failures.append(err)

        if worker_like_failures:
            st.warning(
                "Parallel worker execution failed in this runtime. "
                "Retrying automatically with sequential mode."
            )
            retry_runtime = RuntimeSpec(
                parallelism=ParallelismMode.NONE,
                n_jobs=1,
                chunk_size=1,
                fail_policy=FailPolicy(fail_policy),
                time_series_policy=TimeSeriesPolicy(ts_policy),
                estimator_method=None,
            )
            with st.spinner("Retrying in sequential mode..."):
                result = run_causal(df, specs, retry_runtime)
            fallback_used = True

    summary = result.summary
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Specs", summary.total_specs)
        m2.metric("Succeeded", summary.succeeded)
        m3.metric("Failed", summary.failed)
        m4.metric("Parallelism", summary.parallelism.upper())
        if fallback_used:
            st.caption("Execution note: initial parallel run failed; sequential fallback succeeded.")

    if result.errors:
        error_rows = [
            {
                "Index": err.index,
                "Spec": err.spec_id,
                "Stage": err.stage,
                "Type": err.exception_type,
                "Message": err.message,
            }
            for err in result.errors
        ]
        with st.expander(f"Errors ({len(error_rows)})", expanded=False):
            st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)

    if not result.results:
        st.error("No successful estimands. See error details above.")
        return

    rows = []
    for artifact in result.results:
        ci = artifact.confidence_intervals
        est = _to_scalar_estimate(artifact.estimate)
        ci_low, ci_high = _extract_ci_bounds(ci)
        ci_excludes_zero = (
            ci_low is not None and ci_high is not None and
            ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
        )
        rows.append(
            {
                "Estimand": artifact.estimand_type,
                "Treatment": artifact.spec.treatment,
                "Outcome": artifact.spec.outcome,
                "Estimate": est,
                "Direction": _effect_direction(est),
                "Magnitude": abs(est),
                "CI Low": ci_low,
                "CI High": ci_high,
                "CI Excludes 0": ci_excludes_zero,
                "CI": str(ci) if ci is not None else "N/A",
                "Backend": artifact.backend.get("name"),
                "Method": artifact.backend.get("method_name"),
            }
        )
    results_df = pd.DataFrame(rows).sort_values("Estimand")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    colors = []
    for v in results_df["Estimate"].tolist():
        if v > 0:
            colors.append(theme.accent_success)
        elif v < 0:
            colors.append(theme.accent_danger)
        else:
            colors.append(theme.text_muted)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["Estimand"],
        y=results_df["Estimate"],
        marker_color=colors,
        hovertemplate="Estimand: %{x}<br>Estimate: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        theme,
        height=320,
        xaxis_title="Estimand Type",
        yaxis_title="Estimated Effect",
        title=dict(text="Scarcity Structural Estimates", font=dict(color=theme.text_secondary, size=13)),
    ))
    st.plotly_chart(fig, use_container_width=True)

    pos = int((results_df["Estimate"] > 0).sum())
    neg = int((results_df["Estimate"] < 0).sum())
    zero = int((results_df["Estimate"] == 0).sum())
    total = len(results_df)
    dominant_direction = "increase" if pos >= neg else "decrease"
    agreement_ratio = (max(pos, neg) / max(1, pos + neg)) if (pos + neg) > 0 else 0.0

    robust_mask = results_df["CI Excludes 0"] == True
    robust_count = int(robust_mask.sum())
    robust_pct = (robust_count / total) * 100.0

    strongest = results_df.loc[results_df["Magnitude"].idxmax()]
    strongest_txt = (
        f"{strongest['Estimand']} ({strongest['Estimate']:+.4f}, {strongest['Direction'].lower()})"
    )

    headline = (
        f"Most estimands suggest an **{dominant_direction}** effect of **{treatment}** on **{outcome}** "
        f"(agreement: {agreement_ratio:.0%}; +:{pos}, -:{neg}, ~0:{zero})."
        if (pos + neg) > 0
        else f"Estimated effects are near zero across estimands for **{treatment} -> {outcome}**."
    )
    robustness = (
        f"Confidence intervals exclude zero for **{robust_count}/{total} estimands** ({robust_pct:.0f}%)."
    )
    caveat = (
        "Interpretation confidence is **low** because estimands disagree in sign."
        if pos > 0 and neg > 0
        else "Sign agreement is **coherent** across estimands."
    )

    _interpretation_box(
        "Estimand Interpretation",
        f"{headline}\n\n"
        f"Strongest estimated effect: **{strongest_txt}**.\n\n"
        f"{robustness}\n\n"
        f"{caveat}\n\n"
        "Decision guidance: prioritize action only when sign agreement is high and CI exclusion is broad.",
        theme,
    )

    _interpretation_box(
        "Execution Summary",
        f"Executed **{len(specs)} estimands** for **{treatment} -> {outcome}** using "
        f"Scarcity causal engine with **{parallelism}** parallelism.",
        theme,
    )


def _has_directed_path(
    adjacency: Dict[str, set], start: str, target: str
) -> bool:
    """Return True if a directed path exists from start to target."""
    if start == target:
        return True
    if start not in adjacency:
        return False
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, set()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _select_acyclic_granger_edges(
    sig: pd.DataFrame,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Keep strongest Granger edges while enforcing acyclicity (DAG).

    Edges are ranked by lowest p-value, then highest F-statistic.
    """
    if sig.empty:
        return [], 0

    ranked = sig.sort_values(["p-value", "F-statistic"], ascending=[True, False])
    adjacency: Dict[str, set] = {}
    kept: List[Tuple[str, str]] = []
    dropped_cycles = 0

    for _, row in ranked.iterrows():
        src = str(row["Cause"]).replace('"', "")
        tgt = str(row["Effect"]).replace('"', "")
        if not src or not tgt or src == tgt:
            continue
        # Adding src -> tgt creates a cycle iff tgt already reaches src.
        if _has_directed_path(adjacency, tgt, src):
            dropped_cycles += 1
            continue
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set())
        kept.append((src, tgt))

    return kept, dropped_cycles


def _build_graph_from_granger(df: pd.DataFrame, cols: List[str]) -> str:
    """Build a DAG-safe DOT graph string from Granger causality results."""
    if not HAS_STATSMODELS:
        nodes = '"; "'.join(cols)
        return f'digraph {{ "{nodes}"; }}'

    # Run Granger for all pairs (max lag 3 for graph structure)
    granger_df = run_granger_analysis(df, cols, max_lag=3)
    sig = granger_df[granger_df["Significant"]] if not granger_df.empty else pd.DataFrame()
    dag_edges, dropped_cycles = _select_acyclic_granger_edges(sig)

    # Start DOT string
    dot = ["digraph {"]
    dot.append('  graph [rankdir=LR, bgcolor="transparent"];')
    dot.append('  node [fontname="Space Mono", shape=box, style=filled, '
               'fillcolor="#1f331d", fontcolor="white", color="#00ff88"];')
    dot.append('  edge [color="#6e8a70", arrowsize=0.8];')

    if dag_edges:
        for src, tgt in dag_edges:
            dot.append(f'  "{src}" -> "{tgt}";')
    else:
        for c in cols:
            safe_c = c.replace('"', '')
            dot.append(f'  "{safe_c}";')

    if dropped_cycles:
        st.caption(
            f"Graph pruning: dropped {dropped_cycles} cyclic edge(s) to keep a valid DAG."
        )

    dot.append("}")
    return "\n".join(dot)

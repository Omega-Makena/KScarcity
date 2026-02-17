"""
K-SHIELD: Policy Terrain Sub-page

CSV-driven policy terrain analytics with no hardcoded indicator assumptions.
Data sources:
- World Bank Kenya CSV (default)
- User uploaded clean CSV
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

try:
    import numpy as np
    import pandas as pd

    HAS_DATA_STACK = True
except Exception:
    np = None
    pd = None
    HAS_DATA_STACK = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False

SHARED_DF_KEY = "kshield_shared_df"
SHARED_SOURCE_KEY = "kshield_shared_source"
SHARED_OWNER_KEY = "kshield_shared_owner"


def _set_shared_dataset(df: "pd.DataFrame", source: str, owner: str) -> None:
    """Publish a DataFrame for reuse across K-SHIELD cards in current session."""
    if not HAS_DATA_STACK or df is None or df.empty:
        return
    st.session_state[SHARED_DF_KEY] = df.copy(deep=True)
    st.session_state[SHARED_SOURCE_KEY] = source
    st.session_state[SHARED_OWNER_KEY] = owner


def _get_shared_dataset() -> Tuple[Optional["pd.DataFrame"], str]:
    """Read shared K-SHIELD DataFrame from session state."""
    if not HAS_DATA_STACK:
        return None, ""
    candidate = st.session_state.get(SHARED_DF_KEY)
    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
        source = str(st.session_state.get(SHARED_SOURCE_KEY, "Unknown source"))
        owner = str(st.session_state.get(SHARED_OWNER_KEY, "Unknown card"))
        return candidate, f"{source} via {owner}"
    return None, ""


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Loading World Bank data...")
def load_world_bank_data() -> "pd.DataFrame":
    """Auto-discover and load all indicators from the bundled World Bank CSV."""
    if not HAS_DATA_STACK:
        return pd.DataFrame()  # type: ignore[return-value]

    csv_path = _find_world_bank_csv()
    if csv_path is None:
        return pd.DataFrame()

    try:
        raw = pd.read_csv(csv_path, skiprows=4, encoding="utf-8-sig")
    except Exception:
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    return _to_timeseries_dataframe(raw)


def _find_world_bank_csv() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
        Path(os.getcwd()) / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _to_timeseries_dataframe(raw: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize World Bank or simple CSV format to year-indexed numeric frame."""
    if not HAS_DATA_STACK or raw is None or raw.empty:
        return pd.DataFrame()

    # World Bank wide format.
    if "Indicator Name" in raw.columns:
        year_cols = [
            c
            for c in raw.columns
            if isinstance(c, str) and c.strip().isdigit() and 1900 <= int(c.strip()) <= 2100
        ]
        if year_cols:
            melted = raw.melt(
                id_vars=["Indicator Name"],
                value_vars=year_cols,
                var_name="Year",
                value_name="Value",
            )
            melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
            melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
            melted = melted.dropna(subset=["Year"])
            pivoted = melted.pivot_table(index="Year", columns="Indicator Name", values="Value")
            pivoted.index = pivoted.index.astype(int)
            return _finalize_clean_frame(pivoted)

    # Simple format: first column is date/year, remaining are numeric series.
    if len(raw.columns) < 2:
        return pd.DataFrame()

    frame = raw.copy()
    idx_col = frame.columns[0]

    idx_num = pd.to_numeric(frame[idx_col], errors="coerce")
    if idx_num.notna().sum() >= max(3, int(0.5 * len(frame))):
        frame[idx_col] = idx_num
        frame = frame.dropna(subset=[idx_col]).set_index(idx_col)
    else:
        try:
            frame[idx_col] = pd.to_datetime(frame[idx_col], errors="coerce")
            frame = frame.dropna(subset=[idx_col]).set_index(idx_col)
            frame.index = frame.index.year
        except Exception:
            return pd.DataFrame()

    frame = frame.sort_index()
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    frame = frame[numeric_cols]
    return _finalize_clean_frame(frame)


def _finalize_clean_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    """Apply common cleaning rules to enforce analysis-ready data."""
    if not HAS_DATA_STACK or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]

    # Keep columns with enough support.
    good_cols = out.columns[out.notna().sum() >= 10]
    out = out[good_cols]
    if out.empty:
        return out

    out = out.interpolate(method="linear", limit=3, axis=0)
    out = out.replace([np.inf, -np.inf], np.nan)

    # Remove nearly constant columns (not useful for terrain estimation).
    usable_cols = []
    for c in out.columns:
        series = out[c].dropna()
        if len(series) < 10:
            continue
        if float(series.std()) < 1e-8:
            continue
        usable_cols.append(c)

    out = out[usable_cols]
    return out


def _validate_and_load_upload(uploaded_file) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """Validate and load uploaded CSV into clean year-indexed numeric frame."""
    if not HAS_DATA_STACK:
        return None, "pandas/numpy not installed"

    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as exc:
        return None, f"Could not parse CSV: {exc}"

    clean_df = _to_timeseries_dataframe(raw)
    if clean_df.empty:
        return None, (
            "CSV is not usable. Provide either: "
            "(1) World Bank format with 'Indicator Name' and year columns, or "
            "(2) first column as year/date and at least 2 numeric columns."
        )

    if clean_df.shape[1] < 2:
        return None, "Need at least 2 usable numeric columns after cleaning."

    return clean_df, None


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_z(values: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _ols_slope(x: "np.ndarray", y: "np.ndarray") -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    xz = _safe_z(x)
    yz = _safe_z(y)
    denom = float(np.dot(xz, xz))
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.dot(xz, yz) / denom)


def _ols_fit(X: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def _ols_predict(X: "np.ndarray", beta: "np.ndarray") -> "np.ndarray":
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])
    return Xd @ beta


def _fit_poly2_predict(
    x: "np.ndarray", y: "np.ndarray", grid: "np.ndarray", n_boot: int = 32
) -> Tuple["np.ndarray", "np.ndarray", Tuple[float, float, float]]:
    """Fit quadratic relation and return predictions + confidence map."""
    if len(x) < 8:
        return np.zeros_like(grid), np.zeros_like(grid), (0.0, 0.0, 0.0)

    xz = _safe_z(x)
    yz = _safe_z(y)

    try:
        coeff = np.polyfit(xz, yz, deg=2)
    except Exception:
        coeff = np.array([0.0, _ols_slope(x, y), 0.0])

    g = _safe_z(grid)
    pred = np.polyval(coeff, g)

    # Bootstrap uncertainty on predicted curve.
    rng = np.random.default_rng(7)
    sims: List[np.ndarray] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xz), len(xz))
        xb = xz[idx]
        yb = yz[idx]
        if len(np.unique(xb)) < 3:
            continue
        try:
            cb = np.polyfit(xb, yb, deg=2)
            sims.append(np.polyval(cb, g))
        except Exception:
            continue

    if sims:
        spread = np.nanstd(np.vstack(sims), axis=0)
        conf = 1.0 / (1.0 + spread)
    else:
        conf = np.full_like(pred, 0.4)

    return pred, np.clip(conf, 0.0, 1.0), (float(coeff[0]), float(coeff[1]), float(coeff[2]))


def _binned_surface(
    x: "np.ndarray", y: "np.ndarray", z: "np.ndarray", bins_x: int = 12, bins_y: int = 10
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Aggregate scattered x,y,z into a rectangular mean surface."""
    if len(x) < 5:
        gx = np.linspace(0.0, 1.0, 5)
        gy = np.linspace(0.0, 1.0, 5)
        return gx, gy, np.zeros((len(gy), len(gx)))

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(x_max - x_min) < 1e-8:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-8:
        y_min -= 0.5
        y_max += 0.5

    x_edges = np.linspace(x_min, x_max, bins_x + 1)
    y_edges = np.linspace(y_min, y_max, bins_y + 1)
    grid = np.full((bins_y, bins_x), np.nan)

    for i in range(bins_x):
        x_mask = (x >= x_edges[i]) & (x <= x_edges[i + 1] if i == bins_x - 1 else x < x_edges[i + 1])
        for j in range(bins_y):
            y_mask = (y >= y_edges[j]) & (y <= y_edges[j + 1] if j == bins_y - 1 else y < y_edges[j + 1])
            m = x_mask & y_mask
            if np.any(m):
                grid[j, i] = float(np.nanmean(z[m]))

    if np.isnan(grid).all():
        grid = np.zeros_like(grid)
    else:
        fill_val = float(np.nanmean(grid))
        grid = np.where(np.isnan(grid), fill_val, grid)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, grid


def _rolling_stat_pairs(series: "np.ndarray", window: int) -> Tuple["np.ndarray", "np.ndarray"]:
    """Return rolling variance and lag-1 autocorrelation."""
    if len(series) < window + 2:
        return np.array([]), np.array([])

    vars_: List[float] = []
    acf1: List[float] = []
    for i in range(window, len(series) + 1):
        seg = series[i - window : i]
        vars_.append(float(np.var(seg)))
        if np.std(seg[:-1]) < 1e-8 or np.std(seg[1:]) < 1e-8:
            acf1.append(0.0)
        else:
            acf1.append(float(np.corrcoef(seg[1:], seg[:-1])[0, 1]))
    return np.asarray(vars_), np.asarray(acf1)


def _pareto_frontier(points: "np.ndarray", maximize_x: bool, maximize_y: bool) -> "np.ndarray":
    """Compute Pareto frontier for 2D points."""
    if len(points) == 0:
        return points

    frontier: List[np.ndarray] = []
    for p in points:
        dominated = False
        for q in points:
            better_x = q[0] >= p[0] if maximize_x else q[0] <= p[0]
            better_y = q[1] >= p[1] if maximize_y else q[1] <= p[1]
            strict = (q[0] != p[0]) or (q[1] != p[1])
            if better_x and better_y and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    if not frontier:
        return np.empty((0, 2))
    arr = np.vstack(frontier)
    order = np.argsort(arr[:, 0])
    return arr[order]


# -----------------------------------------------------------------------------
# Analysis core
# -----------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Computing policy terrain analytics...")
def compute_policy_terrain_analytics(
    df: "pd.DataFrame",
    policy_cols: Sequence[str],
    outcome_cols: Sequence[str],
    shock_cols: Sequence[str],
    max_lag: int,
    grid_points: int,
    rolling_window: int,
) -> Dict[str, Any]:
    if not HAS_DATA_STACK:
        return {}

    frame = df.copy()
    frame = frame.sort_index()

    policy_cols = [c for c in policy_cols if c in frame.columns]
    outcome_cols = [c for c in outcome_cols if c in frame.columns]
    shock_cols = [c for c in shock_cols if c in frame.columns and c not in policy_cols and c not in outcome_cols]

    if not policy_cols or not outcome_cols:
        return {}

    primary_policy = policy_cols[0]
    x_raw = frame[primary_policy].to_numpy(dtype=float)
    valid_x = x_raw[np.isfinite(x_raw)]
    if len(valid_x) < 8:
        return {}

    policy_grid = np.linspace(float(np.nanmin(valid_x)), float(np.nanmax(valid_x)), grid_points)

    # Simple causal-network weight from policy-outcome absolute correlation.
    corr = frame[policy_cols + outcome_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    network_weight: Dict[str, float] = {}
    for o in outcome_cols:
        vals = [abs(float(corr.loc[p, o])) for p in policy_cols if p in corr.index and o in corr.columns]
        network_weight[o] = float(np.mean(vals)) if vals else 0.0

    max_w = max(network_weight.values()) if network_weight else 1.0
    if max_w < 1e-8:
        max_w = 1.0

    effect_surface = np.zeros((len(outcome_cols), grid_points))
    confidence_map = np.zeros((len(outcome_cols), grid_points))
    lag_surface = np.zeros((len(outcome_cols), max_lag + 1))
    sensitivity_surface = np.zeros((len(outcome_cols), grid_points))
    saturation_surface = np.zeros((len(outcome_cols), grid_points))

    estimand_rows: List[Dict[str, Any]] = []
    poly_coeffs: Dict[str, Tuple[float, float, float]] = {}

    for oi, out_col in enumerate(outcome_cols):
        subset = frame[[primary_policy, out_col]].dropna()
        if len(subset) < 10:
            continue

        x = subset[primary_policy].to_numpy(dtype=float)
        y = subset[out_col].to_numpy(dtype=float)

        # Estimand-style summary: level slope, differenced slope, best-lag slope.
        level = _ols_slope(x, y)

        dx = np.diff(x)
        dy = np.diff(y)
        diff = _ols_slope(dx, dy) if len(dx) >= 6 else 0.0

        lag_coefs: List[float] = []
        lag_corrs: List[float] = []
        for lag in range(0, max_lag + 1):
            if lag == 0:
                xl, yl = x, y
            else:
                xl, yl = x[:-lag], y[lag:]
            if len(xl) < 8:
                lag_coefs.append(0.0)
                lag_corrs.append(0.0)
                continue
            lag_coefs.append(_ols_slope(xl, yl))
            if np.std(xl) < 1e-8 or np.std(yl) < 1e-8:
                lag_corrs.append(0.0)
            else:
                lag_corrs.append(float(np.corrcoef(_safe_z(xl), _safe_z(yl))[0, 1]))

        best_idx = int(np.argmax(np.abs(np.asarray(lag_corrs)))) if lag_corrs else 0
        best_lag = int(best_idx)
        lag_effect = float(lag_coefs[best_idx]) if lag_coefs else 0.0

        estimates = np.asarray([level, diff, lag_effect], dtype=float)
        if np.allclose(estimates, 0.0):
            effect_scalar = 0.0
            agreement = 0.0
            consistency = 0.0
        else:
            effect_scalar = float(np.nanmean(estimates))
            signs = np.sign(estimates)
            agreement = float(np.mean(signs == np.sign(effect_scalar)))
            consistency = 1.0 / (1.0 + float(np.nanstd(estimates)))

        conf_scalar = float(np.clip(0.5 * agreement + 0.5 * consistency, 0.0, 1.0))
        n_weight = network_weight.get(out_col, 0.0) / max_w

        pred, conf_curve, coeff = _fit_poly2_predict(x, y, policy_grid)
        base_at_median = float(np.interp(float(np.nanmedian(x)), policy_grid, pred))
        surface = (pred - base_at_median) * (0.5 + 0.5 * n_weight)

        effect_surface[oi, :] = surface
        confidence_map[oi, :] = conf_curve * conf_scalar
        lag_surface[oi, :] = np.asarray(lag_coefs, dtype=float)

        grad = np.gradient(surface)
        sensitivity_surface[oi, :] = np.abs(grad)

        # derivative of polynomial wrt standardized policy axis
        g = _safe_z(policy_grid)
        deriv = 2.0 * coeff[0] * g + coeff[1]
        saturation_surface[oi, :] = deriv
        poly_coeffs[out_col] = coeff

        estimand_rows.append(
            {
                "Outcome": out_col,
                "ATE-like": round(level, 4),
                "Diff-effect": round(diff, 4),
                "Lag-effect": round(lag_effect, 4),
                "Best Lag": best_lag,
                "Combined Effect": round(effect_scalar, 4),
                "Confidence": round(conf_scalar, 4),
                "Network Weight": round(float(n_weight), 4),
            }
        )

    if not estimand_rows:
        return {}

    # Aggregate outcome series for temporal/risk models.
    outcome_frame = frame[outcome_cols].copy()
    outcome_frame = outcome_frame.interpolate(method="linear", axis=0)
    agg_outcome = _safe_z(outcome_frame.mean(axis=1).to_numpy(dtype=float))

    primary_series = frame[primary_policy].to_numpy(dtype=float)
    primary_z = _safe_z(primary_series)

    # Interaction terrain: policy x secondary driver.
    secondary_name = None
    if len(policy_cols) > 1:
        secondary_name = policy_cols[1]
    elif shock_cols:
        secondary_name = shock_cols[0]

    if secondary_name and secondary_name in frame.columns:
        inter_subset = frame[[primary_policy, secondary_name] + outcome_cols].dropna()
        x1 = _safe_z(inter_subset[primary_policy].to_numpy(dtype=float))
        x2 = _safe_z(inter_subset[secondary_name].to_numpy(dtype=float))
        y_inter = _safe_z(inter_subset[outcome_cols].mean(axis=1).to_numpy(dtype=float))
        X = np.column_stack([x1, x2, x1 * x2])
        beta = _ols_fit(X, y_inter)
        interaction_strength = float(beta[-1])

        p1g = np.linspace(float(np.nanmin(x1)), float(np.nanmax(x1)), 28)
        p2g = np.linspace(float(np.nanmin(x2)), float(np.nanmax(x2)), 24)
        P1, P2 = np.meshgrid(p1g, p2g)
        Xg = np.column_stack([P1.ravel(), P2.ravel(), (P1 * P2).ravel()])
        inter_pred = _ols_predict(Xg, beta).reshape(P1.shape)
    else:
        secondary_name = "Secondary Driver"
        p1g = np.linspace(-2, 2, 28)
        p2g = np.linspace(-2, 2, 24)
        P1, P2 = np.meshgrid(p1g, p2g)
        inter_pred = np.zeros_like(P1)
        interaction_strength = 0.0

    # Persistence / decay curve.
    target_outcome = outcome_cols[0]
    persist_subset = frame[[primary_policy, target_outcome]].dropna()
    px = _safe_z(persist_subset[primary_policy].to_numpy(dtype=float))
    py = _safe_z(persist_subset[target_outcome].to_numpy(dtype=float))
    decay_lags = np.arange(0, max_lag + 1)
    decay_vals = []
    for lag in decay_lags:
        if lag == 0:
            xl, yl = px, py
        else:
            xl, yl = px[:-lag], py[lag:]
        decay_vals.append(_ols_slope(xl, yl) if len(xl) >= 8 else 0.0)
    decay_vals = np.asarray(decay_vals, dtype=float)

    # Shock response slice via interaction with first shock (or secondary driver fallback).
    shock_name = shock_cols[0] if shock_cols else (secondary_name if secondary_name in frame.columns else None)
    if shock_name and shock_name in frame.columns:
        shock_surface = np.zeros((len(outcome_cols), grid_points))
        for oi, out_col in enumerate(outcome_cols):
            sub = frame[[primary_policy, shock_name, out_col]].dropna()
            if len(sub) < 12:
                continue
            sx = _safe_z(sub[primary_policy].to_numpy(dtype=float))
            sz = _safe_z(sub[shock_name].to_numpy(dtype=float))
            sy = _safe_z(sub[out_col].to_numpy(dtype=float))

            X = np.column_stack([sx, sz, sx * sz])
            beta = _ols_fit(X, sy)

            g = _safe_z(policy_grid)
            high = np.quantile(sz, 0.8)
            low = np.quantile(sz, 0.2)
            X_hi = np.column_stack([g, np.full_like(g, high), g * high])
            X_lo = np.column_stack([g, np.full_like(g, low), g * low])
            shock_surface[oi, :] = _ols_predict(X_hi, beta) - _ols_predict(X_lo, beta)
    else:
        shock_name = "Shock"
        shock_surface = np.zeros((len(outcome_cols), grid_points))

    # Structural break timeline on rolling slope.
    roll_subset = frame[[primary_policy] + outcome_cols].dropna()
    years = roll_subset.index.to_numpy()
    rx = _safe_z(roll_subset[primary_policy].to_numpy(dtype=float))
    ry = _safe_z(roll_subset[outcome_cols].mean(axis=1).to_numpy(dtype=float))

    centers: List[Any] = []
    roll_beta: List[float] = []
    w = max(6, min(rolling_window, max(6, len(rx) - 2)))
    for end in range(w, len(rx) + 1):
        xs = rx[end - w : end]
        ys = ry[end - w : end]
        centers.append(years[end - 1])
        roll_beta.append(_ols_slope(xs, ys))

    roll_beta_arr = np.asarray(roll_beta, dtype=float)
    if len(roll_beta_arr) >= 2:
        beta_change = np.abs(np.diff(roll_beta_arr))
        threshold = float(np.mean(beta_change) + 1.5 * np.std(beta_change))
        break_flags = beta_change > threshold
        break_years = [centers[i + 1] for i, f in enumerate(break_flags) if f]
    else:
        beta_change = np.array([])
        threshold = 0.0
        break_years = []

    # Failure probability surface.
    fail_subset_cols = [primary_policy] + outcome_cols
    second_axis_name = shock_name if shock_name in frame.columns else secondary_name
    if second_axis_name and second_axis_name in frame.columns:
        fail_subset_cols.append(second_axis_name)
    fail_subset = frame[fail_subset_cols].dropna()

    fx = _safe_z(fail_subset[primary_policy].to_numpy(dtype=float))
    if second_axis_name and second_axis_name in fail_subset.columns:
        fy = _safe_z(fail_subset[second_axis_name].to_numpy(dtype=float))
    else:
        fy = _safe_z(np.gradient(fx))
        second_axis_name = "Policy Volatility"

    f_out = _safe_z(fail_subset[outcome_cols].mean(axis=1).to_numpy(dtype=float))
    failure = (np.abs(f_out) > 1.0).astype(float)
    fail_xc, fail_yc, fail_z = _binned_surface(fx, fy, failure, bins_x=14, bins_y=12)

    # Basin of stability map from state speed.
    if len(outcome_cols) >= 2:
        s1 = _safe_z(frame[outcome_cols[0]].interpolate().dropna().to_numpy(dtype=float))
        s2 = _safe_z(frame[outcome_cols[1]].interpolate().dropna().to_numpy(dtype=float))
        m = min(len(s1), len(s2))
        s1, s2 = s1[:m], s2[:m]
    else:
        s1 = agg_outcome
        s2 = np.roll(agg_outcome, 1)

    if len(s1) >= 3:
        vx = np.diff(s1)
        vy = np.diff(s2)
        speed = np.sqrt(vx * vx + vy * vy)
        bas_x, bas_y, bas_speed = _binned_surface(s1[:-1], s2[:-1], speed, bins_x=18, bins_y=14)
        basin_map = np.exp(-bas_speed)
    else:
        bas_x = np.linspace(-2, 2, 10)
        bas_y = np.linspace(-2, 2, 8)
        basin_map = np.zeros((len(bas_y), len(bas_x)))

    # Early warning signal terrain.
    var_roll, acf_roll = _rolling_stat_pairs(agg_outcome, w)
    if len(var_roll) > 0:
        ews = _safe_z(var_roll) + np.maximum(_safe_z(acf_roll), 0)
        yr = frame.index.to_numpy()[w - 1 :]
        pol = _safe_z(primary_series[w - 1 :])
        warn_x, warn_y, warn_z = _binned_surface(yr.astype(float), pol, ews, bins_x=12, bins_y=10)
    else:
        warn_x = np.linspace(0, 1, 6)
        warn_y = np.linspace(-1, 1, 6)
        warn_z = np.zeros((6, 6))

    # Counterfactual delta surface (default +10% of policy std).
    policy_std = float(np.nanstd(primary_series))
    delta_abs_default = 0.1 * policy_std if policy_std > 1e-8 else 0.1
    cf_surface = np.zeros((len(outcome_cols), grid_points))
    for oi, out_col in enumerate(outcome_cols):
        coeff = poly_coeffs.get(out_col, (0.0, 0.0, 0.0))
        g = _safe_z(policy_grid)
        g2 = _safe_z(policy_grid + delta_abs_default)
        pred_base = np.polyval(np.asarray(coeff), g)
        pred_cf = np.polyval(np.asarray(coeff), g2)
        cf_surface[oi, :] = pred_cf - pred_base

    # Efficiency points for objective frontier.
    objective_points = np.zeros((grid_points, len(outcome_cols)))
    for oi, out_col in enumerate(outcome_cols):
        coeff = poly_coeffs.get(out_col, (0.0, 0.0, 0.0))
        objective_points[:, oi] = np.polyval(np.asarray(coeff), _safe_z(policy_grid))

    # Momentum field from observed dynamics.
    m_x = _safe_z(primary_series)
    m_y = agg_outcome
    if len(m_x) >= 3 and len(m_y) >= 3:
        n = min(len(m_x), len(m_y))
        m_x, m_y = m_x[:n], m_y[:n]
        mom = {
            "x": m_x[:-1],
            "y": m_y[:-1],
            "dx": np.diff(m_x),
            "dy": np.diff(m_y),
        }
    else:
        mom = {"x": np.array([]), "y": np.array([]), "dx": np.array([]), "dy": np.array([])}

    # Inequality response terrain via quantile treatment effects (Q1..Q5).
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ineq = np.zeros((len(outcome_cols), len(q_labels)))
    for oi, out_col in enumerate(outcome_cols):
        sub = frame[[primary_policy, out_col]].dropna()
        if len(sub) < 20:
            continue
        x = sub[primary_policy].to_numpy(dtype=float)
        y = sub[out_col].to_numpy(dtype=float)
        try:
            bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        except Exception:
            bins = pd.Series(np.zeros(len(y), dtype=int))
        for qi in range(5):
            mask = np.asarray(bins == qi)
            if np.sum(mask) < 6:
                continue
            ineq[oi, qi] = _ols_slope(x[mask], y[mask])

    return {
        "primary_policy": primary_policy,
        "secondary_name": secondary_name,
        "shock_name": shock_name,
        "second_axis_name": second_axis_name,
        "outcomes": outcome_cols,
        "policy_grid": policy_grid,
        "effect_surface": effect_surface,
        "confidence_map": confidence_map,
        "lag_surface": lag_surface,
        "sensitivity_surface": sensitivity_surface,
        "saturation_surface": saturation_surface,
        "estimands": pd.DataFrame(estimand_rows),
        "interaction": {
            "x": P1,
            "y": P2,
            "z": inter_pred,
            "strength": interaction_strength,
        },
        "persistence": {
            "lags": decay_lags,
            "values": decay_vals,
            "target_outcome": target_outcome,
        },
        "shock_slice": shock_surface,
        "structural_break": {
            "years": centers,
            "rolling_beta": roll_beta_arr,
            "changes": beta_change,
            "threshold": threshold,
            "break_years": break_years,
        },
        "failure": {
            "x": fail_xc,
            "y": fail_yc,
            "z": fail_z,
        },
        "basin": {
            "x": bas_x,
            "y": bas_y,
            "z": basin_map,
        },
        "warning": {
            "x": warn_x,
            "y": warn_y,
            "z": warn_z,
        },
        "counterfactual_surface": cf_surface,
        "counterfactual_delta_abs_default": delta_abs_default,
        "objective_points": objective_points,
        "momentum": mom,
        "inequality": {
            "quantiles": q_labels,
            "z": ineq,
        },
        "row_count": len(frame),
    }


# -----------------------------------------------------------------------------
# Render entry
# -----------------------------------------------------------------------------


def render_terrain(theme, data=None):
    """Render policy terrain workspace and guide."""
    st.markdown('<div class="section-header">POLICY TERRAIN &mdash; CAUSAL POLICY LANDSCAPES</div>', unsafe_allow_html=True)

    nav_mode = st.radio(
        "Terrain Navigation",
        ["Terrain Workspace", "Guide & Tutorial"],
        horizontal=True,
        key="terrain_nav_mode",
    )

    if nav_mode == "Guide & Tutorial":
        _render_terrain_guide(theme)
        return

    if not HAS_DATA_STACK:
        st.error("Missing libraries: pandas, numpy. Install them for Policy Terrain analytics.")
        return

    st.markdown("---")
    source = st.radio(
        "Data Source",
        ["World Bank (Kenya)", "Upload your own CSV", "Shared K-SHIELD Dataset"],
        horizontal=True,
        key="terrain_data_source",
    )

    df = pd.DataFrame()
    shared_df, shared_meta = _get_shared_dataset()
    if source == "World Bank (Kenya)":
        df = load_world_bank_data()
        if df.empty:
            st.error(
                "World Bank CSV not found in data/simulation/ or could not be parsed. "
                "Upload a clean CSV to continue."
            )
            return
        _set_shared_dataset(df, "World Bank (Kenya)", "TERRAIN")
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
        if "terrain_uploaded_df" in st.session_state:
            df = st.session_state["terrain_uploaded_df"]
        if df.empty:
            return
        _set_shared_dataset(df, "Uploaded CSV", "TERRAIN")

    _render_data_profile(df, theme)

    available = sorted(df.columns.tolist())
    if len(available) < 3:
        st.warning("Need at least 3 usable numeric columns to build policy terrains.")
        return

    defaults = _infer_default_roles(available)

    with st.expander("Model Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_cols = st.multiselect(
                "Policy levers (X candidates)",
                options=available,
                default=defaults["policies"],
                key="terrain_policy_cols",
            )
        with c2:
            outcome_cols = st.multiselect(
                "Outcome domains / sectors (Y candidates)",
                options=available,
                default=defaults["outcomes"],
                key="terrain_outcome_cols",
            )
        with c3:
            shock_cols = st.multiselect(
                "Shock / external drivers (optional)",
                options=available,
                default=defaults["shocks"],
                key="terrain_shock_cols",
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            max_lag = st.slider("Max lag for delay models", 1, 12, 6, key="terrain_max_lag")
        with c5:
            grid_points = st.slider("Policy intensity resolution", 16, 80, 36, key="terrain_grid_points")
        with c6:
            max_window = max(6, min(20, max(6, len(df) - 2)))
            if max_window <= 6:
                rolling_window = 6
                st.caption("Regime rolling window fixed at 6 (short dataset).")
            else:
                rolling_window = st.slider(
                    "Regime rolling window",
                    min_value=6,
                    max_value=max_window,
                    value=min(10, max_window),
                    key="terrain_roll_window",
                )

    if len(policy_cols) < 1:
        st.warning("Select at least one policy lever.")
        return
    if len(outcome_cols) < 2:
        st.warning("Select at least two outcome domains/sectors.")
        return

    analytics = compute_policy_terrain_analytics(
        df,
        tuple(policy_cols),
        tuple(outcome_cols),
        tuple(shock_cols),
        int(max_lag),
        int(grid_points),
        int(rolling_window),
    )
    if not analytics:
        st.warning("Unable to compute terrain analytics from current selections. Try different columns.")
        return

    _render_workspace_header(analytics, theme)
    _render_workspace_quick_guide()
    _render_logic_diagnostics(analytics, theme)
    c_mode_1, c_mode_2 = st.columns(2)
    with c_mode_1:
        focus_outcome = st.selectbox(
            "Interpretation focus (outcome domain)",
            analytics["outcomes"],
            key="terrain_focus_outcome",
        )
    with c_mode_2:
        st.radio(
            "Chart Style",
            ["3D-heavy", "2D Heatmaps"],
            horizontal=True,
            key="terrain_style_mode",
        )

    tabs = st.tabs([
        "Core Terrain",
        "Temporal Dynamics",
        "Risk & Stability",
        "Counterfactual Space",
    ])
    with tabs[0]:
        _render_core_terrain_graphs(analytics, theme, focus_outcome)
    with tabs[1]:
        _render_temporal_graphs(analytics, theme, focus_outcome)
    with tabs[2]:
        _render_risk_graphs(analytics, theme, focus_outcome)
    with tabs[3]:
        _render_counterfactual_graphs(analytics, theme, focus_outcome)


# -----------------------------------------------------------------------------
# UI render helpers
# -----------------------------------------------------------------------------


def _infer_default_roles(columns: List[str]) -> Dict[str, List[str]]:
    lower = [c.lower() for c in columns]

    def pick_by_keywords(keys: Sequence[str], limit: int) -> List[str]:
        picked = []
        for c, cl in zip(columns, lower):
            if any(k in cl for k in keys):
                picked.append(c)
        return picked[:limit]

    policy_keys = ["policy", "tax", "interest", "rate", "spend", "debt", "credit", "budget"]
    outcome_keys = ["gdp", "growth", "inflation", "unemployment", "poverty", "income", "trade"]
    shock_keys = ["shock", "price", "exchange", "commodity", "import", "export", "volatility"]

    policies = pick_by_keywords(policy_keys, 3)
    outcomes = pick_by_keywords(outcome_keys, 8)
    shocks = pick_by_keywords(shock_keys, 3)

    if not policies:
        policies = columns[:1]
    if not outcomes:
        outcomes = [c for c in columns if c not in policies][: min(5, max(2, len(columns) - len(policies)))]
    if len(outcomes) < 2:
        extras = [c for c in columns if c not in policies and c not in outcomes]
        outcomes.extend(extras[: 2 - len(outcomes)])

    shocks = [c for c in shocks if c not in policies and c not in outcomes]
    if not shocks:
        shocks = [c for c in columns if c not in policies and c not in outcomes][:1]

    return {"policies": policies, "outcomes": outcomes, "shocks": shocks}


def _render_upload_section(theme) -> None:
    st.markdown(
        f'<div style="background:{theme.bg_card}; border:1px solid {theme.border_default}; '
        f'border-radius:12px; padding:1rem; margin-bottom:1rem;">'
        f'<div style="color:{theme.text_secondary}; font-weight:600; font-size:0.85rem; '
        f'text-transform:uppercase; letter-spacing:0.4px; margin-bottom:0.4rem;">'
        "Upload Clean CSV</div>"
        f'<div style="color:{theme.text_muted}; font-size:0.8rem;">'
        "Accepted:<br>"
        "- World Bank CSV (Indicator Name + year columns)<br>"
        "- Simple CSV (first column year/date, remaining numeric series)"
        "</div></div>",
        unsafe_allow_html=True,
    )

    up = st.file_uploader("Upload CSV", type=["csv"], key="terrain_csv_upload")
    if up is None:
        return

    df, err = _validate_and_load_upload(up)
    if err:
        st.error(f"Validation failed: {err}")
        st.session_state.pop("terrain_uploaded_df", None)
        return

    st.session_state["terrain_uploaded_df"] = df
    _set_shared_dataset(df, "Uploaded CSV", "TERRAIN")
    st.success(
        f"Loaded {len(df.columns)} columns across {len(df)} rows "
        f"({df.index.min()} to {df.index.max()})."
    )
    with st.expander("Preview (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)


def _render_data_profile(df: "pd.DataFrame", theme) -> None:
    years_min = df.index.min()
    years_max = df.index.max()
    density = float(df.notna().sum().sum() / max(1, df.shape[0] * df.shape[1]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df)}")
    c2.metric("Numeric Series", f"{len(df.columns)}")
    c3.metric("Period", f"{years_min} - {years_max}")
    c4.metric("Data Density", f"{density*100:.1f}%")

    st.caption(
        "All terrains below are computed from selected columns in this dataset. "
        "No indicator names are hardcoded into the estimations."
    )


def _render_workspace_header(analytics: Dict[str, Any], theme) -> None:
    estimands = analytics["estimands"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary Policy", analytics["primary_policy"])
    c2.metric("Outcomes Modeled", f"{len(analytics['outcomes'])}")
    c3.metric("Estimands Built", f"{len(estimands)}")
    mean_conf = float(estimands["Confidence"].mean()) if not estimands.empty else 0.0
    c4.metric("Mean Confidence", f"{mean_conf:.2f}")


def _render_workspace_quick_guide():
    st.caption(
        "Read order: Effect Surface -> Confidence -> Lag -> Interaction -> Risk -> Counterfactuals."
    )


# -----------------------------------------------------------------------------
# 1) Core Terrain Graphs (5)
# -----------------------------------------------------------------------------


def _render_core_terrain_graphs(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    st.markdown("#### 1) Policy Effect Surface (Primary Terrain)")
    _plot_effect_surface(analytics, theme)
    _interpret_effect_surface(analytics, theme, focus_outcome)

    st.markdown("#### 2) Policy Stability / Confidence Map")
    _plot_confidence_map(analytics, theme)
    _interpret_confidence_map(analytics, theme, focus_outcome)

    st.markdown("#### 3) Policy Lag / Delay Surface")
    _plot_lag_surface(analytics, theme)
    _interpret_lag_surface(analytics, theme, focus_outcome)

    st.markdown("#### 4) Policy Interaction Terrain")
    _plot_interaction_surface(analytics, theme)
    _interpret_interaction(analytics, theme)

    st.markdown("#### 5) Sector Sensitivity Terrain")
    _plot_sensitivity_surface(analytics, theme)
    _interpret_sensitivity(analytics, theme, focus_outcome)

    st.markdown("**Estimand Summary**")
    st.dataframe(analytics["estimands"], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 2) Temporal Dynamics Graphs (3)
# -----------------------------------------------------------------------------


def _render_temporal_graphs(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    st.markdown("#### 6) Policy Persistence / Decay Curve")
    _plot_persistence_curve(analytics, theme)
    _interpret_persistence(analytics, theme)

    st.markdown("#### 7) Shock Response Terrain Slice")
    _plot_shock_slice(analytics, theme)
    _interpret_shock_slice(analytics, theme, focus_outcome)

    st.markdown("#### 8) Structural Break / Regime Shift Timeline")
    _plot_structural_break(analytics, theme)
    _interpret_structural_break(analytics, theme)


# -----------------------------------------------------------------------------
# 3) Risk and Stability Graphs (3)
# -----------------------------------------------------------------------------


def _render_risk_graphs(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    st.markdown("#### 9) Policy Failure Probability Surface")
    _plot_failure_surface(analytics, theme)
    _interpret_failure_surface(analytics, theme)

    st.markdown("#### 10) Basin of Stability Map")
    _plot_basin_map(analytics, theme)
    _interpret_basin(analytics, theme)

    st.markdown("#### 11) Early Warning Signal Terrain")
    _plot_warning_map(analytics, theme)
    _interpret_warning(analytics, theme)


# -----------------------------------------------------------------------------
# 4) Counterfactual and policy-space graphs (5)
# -----------------------------------------------------------------------------


def _render_counterfactual_graphs(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    st.markdown("#### 12) Counterfactual Policy Delta Surface")
    _plot_counterfactual_surface(analytics, theme)
    _interpret_counterfactual(analytics, theme, focus_outcome)

    st.markdown("#### 13) Policy Efficiency Frontier")
    _plot_efficiency_frontier(analytics, theme)
    _interpret_efficiency(analytics, theme)

    st.markdown("#### 14) Policy Momentum Field")
    _plot_momentum_field(analytics, theme)
    _interpret_momentum(analytics, theme)

    st.markdown("#### 15) Policy Saturation Surface")
    _plot_saturation_surface(analytics, theme)
    _interpret_saturation(analytics, theme, focus_outcome)

    st.markdown("#### 16) Policy Inequality Response Terrain")
    _plot_inequality_surface(analytics, theme)
    _interpret_inequality(analytics, theme, focus_outcome)


# -----------------------------------------------------------------------------
# Interpretation and diagnostics helpers
# -----------------------------------------------------------------------------


def _render_logic_diagnostics(analytics: Dict[str, Any], theme) -> None:
    """Surface quality checks so users know when terrain logic is weak."""
    estimands = analytics.get("estimands", pd.DataFrame())
    row_count = int(analytics.get("row_count", 0))
    mean_conf = float(estimands["Confidence"].mean()) if not estimands.empty else 0.0
    low_conf_share = (
        float((estimands["Confidence"] < 0.45).mean()) if not estimands.empty else 1.0
    )

    flags: List[str] = []
    if row_count < 18:
        flags.append("Low sample size: fewer than 18 rows can make surfaces unstable.")
    if mean_conf < 0.45:
        flags.append("Low average confidence: many effects may be weak/noisy.")
    if low_conf_share > 0.6:
        flags.append("Most estimands are low-confidence; interpret directions cautiously.")

    st.markdown(
        f"**Logic Check**: rows={row_count}, mean_confidence={mean_conf:.2f}, "
        f"low_conf_share={low_conf_share*100:.0f}%"
    )
    if flags:
        for msg in flags:
            st.warning(msg)
    else:
        st.success("Diagnostics look acceptable for exploratory terrain interpretation.")

    st.caption(
        "Core logic: each outcome uses level, differenced, and lagged effect estimates, "
        "then blends them with dependency strength from policy-outcome structure."
    )


def _outcome_idx(analytics: Dict[str, Any], focus_outcome: str) -> int:
    outcomes = list(analytics.get("outcomes", []))
    if not outcomes:
        return 0
    return outcomes.index(focus_outcome) if focus_outcome in outcomes else 0


def _policy_at(analytics: Dict[str, Any], idx: int) -> float:
    grid = np.asarray(analytics.get("policy_grid", []), dtype=float)
    if grid.size == 0:
        return 0.0
    i = int(np.clip(idx, 0, len(grid) - 1))
    return float(grid[i])


def _direction_label(v: float, tol: float = 1e-6) -> str:
    if v > tol:
        return "increase"
    if v < -tol:
        return "decrease"
    return "near-neutral"


def _interpretation_box(title: str, lines: List[str], theme) -> None:
    st.markdown(f"**{title}**")
    for line in lines:
        if line:
            st.markdown(f"- {line}")


def _interpret_effect_surface(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    z = np.asarray(analytics["effect_surface"], dtype=float)
    row = z[i] if z.ndim == 2 and i < z.shape[0] else np.array([0.0])
    peak_idx = int(np.argmax(np.abs(row))) if row.size else 0
    peak_val = float(row[peak_idx]) if row.size else 0.0
    pol = _policy_at(analytics, peak_idx)

    _interpretation_box(
        "Interpretation",
        [
            f"For `{focus_outcome}`, strongest modeled effect is `{_direction_label(peak_val)}` at policy intensity ~{pol:.3g}.",
            f"Effect magnitude at that point: {peak_val:+.3f} (standardized response units).",
            "Read this with confidence and lag charts before making policy conclusions.",
        ],
        theme,
    )


def _interpret_confidence_map(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    conf = np.asarray(analytics["confidence_map"], dtype=float)
    row = conf[i] if conf.ndim == 2 and i < conf.shape[0] else np.array([0.0])
    mean_c = float(np.mean(row)) if row.size else 0.0
    low_share = float(np.mean(row < 0.45)) if row.size else 1.0
    _interpretation_box(
        "Interpretation",
        [
            f"Mean confidence for `{focus_outcome}`: {mean_c:.2f}.",
            f"Low-confidence region share (<0.45): {low_share*100:.0f}%.",
            "If low-confidence share is high, treat effect direction as exploratory.",
        ],
        theme,
    )


def _interpret_lag_surface(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    lags = np.asarray(analytics["lag_surface"], dtype=float)
    row = lags[i] if lags.ndim == 2 and i < lags.shape[0] else np.array([0.0])
    best_lag = int(np.argmax(np.abs(row))) if row.size else 0
    best_val = float(row[best_lag]) if row.size else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"`{focus_outcome}` peaks at lag {best_lag} with `{_direction_label(best_val)}` effect ({best_val:+.3f}).",
            "Lag 0 indicates immediate response; higher lag indicates delayed transmission.",
        ],
        theme,
    )


def _interpret_interaction(analytics: Dict[str, Any], theme) -> None:
    s = float(analytics["interaction"]["strength"])
    mode = "reinforcing" if s > 0 else "offsetting" if s < 0 else "weak/neutral"
    _interpretation_box(
        "Interpretation",
        [
            f"Interaction strength is {s:+.3f}, indicating mostly {mode} policy interaction.",
            f"Secondary axis used: `{analytics['secondary_name']}`.",
        ],
        theme,
    )


def _interpret_sensitivity(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    sens = np.asarray(analytics["sensitivity_surface"], dtype=float)
    row = sens[i] if sens.ndim == 2 and i < sens.shape[0] else np.array([0.0])
    idx = int(np.argmax(row)) if row.size else 0
    _interpretation_box(
        "Interpretation",
        [
            f"`{focus_outcome}` is most sensitive around policy intensity ~{_policy_at(analytics, idx):.3g}.",
            f"Peak local sensitivity: {float(row[idx]) if row.size else 0.0:.3f}.",
            "This zone is where small policy changes may create large outcome shifts.",
        ],
        theme,
    )


def _interpret_persistence(analytics: Dict[str, Any], theme) -> None:
    vals = np.asarray(analytics["persistence"]["values"], dtype=float)
    if vals.size == 0:
        return
    base = abs(float(vals[0])) if vals.size else 0.0
    half_idx = next((k for k, v in enumerate(vals[1:], start=1) if abs(float(v)) <= 0.5 * base), None)
    half_txt = str(half_idx) if half_idx is not None else "not reached in tested lags"
    _interpretation_box(
        "Interpretation",
        [
            f"Initial impact: {vals[0]:+.3f}. Estimated half-decay lag: {half_txt}.",
            "Slow decay suggests persistent policy transmission; fast decay suggests short-lived effects.",
        ],
        theme,
    )


def _interpret_shock_slice(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    z = np.asarray(analytics["shock_slice"], dtype=float)
    row = z[i] if z.ndim == 2 and i < z.shape[0] else np.array([0.0])
    avg = float(np.mean(row)) if row.size else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"Under higher `{analytics['shock_name']}`, `{focus_outcome}` shifts on average toward `{_direction_label(avg)}` ({avg:+.3f}).",
            "Positive means high-shock scenarios amplify the modeled response vs low-shock baseline.",
        ],
        theme,
    )


def _interpret_structural_break(analytics: Dict[str, Any], theme) -> None:
    sb = analytics["structural_break"]
    years = sb.get("break_years", [])
    _interpretation_box(
        "Interpretation",
        [
            f"Detected regime shifts: {len(years)}.",
            f"Break years: {', '.join(map(str, years)) if years else 'none in current window'}.",
            "Around break years, older causal relationships may not extrapolate well.",
        ],
        theme,
    )


def _interpret_failure_surface(analytics: Dict[str, Any], theme) -> None:
    f = analytics["failure"]
    z = np.asarray(f["z"], dtype=float)
    if z.size == 0:
        return
    j, i = np.unravel_index(int(np.argmax(z)), z.shape)
    x_val = float(np.asarray(f["x"], dtype=float)[i]) if len(f["x"]) else 0.0
    y_val = float(np.asarray(f["y"], dtype=float)[j]) if len(f["y"]) else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"Highest modeled failure-risk zone is near ({x_val:.2f}, {y_val:.2f}) with probability score {float(z[j, i]):.2f}.",
            "Treat this as a risk hotspot for stress-testing policy plans.",
        ],
        theme,
    )


def _interpret_basin(analytics: Dict[str, Any], theme) -> None:
    b = analytics["basin"]
    z = np.asarray(b["z"], dtype=float)
    if z.size == 0:
        return
    j, i = np.unravel_index(int(np.argmax(z)), z.shape)
    x_val = float(np.asarray(b["x"], dtype=float)[i]) if len(b["x"]) else 0.0
    y_val = float(np.asarray(b["y"], dtype=float)[j]) if len(b["y"]) else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"Strongest stability basin appears near state ({x_val:.2f}, {y_val:.2f}).",
            f"Basin score at that point: {float(z[j, i]):.2f}.",
        ],
        theme,
    )


def _interpret_warning(analytics: Dict[str, Any], theme) -> None:
    w = analytics["warning"]
    z = np.asarray(w["z"], dtype=float)
    if z.size == 0:
        return
    j, i = np.unravel_index(int(np.argmax(z)), z.shape)
    x_val = float(np.asarray(w["x"], dtype=float)[i]) if len(w["x"]) else 0.0
    y_val = float(np.asarray(w["y"], dtype=float)[j]) if len(w["y"]) else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"Highest early-warning signal at time ~{x_val:.0f}, policy-z ~{y_val:.2f}.",
            f"Warning index there: {float(z[j, i]):.2f}.",
        ],
        theme,
    )


def _interpret_counterfactual(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    base = np.asarray(analytics["counterfactual_surface"], dtype=float)
    row = base[i] if base.ndim == 2 and i < base.shape[0] else np.array([0.0])
    default_delta = float(analytics["counterfactual_delta_abs_default"])
    chosen_delta = float(st.session_state.get("terrain_counterfactual_delta", default_delta))
    ratio = chosen_delta / max(default_delta, 1e-8)
    row = row * ratio
    idx = int(np.argmax(np.abs(row))) if row.size else 0
    _interpretation_box(
        "Interpretation",
        [
            f"With selected counterfactual shift ({chosen_delta:.3g}), `{focus_outcome}` changes most around policy ~{_policy_at(analytics, idx):.3g}.",
            f"Peak counterfactual delta: {float(row[idx]) if row.size else 0.0:+.3f}.",
        ],
        theme,
    )


def _interpret_efficiency(analytics: Dict[str, Any], theme) -> None:
    outcomes = list(analytics["outcomes"])
    if not outcomes:
        return
    o1 = st.session_state.get("terrain_obj_a", outcomes[0])
    o2 = st.session_state.get("terrain_obj_b", outcomes[1] if len(outcomes) > 1 else outcomes[0])
    dir1 = st.session_state.get("terrain_obj_a_dir", "Maximize")
    dir2 = st.session_state.get("terrain_obj_b_dir", "Maximize")
    if o1 not in outcomes or o2 not in outcomes:
        return
    i1, i2 = outcomes.index(o1), outcomes.index(o2)
    points = np.asarray(analytics["objective_points"], dtype=float)
    xy = np.column_stack([points[:, i1], points[:, i2]])
    frontier = _pareto_frontier(
        xy,
        maximize_x=(dir1 == "Maximize"),
        maximize_y=(dir2 == "Maximize"),
    )
    _interpretation_box(
        "Interpretation",
        [
            f"Frontier size: {len(frontier)} non-dominated policy states for `{o1}` vs `{o2}`.",
            "Points off the frontier are dominated by better tradeoff alternatives.",
        ],
        theme,
    )


def _interpret_momentum(analytics: Dict[str, Any], theme) -> None:
    m = analytics["momentum"]
    if len(m["dx"]) == 0:
        return
    mean_dx = float(np.mean(m["dx"]))
    mean_dy = float(np.mean(m["dy"]))
    _interpretation_box(
        "Interpretation",
        [
            f"Average policy-direction drift: {mean_dx:+.3f} (z-units/step).",
            f"Average outcome-direction drift: {mean_dy:+.3f} (z-units/step).",
            "Same-sign drifts suggest aligned movement; opposite signs suggest tension/tradeoff.",
        ],
        theme,
    )


def _interpret_saturation(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    sat = np.asarray(analytics["saturation_surface"], dtype=float)
    row = sat[i] if sat.ndim == 2 and i < sat.shape[0] else np.array([0.0])
    nonpos = float(np.mean(row <= 0.0)) if row.size else 0.0
    _interpretation_box(
        "Interpretation",
        [
            f"For `{focus_outcome}`, non-positive marginal-return share is {nonpos*100:.0f}%.",
            "High share indicates saturation zones where additional policy input yields limited gains.",
        ],
        theme,
    )


def _interpret_inequality(analytics: Dict[str, Any], theme, focus_outcome: str) -> None:
    i = _outcome_idx(analytics, focus_outcome)
    z = np.asarray(analytics["inequality"]["z"], dtype=float)
    row = z[i] if z.ndim == 2 and i < z.shape[0] else np.array([0.0])
    spread = float(np.max(row) - np.min(row)) if row.size else 0.0
    q = analytics["inequality"]["quantiles"]
    hi = int(np.argmax(row)) if row.size else 0
    lo = int(np.argmin(row)) if row.size else 0
    _interpretation_box(
        "Interpretation",
        [
            f"Quantile effect spread for `{focus_outcome}`: {spread:.3f}.",
            f"Strongest positive quantile: {q[hi] if q else 'n/a'}; weakest: {q[lo] if q else 'n/a'}.",
            "Larger spread implies uneven/distribution-sensitive policy impact.",
        ],
        theme,
    )


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def _terrain_3d_enabled() -> bool:
    return st.session_state.get("terrain_style_mode", "3D-heavy") == "3D-heavy"


def _plotly_chart(fig) -> None:
    """Render Plotly charts without modebar controls."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "responsive": True,
        },
    )


def _axis_coords(values: Sequence[Any]) -> Tuple["np.ndarray", bool, List[float], List[str]]:
    """Convert labels to numeric coordinates for optional 3D surfaces."""
    try:
        arr = np.asarray(values, dtype=float)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            return arr, True, arr.astype(float).tolist(), [str(v) for v in values]
    except Exception:
        pass
    coords = np.arange(len(values), dtype=float)
    return coords, False, coords.tolist(), [str(v) for v in values]


def _plot_matrix_chart(
    *,
    x: Sequence[Any],
    y: Sequence[Any],
    z: "np.ndarray",
    theme,
    x_title: str,
    y_title: str,
    z_title: str,
    height: int = 430,
    colorscale: str = "RdBu_r",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    zmid: Optional[float] = None,
    title: Optional[str] = None,
):
    """Render a matrix as either heatmap (2D) or surface (3D-heavy mode)."""
    if not HAS_PLOTLY:
        st.dataframe(pd.DataFrame(z, index=list(y), columns=list(x)), use_container_width=True)
        return

    z_arr = np.asarray(z, dtype=float)
    x_coords, x_is_numeric, x_tickvals, x_ticktext = _axis_coords(x)
    y_coords, y_is_numeric, y_tickvals, y_ticktext = _axis_coords(y)

    if _terrain_3d_enabled():
        X, Y = np.meshgrid(x_coords, y_coords)
        surf = go.Surface(
            x=X,
            y=Y,
            z=z_arr,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=z_title),
        )
        if zmid is not None:
            surf.update(cmid=zmid)

        fig = go.Figure(data=surf)
        fig.update_layout(
            height=max(460, height),
            margin=dict(l=0, r=0, t=35 if title else 10, b=10),
            title=title,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
            scene=dict(
                xaxis=dict(
                    title=x_title,
                    tickvals=None if x_is_numeric else x_tickvals,
                    ticktext=None if x_is_numeric else x_ticktext,
                ),
                yaxis=dict(
                    title=y_title,
                    tickvals=None if y_is_numeric else y_tickvals,
                    ticktext=None if y_is_numeric else y_ticktext,
                ),
                zaxis=dict(title=z_title),
            ),
        )
        _plotly_chart(fig)
        return

    fig = go.Figure(
        data=go.Heatmap(
            x=list(x),
            y=list(y),
            z=z_arr,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            colorbar=dict(title=z_title),
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35 if title else 20, b=40),
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.text_muted),
    )
    _plotly_chart(fig)


def _plot_effect_surface(analytics: Dict[str, Any], theme) -> None:
    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    z = analytics["effect_surface"]
    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} intensity",
        y_title="Outcome domain",
        z_title="Estimated Effect",
        height=460,
        colorscale="RdBu_r",
        zmid=0.0,
    )



def _plot_confidence_map(analytics: Dict[str, Any], theme) -> None:
    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    z = analytics["confidence_map"]
    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} intensity",
        y_title="Outcome domain",
        z_title="Confidence",
        height=430,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
    )



def _plot_lag_surface(analytics: Dict[str, Any], theme) -> None:
    z = analytics["lag_surface"]
    y = analytics["outcomes"]
    lags = list(range(z.shape[1]))
    _plot_matrix_chart(
        x=lags,
        y=y,
        z=z,
        theme=theme,
        x_title="Lag steps",
        y_title="Outcome domain",
        z_title="Lag effect",
        height=420,
        colorscale="RdBu_r",
        zmid=0.0,
    )



def _plot_interaction_surface(analytics: Dict[str, Any], theme) -> None:
    inter = analytics["interaction"]

    c1, c2 = st.columns([3, 1])
    with c2:
        st.metric("Interaction Strength", f"{inter['strength']:.3f}")
        st.caption(
            f"Secondary axis: {analytics['secondary_name']}. "
            "Positive means reinforcing policies; negative means cancellation."
        )

    with c1:
        if HAS_PLOTLY:
            fig = go.Figure(
                data=go.Surface(
                    x=inter["x"],
                    y=inter["y"],
                    z=inter["z"],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(title="Interaction response"),
                )
            )
            fig.update_layout(
                height=480,
                margin=dict(l=0, r=0, t=10, b=10),
                scene=dict(
                    xaxis_title=analytics["primary_policy"],
                    yaxis_title=analytics["secondary_name"],
                    zaxis_title="Response",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme.text_muted),
            )
            _plotly_chart(fig)
        else:
            st.dataframe(pd.DataFrame(inter["z"]), use_container_width=True)



def _plot_sensitivity_surface(analytics: Dict[str, Any], theme) -> None:
    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    z = analytics["sensitivity_surface"]
    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} intensity",
        y_title="Outcome domain",
        z_title="Sensitivity",
        height=430,
        colorscale="YlOrRd",
    )



def _plot_persistence_curve(analytics: Dict[str, Any], theme) -> None:
    p = analytics["persistence"]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=p["lags"],
                y=p["values"],
                mode="lines+markers",
                line=dict(color=theme.accent_primary, width=3),
                marker=dict(size=7),
                name="Decay",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(180,180,180,0.6)")
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Lag step",
            yaxis_title="Estimated effect",
            title=f"Policy persistence on {p['target_outcome']}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
        )
        _plotly_chart(fig)
    else:
        st.line_chart(pd.DataFrame({"Lag": p["lags"], "Effect": p["values"]}).set_index("Lag"))



def _plot_shock_slice(analytics: Dict[str, Any], theme) -> None:
    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    z = analytics["shock_slice"]
    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} intensity",
        y_title="Outcome domain",
        z_title="High-shock minus low-shock",
        title=f"Shock axis: {analytics['shock_name']}",
        height=420,
        colorscale="RdBu_r",
        zmid=0.0,
    )



def _plot_structural_break(analytics: Dict[str, Any], theme) -> None:
    sb = analytics["structural_break"]
    years = sb["years"]
    beta = sb["rolling_beta"]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=years,
                y=beta,
                mode="lines+markers",
                name="Rolling causal slope",
                line=dict(color=theme.accent_primary, width=2),
            )
        )
        if sb["break_years"]:
            by = sb["break_years"]
            bv = [beta[years.index(y)] for y in by if y in years]
            fig.add_trace(
                go.Scatter(
                    x=by,
                    y=bv,
                    mode="markers",
                    marker=dict(color=theme.accent_danger, size=10, symbol="x"),
                    name="Regime shift",
                )
            )
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Time",
            yaxis_title="Rolling policy-outcome slope",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
        )
        _plotly_chart(fig)
    else:
        st.line_chart(pd.DataFrame({"Year": years, "Slope": beta}).set_index("Year"))

    st.caption(
        f"Break threshold: {sb['threshold']:.4f}. "
        f"Detected breakpoints: {', '.join(map(str, sb['break_years'])) if sb['break_years'] else 'None'}"
    )



def _plot_failure_surface(analytics: Dict[str, Any], theme) -> None:
    f = analytics["failure"]
    if HAS_PLOTLY:
        fig = go.Figure(
            data=go.Surface(
                x=f["x"],
                y=f["y"],
                z=f["z"],
                colorscale="Reds",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Failure probability"),
            )
        )
        fig.update_layout(
            height=470,
            margin=dict(l=0, r=0, t=10, b=10),
            scene=dict(
                xaxis_title=analytics["primary_policy"],
                yaxis_title=analytics["second_axis_name"],
                zaxis_title="P(Failure)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
        )
        _plotly_chart(fig)
    else:
        st.dataframe(pd.DataFrame(f["z"]), use_container_width=True)



def _plot_basin_map(analytics: Dict[str, Any], theme) -> None:
    b = analytics["basin"]
    _plot_matrix_chart(
        x=b["x"],
        y=b["y"],
        z=b["z"],
        theme=theme,
        x_title="State axis 1",
        y_title="State axis 2",
        z_title="Stability basin score",
        height=410,
        colorscale="GnBu",
        zmin=0.0,
        zmax=1.0,
    )



def _plot_warning_map(analytics: Dict[str, Any], theme) -> None:
    w = analytics["warning"]
    _plot_matrix_chart(
        x=w["x"],
        y=w["y"],
        z=w["z"],
        theme=theme,
        x_title="Time",
        y_title=f"{analytics['primary_policy']} (z)",
        z_title="Early warning index",
        height=410,
        colorscale="Magma",
    )



def _plot_counterfactual_surface(analytics: Dict[str, Any], theme) -> None:
    default_delta = analytics["counterfactual_delta_abs_default"]
    delta_abs = st.slider(
        "Counterfactual policy shift (absolute units)",
        min_value=0.0,
        max_value=max(0.1, float(np.nanstd(analytics["policy_grid"])) * 2.0),
        value=float(default_delta),
        step=max(0.01, float(default_delta) / 10.0),
        key="terrain_counterfactual_delta",
    )

    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    # Scale default surface by chosen delta ratio.
    ratio = delta_abs / max(default_delta, 1e-8)
    z = analytics["counterfactual_surface"] * ratio

    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} baseline intensity",
        y_title="Outcome domain",
        z_title="Counterfactual delta",
        height=420,
        colorscale="RdBu_r",
        zmid=0.0,
    )



def _plot_efficiency_frontier(analytics: Dict[str, Any], theme) -> None:
    outcomes = analytics["outcomes"]
    points = analytics["objective_points"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        o1 = st.selectbox("Objective A", outcomes, index=0, key="terrain_obj_a")
    with c2:
        o2_idx = 1 if len(outcomes) > 1 else 0
        o2 = st.selectbox("Objective B", outcomes, index=o2_idx, key="terrain_obj_b")
    with c3:
        dir1 = st.selectbox("A direction", ["Maximize", "Minimize"], key="terrain_obj_a_dir")
    with c4:
        dir2 = st.selectbox("B direction", ["Maximize", "Minimize"], key="terrain_obj_b_dir")

    if o1 == o2:
        st.warning("Choose two different objectives for a meaningful efficiency frontier.")
        return

    i1 = outcomes.index(o1)
    i2 = outcomes.index(o2)
    xy = np.column_stack([points[:, i1], points[:, i2]])

    frontier = _pareto_frontier(
        xy,
        maximize_x=(dir1 == "Maximize"),
        maximize_y=(dir2 == "Maximize"),
    )

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker=dict(color=theme.accent_primary, size=7, opacity=0.7),
                name="Policy states",
            )
        )
        if len(frontier) > 0:
            fig.add_trace(
                go.Scatter(
                    x=frontier[:, 0],
                    y=frontier[:, 1],
                    mode="lines+markers",
                    marker=dict(color=theme.accent_success, size=8),
                    line=dict(color=theme.accent_success, width=3),
                    name="Efficiency frontier",
                )
            )
        fig.update_layout(
            height=370,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title=o1,
            yaxis_title=o2,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
        )
        _plotly_chart(fig)
    else:
        st.dataframe(pd.DataFrame(xy, columns=[o1, o2]), use_container_width=True)



def _plot_momentum_field(analytics: Dict[str, Any], theme) -> None:
    m = analytics["momentum"]
    if len(m["x"]) == 0:
        st.info("Not enough points to compute momentum field.")
        return

    # Downsample for readability.
    step = max(1, len(m["x"]) // 40)
    xs = m["x"][::step]
    ys = m["y"][::step]
    dx = m["dx"][::step]
    dy = m["dy"][::step]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=7, color=theme.accent_primary),
                name="State",
            )
        )
        for i in range(len(xs)):
            fig.add_trace(
                go.Scatter(
                    x=[xs[i], xs[i] + dx[i]],
                    y=[ys[i], ys[i] + dy[i]],
                    mode="lines",
                    line=dict(color=theme.accent_success, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title=f"{analytics['primary_policy']} (z)",
            yaxis_title="Aggregate outcome (z)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_muted),
        )
        _plotly_chart(fig)
    else:
        st.line_chart(pd.DataFrame({"x": xs, "y": ys}).set_index("x"))



def _plot_saturation_surface(analytics: Dict[str, Any], theme) -> None:
    x = analytics["policy_grid"]
    y = analytics["outcomes"]
    z = analytics["saturation_surface"]
    _plot_matrix_chart(
        x=x,
        y=y,
        z=z,
        theme=theme,
        x_title=f"{analytics['primary_policy']} intensity",
        y_title="Outcome domain",
        z_title="Marginal return",
        height=420,
        colorscale="RdBu_r",
        zmid=0.0,
    )



def _plot_inequality_surface(analytics: Dict[str, Any], theme) -> None:
    q = analytics["inequality"]["quantiles"]
    z = analytics["inequality"]["z"]
    y = analytics["outcomes"]
    _plot_matrix_chart(
        x=q,
        y=y,
        z=z,
        theme=theme,
        x_title="Outcome quantile group",
        y_title="Outcome domain",
        z_title="Quantile effect",
        height=390,
        colorscale="RdBu_r",
        zmid=0.0,
    )


# -----------------------------------------------------------------------------
# Guide
# -----------------------------------------------------------------------------


def _render_terrain_guide(theme):
    """Detailed plain-language guide for policy terrain analytics."""
    st.markdown('<div class="section-header">POLICY TERRAIN GUIDE</div>', unsafe_allow_html=True)

    sections = [
        "1) Why this card exists",
        "2) What data it accepts",
        "3) How estimands feed the terrain",
        "4) What each graph tells you",
        "5) How to avoid bad interpretation",
    ]
    selected = st.radio("Guide Sections", sections, key="terrain_guide_section")

    if selected == "1) Why this card exists":
        st.markdown(
            """
            This card turns raw policy and macro time series into a decision terrain.

            It answers:
            - Which policy intensity regions are associated with stronger effects?
            - Which outcomes are responsive vs resistant?
            - Where are results statistically fragile?
            - Which zones look stable, unstable, or failure-prone?
            """
        )

    elif selected == "2) What data it accepts":
        st.markdown(
            """
            Accepted clean inputs:
            - World Bank format CSV (`Indicator Name` + year columns).
            - Simple time-series CSV (first column year/date, remaining numeric).

            Cleaning rules:
            - Keeps numeric columns with enough observations.
            - Interpolates small gaps.
            - Drops constant/empty columns.
            """
        )

    elif selected == "3) How estimands feed the terrain":
        st.markdown(
            """
            For each policy-outcome pair, the system builds multiple effect estimates:
            - Level effect (ATE-like slope)
            - Differenced effect (change-on-change)
            - Lagged effect (delay-aware)

            These are combined into a terrain effect signal, then weighted by a
            network-strength proxy from policy-outcome dependency structure.
            """
        )

    elif selected == "4) What each graph tells you":
        st.markdown(
            """
            Core Terrain:
            - Effect Surface: magnitude and direction by policy intensity and outcome.
            - Confidence Map: where estimates are more reliable.
            - Lag Surface: when effects materialize.
            - Interaction Terrain: reinforcement vs cancellation between levers.
            - Sensitivity Terrain: where small policy moves have large impact.

            Temporal:
            - Persistence/Decay: how long effects last.
            - Shock Slice: how terrain shifts under stress.
            - Regime Timeline: where structural behavior changes.

            Risk:
            - Failure Surface: regions with higher destabilization probability.
            - Basin Map: attractor zones vs divergence zones.
            - Early Warning: rising variance/autocorrelation zones.

            Counterfactual:
            - Delta Surface: expected change if policy shifts.
            - Efficiency Frontier: trade-off boundary across objectives.
            - Momentum Field: direction of evolving state.
            - Saturation: where extra policy effort loses marginal benefit.
            - Inequality Terrain: quantile-level heterogeneous response.
            """
        )

    elif selected == "5) How to avoid bad interpretation":
        st.markdown(
            """
            - Do not use one chart in isolation; read effect + confidence + risk together.
            - High effect with low confidence is exploratory, not operational.
            - Regime shifts can invalidate earlier policy relationships.
            - Treat these as decision-support signals, not deterministic forecasts.
            """
        )

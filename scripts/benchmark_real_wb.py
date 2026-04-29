"""
benchmark_real_wb.py
====================
Tests OnlineDiscoveryEngine against real World Bank data (Kenya 1995-2023).

Two evaluation protocols:

  1. Rolling leave-one-year-out direction forecast
     - Warmup on 1995..T-1, then predict direction of ACTUAL change at year T.
     - Each year T: engine predicts next state from state at T-1.
       Predicted direction compared to actual sign of (y_T - y_{T-1}).
     - Reports: determination rate + direction accuracy vs AR(1) baseline.

  2. Shock simulation on real baseline
     - Train engine on full Kenya history (all available years).
     - Apply same 6 policy shocks as simulation_v2.py to the real baseline state.
     - Compare predicted directions against economic theory.
     - Also runs federated (KEN + TZA + UGA) condition.

Usage:
    python scripts/benchmark_real_wb.py            # dry-run (offline mock)
    python scripts/benchmark_real_wb.py --live     # real World Bank API
    python scripts/benchmark_real_wb.py --live --start 2000 --end 2022
    python scripts/benchmark_real_wb.py --live --no-federated
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.real_wb")

OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"

# ---------------------------------------------------------------------------
# Shared indicator set (matches simulation_v2 + east_africa_federation)
# ---------------------------------------------------------------------------

INDICATORS = [
    "gdp_growth", "inflation", "unemployment", "exports_gdp", "imports_gdp",
    "current_account", "govt_consumption", "tax_revenue", "govt_debt",
    "real_interest_rate", "broad_money", "private_credit",
    "urban_population", "school_enrollment", "life_expectancy",
    "electricity_access", "internet_users",
]
SCHEMA = {"fields": [{"name": v, "type": "float"} for v in INDICATORS]}

# Minimum warmup years before we start evaluating (engine needs time to learn)
WARMUP_YEARS = 15

# ---------------------------------------------------------------------------
# Shock library — same as simulation_v2.py (theory-grounded)
# ---------------------------------------------------------------------------

SHOCKS: List[Dict[str, Any]] = [
    {
        "id": "S1_electricity",
        "label": "Electricity access +20 pp  (infrastructure investment)",
        "var": "electricity_access",
        "delta": +20.0,
        "predictions": [
            ("gdp_growth",        +1, "Higher productivity from electrification"),
            ("internet_users",    +1, "Digital infrastructure piggybacks on power"),
            ("life_expectancy",   +1, "Healthcare and food storage improvements"),
            ("private_credit",    +1, "More economic activity -> more bankable projects"),
            ("urban_population",  +1, "Electrified areas attract migration"),
        ],
    },
    {
        "id": "S2_govt_debt",
        "label": "Government debt +15 pp GDP  (fiscal expansion)",
        "var": "govt_debt",
        "delta": +15.0,
        "predictions": [
            ("real_interest_rate", +1, "Crowding-out raises cost of capital"),
            ("inflation",          +1, "Monetization risk and demand pressure"),
            ("private_credit",     -1, "Crowded out by govt borrowing"),
            ("gdp_growth",         +1, "Short-run Keynesian stimulus multiplier"),
            ("tax_revenue",        +1, "Higher income base boosts tax receipts"),
        ],
    },
    {
        "id": "S3_inflation",
        "label": "Inflation shock +5 pp  (external price pressure)",
        "var": "inflation",
        "delta": +5.0,
        "predictions": [
            ("real_interest_rate", +1, "Central bank raises policy rate"),
            ("exports_gdp",        -1, "Competitiveness loss via REER appreciation"),
            ("broad_money",        +1, "Monetization -- more money in system"),
            ("private_credit",     -1, "Higher rates discourage borrowing"),
            ("current_account",    -1, "Import costs rise, trade balance deteriorates"),
            ("unemployment",       +1, "Stagflation dynamic at high inflation"),
        ],
    },
    {
        "id": "S4_private_credit",
        "label": "Private credit +5 pp GDP  (FDI / capital availability surge)",
        "var": "private_credit",
        "delta": +5.0,
        "predictions": [
            ("gdp_growth",         +1, "Investment multiplier"),
            ("exports_gdp",        +1, "New productive capacity expands export base"),
            ("unemployment",       -1, "Capital-financed expansion creates jobs"),
            ("electricity_access", +1, "Investable capital flows into infrastructure"),
        ],
    },
    {
        "id": "S5_export_drought",
        "label": "Exports -10 pp GDP  (drought / commodity shock)",
        "var": "exports_gdp",
        "delta": -10.0,
        "predictions": [
            ("current_account", -1, "Trade balance deteriorates directly"),
            ("gdp_growth",      -1, "Output falls with agricultural sector"),
            ("unemployment",    +1, "Rural labour market shock"),
            ("inflation",       +1, "Food price pressure from supply shortage"),
            ("tax_revenue",     -1, "Lower income base shrinks fiscal revenues"),
        ],
    },
    {
        "id": "S6_interest_rate",
        "label": "Real interest rate +3 pp  (monetary tightening)",
        "var": "real_interest_rate",
        "delta": +3.0,
        "predictions": [
            ("private_credit", -1, "Borrowing more expensive -> credit contraction"),
            ("gdp_growth",     -1, "Investment falls with higher cost of capital"),
            ("unemployment",   +1, "Businesses cut expansion plans"),
            ("inflation",      -1, "Monetary tightening suppresses demand inflation"),
            ("broad_money",    -1, "Monetary contraction reduces money supply"),
            ("govt_debt",      +1, "Higher debt service increases nominal debt stock"),
        ],
    },
]

TOTAL_SHOCK_PREDS = sum(len(s["predictions"]) for s in SHOCKS)

# ---------------------------------------------------------------------------
# WB data fetching (live or mock)
# ---------------------------------------------------------------------------

WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG":    "inflation",
    "SL.UEM.TOTL.ZS":    "unemployment",
    "NE.EXP.GNFS.ZS":    "exports_gdp",
    "NE.IMP.GNFS.ZS":    "imports_gdp",
    "NE.TRD.GNFS.ZS":    "trade_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account",
    "NE.CON.GOVT.ZS":    "govt_consumption",
    "GC.TAX.TOTL.GD.ZS": "tax_revenue",
    "GC.DOD.TOTL.GD.ZS": "govt_debt",
    "FR.INR.RINR":        "real_interest_rate",
    "FM.LBL.BMNY.GD.ZS": "broad_money",
    "FS.AST.PRVT.GD.ZS": "private_credit",
    "SP.URB.TOTL.IN.ZS":  "urban_population",
    "SE.PRM.ENRR":        "school_enrollment",
    "SP.DYN.LE00.IN":     "life_expectancy",
    "EG.ELC.ACCS.ZS":    "electricity_access",
    "IT.NET.USER.ZS":     "internet_users",
    "IT.CEL.SETS.P2":     "mobile_subscriptions",
}


def _fetch_indicator(country: str, wb_code: str, start: int, end: int) -> Dict[int, Optional[float]]:
    import requests
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{wb_code}"
        f"?format=json&per_page=100&date={start}:{end}"
    )
    try:
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            return {}
        return {
            int(e["date"]): (float(e["value"]) if e["value"] is not None else None)
            for e in payload[1]
            if str(e.get("date", "")).isdigit()
        }
    except Exception as exc:
        logger.warning("WB fetch failed %s/%s: %s", country, wb_code, exc)
        return {}


def fetch_live(country: str, start: int, end: int) -> Dict[int, Dict[str, float]]:
    """Fetch all WB indicators for one country. Returns {year: {name: val}}."""
    yearly: Dict[int, Dict[str, float]] = {}
    for wb_code, short in WB_INDICATORS.items():
        vals = _fetch_indicator(country, wb_code, start, end)
        for yr, v in vals.items():
            if v is not None:
                yearly.setdefault(yr, {})[short] = v
        time.sleep(0.4)
    return {yr: row for yr, row in yearly.items() if len(row) >= 3}


def fetch_mock(country: str, start: int, end: int, seed: int = 0) -> Dict[int, Dict[str, float]]:
    """Synthetic WB-style data with AR(1) structure and realistic Kenya ranges."""
    rng = np.random.default_rng(seed + abs(hash(country)) % 10000)
    country_factor = {"KEN": 1.0, "TZA": 0.85, "UGA": 0.90}.get(country, 1.0)

    # Kenya-realistic baseline levels
    base = {
        "gdp_growth":          5.0,
        "inflation":           7.0,
        "unemployment":        5.5,
        "exports_gdp":        17.0,
        "imports_gdp":        24.0,
        "trade_gdp":          41.0,
        "current_account":    -5.0,
        "govt_consumption":   14.0,
        "tax_revenue":        15.0,
        "govt_debt":          52.0,
        "real_interest_rate":  5.5,
        "broad_money":        35.0,
        "private_credit":     28.0,
        "urban_population":   27.0,
        "school_enrollment":  94.0,
        "life_expectancy":    64.0,
        "electricity_access": 55.0,
        "internet_users":     22.0,
        "mobile_subscriptions": 85.0,
    }
    ar = {
        "gdp_growth": 0.55, "inflation": 0.72, "unemployment": 0.65,
        "exports_gdp": 0.50, "imports_gdp": 0.55, "trade_gdp": 0.60,
        "current_account": 0.45, "govt_consumption": 0.70, "tax_revenue": 0.68,
        "govt_debt": 0.80, "real_interest_rate": 0.60, "broad_money": 0.70,
        "private_credit": 0.68, "urban_population": 0.92, "school_enrollment": 0.88,
        "life_expectancy": 0.91, "electricity_access": 0.82, "internet_users": 0.75,
        "mobile_subscriptions": 0.78,
    }
    cross = [
        ("electricity_access", "gdp_growth",        0.15),
        ("electricity_access", "internet_users",     0.25),
        ("electricity_access", "private_credit",     0.10),
        ("govt_debt",          "real_interest_rate", 0.18),
        ("govt_debt",          "inflation",          0.10),
        ("govt_debt",          "private_credit",    -0.15),
        ("inflation",          "real_interest_rate", 0.30),
        ("inflation",          "exports_gdp",       -0.20),
        ("inflation",          "private_credit",    -0.12),
        ("inflation",          "current_account",   -0.18),
        ("private_credit",     "gdp_growth",         0.20),
        ("private_credit",     "unemployment",      -0.15),
        ("exports_gdp",        "current_account",    0.60),
        ("exports_gdp",        "gdp_growth",         0.25),
        ("real_interest_rate", "private_credit",    -0.25),
        ("real_interest_rate", "gdp_growth",        -0.15),
        ("real_interest_rate", "inflation",         -0.20),
        ("gdp_growth",         "tax_revenue",        0.35),
        ("gdp_growth",         "unemployment",      -0.20),
    ]
    cross_map: Dict[str, list] = {k: [] for k in base}
    for src, tgt, beta in cross:
        cross_map[tgt].append((src, beta))

    state = dict(base)
    result: Dict[int, Dict[str, float]] = {}
    for year in range(start, end + 1):
        new_state: Dict[str, float] = {}
        for v, bv in base.items():
            noise_scale = abs(bv) * 0.08 / country_factor
            ar_part = ar.get(v, 0.7) * (state[v] - bv)
            cx_part = sum(beta * (state[src] - base.get(src, 0)) for src, beta in cross_map.get(v, []))
            noise = float(rng.standard_normal()) * noise_scale
            new_state[v] = bv + ar_part + cx_part * country_factor + noise
        state = new_state
        result[year] = {k: v for k, v in state.items() if k in set(WB_INDICATORS.values())}
    return result


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _build_engine():
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    eng = OnlineDiscoveryEngine(mode="balanced")
    eng.initialize_v2(SCHEMA, use_causal=True)
    return eng


def _rows_from_yearly(yearly: Dict[int, Dict[str, float]]) -> List[Tuple[int, Dict[str, float]]]:
    """Return sorted (year, row) pairs, restricted to INDICATORS columns."""
    rows = []
    for yr in sorted(yearly.keys()):
        row = {k: v for k, v in yearly[yr].items() if k in set(INDICATORS)}
        if row:
            rows.append((yr, row))
    return rows


# ---------------------------------------------------------------------------
# Protocol 1: Rolling leave-one-year-out direction forecast
# ---------------------------------------------------------------------------

def _ar1_predict_direction(history: List[Dict[str, float]], var: str) -> Optional[int]:
    """
    Simple AR(1) baseline: predict direction = sign of last change.
    Returns +1, -1, or None if not enough history.
    """
    if len(history) < 2:
        return None
    prev, curr = history[-2].get(var), history[-1].get(var)
    if prev is None or curr is None:
        return None
    delta = curr - prev
    if abs(delta) < 1e-9:
        return None
    return +1 if delta > 0 else -1


def run_rolling_forecast(
    yearly: Dict[int, Dict[str, float]],
    label: str,
    peer_data: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Leave-one-year-out: train on 1995..T-1, predict direction at T.
    Returns one record per (year, indicator) test point.
    """
    rows_by_year = _rows_from_yearly(yearly)
    if len(rows_by_year) < WARMUP_YEARS + 1:
        logger.warning("%s: only %d years — not enough for rolling eval", label, len(rows_by_year))
        return []

    records = []
    engine_history: List[Dict[str, float]] = []

    # Build fresh engine for the full rolling evaluation
    eng = _build_engine()

    for i, (year, row) in enumerate(rows_by_year):
        if i < WARMUP_YEARS:
            eng.process_row(row)
            engine_history.append(row)
            if peer_data:
                for peer_code, peer_yearly in peer_data.items():
                    if year in peer_yearly:
                        peer_row = {k: v for k, v in peer_yearly[year].items() if k in set(INDICATORS)}
                        if peer_row and hasattr(eng, "process_peer_row"):
                            eng.process_peer_row(peer_code, peer_row, peer_weight=0.70)
            continue

        # We have warmup. Now: engine knows history up to year-1.
        # Use last known state as the starting point for prediction.
        prev_row = engine_history[-1]

        # Engine predicts next state
        try:
            predicted = eng.predict(prev_row)
        except Exception:
            predicted = {}

        # Evaluate per indicator
        for var in INDICATORS:
            actual_val = row.get(var)
            prev_val   = prev_row.get(var)
            if actual_val is None or prev_val is None:
                continue

            actual_delta = actual_val - prev_val
            if abs(actual_delta) < 1e-9:
                continue  # no change — skip (ambiguous)

            actual_dir = +1 if actual_delta > 0 else -1

            # Engine direction
            pred_val = predicted.get(var)
            if pred_val is not None and abs(pred_val - prev_val) > 1e-9:
                engine_dir: Optional[int] = +1 if pred_val > prev_val else -1
                engine_match: Optional[bool] = (engine_dir == actual_dir)
            else:
                engine_dir = None
                engine_match = None

            # AR(1) direction
            ar1_dir = _ar1_predict_direction(engine_history, var)
            ar1_match = (ar1_dir == actual_dir) if ar1_dir is not None else None

            records.append({
                "condition":     label,
                "year":          year,
                "variable":      var,
                "actual_delta":  round(actual_delta, 4),
                "actual_dir":    actual_dir,
                "engine_dir":    engine_dir,
                "engine_match":  engine_match,
                "ar1_dir":       ar1_dir,
                "ar1_match":     ar1_match,
            })

        # Now ingest this year's row before moving on
        eng.process_row(row)
        engine_history.append(row)
        if peer_data:
            for peer_code, peer_yearly in peer_data.items():
                if year in peer_yearly:
                    peer_row = {k: v for k, v in peer_yearly[year].items() if k in set(INDICATORS)}
                    if peer_row and hasattr(eng, "process_peer_row"):
                        eng.process_peer_row(peer_code, peer_row, peer_weight=0.70)

    return records


# ---------------------------------------------------------------------------
# Protocol 2: Shock simulation on real baseline
# ---------------------------------------------------------------------------

def _predict_delta(eng, baseline: Dict[str, float], shock_var: str, shock_delta: float) -> Dict[str, float]:
    try:
        pred_base = eng.predict(baseline)
    except Exception:
        pred_base = {}

    shocked = dict(baseline)
    shocked[shock_var] = baseline.get(shock_var, 0.0) + shock_delta
    try:
        pred_shock = eng.predict(shocked)
    except Exception:
        pred_shock = {}

    delta: Dict[str, float] = {}
    for v in INDICATORS:
        b = pred_base.get(v, baseline.get(v, 0.0))
        s = pred_shock.get(v, baseline.get(v, 0.0))
        delta[v] = s - b
    return delta


def run_shock_simulation(
    yearly: Dict[int, Dict[str, float]],
    label: str,
    peer_data: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Train on full history, then evaluate 6 shocks. Returns (records, n_hyps)."""
    rows_by_year = _rows_from_yearly(yearly)
    if not rows_by_year:
        return [], 0

    eng = _build_engine()
    rows_only = [r for _, r in rows_by_year]

    for row in rows_only:
        eng.process_row(row)

    if peer_data:
        for peer_code, peer_yearly in peer_data.items():
            for yr, peer_row_full in sorted(peer_yearly.items()):
                peer_row = {k: v for k, v in peer_row_full.items() if k in set(INDICATORS)}
                if peer_row and hasattr(eng, "process_peer_row"):
                    eng.process_peer_row(peer_code, peer_row, peer_weight=0.70)

    # Baseline: mean of last 5 available years
    tail = rows_only[-5:]
    baseline: Dict[str, float] = {}
    for v in INDICATORS:
        vals = [r[v] for r in tail if v in r]
        if vals:
            baseline[v] = float(np.mean(vals))

    n_hyps = len(eng.hypotheses.population)

    records: List[Dict[str, Any]] = []
    for shock in SHOCKS:
        delta = _predict_delta(eng, baseline, shock["var"], shock["delta"])
        for tgt_var, expected_dir, rationale in shock["predictions"]:
            actual_delta = delta.get(tgt_var, 0.0)
            if abs(actual_delta) < 1e-9:
                match: Optional[bool] = None
            else:
                match = (math.copysign(1.0, actual_delta) == float(expected_dir))
            records.append({
                "condition":    label,
                "shock_id":     shock["id"],
                "shock_label":  shock["label"],
                "shock_var":    shock["var"],
                "shock_delta":  shock["delta"],
                "target_var":   tgt_var,
                "expected_dir": "+" if expected_dir > 0 else "-",
                "actual_delta": round(actual_delta, 6),
                "match":        match,
                "rationale":    rationale,
            })

    return records, n_hyps


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta as sp_beta
        lo = float(sp_beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
        hi = float(sp_beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    except ImportError:
        lo = k / n - 0.1
        hi = k / n + 0.1
    return max(0.0, lo), min(1.0, hi)


def _summarise_rolling(records: List[Dict[str, Any]], label: str) -> List[str]:
    if not records:
        return [f"  {label}: no data"]

    det   = [r for r in records if r["engine_match"] is not None]
    match = [r for r in det if r["engine_match"] is True]
    k, n  = len(match), len(det)
    det_rate = n / len(records) if records else 0.0
    acc = k / n if n > 0 else 0.0
    lo, hi = _clopper_pearson(k, n)

    ar1_det   = [r for r in records if r["ar1_match"] is not None]
    ar1_match = [r for r in ar1_det if r["ar1_match"] is True]
    ar1_acc = len(ar1_match) / len(ar1_det) if ar1_det else 0.0

    lines = [
        f"  {label}",
        f"    Total direction tests    : {len(records)}",
        f"    Determined (engine made call) : {n} / {len(records)}  ({100*det_rate:.1f}%)",
        f"    Direction accuracy       : {k} / {n}  ({100*acc:.1f}%)",
        f"    95% CI                   : [{100*lo:.1f}%, {100*hi:.1f}%]",
        f"    AR(1) baseline accuracy  : {len(ar1_match)} / {len(ar1_det)}  ({100*ar1_acc:.1f}%)",
        f"    Engine lift vs AR(1)     : {100*(acc - ar1_acc):+.1f} pp",
    ]

    # Per-year breakdown
    years = sorted(set(r["year"] for r in records))
    lines.append(f"    Per-year accuracy (determined only):")
    for yr in years:
        yr_det   = [r for r in records if r["year"] == yr and r["engine_match"] is not None]
        yr_match = [r for r in yr_det if r["engine_match"] is True]
        yr_ar1   = [r for r in records if r["year"] == yr and r["ar1_match"] is not None]
        yr_ar1m  = [r for r in yr_ar1 if r["ar1_match"] is True]
        e_str  = f"{len(yr_match)}/{len(yr_det)}" if yr_det else "—/—"
        a_str  = f"{len(yr_ar1m)}/{len(yr_ar1)}" if yr_ar1 else "—/—"
        lines.append(f"      {yr}  engine={e_str}  AR(1)={a_str}")

    return lines


def _summarise_shock(records: List[Dict[str, Any]], n_hyps: int, label: str) -> List[str]:
    if not records:
        return [f"  {label}: no data"]

    det   = [r for r in records if r["match"] is not None]
    match = [r for r in det if r["match"] is True]
    k, n  = len(match), len(det)
    lo, hi = _clopper_pearson(k, n)

    lines = [
        f"  {label}",
        f"    Hypotheses in pool      : {n_hyps}",
        f"    Determined predictions  : {n} / {len(records)}  ({len(records)-n} indeterminate)",
        f"    Direction matches       : {k} / {n}  ({100*k/n:.1f}%)" if n > 0 else "    No determined predictions",
        f"    95% CI                  : [{100*lo:.1f}%, {100*hi:.1f}%]",
    ]

    lines.append(f"    Per-shock:")
    for shock in SHOCKS:
        sr = [r for r in det if r["shock_id"] == shock["id"]]
        sm = [r for r in sr if r["match"] is True]
        pct = 100 * len(sm) / len(sr) if sr else 0.0
        lines.append(f"      {shock['label'][:52]:52s}  {len(sm)}/{len(sr):1d}  {pct:.0f}%")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",   action="store_true",  help="Fetch real World Bank data (requires internet)")
    p.add_argument("--start",  type=int, default=1995)
    p.add_argument("--end",    type=int, default=2023)
    p.add_argument("--no-federated", dest="federated", action="store_false")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.set_defaults(federated=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_mode = "LIVE World Bank API" if args.live else "DRY-RUN (synthetic mock)"
    logger.info("=== Real-data benchmark | %s | %d-%d ===", data_mode, args.start, args.end)

    # ── Fetch data ────────────────────────────────────────────────────────────
    fetch = fetch_live if args.live else fetch_mock

    logger.info("Fetching Kenya (KEN) ...")
    ken = fetch("KEN", args.start, args.end)
    logger.info("  Kenya: %d years", len(ken))

    tza = uga = {}
    if args.federated:
        logger.info("Fetching Tanzania (TZA) ...")
        tza = fetch("TZA", args.start, args.end)
        logger.info("  Tanzania: %d years", len(tza))
        logger.info("Fetching Uganda (UGA) ...")
        uga = fetch("UGA", args.start, args.end)
        logger.info("  Uganda: %d years", len(uga))

    if not ken:
        logger.error("No Kenya data — cannot proceed")
        sys.exit(1)

    peer_data = {}
    if args.federated:
        if tza: peer_data["TZA"] = tza
        if uga: peer_data["UGA"] = uga

    years_avail = sorted(ken.keys())
    logger.info("Kenya years available: %s – %s (%d total)",
                years_avail[0], years_avail[-1], len(years_avail))
    logger.info("Indicators with data (sample year %s): %s",
                years_avail[-1], list(ken[years_avail[-1]].keys())[:8])

    # ── Protocol 1: Rolling direction forecast ────────────────────────────────
    logger.info("--- Protocol 1: Rolling leave-one-year-out direction forecast ---")

    recs_local = run_rolling_forecast(ken, "LOCAL (Kenya only)")
    recs_fed   = run_rolling_forecast(ken, "FEDERATED (KEN+TZA+UGA)", peer_data if peer_data else None)

    # ── Protocol 2: Shock simulation on real baseline ─────────────────────────
    logger.info("--- Protocol 2: Shock simulation on real Kenya baseline ---")

    shock_recs_local, n_hyps_local = run_shock_simulation(ken, "LOCAL")
    shock_recs_fed, n_hyps_fed     = run_shock_simulation(
        ken, "FEDERATED", peer_data if peer_data else None
    )

    # ── Report ────────────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 72,
        "SCARCITY -- REAL WORLD BANK DATA BENCHMARK",
        f"  Data source  : {data_mode}",
        f"  Kenya years  : {years_avail[0]}-{years_avail[-1]}  ({len(years_avail)} observations)",
        f"  Peers        : {', '.join(peer_data.keys()) if peer_data else 'none'}",
        f"  Warmup years : {WARMUP_YEARS}",
        "=" * 72,
        "",
        "PROTOCOL 1: Rolling Leave-One-Year-Out Direction Forecast",
        "  Prediction: direction of ACTUAL annual change for each indicator.",
        "  Baseline: AR(1) (predict same direction as last year's change).",
        "",
    ]
    summary_lines += _summarise_rolling(recs_local, "LOCAL  (Kenya only)")
    summary_lines.append("")
    if recs_fed:
        summary_lines += _summarise_rolling(recs_fed, "FEDERATED (KEN+TZA+UGA)")
    summary_lines += [
        "",
        "-" * 72,
        "PROTOCOL 2: Shock Simulation on Real Baseline",
        "  Baseline: mean of last 5 Kenya data years.",
        "  Shocks: 6 policy shocks x theory-predicted directions.",
        "",
    ]
    summary_lines += _summarise_shock(shock_recs_local, n_hyps_local, "LOCAL")
    summary_lines.append("")
    if shock_recs_fed:
        summary_lines += _summarise_shock(shock_recs_fed, n_hyps_fed, "FEDERATED")
    summary_lines += [
        "",
        "-" * 72,
        "Notes:",
        "  Protocol 1 tests out-of-sample forecasting on real annual macro data.",
        "  Protocol 2 tests theory alignment using the engine's learned relationships.",
        "  Random direction guessing = 50%.  AR(1) baseline ~ 52-55%.",
        "  Target: engine accuracy > AR(1) with 95% CI lower bound > 60%.",
        "-" * 72,
    ]

    text = "\n".join(summary_lines)

    out_path = args.out_dir / "real_wb_benchmark.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Summary -> %s", out_path)

    # Write CSV for rolling records
    if recs_local or recs_fed:
        csv_path = args.out_dir / "real_wb_rolling.csv"
        all_recs = recs_local + recs_fed
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_recs[0].keys()))
            w.writeheader(); w.writerows(all_recs)
        logger.info("Rolling CSV -> %s", csv_path)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()

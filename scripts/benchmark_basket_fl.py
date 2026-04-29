"""
benchmark_basket_fl.py
======================
Demonstrates the full basket + federated learning architecture.

Architecture:
  - 4 baskets: macro, financial, infrastructure, human_capital
  - 3 nodes: KEN (target), TZA, UGA (peers)
  - FederationHub routes peer rows strictly within same basket
  - Pretraining: feed all SSA countries (or mock corpus) into basket engines
    BEFORE live Kenya evaluation begins

Evaluation:
  - Protocol: rolling leave-one-year-out direction forecast (Kenya 2010-2023)
  - Three conditions compared:
      A. COLD-START   — no pretraining, fresh engine (baseline)
      B. PRETRAINED   — basket engines warmed on broad corpus before Kenya data
      C. AR(1)        — naive "same direction as last year" (reference floor)

Usage:
    python scripts/benchmark_basket_fl.py                  # dry-run (fast)
    python scripts/benchmark_basket_fl.py --live           # real WB data
    python scripts/benchmark_basket_fl.py --live --pretrain-live  # real pretraining corpus too
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
logger = logging.getLogger("benchmark.basket_fl")

OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"

# All indicators used in evaluation
ALL_INDICATORS = [
    "gdp_growth", "inflation", "unemployment", "exports_gdp", "imports_gdp",
    "current_account", "govt_consumption", "tax_revenue", "govt_debt",
    "real_interest_rate", "broad_money", "private_credit",
    "urban_population", "school_enrollment", "life_expectancy",
    "electricity_access", "internet_users",
]

WARMUP_YEARS = 15   # years before rolling evaluation starts

# Sub-Saharan Africa country codes (WB) used for pretraining corpus
SSA_COUNTRIES = [
    "ETH", "NGA", "GHA", "ZAF", "MOZ", "ZMB", "MWI", "MDG",
    "CMR", "CIV", "SEN", "MLI", "BFA", "TCD", "NER", "GIN",
    "BEN", "TGO", "SLE", "LBR", "ZWE", "BWA", "NAM", "LSO",
    "SWZ", "RWA", "BDI", "DJI", "ERI", "SOM",
]
# Exclude KEN/TZA/UGA from pretraining — they are the live evaluation nodes
_EVAL_COUNTRIES = {"KEN", "TZA", "UGA"}
PRETRAIN_COUNTRIES = [c for c in SSA_COUNTRIES if c not in _EVAL_COUNTRIES]


# ---------------------------------------------------------------------------
# WB data fetching (reuses logic from experiment_east_africa_federation.py)
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


def _fetch_one(country: str, wb_code: str, start: int, end: int) -> Dict[int, Optional[float]]:
    import requests
    url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{wb_code}"
           f"?format=json&per_page=100&date={start}:{end}")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return {}
        return {int(e["date"]): (float(e["value"]) if e["value"] is not None else None)
                for e in payload[1] if str(e.get("date", "")).isdigit()}
    except Exception as exc:
        logger.debug("WB fetch failed %s/%s: %s", country, wb_code, exc)
        return {}


def fetch_live(country: str, start: int, end: int) -> Dict[int, Dict[str, float]]:
    yearly: Dict[int, Dict[str, float]] = {}
    for wb_code, short in WB_INDICATORS.items():
        for yr, v in _fetch_one(country, wb_code, start, end).items():
            if v is not None:
                yearly.setdefault(yr, {})[short] = v
        time.sleep(0.35)
    return {yr: row for yr, row in yearly.items() if len(row) >= 3}


def fetch_mock(country: str, start: int, end: int, seed: int = 0) -> Dict[int, Dict[str, float]]:
    """Structured mock with Kenya-realistic levels and economic cross-effects."""
    rng = np.random.default_rng(seed + abs(hash(country)) % 99991)
    cf = {"KEN": 1.0, "TZA": 0.85, "UGA": 0.90}.get(country, 0.80 + rng.uniform(0, 0.15))
    base = {
        "gdp_growth": 5.0, "inflation": 7.0, "unemployment": 5.5,
        "exports_gdp": 17.0, "imports_gdp": 24.0, "trade_gdp": 41.0,
        "current_account": -5.0, "govt_consumption": 14.0,
        "tax_revenue": 15.0, "govt_debt": 52.0,
        "real_interest_rate": 5.5, "broad_money": 35.0, "private_credit": 28.0,
        "urban_population": 27.0, "school_enrollment": 94.0,
        "life_expectancy": 64.0, "electricity_access": 55.0,
        "internet_users": 22.0, "mobile_subscriptions": 85.0,
    }
    # Vary baseline by country
    for k in base:
        base[k] = base[k] * (0.85 + rng.uniform(0, 0.30))
    ar = {k: 0.65 for k in base}
    ar.update({"govt_debt": 0.82, "urban_population": 0.93, "life_expectancy": 0.92,
                "school_enrollment": 0.89, "electricity_access": 0.84})
    cross = [
        ("electricity_access", "gdp_growth", 0.15), ("electricity_access", "internet_users", 0.25),
        ("electricity_access", "private_credit", 0.10), ("govt_debt", "real_interest_rate", 0.18),
        ("govt_debt", "inflation", 0.10), ("govt_debt", "private_credit", -0.15),
        ("inflation", "real_interest_rate", 0.30), ("inflation", "exports_gdp", -0.20),
        ("inflation", "private_credit", -0.12), ("inflation", "current_account", -0.18),
        ("private_credit", "gdp_growth", 0.20), ("private_credit", "unemployment", -0.15),
        ("exports_gdp", "current_account", 0.60), ("exports_gdp", "gdp_growth", 0.25),
        ("real_interest_rate", "private_credit", -0.25), ("real_interest_rate", "gdp_growth", -0.15),
        ("gdp_growth", "tax_revenue", 0.35), ("gdp_growth", "unemployment", -0.20),
    ]
    cm: Dict[str, list] = {k: [] for k in base}
    for s, t, b in cross:
        cm[t].append((s, b))
    state = dict(base)
    result: Dict[int, Dict[str, float]] = {}
    for year in range(start, end + 1):
        new: Dict[str, float] = {}
        for v, bv in base.items():
            noise = float(rng.standard_normal()) * abs(bv) * 0.08 / cf
            ar_part = ar.get(v, 0.65) * (state[v] - bv)
            cx = sum(beta * (state[src] - base.get(src, 0)) for src, beta in cm.get(v, []))
            new[v] = bv + ar_part + cx * cf + noise
        state = new
        result[year] = {k: v for k, v in state.items() if k in set(WB_INDICATORS.values())}
    return result


def _to_rows(yearly: Dict[int, Dict[str, float]]) -> List[Tuple[int, Dict[str, float]]]:
    result = []
    for yr in sorted(yearly):
        row = {k: v for k, v in yearly[yr].items() if k in set(ALL_INDICATORS)}
        if row:
            result.append((yr, row))
    return result


# ---------------------------------------------------------------------------
# AR(1) baseline
# ---------------------------------------------------------------------------

def _ar1_direction(history: List[Dict[str, float]], var: str) -> Optional[int]:
    if len(history) < 2:
        return None
    p, c = history[-2].get(var), history[-1].get(var)
    if p is None or c is None or abs(c - p) < 1e-9:
        return None
    return +1 if c > p else -1


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta as sp_beta
        lo = float(sp_beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(sp_beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    except ImportError:
        lo, hi = max(0.0, k/n - 0.1), min(1.0, k/n + 0.1)
    return lo, hi


# ---------------------------------------------------------------------------
# Single rolling evaluation run
# ---------------------------------------------------------------------------

def run_rolling(
    label: str,
    ken_yearly: Dict[int, Dict[str, float]],
    hub: Optional[Any] = None,     # FederationHub (pretrained condition)
    node_id: str = "KEN",
    fan_out: bool = True,
    peer_yearly: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Rolling leave-one-year-out evaluation.

    If hub is provided, uses the hub's pretrained FederationNode for KEN.
    Otherwise, creates a fresh cold-start FederationNode.
    """
    from scarcity.engine.federation_node import FederationNode
    from scarcity.engine.federation_hub import FederationHub

    rows = _to_rows(ken_yearly)
    if len(rows) < WARMUP_YEARS + 1:
        logger.warning("%s: only %d years — need %d+1", label, len(rows), WARMUP_YEARS)
        return []

    # Use provided hub or build a fresh one
    if hub is not None:
        active_hub = hub
        ken_node = hub.node(node_id)
    else:
        # Cold-start
        active_hub = FederationHub()
        ken_node = FederationNode(node_id)
        active_hub.register(ken_node)
        if peer_yearly:
            for pid in peer_yearly:
                active_hub.register(FederationNode(pid))

    records: List[Dict[str, Any]] = []
    history: List[Dict[str, float]] = []

    for i, (year, row) in enumerate(rows):
        if i < WARMUP_YEARS:
            # Warmup: feed to hub (also fans out to peers if present)
            if peer_yearly:
                peer_rows_this_year = {
                    pid: {k: v for k, v in pyr.get(year, {}).items() if k in set(ALL_INDICATORS)}
                    for pid, pyr in peer_yearly.items()
                }
                obs_all = {node_id: row, **{p: r for p, r in peer_rows_this_year.items() if r}}
                active_hub.observe_all(obs_all, fan_out=fan_out)
            else:
                active_hub.observe(node_id, row, fan_out=False)
            history.append(row)
            continue

        # Evaluation step: predict direction before ingesting this year
        prev_row = history[-1]
        try:
            predicted = ken_node.predict(prev_row)
        except Exception:
            predicted = {}

        for var in ALL_INDICATORS:
            actual_val = row.get(var)
            prev_val   = prev_row.get(var)
            if actual_val is None or prev_val is None:
                continue
            delta = actual_val - prev_val
            if abs(delta) < 1e-9:
                continue
            actual_dir = +1 if delta > 0 else -1

            pred_val = predicted.get(var)
            if pred_val is not None and abs(pred_val - prev_val) > 1e-9:
                engine_dir: Optional[int] = +1 if pred_val > prev_val else -1
                engine_match: Optional[bool] = (engine_dir == actual_dir)
            else:
                engine_dir = None
                engine_match = None

            ar1_dir = _ar1_direction(history, var)
            ar1_match = (ar1_dir == actual_dir) if ar1_dir is not None else None

            records.append({
                "condition":    label,
                "year":         year,
                "variable":     var,
                "actual_delta": round(delta, 4),
                "actual_dir":   actual_dir,
                "engine_dir":   engine_dir,
                "engine_match": engine_match,
                "ar1_dir":      ar1_dir,
                "ar1_match":    ar1_match,
            })

        # Ingest this year's data
        if peer_yearly:
            peer_rows_this_year = {
                pid: {k: v for k, v in pyr.get(year, {}).items() if k in set(ALL_INDICATORS)}
                for pid, pyr in peer_yearly.items()
            }
            obs_all = {node_id: row, **{p: r for p, r in peer_rows_this_year.items() if r}}
            active_hub.observe_all(obs_all, fan_out=fan_out)
        else:
            active_hub.observe(node_id, row, fan_out=False)
        history.append(row)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report_condition(records: List[Dict[str, Any]], label: str) -> List[str]:
    if not records:
        return [f"  {label}: no records"]

    det   = [r for r in records if r["engine_match"] is not None]
    match = [r for r in det if r["engine_match"] is True]
    k, n  = len(match), len(det)
    det_rate = n / len(records) if records else 0.0
    acc = k / n if n > 0 else 0.0
    lo, hi = _clopper_pearson(k, n)

    ar1_det   = [r for r in records if r["ar1_match"] is not None]
    ar1_match = [r for r in ar1_det if r["ar1_match"] is True]
    ar1_acc = len(ar1_match) / len(ar1_det) if ar1_det else 0.0

    lift = acc - ar1_acc
    pass_fail = "PASS" if lo >= 0.50 else ("MARGINAL" if lo >= 0.40 else "FAIL")

    lines = [
        f"  Condition: {label}  [{pass_fail}]",
        f"    Total direction tests  : {len(records)}",
        f"    Engine determined      : {n} / {len(records)}  ({100*det_rate:.1f}%)",
        f"    Direction accuracy     : {k} / {n}  ({100*acc:.1f}%)",
        f"    95% CI                 : [{100*lo:.1f}%, {100*hi:.1f}%]",
        f"    AR(1) accuracy         : {len(ar1_match)} / {len(ar1_det)}  ({100*ar1_acc:.1f}%)",
        f"    Engine lift vs AR(1)   : {lift:+.1%}",
    ]

    # Per-basket breakdown
    from scarcity.engine.baskets import REGISTRY
    lines.append(f"    Per-basket accuracy (determined only):")
    for bid in REGISTRY.all_ids():
        basket = REGISTRY.get(bid)
        br = [r for r in det if r["variable"] in basket.variables]
        bm = [r for r in br if r["engine_match"] is True]
        if br:
            lines.append(f"      {bid:20s}  {len(bm)}/{len(br)}  ({100*len(bm)/len(br):.0f}%)")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",          action="store_true", help="Real WB data for KEN/TZA/UGA")
    p.add_argument("--pretrain-live", action="store_true", help="Real WB pretraining corpus (SSA countries)")
    p.add_argument("--start",  type=int, default=1995)
    p.add_argument("--end",    type=int, default=2023)
    p.add_argument("--pretrain-start", type=int, default=1975,
                   help="Start year for pretraining corpus")
    p.add_argument("--pretrain-end",   type=int, default=1994,
                   help="End year for pretraining corpus (must be before --start)")
    p.add_argument("--pretrain-countries", type=int, default=10,
                   help="Number of SSA countries to use for pretraining corpus (default 10)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from scarcity.engine.federation_node import FederationNode
    from scarcity.engine.federation_hub import FederationHub
    from scarcity.engine.baskets import REGISTRY

    data_mode = "LIVE WB API" if args.live else "DRY-RUN (mock)"
    pretrain_mode = ("LIVE WB API" if args.pretrain_live else "DRY-RUN (mock)") if True else "none"
    logger.info("=== Basket FL Benchmark | eval=%s | pretrain=%s ===", data_mode, pretrain_mode)

    fetch = fetch_live if args.live else fetch_mock

    # ── Fetch eval data (KEN, TZA, UGA) ─────────────────────────────────────
    logger.info("Fetching evaluation data ...")
    ken = fetch("KEN", args.start, args.end)
    tza = fetch("TZA", args.start, args.end)
    uga = fetch("UGA", args.start, args.end)
    logger.info("  KEN=%d yrs  TZA=%d yrs  UGA=%d yrs", len(ken), len(tza), len(uga))

    peer_yearly = {}
    if tza: peer_yearly["TZA"] = tza
    if uga: peer_yearly["UGA"] = uga

    # ── Build pretraining corpus ──────────────────────────────────────────────
    logger.info("Building pretraining corpus (%s) ...", pretrain_mode)
    fetch_pre = fetch_live if args.pretrain_live else fetch_mock
    pretrain_countries = PRETRAIN_COUNTRIES[:args.pretrain_countries]
    corpus_rows: List[Dict[str, float]] = []

    for i, code in enumerate(pretrain_countries):
        logger.info("  Pretrain corpus: %s (%d/%d)", code, i+1, len(pretrain_countries))
        cdata = fetch_pre(code, args.pretrain_start, args.pretrain_end, seed=i+100) \
                if not args.pretrain_live else fetch_pre(code, args.pretrain_start, args.pretrain_end)
        for yr in sorted(cdata):
            row = {k: v for k, v in cdata[yr].items() if k in set(ALL_INDICATORS)}
            if row:
                corpus_rows.append(row)

    logger.info("Pretraining corpus: %d rows from %d countries (%d-%d)",
                len(corpus_rows), len(pretrain_countries), args.pretrain_start, args.pretrain_end)

    # ── Condition A: COLD-START (no pretraining) ─────────────────────────────
    logger.info("--- Condition A: COLD-START ---")
    recs_cold = run_rolling(
        "COLD-START", ken,
        hub=None, fan_out=False, peer_yearly=None,
    )

    # ── Condition B: COLD-START + FEDERATED (no pretraining, with TZA/UGA) ──
    logger.info("--- Condition B: COLD-START + FEDERATED ---")
    hub_cold_fed = FederationHub()
    for nid in ["KEN", "TZA", "UGA"]:
        hub_cold_fed.register(FederationNode(nid))
    recs_cold_fed = run_rolling(
        "COLD-START+FED", ken,
        hub=hub_cold_fed, node_id="KEN", fan_out=True, peer_yearly=peer_yearly,
    )

    # ── Condition C: PRETRAINED (basket engines warmed, no live federation) ──
    logger.info("--- Condition C: PRETRAINED (no live federation) ---")
    hub_pre = FederationHub()
    ken_pre = FederationNode("KEN")
    hub_pre.register(ken_pre)

    # Pretrain all baskets
    pretrain_results = hub_pre.pretrain_all_baskets(
        {bid: corpus_rows for bid in REGISTRY.all_ids()}
    )
    logger.info("Pretraining complete: %s", pretrain_results)

    recs_pretrained = run_rolling(
        "PRETRAINED", ken,
        hub=hub_pre, node_id="KEN", fan_out=False, peer_yearly=None,
    )

    # ── Condition D: PRETRAINED + FEDERATED ───────────────────────────────────
    logger.info("--- Condition D: PRETRAINED + FEDERATED ---")
    hub_pre_fed = FederationHub()
    for nid in ["KEN", "TZA", "UGA"]:
        node = FederationNode(nid)
        hub_pre_fed.register(node)
        # Pretrain all baskets for each node
        for bid in REGISTRY.all_ids():
            node.pretrain(bid, corpus_rows)

    recs_pre_fed = run_rolling(
        "PRETRAINED+FED", ken,
        hub=hub_pre_fed, node_id="KEN", fan_out=True, peer_yearly=peer_yearly,
    )

    # ── System stats ──────────────────────────────────────────────────────────
    logger.info("\n%s", hub_pre_fed.summary())

    # ── Report ────────────────────────────────────────────────────────────────
    years_ken = sorted(ken.keys())
    summary_lines = [
        "=" * 72,
        "SCARCITY — BASKET + FEDERATED LEARNING BENCHMARK",
        f"  Data mode    : {data_mode}",
        f"  Pretrain mode: {pretrain_mode}",
        f"  Kenya years  : {years_ken[0]}-{years_ken[-1]}  ({len(years_ken)} obs)",
        f"  Pretrain rows: {len(corpus_rows)}  ({len(pretrain_countries)} SSA countries, "
        f"{args.pretrain_start}-{args.pretrain_end})",
        f"  Baskets      : {REGISTRY.all_ids()}",
        f"  Warmup years : {WARMUP_YEARS}",
        "=" * 72,
        "",
        "Rolling leave-one-year-out direction forecast (Kenya 2010-2023)",
        "Metric: direction accuracy on determined predictions vs AR(1) baseline",
        "",
    ]
    for recs, lbl in [
        (recs_cold,      "A. COLD-START             (no pretraining, no federation)"),
        (recs_cold_fed,  "B. COLD-START + FEDERATED (no pretraining, with TZA/UGA)"),
        (recs_pretrained,"C. PRETRAINED             (basket warm-start, no federation)"),
        (recs_pre_fed,   "D. PRETRAINED + FEDERATED (basket warm-start + TZA/UGA)"),
    ]:
        summary_lines += _report_condition(recs, lbl)
        summary_lines.append("")

    summary_lines += [
        "-" * 72,
        "Target: direction accuracy CI lower bound > 50%  (50% = random guessing).",
        "PASS threshold deliberately modest for 14 test years of annual data.",
        "AR(1) is the meaningful real-world baseline (~55% on annual macro data).",
        "-" * 72,
    ]

    text = "\n".join(summary_lines)
    out_path = args.out_dir / "basket_fl_benchmark.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Summary -> %s", out_path)

    # CSV
    all_recs = recs_cold + recs_cold_fed + recs_pretrained + recs_pre_fed
    if all_recs:
        csv_path = args.out_dir / "basket_fl_rolling.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_recs[0].keys()))
            w.writeheader(); w.writerows(all_recs)
        logger.info("CSV -> %s", csv_path)

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

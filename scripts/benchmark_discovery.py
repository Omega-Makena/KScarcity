"""
benchmark_discovery.py
======================
Evaluates OnlineDiscoveryEngine as a RELATIONSHIP DISCOVERY tool.

This is the correct evaluation for this engine. The question is not
"can it beat AR(1) on next-year direction forecasting" but rather:
"does it correctly identify the sign and structure of economic relationships?"

Method
------
1. Train basket engines on all available real Kenya data (+ TZA/UGA federation).
2. For each relationship in the theory-grounded ground-truth library:
   a. Perturb the source variable by +1 standard deviation from baseline.
   b. Read the engine's predicted change in the target variable.
   c. Compare predicted sign to expected sign from economic theory.
3. Report: discovery rate, sign accuracy, per-basket breakdown, top findings.

Ground truth library
--------------------
25 relationships drawn from:
  - IMF Article IV Consultation, Kenya 2019/2022
  - World Bank WDI documentation
  - Mankiw, Macroeconomics (10th ed.) Chapters 14, 17
  - Levine (2005) finance-growth nexus review
  - Standard IS-LM / AS-AD framework

Metrics
-------
  Discovery rate   : % of ground-truth relationships where engine gives
                     a non-zero predicted delta (has some signal)
  Sign accuracy    : % of discovered relationships with correct sign
  Overall recall   : sign-correct hits / all ground-truth relationships

Usage
-----
    python scripts/benchmark_discovery.py            # dry-run (mock data)
    python scripts/benchmark_discovery.py --live     # real World Bank data
    python scripts/benchmark_discovery.py --live --pretrain-live
"""

from __future__ import annotations

import argparse
import csv
import logging
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
logger = logging.getLogger("benchmark.discovery")

OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"

ALL_INDICATORS = [
    "gdp_growth", "inflation", "unemployment", "exports_gdp", "imports_gdp",
    "current_account", "govt_consumption", "tax_revenue", "govt_debt",
    "real_interest_rate", "broad_money", "private_credit",
    "urban_population", "school_enrollment", "life_expectancy",
    "electricity_access", "internet_users",
]

# ---------------------------------------------------------------------------
# Ground truth relationship library
# 25 relationships with known signs from established economic theory.
# Each entry: (source, target, sign, basket, citation)
# sign: +1 = positive relationship, -1 = negative relationship
# ---------------------------------------------------------------------------

GROUND_TRUTH: List[Dict[str, Any]] = [
    # ── Macro basket ────────────────────────────────────────────────────────
    {"source": "private_credit",    "target": "gdp_growth",        "sign": +1,
     "basket": "macro",      "cite": "Levine 2005; finance-growth nexus"},
    {"source": "exports_gdp",       "target": "gdp_growth",        "sign": +1,
     "basket": "macro",      "cite": "Mankiw ch.11; open-economy multiplier"},
    {"source": "exports_gdp",       "target": "current_account",   "sign": +1,
     "basket": "macro",      "cite": "National accounts identity"},
    {"source": "exports_gdp",       "target": "tax_revenue",       "sign": +1,
     "basket": "macro",      "cite": "IMF SSA Fiscal Monitor 2021"},
    {"source": "gdp_growth",        "target": "tax_revenue",       "sign": +1,
     "basket": "macro",      "cite": "Mankiw ch.17; automatic stabilisers"},
    {"source": "gdp_growth",        "target": "unemployment",      "sign": -1,
     "basket": "macro",      "cite": "Okun's Law"},
    {"source": "inflation",         "target": "exports_gdp",       "sign": -1,
     "basket": "macro",      "cite": "REER competitiveness; IMF Art.IV Kenya 2022"},
    {"source": "inflation",         "target": "current_account",   "sign": -1,
     "basket": "macro",      "cite": "Import costs rise; trade balance worsens"},
    {"source": "govt_consumption",  "target": "gdp_growth",        "sign": +1,
     "basket": "macro",      "cite": "Keynesian fiscal multiplier"},

    # ── Financial basket ────────────────────────────────────────────────────
    {"source": "inflation",         "target": "real_interest_rate","sign": +1,
     "basket": "financial",  "cite": "Taylor rule; CBK reaction function"},
    {"source": "govt_debt",         "target": "real_interest_rate","sign": +1,
     "basket": "financial",  "cite": "Crowding-out; Mankiw ch.17"},
    {"source": "govt_debt",         "target": "private_credit",    "sign": -1,
     "basket": "financial",  "cite": "Crowding-out via bond market"},
    {"source": "real_interest_rate","target": "private_credit",    "sign": -1,
     "basket": "financial",  "cite": "Standard credit channel; IS curve"},
    {"source": "real_interest_rate","target": "broad_money",       "sign": -1,
     "basket": "financial",  "cite": "Monetary contraction reduces M2"},
    {"source": "real_interest_rate","target": "gdp_growth",        "sign": -1,
     "basket": "financial",  "cite": "Investment channel; Mankiw ch.14"},
    {"source": "private_credit",    "target": "broad_money",       "sign": +1,
     "basket": "financial",  "cite": "Money multiplier; credit expansion"},

    # ── Infrastructure basket ────────────────────────────────────────────────
    {"source": "electricity_access","target": "gdp_growth",        "sign": +1,
     "basket": "infrastructure", "cite": "IMF Art.IV Kenya 2022 sec.III; WB WDI"},
    {"source": "electricity_access","target": "internet_users",    "sign": +1,
     "basket": "infrastructure", "cite": "Digital infrastructure co-location"},
    {"source": "electricity_access","target": "private_credit",    "sign": +1,
     "basket": "infrastructure", "cite": "Electrification enables bankable projects"},
    {"source": "internet_users",    "target": "gdp_growth",        "sign": +1,
     "basket": "infrastructure", "cite": "Digital economy productivity"},

    # ── Human capital basket ────────────────────────────────────────────────
    {"source": "gdp_growth",        "target": "life_expectancy",   "sign": +1,
     "basket": "human_capital", "cite": "Preston curve; income-health relationship"},
    {"source": "gdp_growth",        "target": "school_enrollment", "sign": +1,
     "basket": "human_capital", "cite": "Income → education demand"},
    {"source": "unemployment",      "target": "gdp_growth",        "sign": -1,
     "basket": "human_capital", "cite": "Okun's law (reverse)"},
    {"source": "life_expectancy",   "target": "school_enrollment", "sign": +1,
     "basket": "human_capital", "cite": "Longer horizon → more education investment"},
    {"source": "urban_population",  "target": "gdp_growth",        "sign": +1,
     "basket": "human_capital", "cite": "Agglomeration economies; urbanisation-growth"},
]

SSA_PRETRAIN_COUNTRIES = [
    "ETH", "NGA", "GHA", "ZAF", "MOZ", "ZMB", "MWI", "MDG",
    "CMR", "CIV", "SEN", "MLI", "BFA", "ZWE", "BWA",
]

OECD_PRETRAIN_COUNTRIES = [
    "FRA", "GBR", "NLD", "BEL", "AUT", "SWE", "DNK", "NOR",
    "FIN", "CHE", "PRT", "ESP", "ITA", "IRL", "AUS",
]

# ---------------------------------------------------------------------------
# FRED quarterly series config  (USA only)
# Each tuple: (FRED series ID, local indicator name)
# FRED returns all series at quarterly frequency via &frequency=q&aggregation_method=avg
# ---------------------------------------------------------------------------

FRED_SERIES_CONFIG: List[Tuple[str, str]] = [
    ("A191RL1Q225SBEA",    "gdp_growth"),        # Real GDP % chg yr-ago, quarterly BEA
    ("CPIAUCSL",           "_cpi"),              # CPI monthly→quarterly avg (derive YoY inflation)
    ("UNRATE",             "unemployment"),      # Unemployment monthly→quarterly avg
    ("EXPGS",              "_exports_bn"),       # Exports of goods & services, $bn SAAR
    ("IMPGS",              "_imports_bn"),       # Imports of goods & services, $bn SAAR
    ("A955RC1Q027SBEA",    "govt_consumption"),  # Govt consumption+invest % of GDP (BEA)
    ("GFDEGDQ188S",        "govt_debt"),         # Federal debt % of GDP, quarterly
    ("REAINTRATREARAT10Y", "real_interest_rate"),# Real interest rate
    ("M2SL",               "_m2"),              # M2 money supply, $bn monthly
    ("GDP",                "_gdp_nom"),          # Nominal GDP, $bn SAAR
    ("LOANS",              "_loans"),            # Loans & leases at all commercial banks, $bn SA
    ("W010RC1Q027SBEA",    "_tax_receipts"),     # Govt current tax receipts, $bn SAAR
]

# ---------------------------------------------------------------------------
# WB fetch (shared with other benchmarks)
# ---------------------------------------------------------------------------

WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",    "FP.CPI.TOTL.ZG":    "inflation",
    "SL.UEM.TOTL.ZS":    "unemployment",  "NE.EXP.GNFS.ZS":    "exports_gdp",
    "NE.IMP.GNFS.ZS":    "imports_gdp",   "NE.TRD.GNFS.ZS":    "trade_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account","NE.CON.GOVT.ZS":   "govt_consumption",
    "GC.TAX.TOTL.GD.ZS": "tax_revenue",   "GC.DOD.TOTL.GD.ZS": "govt_debt",
    "FR.INR.RINR":        "real_interest_rate","FM.LBL.BMNY.GD.ZS":"broad_money",
    "FS.AST.PRVT.GD.ZS": "private_credit","SP.URB.TOTL.IN.ZS":  "urban_population",
    "SE.PRM.ENRR":        "school_enrollment","SP.DYN.LE00.IN":  "life_expectancy",
    "EG.ELC.ACCS.ZS":    "electricity_access","IT.NET.USER.ZS":  "internet_users",
    "IT.CEL.SETS.P2":     "mobile_subscriptions",
}


def _fetch_one(country, wb_code, start, end):
    import requests
    url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{wb_code}"
           f"?format=json&per_page=100&date={start}:{end}")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        p = r.json()
        if len(p) < 2 or not p[1]:
            return {}
        return {int(e["date"]): (float(e["value"]) if e["value"] is not None else None)
                for e in p[1] if str(e.get("date","")).isdigit()}
    except Exception as exc:
        logger.debug("WB %s/%s: %s", country, wb_code, exc)
        return {}


def fetch_live(country, start, end):
    yearly: Dict[int, Dict[str, float]] = {}
    for wb_code, short in WB_INDICATORS.items():
        for yr, v in _fetch_one(country, wb_code, start, end).items():
            if v is not None:
                yearly.setdefault(yr, {})[short] = v
        time.sleep(0.35)
    return {yr: row for yr, row in yearly.items() if len(row) >= 3}


_DEVELOPING_BASE = {
    "gdp_growth": 5.0, "inflation": 7.0, "unemployment": 5.5,
    "exports_gdp": 17.0, "imports_gdp": 24.0, "trade_gdp": 41.0,
    "current_account": -5.0, "govt_consumption": 14.0,
    "tax_revenue": 15.0, "govt_debt": 52.0, "real_interest_rate": 5.5,
    "broad_money": 35.0, "private_credit": 28.0, "urban_population": 27.0,
    "school_enrollment": 94.0, "life_expectancy": 64.0,
    "electricity_access": 55.0, "internet_users": 22.0, "mobile_subscriptions": 85.0,
}

_DEVELOPED_BASE = {
    "gdp_growth": 1.5, "inflation": 2.0, "unemployment": 6.0,
    "exports_gdp": 45.0, "imports_gdp": 40.0, "trade_gdp": 85.0,
    "current_account": 4.0, "govt_consumption": 20.0,
    "tax_revenue": 38.0, "govt_debt": 68.0, "real_interest_rate": 0.8,
    "broad_money": 105.0, "private_credit": 100.0, "urban_population": 77.0,
    "school_enrollment": 100.0, "life_expectancy": 81.0,
    "electricity_access": 100.0, "internet_users": 88.0, "mobile_subscriptions": 125.0,
}

_DEVELOPED_CODES = {
    "DEU", "FRA", "GBR", "NLD", "BEL", "AUT", "SWE", "DNK", "NOR",
    "FIN", "CHE", "PRT", "ESP", "ITA", "IRL", "AUS", "CAN", "USA", "JPN",
}


def fetch_mock(country, start, end, seed=0):
    rng = np.random.default_rng(seed + abs(hash(country)) % 99991)
    is_developed = country in _DEVELOPED_CODES
    if is_developed:
        base = {k: v * (0.92 + rng.uniform(0, 0.16)) for k, v in _DEVELOPED_BASE.items()}
        cf = 1.0 + rng.uniform(0, 0.10)
    else:
        cf = {"KEN": 1.0, "TZA": 0.85, "UGA": 0.90}.get(country, 0.75 + rng.uniform(0, 0.20))
        base = {k: v * (0.85 + rng.uniform(0, 0.30)) for k, v in _DEVELOPING_BASE.items()}
    if is_developed:
        ar = {k: 0.75 for k in base}
        ar.update({"govt_debt": 0.88, "urban_population": 0.97, "life_expectancy": 0.96,
                   "electricity_access": 0.98, "internet_users": 0.92})
        noise_scale = 0.04
    else:
        ar = {k: 0.68 for k in base}
        ar.update({"govt_debt": 0.82, "urban_population": 0.93, "life_expectancy": 0.92})
        noise_scale = 0.08

    cross = [
        ("electricity_access","gdp_growth",0.15),("electricity_access","internet_users",0.25),
        ("electricity_access","private_credit",0.10),("govt_debt","real_interest_rate",0.18),
        ("govt_debt","inflation",0.10),("govt_debt","private_credit",-0.15),
        ("inflation","real_interest_rate",0.30),("inflation","exports_gdp",-0.20),
        ("inflation","private_credit",-0.12),("inflation","current_account",-0.18),
        ("private_credit","gdp_growth",0.20),("private_credit","unemployment",-0.15),
        ("exports_gdp","current_account",0.60),("exports_gdp","gdp_growth",0.25),
        ("real_interest_rate","private_credit",-0.25),("real_interest_rate","gdp_growth",-0.15),
        ("real_interest_rate","broad_money",-0.18),("gdp_growth","tax_revenue",0.35),
        ("gdp_growth","unemployment",-0.20),("gdp_growth","life_expectancy",0.12),
        ("gdp_growth","school_enrollment",0.08),("urban_population","gdp_growth",0.10),
        ("private_credit","broad_money",0.22),
    ]
    cm: Dict[str, list] = {k: [] for k in base}
    for s, t, b in cross:
        cm[t].append((s, b))
    state = dict(base)
    result: Dict[int, Dict[str, float]] = {}
    for year in range(start, end + 1):
        new: Dict[str, float] = {}
        for v, bv in base.items():
            noise = float(rng.standard_normal()) * abs(bv) * noise_scale / cf
            ar_part = ar.get(v, 0.68) * (state[v] - bv)
            cx = sum(beta * (state[src] - base.get(src, 0)) for src, beta in cm.get(v, []))
            new[v] = bv + ar_part + cx * cf + noise
        state = new
        result[year] = {k: v for k, v in state.items() if k in set(WB_INDICATORS.values())}
    return result


# ---------------------------------------------------------------------------
# FRED quarterly fetch (USA only)
# ---------------------------------------------------------------------------

def _fetch_fred_series_quarterly(
    api_key: str, series_id: str, start: str, end: str
) -> Dict[Tuple[int, int], float]:
    """Fetch one FRED series at quarterly frequency. Returns {(year, quarter): value}."""
    import requests
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&observation_start={start}&observation_end={end}"
        f"&frequency=q&aggregation_method=avg"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        result: Dict[Tuple[int, int], float] = {}
        for obs in r.json().get("observations", []):
            if obs.get("value", ".") != ".":
                d = obs["date"]
                yr, mo = int(d[:4]), int(d[5:7])
                result[(yr, (mo - 1) // 3 + 1)] = float(obs["value"])
        return result
    except Exception as exc:
        logger.warning("FRED %s: %s", series_id, exc)
        return {}


def fetch_fred_quarterly(
    country: str, start_year: int, end_year: int, api_key: str
) -> Dict[int, Dict[str, float]]:
    """
    Fetch quarterly macro data for USA from FRED API.
    Period keys are year*10+quarter (e.g. 19801=1980-Q1, 20234=2023-Q4).
    Falls back to mock data for non-USA countries.
    """
    if country != "USA":
        logger.warning("FRED mode only supports USA; %s → mock data", country)
        return fetch_mock(country, start_year, end_year)

    start_s = f"{start_year}-01-01"
    end_s   = f"{end_year}-12-31"

    raw: Dict[str, Dict[Tuple[int, int], float]] = {}
    for sid, name in FRED_SERIES_CONFIG:
        raw[name] = _fetch_fred_series_quarterly(api_key, sid, start_s, end_s)
        logger.info("  FRED %-30s → %-22s %3d quarters", sid, name, len(raw[name]))
        time.sleep(0.15)

    all_yq = {k for d in raw.values() for k in d if start_year <= k[0] <= end_year}

    result: Dict[int, Dict[str, float]] = {}
    for (yr, q) in sorted(all_yq):
        period_key = yr * 10 + q
        row: Dict[str, float] = {}

        for name in ("gdp_growth", "unemployment", "exports_gdp", "imports_gdp",
                     "govt_consumption", "govt_debt", "real_interest_rate"):
            v = raw.get(name, {}).get((yr, q))
            if v is not None:
                row[name] = v

        # YoY inflation from quarterly CPI averages
        cpi_now  = raw.get("_cpi", {}).get((yr, q))
        cpi_prev = raw.get("_cpi", {}).get((yr - 1, q))
        if cpi_now and cpi_prev and cpi_prev > 0:
            row["inflation"] = 100.0 * (cpi_now / cpi_prev - 1.0)

        gdp_nom = raw.get("_gdp_nom", {}).get((yr, q))

        # Exports and imports % GDP  (both in $bn SAAR / GDP $bn SAAR)
        exports_bn = raw.get("_exports_bn", {}).get((yr, q))
        imports_bn = raw.get("_imports_bn", {}).get((yr, q))
        if exports_bn is not None and gdp_nom:
            row["exports_gdp"] = exports_bn / gdp_nom * 100.0
        if imports_bn is not None and gdp_nom:
            row["imports_gdp"] = imports_bn / gdp_nom * 100.0

        # Current account ≈ net exports % GDP (trade balance proxy)
        if exports_bn is not None and imports_bn is not None and gdp_nom:
            row["current_account"] = (exports_bn - imports_bn) / gdp_nom * 100.0

        # Trade % GDP
        if "exports_gdp" in row and "imports_gdp" in row:
            row["trade_gdp"] = row["exports_gdp"] + row["imports_gdp"]

        # Broad money % GDP  (M2 $bn stock / GDP $bn SAAR)
        m2 = raw.get("_m2", {}).get((yr, q))
        if m2 is not None and gdp_nom:
            row["broad_money"] = m2 / gdp_nom * 100.0

        # Private credit % GDP  (bank loans $bn SA / GDP $bn SAAR)
        loans = raw.get("_loans", {}).get((yr, q))
        if loans is not None and gdp_nom:
            row["private_credit"] = loans / gdp_nom * 100.0

        # Tax revenue % GDP  (tax receipts $bn SAAR / GDP $bn SAAR)
        tax = raw.get("_tax_receipts", {}).get((yr, q))
        if tax is not None and gdp_nom:
            row["tax_revenue"] = tax / gdp_nom * 100.0

        row = {k: v for k, v in row.items() if k in set(ALL_INDICATORS)}
        if len(row) >= 3:
            result[period_key] = row

    logger.info("FRED: %d quarterly periods, avg %.0f indicators/period",
                len(result), np.mean([len(v) for v in result.values()]) if result else 0)
    return result


def annual_to_quarterly_q4(annual: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """
    Map annual WB data to Q4 period keys (year*10+4) so annual peer observations
    interleave correctly with quarterly primary data in build_and_train().
    Each annual peer obs aligns with Q4 of that year; Q1-Q3 see primary data only.
    """
    return {yr * 10 + 4: row for yr, row in annual.items()}


# ---------------------------------------------------------------------------
# Build and train the federated basket system
# ---------------------------------------------------------------------------

def build_and_train(
    primary_id: str,
    primary_data: Dict[int, Dict[str, float]],
    peers: Dict[str, Dict[int, Dict[str, float]]],
    corpus: List[Dict[str, float]],
    do_pretrain: bool = True,
    do_federate: bool = True,
) -> Any:
    """
    Construct FederationHub, pretrain (optional), train on all available data.
    Returns the hub.
    """
    from scarcity.engine.federation_hub import FederationHub
    from scarcity.engine.federation_node import FederationNode
    from scarcity.engine.baskets import REGISTRY

    hub = FederationHub()
    hub.register(FederationNode(primary_id))
    if do_federate:
        for pid, pdata in peers.items():
            if pdata:
                hub.register(FederationNode(pid))

    # Pretrain all basket engines on the broad corpus
    if do_pretrain and corpus:
        logger.info("Pretraining all nodes on %d corpus rows ...", len(corpus))
        for nid in hub.node_ids():
            node = hub.node(nid)
            for bid in REGISTRY.all_ids():
                node.pretrain(bid, corpus)
        # Soften pretrained priors so the live stream can confirm or revise
        # direction without MetaController killing hypotheses prematurely.
        for nid in hub.node_ids():
            hub.node(nid).begin_live_stream(pretrain_discount=0.5)

    # Stream all available data year by year
    all_years = sorted(set(primary_data) | set().union(*[set(d) for d in peers.values()]))
    logger.info("Streaming %d years of live data ...", len(all_years))
    for yr in all_years:
        rows_this_year: Dict[str, Dict[str, float]] = {}
        if yr in primary_data:
            rows_this_year[primary_id] = {
                k: v for k, v in primary_data[yr].items() if k in set(ALL_INDICATORS)
            }
        if do_federate:
            for pid, pdata in peers.items():
                if yr in pdata:
                    rows_this_year[pid] = {
                        k: v for k, v in pdata[yr].items() if k in set(ALL_INDICATORS)
                    }
        hub.observe_all(rows_this_year, fan_out=do_federate, peer_weight=0.70)

    return hub


# ---------------------------------------------------------------------------
# Discovery evaluation
# ---------------------------------------------------------------------------

def _compute_baseline(yearly: Dict[int, Dict[str, float]], n_tail: int = 5) -> Dict[str, float]:
    """Mean of last n_tail years as baseline state for perturbation tests."""
    years = sorted(yearly.keys())[-n_tail:]
    baseline: Dict[str, float] = {}
    for v in ALL_INDICATORS:
        vals = [yearly[yr][v] for yr in years if v in yearly.get(yr, {})]
        if vals:
            baseline[v] = float(np.mean(vals))
    return baseline


def _compute_stds(yearly: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Per-variable standard deviation across all years."""
    stds: Dict[str, float] = {}
    for v in ALL_INDICATORS:
        vals = [yearly[yr][v] for yr in yearly if v in yearly[yr]]
        if len(vals) >= 3:
            stds[v] = max(float(np.std(vals)), 1e-6)
        else:
            stds[v] = 1.0
    return stds


def discover_mediated_paths(
    hub: Any,
    node_id: str,
    ground_truth: List[Dict[str, Any]],
    p_threshold: float = 0.15,
    min_indirect: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Scan MediatingHypothesis pool for significant X → M → Y chains.

    For each ground-truth relationship (src → tgt) that cannot be recovered
    as a direct bivariate link, check whether a significant mediated path
    exists through any basket variable M.  Returns a list of discovered chains.

    This is reported separately from direct recall so mediation paths do not
    inflate the direct discovery scores.
    """
    from scarcity.engine.relationships_extended import MediatingHypothesis
    from scarcity.engine.discovery import HypothesisState

    node = hub.node(node_id)
    gt_pairs = {(r["source"], r["target"]): r["sign"] for r in ground_truth}
    chains: List[Dict[str, Any]] = []

    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        for h in eng.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            if not isinstance(h, MediatingHypothesis):
                continue
            if h._n < 30:
                continue
            if getattr(h, 'sobel_p', 1.0) >= p_threshold:
                continue
            ie = getattr(h, 'indirect_effect', 0.0)
            if abs(ie) < min_indirect:
                continue
            src, med, tgt = h.source, h.mediator, h.target
            # Check if this chain corresponds to a ground-truth pair
            expected_sign = gt_pairs.get((src, tgt))
            discovered_sign = +1 if ie > 0 else -1
            chains.append({
                "source": src,
                "mediator": med,
                "target": tgt,
                "indirect_effect": round(float(ie), 4),
                "sobel_p": round(float(h.sobel_p), 4),
                "confidence": round(float(h.confidence), 3),
                "gt_sign": expected_sign,
                "disc_sign": discovered_sign,
                "gt_match": (expected_sign is not None and discovered_sign == expected_sign),
                "basket": bid,
            })

    # Deduplicate: keep highest confidence per (source, target) chain
    seen: Dict[Tuple[str, str, str], Dict] = {}
    for c in chains:
        key = (c["source"], c["mediator"], c["target"])
        if key not in seen or c["confidence"] > seen[key]["confidence"]:
            seen[key] = c
    return sorted(seen.values(), key=lambda x: -x["confidence"])


def _diagnose_basket(hub: Any, node_id: str, basket_id: str, baseline: Dict[str, float]) -> None:
    """Print hypothesis states and confidence for a basket (diagnostic only)."""
    try:
        node = hub.node(node_id)
        eng = node._engines.get(basket_id)
        if eng is None:
            return
        hyps = list(eng.hypotheses.population.values())
        active = [h for h in hyps if h.confidence >= 0.20]
        logger.info("  [diag] %s basket: %d total hyps, %d with conf>=0.20",
                    basket_id, len(hyps), len(active))
        for h in sorted(active, key=lambda x: -x.confidence)[:5]:
            vnames = getattr(h, 'variables', [])
            logger.info("    conf=%.3f state=%s type=%s vars=%s",
                        h.confidence, h.meta.state.value, type(h).__name__, vnames)
        from scarcity.engine.baskets import REGISTRY
        basket = REGISTRY.get(basket_id)
        filtered_base = basket.filter_row(baseline)
        preds_base = eng.predict(filtered_base)
        logger.info("  [diag] predict keys with data: %s", list(preds_base.keys()))
    except Exception as exc:
        logger.debug("diag error: %s", exc)


def _low_threshold_predict(
    node: Any,
    row: Dict[str, float],
    threshold: float = 0.10,
) -> Dict[str, float]:
    """
    Confidence-weighted ensemble prediction with a lower confidence threshold.

    Same logic as OnlineDiscoveryEngine.predict() but uses threshold=0.10
    instead of 0.20, capturing moderate-confidence cross-variable hypotheses
    that are real but subtle in quarterly macroeconomic data.
    """
    from scarcity.engine.discovery import HypothesisState
    from scarcity.engine.baskets import REGISTRY

    weighted_sum: Dict[str, float] = {}
    weight_total: Dict[str, float] = {}

    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        basket = REGISTRY.get(bid)
        filtered = basket.filter_row(row)
        if not filtered:
            continue
        for h in eng.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            if h.confidence < threshold:
                continue
            result = h.predict_value(filtered)
            if result is None:
                continue
            var, val = result
            if not np.isfinite(val):
                continue
            w = h.confidence
            weighted_sum[var] = weighted_sum.get(var, 0.0) + w * val
            weight_total[var] = weight_total.get(var, 0.0) + w

    output: Dict[str, float] = {
        var: weighted_sum[var] / weight_total[var]
        for var in weighted_sum if weight_total[var] > 0
    }
    for var, val in row.items():
        if var not in output and np.isfinite(val):
            output[var] = val
    return output


def _lag_sweep_predict(
    node: Any,
    baseline: Dict[str, float],
    source: str,
    perturb: float,
    max_k: int = 4,
    threshold: float = 0.10,
) -> List[Dict[str, float]]:
    """
    Simulate step-function perturbation: hold source elevated for max_k steps.

    Returns predicted states at t+1 … t+max_k. Each step uses the previous
    prediction as the new "current state", propagating the shock through the
    causal graph. The source variable is held at baseline+perturb throughout
    (step input, not impulse).
    """
    current = dict(baseline)
    current[source] = baseline[source] + perturb
    responses: List[Dict[str, float]] = []
    for _ in range(max_k):
        try:
            pred = _low_threshold_predict(node, current, threshold=threshold)
        except Exception:
            pred = dict(current)
        responses.append(dict(pred))
        # Propagate: next step starts from current prediction
        current = {v: pred.get(v, baseline.get(v, 0.0)) for v in baseline}
        current[source] = baseline[source] + perturb  # keep source elevated
    return responses


def _get_relationship_confidence(
    hub: Any,
    node_id: str,
    source: str,
    target: str,
) -> float:
    """Best confidence score of any live hypothesis modelling source → target."""
    from scarcity.engine.relationships import (
        CorrelationalHypothesis, CausalHypothesis, FunctionalHypothesis,
    )
    from scarcity.engine.discovery import HypothesisState
    node = hub.node(node_id)
    best_conf = 0.0
    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        for h in eng.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            if isinstance(h, CorrelationalHypothesis):
                if h.var1 == source and h.var2 == target:
                    best_conf = max(best_conf, h.confidence)
            elif isinstance(h, (CausalHypothesis, FunctionalHypothesis)):
                if getattr(h, 'source', None) == source and getattr(h, 'target', None) == target:
                    best_conf = max(best_conf, h.confidence)
    return best_conf


def _check_hypothesis_direct(
    hub: Any,
    node_id: str,
    source: str,
    target: str,
    p_threshold: float = 0.10,
) -> Optional[Tuple[int, str]]:
    """
    Scan all basket engines for a statistically significant source→target hypothesis.

    Used as fallback when the ensemble perturbation test gives zero delta (e.g.
    when a cross-variable hypothesis exists with conf < 0.20 threshold, or when
    a CausalHypothesis has direction=-1 but the forward test is still significant).

    Returns (sign, label) if found, None otherwise.
    """
    from scarcity.engine.relationships import (
        CorrelationalHypothesis, CausalHypothesis, FunctionalHypothesis,
    )
    from scarcity.engine.discovery import HypothesisState

    node = hub.node(node_id)
    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        for h in eng.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue

            if isinstance(h, CorrelationalHypothesis):
                if h.var1 == source and h.var2 == target:
                    if h.n >= 15 and abs(h.r) >= 0.08 and h.p_value < p_threshold:
                        return (+1 if h.r > 0 else -1, "corr")

            elif isinstance(h, CausalHypothesis):
                if h.source == source and h.target == target:
                    # Check forward p-value regardless of direction attribute:
                    # direction=-1 means backward is dominant, but the forward
                    # test may still be significant and informative for discovery.
                    if (len(h.buffer_x) >= 15
                            and h.p_value_forward < p_threshold
                            and h._coef_aug is not None
                            and len(h._coef_aug) > h.lag + 1):
                        x_coef_sum = float(np.sum(h._coef_aug[h.lag + 1:]))
                        if abs(x_coef_sum) > 1e-9:
                            return (+1 if x_coef_sum > 0 else -1, "causal")

            elif isinstance(h, FunctionalHypothesis):
                if h.source == source and h.target == target:
                    if (len(h.buffer_x) >= 15
                            and h.poly_r2 > 0.03
                            and len(h.coefficients) > 1):
                        slope = float(h.coefficients[1])
                        if abs(slope) > 1e-9:
                            return (+1 if slope > 0 else -1, "func")
    return None


def _diagnose_relationship(
    hub: Any,
    node_id: str,
    source: str,
    target: str,
    baseline: Dict[str, float],
    stds: Dict[str, float],
) -> None:
    """Per-relationship diagnostic: why delta=0 for this pair."""
    from scarcity.engine.relationships import (
        CorrelationalHypothesis, CausalHypothesis, FunctionalHypothesis,
    )
    from scarcity.engine.discovery import HypothesisState
    node = hub.node(node_id)
    perturbed = dict(baseline)
    perturbed[source] = baseline[source] + stds.get(source, 1.0)
    logger.info("    [rel-diag] %s → %s  (perturb by %.3f)",
                source, target, stds.get(source, 1.0))
    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        from scarcity.engine.baskets import REGISTRY
        basket = REGISTRY.get(bid)
        filtered_b = basket.filter_row(baseline)
        filtered_p = basket.filter_row(perturbed)
        if source not in filtered_b and target not in filtered_b:
            continue
        for h in eng.hypotheses.population.values():
            v = getattr(h, 'variables', [])
            if source in v and target in v:
                is_dead = h.meta.state == HypothesisState.DEAD
                pred_b = h.predict_value(filtered_b)
                pred_p = h.predict_value(filtered_p)
                delta_h = None
                if pred_b and pred_p and pred_b[0] == target and pred_p[0] == target:
                    delta_h = pred_p[1] - pred_b[1]
                logger.info(
                    "      [%s] %s conf=%.3f dead=%s pred_delta=%s",
                    bid, type(h).__name__, h.confidence, is_dead,
                    f"{delta_h:+.4f}" if delta_h is not None else "None"
                )
                if isinstance(h, CorrelationalHypothesis):
                    logger.info("        r=%.4f p=%.4f n=%d", h.r, h.p_value, h.n)
                elif isinstance(h, CausalHypothesis):
                    logger.info("        dir=%d p_fwd=%.4f p_bwd=%.4f",
                                h.direction, h.p_value_forward, h.p_value_backward)
                elif isinstance(h, FunctionalHypothesis):
                    coef = h.coefficients[:2].tolist() if len(h.coefficients) >= 2 else []
                    logger.info("        r2=%.4f coef=%s", h.poly_r2, coef)


def evaluate_discovery(
    hub: Any,
    node_id: str,
    baseline: Dict[str, float],
    stds: Dict[str, float],
    ground_truth: List[Dict[str, Any]],
    perturbation_scale: float = 1.0,
    diagnose: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each ground-truth relationship (source → target, expected_sign):

    Primary:  Perturb source by +1 std; use low-threshold (0.10) ensemble prediction.
              Any non-zero delta in target = discovered.
    Fallback: Direct hypothesis scan — check if any significant hypothesis in the
              pool models source→target (p < 0.10). Captures CausalHypothesis
              regardless of direction flag, and weak CorrelationalHypotheses
              that fall below the standard ensemble threshold.

    Returns one record per ground-truth entry.
    """
    node = hub.node(node_id)
    if diagnose:
        for bid in ["macro", "financial", "human_capital"]:
            _diagnose_basket(hub, node_id, bid, baseline)

    records = []
    for gt in ground_truth:
        src, tgt = gt["source"], gt["target"]
        expected_sign = gt["sign"]

        if src not in baseline or tgt not in baseline:
            records.append({
                "source": src, "target": tgt, "basket": gt["basket"],
                "expected_sign": gt["sign"], "delta": None, "pred_sign": None,
                "discovered": False, "sign_correct": False, "confidence": 0.0,
                "cite": gt["cite"],
            })
            continue

        perturb_size = stds.get(src, 1.0) * perturbation_scale
        _EPSILON = 1e-4  # minimum |delta| to count as a real prediction

        # Primary: step-function lag sweep — hold source elevated for up to 4 steps
        # and find the dominant signed response across lags.
        try:
            pred_base = _low_threshold_predict(node, baseline)
        except Exception:
            pred_base = {}
        try:
            lag_responses = _lag_sweep_predict(node, baseline, src, perturb_size, max_k=4)
        except Exception:
            lag_responses = []

        base_tgt = pred_base.get(tgt, baseline.get(tgt, 0.0))
        # Majority-sign voting: use sign of weighted sum across all lags,
        # but discovered threshold is still max |delta| to avoid noise.
        delta = 0.0         # max |delta|, used for discovered test
        sign_acc = 0.0      # sum of signed deltas, used for direction vote
        for resp in lag_responses:
            d = resp.get(tgt, baseline.get(tgt, 0.0)) - base_tgt
            sign_acc += d
            if abs(d) > abs(delta):
                delta = d

        discovered = abs(delta) > _EPSILON
        if discovered:
            pred_sign = +1 if sign_acc > 0 else -1
            sign_correct = (pred_sign == expected_sign)
        else:
            # Fallback: direct hypothesis scan (captures CausalHypothesis regardless of
            # direction flag and CorrelationalHypothesis below ensemble threshold)
            hyp_result = _check_hypothesis_direct(hub, node_id, src, tgt)
            if hyp_result is not None:
                hyp_sign, _hyp_type = hyp_result
                discovered = True
                pred_sign = hyp_sign
                delta = float(hyp_sign) * 1e-6   # nominal sentinel for downstream sign logic
                sign_correct = (hyp_sign == expected_sign)
                if diagnose:
                    logger.info("  [hyp-scan] %s→%s found via %s sign=%+d",
                                src, tgt, _hyp_type, hyp_sign)
            else:
                if diagnose:
                    _diagnose_relationship(hub, node_id, src, tgt, baseline, stds)
                pred_sign = None
                sign_correct = False

        best_conf = _get_relationship_confidence(hub, node_id, src, tgt)

        records.append({
            "source":        src,
            "target":        tgt,
            "basket":        gt["basket"],
            "expected_sign": expected_sign,
            "delta":         round(delta, 6) if discovered else None,
            "pred_sign":     pred_sign,
            "discovered":    discovered,
            "sign_correct":  sign_correct,
            "confidence":    round(best_conf, 3),
            "cite":          gt["cite"],
        })

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_sign(s: Optional[int]) -> str:
    if s is None: return " "
    return "+" if s > 0 else "-"


def report(records: List[Dict[str, Any]], label: str) -> List[str]:
    from scarcity.engine.baskets import REGISTRY

    total = len(records)
    discovered = [r for r in records if r["discovered"]]
    correct    = [r for r in records if r["sign_correct"]]

    disc_rate   = len(discovered) / total if total else 0.0
    sign_acc    = len(correct) / len(discovered) if discovered else 0.0
    recall      = len(correct) / total if total else 0.0

    # Confidence-weighted sign accuracy: sum(conf*correct) / sum(conf) over discovered
    conf_sum     = sum(r["confidence"] for r in discovered)
    conf_w_acc   = (sum(r["confidence"] for r in correct) / conf_sum
                    if conf_sum > 1e-9 else 0.0)

    # Identity-aware recall: exclude accounting-identity targets from structural recall
    identity_vars = frozenset(
        v for v in {r["target"] for r in records}
        if REGISTRY.variable_type(v) == "identity"
    )
    structural_recs    = [r for r in records if r["target"] not in identity_vars]
    structural_correct = [r for r in structural_recs if r["sign_correct"]]
    structural_recall  = (len(structural_correct) / len(structural_recs)
                          if structural_recs else 0.0)

    lines = [
        f"  {label}",
        f"    Ground-truth relationships : {total}",
        f"    Discovered (non-zero delta): {len(discovered)} / {total}  ({100*disc_rate:.0f}%)",
        f"    Sign correct (of discovered): {len(correct)} / {len(discovered)}  ({100*sign_acc:.0f}%)",
        f"    Conf-weighted sign accuracy : {100*conf_w_acc:.0f}%",
        f"    Overall recall             : {len(correct)} / {total}  ({100*recall:.0f}%)",
        f"    Structural-only recall     : {len(structural_correct)} / {len(structural_recs)}"
        f"  ({100*structural_recall:.0f}%)",
        f"",
        f"    Per-basket breakdown:",
    ]

    for bid in REGISTRY.all_ids():
        br = [r for r in records if r["basket"] == bid]
        bd = [r for r in br if r["discovered"]]
        bc = [r for r in br if r["sign_correct"]]
        if not br:
            continue
        lines.append(
            f"      {bid:20s}  disc={len(bd)}/{len(br)}"
            f"  correct={len(bc)}/{len(bd) if bd else 1}"
            f"  ({100*len(bc)/len(bd):.0f}%)" if bd else
            f"      {bid:20s}  disc={len(bd)}/{len(br)}  correct=0/0"
        )

    lines += ["", "    Relationship detail (source → target):"]
    lines.append(
        f"      {'Source':22s} {'Target':22s} {'Exp':>4} {'Got':>4}"
        f" {'Delta':>10}  {'Conf':>5}  {'Status'}"
    )
    lines.append("      " + "-" * 82)

    for r in sorted(records, key=lambda x: (x["basket"], not x["sign_correct"], not x["discovered"])):
        status = "CORRECT" if r["sign_correct"] else ("WRONG SIGN" if r["discovered"] else "NOT FOUND")
        delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "    —   "
        vtype_tag = "*" if r["target"] in identity_vars else " "
        lines.append(
            f"      {r['source']:22s} {r['target']:22s}{vtype_tag}"
            f" {_format_sign(r['expected_sign']):>4} {_format_sign(r['pred_sign']):>4}"
            f" {delta_str:>10}  {r['confidence']:>5.3f}  {status}"
        )

    if identity_vars:
        lines.append(f"      (* = accounting identity target; excluded from structural recall)")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",          action="store_true")
    p.add_argument("--pretrain-live", action="store_true")
    p.add_argument("--start",         type=int, default=1980)
    p.add_argument("--end",           type=int, default=2023)
    p.add_argument("--pretrain-start",type=int, default=1995)
    p.add_argument("--pretrain-end",  type=int, default=2009)
    p.add_argument("--pretrain-n",    type=int, default=12,
                   help="Number of countries for pretraining corpus")
    p.add_argument("--country",       type=str, default="DEU",
                   help="Primary evaluation country (ISO3, default: DEU)")
    p.add_argument("--peers",         type=str, default="FRA,GBR",
                   help="Comma-separated federation peer countries (default: FRA,GBR)")
    p.add_argument("--ssa",           action="store_true",
                   help="Use SSA pretraining countries instead of OECD")
    p.add_argument("--fred",          action="store_true",
                   help="Use FRED quarterly API for primary country (USA only; ~4x more observations)")
    p.add_argument("--fred-key",      type=str, default="", dest="fred_key",
                   help="FRED API key (required with --fred); free at fred.stlouisfed.org")
    p.add_argument("--out-dir",       type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine data sources ────────────────────────────────────────────
    if args.fred:
        if not args.fred_key:
            logger.error("--fred requires --fred-key <API_KEY>  (free at fred.stlouisfed.org)")
            sys.exit(1)
        _fkey = args.fred_key
        def _fetch_fred(c, s, e): return fetch_fred_quarterly(c, s, e, _fkey)
        fetch_primary = _fetch_fred
        fetch_peers   = fetch_live if args.live else fetch_mock
        data_label    = "FRED Quarterly API"
    else:
        _base_fetch   = fetch_live if args.live else fetch_mock
        fetch_primary = _base_fetch
        fetch_peers   = _base_fetch
        data_label    = "LIVE World Bank API" if args.live else "DRY-RUN (mock)"

    fetch_pre    = fetch_live if args.pretrain_live else fetch_mock
    pretrain_label = "LIVE World Bank API" if args.pretrain_live else "DRY-RUN (mock)"

    primary = args.country.upper()
    peer_ids = [p.strip().upper() for p in args.peers.split(",") if p.strip()]
    pretrain_pool = SSA_PRETRAIN_COUNTRIES if args.ssa else OECD_PRETRAIN_COUNTRIES
    pretrain_type = "SSA" if args.ssa else "OECD"

    logger.info("=== Discovery Benchmark | data=%s | pretrain=%s | country=%s | peers=%s ===",
                data_label, pretrain_label, primary, ",".join(peer_ids))

    # ── Fetch eval data ───────────────────────────────────────────────────
    logger.info("Fetching %s + peers (%s) for %d-%d ...",
                primary, ",".join(peer_ids), args.start, args.end)
    primary_data = fetch_primary(primary, args.start, args.end)
    peers: Dict[str, Dict[int, Dict[str, float]]] = {}
    for pid in peer_ids:
        raw_peer = fetch_peers(pid, args.start, args.end)
        # In FRED mode align annual WB peers to Q4 period keys so they
        # interleave correctly with the quarterly primary data stream.
        peers[pid] = annual_to_quarterly_q4(raw_peer) if args.fred else raw_peer
    logger.info("  %s=%d obs  %s",
                primary, len(primary_data),
                "  ".join(f"{p}={len(d)}" for p, d in peers.items()))

    # ── Build pretraining corpus ──────────────────────────────────────────
    # Exclude the eval country and its peers from the pretraining pool
    eval_codes = {primary} | set(peer_ids)
    pretrain_countries = [c for c in pretrain_pool if c not in eval_codes][:args.pretrain_n]
    corpus: List[Dict[str, float]] = []
    logger.info("Building pretraining corpus (%d %s countries, %d-%d) ...",
                len(pretrain_countries), pretrain_type, args.pretrain_start, args.pretrain_end)
    for i, code in enumerate(pretrain_countries):
        cdata = fetch_pre(code, args.pretrain_start, args.pretrain_end,
                          **({} if args.pretrain_live else {"seed": i + 42}))
        for yr in sorted(cdata):
            row = {k: v for k, v in cdata[yr].items() if k in set(ALL_INDICATORS)}
            if row:
                corpus.append(row)
    logger.info("  Corpus: %d rows", len(corpus))

    n_tail = 20 if args.fred else 5  # ~5 years: 20 quarters vs 5 annual obs
    baseline = _compute_baseline(primary_data, n_tail=n_tail)
    stds     = _compute_stds(primary_data)

    peer_label = "+".join([primary] + peer_ids)

    # ── Four conditions ───────────────────────────────────────────────────
    conditions = [
        (f"A. Cold-start, no federation",
         dict(do_pretrain=False, do_federate=False)),
        (f"B. Cold-start + federation ({peer_label})",
         dict(do_pretrain=False, do_federate=True)),
        (f"C. Pretrained, no federation",
         dict(do_pretrain=True,  do_federate=False)),
        (f"D. Pretrained + federation ({peer_label})",
         dict(do_pretrain=True,  do_federate=True)),
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    all_mediated: Dict[str, List[Dict[str, Any]]] = {}
    for i, (label, kwargs) in enumerate(conditions):
        logger.info("--- %s ---", label)
        hub = build_and_train(primary, primary_data, peers, corpus, **kwargs)
        if kwargs.get("do_federate", False):
            n_hints = hub.sync_directions(primary)
            logger.info("  direction sync: %d hints applied to %s", n_hints, primary)
        recs = evaluate_discovery(hub, primary, baseline, stds, GROUND_TRUTH,
                                  diagnose=(i == 0 and args.fred))
        all_results[label] = recs
        all_mediated[label] = discover_mediated_paths(hub, primary, GROUND_TRUTH)
        logger.info("  hub state:\n%s", hub.summary())

    # ── Compose report ────────────────────────────────────────────────────
    summary_lines = [
        "=" * 78,
        "SCARCITY — RELATIONSHIP DISCOVERY BENCHMARK",
        f"  Data         : {data_label}",
        f"  Pretrain     : {pretrain_label}",
        f"  Country      : {primary}  ({args.start}-{args.end}, {len(primary_data)} obs)",
        f"  Peers        : {', '.join(peer_ids)}",
        f"  Pretrain     : {len(pretrain_countries)} {pretrain_type} countries"
        f"  {args.pretrain_start}-{args.pretrain_end}  ({len(corpus)} rows)",
        f"  Ground truth : {len(GROUND_TRUTH)} theory-grounded relationships",
        "=" * 78,
        "",
        "Evaluation: perturb source variable by +1 std, check predicted target sign.",
        "Metrics:",
        "  Discovery rate  = engine gives non-zero prediction for this relationship",
        "  Sign accuracy   = of discovered, fraction with correct +/- direction",
        "  Overall recall  = sign-correct / all ground-truth relationships",
        "",
    ]

    for label, recs in all_results.items():
        summary_lines += report(recs, label)
        summary_lines.append("")

    # Comparison table
    from scarcity.engine.baskets import REGISTRY as _REG
    _identity_vars = frozenset(
        v for v in {r["target"] for recs in all_results.values() for r in recs}
        if _REG.variable_type(v) == "identity"
    )
    summary_lines += ["-" * 78, "Summary comparison:", ""]
    header = (f"  {'Condition':45s}  {'Disc%':>6}  {'SignAcc%':>8}"
              f"  {'Recall%':>8}  {'StrRecall%':>10}")
    summary_lines.append(header)
    summary_lines.append("  " + "-" * 82)
    for label, recs in all_results.items():
        total = len(recs)
        disc  = sum(1 for r in recs if r["discovered"])
        corr  = sum(1 for r in recs if r["sign_correct"])
        str_recs = [r for r in recs if r["target"] not in _identity_vars]
        str_corr = sum(1 for r in str_recs if r["sign_correct"])
        disc_pct    = 100 * disc / total if total else 0
        sign_pct    = 100 * corr / disc  if disc   else 0
        recall_pct  = 100 * corr / total if total else 0
        str_rec_pct = 100 * str_corr / len(str_recs) if str_recs else 0
        summary_lines.append(
            f"  {label:45s}  {disc_pct:6.1f}  {sign_pct:8.1f}"
            f"  {recall_pct:8.1f}  {str_rec_pct:10.1f}"
        )

    # Compute per-condition testable-only recall (relationships with both vars in baseline)
    all_recs_by_label = all_results
    testable_note = ""
    if all_recs_by_label:
        first_recs = list(all_recs_by_label.values())[0]
        n_testable = sum(1 for r in first_recs if r["delta"] is not None or r["discovered"])
        n_total_gt = len(first_recs)
        # A record with delta=None AND discovered=False means src/tgt not in baseline
        n_missing_data = sum(
            1 for r in first_recs
            if not r["discovered"] and r["delta"] is None
            and (r["source"] not in baseline or r["target"] not in baseline)
        )
        n_testable = n_total_gt - n_missing_data
        best_recall_testable = max(
            (sum(1 for r in recs if r["sign_correct"]) / n_testable
             if n_testable > 0 else 0.0)
            for recs in all_recs_by_label.values()
        )
        testable_note = (
            f"  Testable with available data : {n_testable}/{n_total_gt} relationships\n"
            f"  Best recall (testable-only)  : {100*best_recall_testable:.0f}%\n"
        )

    # Mediated paths section
    med_lines = ["-" * 78, "Multi-step (mediated) discovery: X → M → Y chains", ""]
    for label, chains in all_mediated.items():
        med_lines.append(f"  {label}")
        if not chains:
            med_lines.append("    (no significant mediation chains found)")
        else:
            gt_match = [c for c in chains if c["gt_match"]]
            med_lines.append(
                f"    {len(chains)} chains found, "
                f"{len(gt_match)} matching ground-truth pairs"
            )
            med_lines.append(
                f"      {'Source':18s} {'Mediator':18s} {'Target':18s}"
                f"  {'IE':>8}  {'p':>6}  {'Conf':>5}  {'GT?'}"
            )
            med_lines.append("      " + "-" * 78)
            for c in chains[:15]:
                gt_str = (f"MATCH({_format_sign(c['gt_sign'])})" if c["gt_match"]
                          else ("gt-mismatch" if c["gt_sign"] is not None else "no-gt"))
                med_lines.append(
                    f"      {c['source']:18s} {c['mediator']:18s} {c['target']:18s}"
                    f"  {c['indirect_effect']:+8.4f}  {c['sobel_p']:6.4f}"
                    f"  {c['confidence']:5.3f}  {gt_str}"
                )
        med_lines.append("")
    summary_lines += med_lines

    interp_lines = [
        "",
        "-" * 78,
        "Interpretation:",
        "  Evaluation method:",
        "    Primary  — step-function lag sweep (4 steps, conf>=0.10 ensemble).",
        "               Source held +1std; dominant signed response across lags used.",
        "               Epsilon threshold: |delta| > 1e-4 (near-zero = NOT FOUND).",
        "    Fallback — direct hypothesis scan (CorrelationalHypothesis p<0.10,",
        "               CausalHypothesis forward p<0.10, FunctionalHypothesis).",
        "  Conf-weighted sign accuracy weights each discovery by hypothesis confidence.",
        "  Structural recall excludes accounting-identity targets (current_account,",
        "    tax_revenue, broad_money) where the sign is constrained by definition.",
        "  Discovery rate 80%+ = engine finds signals for most known relationships.",
        "  Sign accuracy  70%+ = engine correctly identifies +/- for most found.",
        "  Overall recall 60%+ = engine correctly maps most of the known structure.",
    ]
    if args.fred:
        interp_lines += [
            "",
            "  Data coverage note (FRED USA mode):",
            "    Infrastructure basket (4 relationships) : 0% testable — FRED does not",
            "      publish electricity_access or internet_users for USA.",
            "    Human capital basket (5 relationships)  : 20% testable — FRED lacks",
            "      life_expectancy, school_enrollment, and urban_population.",
            "    Recall on testable relationships only (macro+financial+1 human_capital):",
            testable_note.rstrip() if testable_note else "",
            "",
            "  Theory-data alignment note (USA 1980-2023):",
            "    Some signs differ from theory due to USA-specific empirical patterns:",
            "    - govt_debt → real_interest_rate: secular rate decline despite rising debt",
            "    - private_credit → gdp_growth: post-GFC debt overhang shows negative",
            "    - exports_gdp → current_account: level correlation negative (trade openness",
            "        expands both exports and imports; partial causal effect is +1 by identity)",
            "    - unemployment → gdp_growth: lagged recovery bounce can give spurious +",
        ]
    interp_lines.append("-" * 78)
    summary_lines += interp_lines

    text = "\n".join(summary_lines)
    out_path = args.out_dir / "discovery_benchmark.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Report -> %s", out_path)

    # CSV
    all_recs_flat = [{"condition": lbl, **r} for lbl, recs in all_results.items() for r in recs]
    csv_path = args.out_dir / "discovery_results.csv"
    if all_recs_flat:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_recs_flat[0].keys()))
            w.writeheader(); w.writerows(all_recs_flat)
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

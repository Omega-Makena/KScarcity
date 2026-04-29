"""
East Africa Federation Experiment
==================================
Pulls World Bank annual data for Kenya (KEN), Tanzania (TZA), Uganda (UGA)
and runs three streaming scenarios through OnlineDiscoveryEngine:

  1. Local-only   — each country's engine trained on its own data only
  2. Federated    — each country cross-trains on peer data each year
  3. Late-joiner  — Uganda joins after KEN+TZA have run for N years;
                    Uganda gets a warm-start (all prior KEN+TZA rows)
                    before seeing its own data

Metrics recorded per (country, scenario, year):
  - avg_confidence  : mean confidence of active hypotheses
  - active_hyp      : number of active hypotheses
  - step_count      : cumulative observations fed

Results written to:
  artifacts/meta/east_africa_federation_results.csv
  artifacts/meta/east_africa_late_joiner_results.csv

Usage:
    python scripts/experiment_east_africa_federation.py
    python scripts/experiment_east_africa_federation.py --start 2000 --end 2023
    python scripts/experiment_east_africa_federation.py --dry-run  # offline mock data
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment.east_africa")

# ---------------------------------------------------------------------------
# Indicator set — World Bank API codes  →  short names
# Mirrors KEY_INDICATORS in kshiked/ui/kenya_data_loader.py
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

COUNTRIES = {
    "KEN": "Kenya",
    "TZA": "Tanzania",
    "UGA": "Uganda",
}

WB_API_BASE = "https://api.worldbank.org/v2"
LATE_JOINER_YEARS = 10   # years KEN+TZA run before UGA joins


# ---------------------------------------------------------------------------
# World Bank REST fetch
# ---------------------------------------------------------------------------

def _fetch_wb_indicator(
    country: str,
    indicator_code: str,
    start_year: int,
    end_year: int,
) -> Dict[int, Optional[float]]:
    """Fetch one indicator for one country via World Bank JSON API.

    Returns {year: value_or_None}.
    """
    import requests

    url = (
        f"{WB_API_BASE}/country/{country}/indicator/{indicator_code}"
        f"?format=json&per_page=100&date={start_year}:{end_year}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            return {}
        return {
            int(entry["date"]): (float(entry["value"]) if entry["value"] is not None else None)
            for entry in payload[1]
            if entry.get("date", "").isdigit()
        }
    except Exception as exc:
        logger.warning(f"WB fetch failed {country}/{indicator_code}: {exc}")
        return {}


def fetch_country_data(
    country_code: str,
    start_year: int,
    end_year: int,
    retry_delay: float = 0.4,
) -> Dict[int, Dict[str, float]]:
    """Fetch all WB_INDICATORS for one country.

    Returns {year: {short_name: value}} — only years with ≥ 3 non-null values.
    """
    yearly: Dict[int, Dict[str, float]] = {}

    for wb_code, short_name in WB_INDICATORS.items():
        values = _fetch_wb_indicator(country_code, wb_code, start_year, end_year)
        for year, val in values.items():
            if val is not None:
                yearly.setdefault(year, {})[short_name] = val
        time.sleep(retry_delay)

    # Filter years with too little data
    MIN_FIELDS = 3
    return {yr: row for yr, row in yearly.items() if len(row) >= MIN_FIELDS}


# ---------------------------------------------------------------------------
# Mock data for --dry-run (offline testing)
# ---------------------------------------------------------------------------

def _mock_country_data(
    country_code: str,
    start_year: int,
    end_year: int,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Generate synthetic WB-style rows for offline testing."""
    import random
    rng = random.Random(seed + hash(country_code) % 1000)

    base = {
        "gdp_growth":        rng.uniform(2.0, 7.0),
        "inflation":         rng.uniform(3.0, 12.0),
        "unemployment":      rng.uniform(2.0, 8.0),
        "exports_gdp":       rng.uniform(10.0, 25.0),
        "imports_gdp":       rng.uniform(12.0, 30.0),
        "trade_gdp":         rng.uniform(30.0, 55.0),
        "current_account":   rng.uniform(-8.0, 2.0),
        "govt_consumption":  rng.uniform(8.0, 18.0),
        "tax_revenue":       rng.uniform(10.0, 18.0),
        "govt_debt":         rng.uniform(20.0, 65.0),
        "real_interest_rate": rng.uniform(1.0, 10.0),
        "broad_money":       rng.uniform(20.0, 40.0),
        "private_credit":    rng.uniform(10.0, 30.0),
        "urban_population":  rng.uniform(15.0, 45.0),
        "school_enrollment": rng.uniform(70.0, 110.0),
        "life_expectancy":   rng.uniform(52.0, 68.0),
        "electricity_access": rng.uniform(10.0, 70.0),
        "internet_users":    rng.uniform(0.1, 40.0),
        "mobile_subscriptions": rng.uniform(5.0, 90.0),
    }
    result: Dict[int, Dict[str, float]] = {}
    for year in range(start_year, end_year + 1):
        row = {}
        for k, v in base.items():
            noise = rng.gauss(0, abs(v) * 0.05)
            trend = (year - start_year) * rng.uniform(-0.1, 0.2)
            row[k] = v + noise + trend
        result[year] = row
    return result


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _build_schema(field_names: List[str]) -> Dict:
    return {"fields": [{"name": n, "type": "numeric"} for n in field_names]}


def _build_engine(field_names: List[str], buffer_size: int = 50) -> "OnlineDiscoveryEngine":
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    engine = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=buffer_size)
    engine.initialize(_build_schema(field_names))
    return engine


def _avg_confidence(engine: "OnlineDiscoveryEngine") -> float:
    """Mean confidence of active hypotheses (excludes _not_ready cold-start entries)."""
    confidences = [
        float(getattr(h, "confidence", 0.0))
        for h in engine.hypotheses.population.values()
        if getattr(h, "state", "active") == "active"
        and float(getattr(h, "confidence", 0.0)) > 0.0
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _active_count(engine: "OnlineDiscoveryEngine") -> int:
    return sum(
        1 for h in engine.hypotheses.population.values()
        if getattr(h, "state", "active") == "active"
    )


# ---------------------------------------------------------------------------
# Scenario 1 & 2: local-only vs federated (all 3 countries from year 0)
# ---------------------------------------------------------------------------

def run_local_vs_federated(
    country_data: Dict[str, Dict[int, Dict[str, float]]],
    years: List[int],
) -> List[Dict]:
    """
    Run local-only and federated scenarios for all three countries.

    Federated = each year, after processing its own data, each country also
    processes the same year's row from every other country (cross-training).
    This simulates the parameter-sharing effect of the federation stack
    without requiring the full network transport layer.
    """
    logger.info("=== Scenario: Local vs Federated ===")

    all_fields = sorted({
        field
        for rows in country_data.values()
        for row in rows.values()
        for field in row
    })

    # Build one engine per country × scenario
    engines = {
        "local":   {c: _build_engine(all_fields) for c in COUNTRIES},
        "federated": {c: _build_engine(all_fields) for c in COUNTRIES},
    }

    records: List[Dict] = []

    for year in years:
        for scenario, eng_map in engines.items():
            for country_code in COUNTRIES:
                engine = eng_map[country_code]
                own_row = country_data[country_code].get(year)
                if own_row is None:
                    continue

                # Always train on own data
                engine.process_row(own_row)

                # Federated: also process peers' data from same year
                if scenario == "federated":
                    for peer in COUNTRIES:
                        if peer == country_code:
                            continue
                        peer_row = country_data[peer].get(year)
                        if peer_row is not None:
                            engine.process_row(peer_row)

                records.append({
                    "scenario":        scenario,
                    "country":         COUNTRIES[country_code],
                    "country_code":    country_code,
                    "year":            year,
                    "step_count":      engine.step_count,
                    "avg_confidence":  round(_avg_confidence(engine), 6),
                    "active_hyp":      _active_count(engine),
                })

    return records


# ---------------------------------------------------------------------------
# Scenario 3: late joiner (Uganda joins after N years)
# ---------------------------------------------------------------------------

def run_late_joiner(
    country_data: Dict[str, Dict[int, Dict[str, float]]],
    years: List[int],
    late_joiner: str = "UGA",
    pioneer_countries: Tuple[str, ...] = ("KEN", "TZA"),
    join_after_years: int = LATE_JOINER_YEARS,
) -> List[Dict]:
    """
    Late-joiner experiment.

    Pioneers run for `join_after_years` years first.  Then the late joiner
    appears in two variants:
      - cold   : starts from scratch with zero prior knowledge
      - warm   : gets all pioneer rows seen so far as a warm-start,
                 then continues on its own data
    """
    logger.info(f"=== Scenario: Late Joiner ({COUNTRIES[late_joiner]}) ===")

    all_fields = sorted({
        field
        for rows in country_data.values()
        for row in rows.values()
        for field in row
    })

    pioneer_engines: Dict[str, "OnlineDiscoveryEngine"] = {
        c: _build_engine(all_fields) for c in pioneer_countries
    }

    lj_cold = _build_engine(all_fields)
    lj_warm = _build_engine(all_fields)

    pioneer_years = years[:join_after_years]
    all_years_after = years

    records: List[Dict] = []

    # Phase 1: pioneers run alone
    pioneer_rows_seen: List[Dict[str, float]] = []

    for year in pioneer_years:
        for code in pioneer_countries:
            row = country_data[code].get(year)
            if row is not None:
                pioneer_engines[code].process_row(row)
                pioneer_rows_seen.append(row)

    # Phase 2: warm-start the warm engine with everything pioneers saw
    logger.info(
        f"Warm-starting {COUNTRIES[late_joiner]} with {len(pioneer_rows_seen)} pioneer rows"
    )
    for row in pioneer_rows_seen:
        lj_warm.process_row(row)

    # Phase 3: late joiner (both variants) process own data from join point
    join_year = years[join_after_years] if join_after_years < len(years) else years[-1]
    lj_years = [y for y in all_years_after if y >= join_year]

    for year in lj_years:
        row = country_data[late_joiner].get(year)
        if row is None:
            continue

        for variant, engine in [("cold", lj_cold), ("warm", lj_warm)]:
            engine.process_row(row)
            records.append({
                "variant":         variant,
                "country":         COUNTRIES[late_joiner],
                "country_code":    late_joiner,
                "year":            year,
                "step_count":      engine.step_count,
                "avg_confidence":  round(_avg_confidence(engine), 6),
                "active_hyp":      _active_count(engine),
                "join_year":       join_year,
                "pioneer_rows":    len(pioneer_rows_seen),
            })

    return records


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(records: List[Dict], path: Path) -> None:
    if not records:
        logger.warning(f"No records to save: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1995, help="First year to fetch")
    p.add_argument("--end",   type=int, default=2023, help="Last year to fetch")
    p.add_argument("--late-joiner-years", type=int, default=LATE_JOINER_YEARS,
                   help="Years pioneers run before Uganda joins")
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic data (no network calls)")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "artifacts" / "meta",
                   help="Output directory for CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"East Africa Federation Experiment  [{args.start}–{args.end}]")
    logger.info(f"Mode: {'DRY RUN (synthetic)' if args.dry_run else 'LIVE (World Bank API)'}")

    # ---- Fetch data --------------------------------------------------------
    country_data: Dict[str, Dict[int, Dict[str, float]]] = {}

    for code, name in COUNTRIES.items():
        logger.info(f"Fetching {name} ({code}) …")
        if args.dry_run:
            country_data[code] = _mock_country_data(code, args.start, args.end)
        else:
            country_data[code] = fetch_country_data(code, args.start, args.end)

        year_count = len(country_data[code])
        if year_count == 0:
            logger.error(f"No data for {name}. Aborting — check network or use --dry-run.")
            sys.exit(1)
        logger.info(f"  {name}: {year_count} years with data")

    # Sorted union of all years present in at least one country
    all_years = sorted({
        year
        for rows in country_data.values()
        for year in rows
    })
    logger.info(f"Year range with data: {all_years[0]}–{all_years[-1]}  ({len(all_years)} years)")

    # ---- Run scenarios -----------------------------------------------------
    logger.info("Running local-only vs federated …")
    local_fed_records = run_local_vs_federated(country_data, all_years)

    logger.info("Running late-joiner experiment …")
    late_joiner_records = run_late_joiner(
        country_data,
        all_years,
        late_joiner="UGA",
        pioneer_countries=("KEN", "TZA"),
        join_after_years=args.late_joiner_years,
    )

    # ---- Save results ------------------------------------------------------
    save_csv(local_fed_records, args.out_dir / "east_africa_local_vs_federated.csv")
    save_csv(late_joiner_records, args.out_dir / "east_africa_late_joiner.csv")

    # ---- Quick summary -----------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    if local_fed_records:
        final_year = max(r["year"] for r in local_fed_records)
        print(f"\nFinal year: {final_year}")
        print(f"{'Country':<12} {'Scenario':<12} {'Avg Confidence':>16} {'Active Hyp':>12}")
        print("-" * 55)
        for r in sorted(local_fed_records, key=lambda x: (x["country"], x["scenario"])):
            if r["year"] == final_year:
                print(
                    f"{r['country']:<12} {r['scenario']:<12}"
                    f" {r['avg_confidence']:>16.4f} {r['active_hyp']:>12}"
                )

    if late_joiner_records:
        final_year_lj = max(r["year"] for r in late_joiner_records)
        join_year = late_joiner_records[0]["join_year"]
        pioneer_rows = late_joiner_records[0]["pioneer_rows"]
        print(f"\nLate Joiner — Uganda joins at year {join_year}"
              f" (after {pioneer_rows} pioneer rows)")
        print(f"{'Variant':<10} {'Final Confidence':>18} {'Active Hyp':>12}")
        print("-" * 45)
        for r in late_joiner_records:
            if r["year"] == final_year_lj:
                print(
                    f"{r['variant']:<10} {r['avg_confidence']:>18.4f}"
                    f" {r['active_hyp']:>12}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()

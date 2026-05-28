"""
Data loader for East African World Bank macro data.

Primary source for KEN: local CSV (data/simulation/API_KEN_DS2_en_csv_v2_14659.csv)
Primary source for TZA/UGA: World Bank REST API with local JSON cache.
KEN govt_debt is sourced via fallback chain:
  1. World Bank API (GC.DOD.TOTL.GD.ZS) -- often returns empty for KEN
  2. IMF DataMapper API (GGXWDG_NGDP/KEN) -- reliable, 1998-2023
  3. Hardcoded Kenya National Treasury / IMF WEO anchor values (offline fallback)

All DataFrames are returned with standardised column names matching
ground_truth_typed.py variable names.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CSV_PATH = _ROOT / 'data' / 'simulation' / 'API_KEN_DS2_en_csv_v2_14659.csv'
_CACHE_DIR = _ROOT / 'data' / 'simulation' / 'wb_cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# World Bank API indicator codes -> standardised short names
# (matches ground_truth_typed.py variable names)
# ---------------------------------------------------------------------------

WB_INDICATORS: dict[str, str] = {
    'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
    'FP.CPI.TOTL.ZG':    'inflation_cpi',
    'SL.UEM.TOTL.ZS':    'unemployment',
    'FR.INR.RINR':        'real_interest_rate',
    'FS.AST.PRVT.GD.ZS': 'private_credit',
    'NE.CON.GOVT.ZS':    'govt_consumption',
    'NE.EXP.GNFS.ZS':    'exports_gdp',
    'NE.IMP.GNFS.ZS':    'imports_gdp',
    'BN.CAB.XOKA.GD.ZS': 'current_account',
    'NE.GDI.TOTL.ZS':    'gcf',
    'EG.ELC.ACCS.ZS':    'electricity_access',
    'IT.NET.USER.ZS':    'internet_users',
    'SE.PRM.ENRR':       'school_enrollment',
    'SP.DYN.LE00.IN':    'life_expectancy',
    'GC.DOD.TOTL.GD.ZS': 'govt_debt',
    'FM.LBL.BMNY.GD.ZS': 'broad_money',
    'SP.URB.TOTL.IN.ZS': 'urban_population',
    'IT.CEL.SETS.P2':    'mobile_subscriptions',
    'GC.TAX.TOTL.GD.ZS': 'tax_revenue',
}

# Mapping from World Bank CSV Indicator Name strings to standardised names.
# These are the exact strings that appear in the Kenya CSV after skiprows=4.
CSV_INDICATOR_MAP: dict[str, str] = {
    'GDP growth (annual %)':                                                 'gdp_growth',
    'Inflation, consumer prices (annual %)':                                 'inflation_cpi',
    'Unemployment, total (% of total labor force) (modeled ILO estimate)':  'unemployment',
    'Real interest rate (%)':                                                'real_interest_rate',
    'Domestic credit to private sector (% of GDP)':                         'private_credit',
    'General government final consumption expenditure (% of GDP)':           'govt_consumption',
    'Exports of goods and services (% of GDP)':                              'exports_gdp',
    'Imports of goods and services (% of GDP)':                              'imports_gdp',
    'Current account balance (% of GDP)':                                    'current_account',
    'Gross capital formation (% of GDP)':                                    'gcf',
    'Access to electricity (% of population)':                               'electricity_access',
    'Individuals using the Internet (% of population)':                      'internet_users',
    'School enrollment, primary (% gross)':                                  'school_enrollment',
    'Life expectancy at birth, total (years)':                               'life_expectancy',
    'Broad money (% of GDP)':                                                'broad_money',
    'Urban population (% of total population)':                              'urban_population',
    'Mobile cellular subscriptions (per 100 people)':                        'mobile_subscriptions',
    'Tax revenue (% of GDP)':                                                'tax_revenue',
    # govt_debt not in CSV — fetched from API separately
}

# Variables available in CSV (govt_debt must come from API for KEN too)
_CSV_AVAILABLE = set(CSV_INDICATOR_MAP.values())

# ---------------------------------------------------------------------------
# Kenya govt_debt hardcoded fallback (IMF WEO GGXWDG_NGDP, % of GDP)
# Used when both WB API and IMF DataMapper API are unavailable.
# Values confirmed from IMF DataMapper 2024 vintage.
# ---------------------------------------------------------------------------
_KEN_GOVT_DEBT_FALLBACK: dict[int, float] = {
    1998: 38.5, 1999: 38.4, 2000: 43.1, 2001: 41.3, 2002: 42.0,
    2003: 43.8, 2004: 40.8, 2005: 37.4, 2006: 37.1, 2007: 34.2,
    2008: 34.3, 2009: 36.0, 2010: 36.7, 2011: 35.7, 2012: 37.6,
    2013: 39.8, 2014: 41.3, 2015: 45.8, 2016: 50.4, 2017: 53.9,
    2018: 56.4, 2019: 59.1, 2020: 68.0, 2021: 68.2, 2022: 67.8,
    2023: 73.4,
}


# ---------------------------------------------------------------------------
# World Bank API fetcher with caching
# ---------------------------------------------------------------------------

def _cache_path(country: str, start: int, end: int) -> Path:
    return _CACHE_DIR / f'{country}_{start}_{end}.json'


def _fetch_wb_indicator(country: str, wb_code: str,
                        start: int, end: int) -> dict[int, float | None]:
    """Fetch one indicator from World Bank API. Returns {year: value}."""
    import requests
    url = (f'https://api.worldbank.org/v2/country/{country}/indicator/{wb_code}'
           f'?format=json&per_page=100&date={start}:{end}')
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return {}
        return {
            int(e['date']): (float(e['value']) if e['value'] is not None else None)
            for e in payload[1]
            if str(e.get('date', '')).isdigit()
        }
    except Exception as exc:
        print(f'  WB fetch warning: {country}/{wb_code}: {exc}')
        return {}


def _fetch_imf_govt_debt(country: str, start: int, end: int) -> dict[int, float]:
    """
    Fetch government debt (% of GDP) from IMF DataMapper API.

    Uses GGXWDG_NGDP indicator (General government gross debt as % of GDP).
    Returns {year: value} for years in [start, end] with non-null values.
    Returns empty dict on any error.
    """
    import requests
    url = (f'https://www.imf.org/external/datamapper/api/v1/'
           f'GGXWDG_NGDP/{country}')
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        payload = r.json()
        country_data = (
            payload.get('values', {})
            .get('GGXWDG_NGDP', {})
            .get(country, {})
        )
        result = {}
        for yr_str, val in country_data.items():
            try:
                yr = int(yr_str)
                if start <= yr <= end and val is not None:
                    result[yr] = float(val)
            except (ValueError, TypeError):
                pass
        return result
    except Exception as exc:
        print(f'  IMF DataMapper warning ({country}/GGXWDG_NGDP): {exc}')
        return {}


def fetch_country_api(country: str, start: int = 1990, end: int = 2023,
                      retry_delay: float = 0.35) -> pd.DataFrame:
    """
    Fetch all WB_INDICATORS for one country from the API with local JSON cache.

    Returns:
        DataFrame with years as index (int), standardised short names as columns.
        NaN where data is unavailable.
    """
    cache = _cache_path(country, start, end)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding='utf-8'))
        yearly = {int(yr): row for yr, row in raw.items()}
    else:
        print(f'  Fetching {country} from World Bank API ({start}-{end})...')
        yearly: dict[int, dict[str, float]] = {}
        for wb_code, short in WB_INDICATORS.items():
            vals = _fetch_wb_indicator(country, wb_code, start, end)
            for yr, v in vals.items():
                if v is not None:
                    yearly.setdefault(yr, {})[short] = v
            time.sleep(retry_delay)
        cache.write_text(json.dumps(yearly), encoding='utf-8')
        print(f'  Cached {country} to {cache}')

    # Build DataFrame
    all_years = list(range(start, end + 1))
    df = pd.DataFrame(index=all_years, columns=list(WB_INDICATORS.values()), dtype=float)
    for yr, row in yearly.items():
        if start <= yr <= end:
            for col, val in row.items():
                if col in df.columns:
                    df.loc[yr, col] = val
    df.index.name = 'year'
    return df


def load_kenya_csv(start: int = 1990, end: int = 2023) -> pd.DataFrame:
    """
    Load KEN data from local CSV for indicators available there,
    then fill govt_debt from API cache.

    Returns:
        DataFrame with years as index, standardised column names.
    """
    raw = pd.read_csv(_CSV_PATH, skiprows=4)
    year_cols = [c for c in raw.columns if c.isdigit() and start <= int(c) <= end]

    records: dict[str, list[float]] = {}
    for indicator_name, short_name in CSV_INDICATOR_MAP.items():
        match = raw[raw['Indicator Name'].str.strip() == indicator_name]
        if len(match):
            records[short_name] = match[year_cols].iloc[0].astype(float).tolist()

    int_years = [int(y) for y in year_cols]
    df = pd.DataFrame(records, index=int_years)
    df.index.name = 'year'

    # govt_debt not in CSV — try WB -> IMF -> hardcoded fallback
    if 'govt_debt' not in df.columns:
        gd_cache = _CACHE_DIR / f'KEN_govt_debt_{start}_{end}.json'
        cached_nonempty = (
            gd_cache.exists()
            and bool(json.loads(gd_cache.read_text(encoding='utf-8')))
        )
        if cached_nonempty:
            raw_gd = json.loads(gd_cache.read_text(encoding='utf-8'))
            vals_gd = {int(yr): v for yr, v in raw_gd.items() if v is not None}
            source = 'cache'
        else:
            # 1. Try World Bank API
            print('  Fetching KEN govt_debt from World Bank API...')
            vals_gd = _fetch_wb_indicator('KEN', 'GC.DOD.TOTL.GD.ZS', start, end)
            vals_gd = {yr: v for yr, v in vals_gd.items() if v is not None}
            source = 'WB'

            # 2. If WB empty, try IMF DataMapper
            if not vals_gd:
                print('  WB returned empty for KEN govt_debt; trying IMF DataMapper...')
                vals_gd = _fetch_imf_govt_debt('KEN', start, end)
                source = 'IMF'

            # 3. If IMF also empty, use hardcoded fallback
            if not vals_gd:
                print('  IMF also unavailable; using hardcoded Kenya govt_debt fallback.')
                vals_gd = {
                    yr: v for yr, v in _KEN_GOVT_DEBT_FALLBACK.items()
                    if start <= yr <= end
                }
                source = 'hardcoded'

            gd_cache.write_text(json.dumps(vals_gd), encoding='utf-8')
            print(f'  KEN govt_debt loaded from {source}: {len(vals_gd)} years '
                  f'({min(vals_gd) if vals_gd else "N/A"}-'
                  f'{max(vals_gd) if vals_gd else "N/A"})')

        df['govt_debt'] = pd.Series(vals_gd, dtype=float).reindex(df.index)

    return df.sort_index()


def load_country_data(country_code: str, start: int = 1990,
                      end: int = 2023) -> pd.DataFrame:
    """
    Load World Bank macro data for one country.

    KEN is loaded from the local CSV (faster, offline).
    TZA/UGA are fetched from the World Bank API (with local JSON cache).

    Args:
        country_code: 'KEN', 'TZA', or 'UGA' (ISO 3166-1 alpha-3)
        start, end: inclusive year range

    Returns:
        DataFrame with years as index, standardised variable names as columns.
        Missing values are NaN.
    """
    if country_code == 'KEN':
        return load_kenya_csv(start, end)
    else:
        return fetch_country_api(country_code, start, end)


def prepare_multi_country_data(
    countries: list[str] | None = None,
    start: int = 1990,
    end: int = 2023,
) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple countries. Returns {country_code: DataFrame}.

    All DataFrames have the same standardised column names.
    Rows with ALL-NaN are dropped; remaining gaps remain as NaN.
    """
    if countries is None:
        countries = ['KEN', 'TZA', 'UGA']

    result: dict[str, pd.DataFrame] = {}
    for cc in countries:
        df = load_country_data(cc, start, end)
        # Drop years where every column is NaN
        df = df.dropna(how='all')
        result[cc] = df

    # Summary
    for cc, df in result.items():
        n_rows = len(df)
        n_cols = len(df.columns)
        pct_missing = 100 * df.isnull().values.mean()
        yr_min = df.index.min() if n_rows else start
        yr_max = df.index.max() if n_rows else end
        print(f'  {cc}: {n_rows} years ({yr_min}-{yr_max}), '
              f'{n_cols} indicators, {pct_missing:.1f}% missing')

    return result


def get_variable_name_mapping() -> dict[str, str]:
    """
    Map ground truth variable names to their standardised column names.

    The standardised names already match ground truth names, so this is
    an identity mapping — provided for documentation and cross-checks.
    """
    return {name: name for name in WB_INDICATORS.values()}


def truncate_to_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the first N complete rows (all non-NaN columns present)."""
    # Use rows where all columns have data; take first n
    complete = df.dropna()
    if len(complete) >= n:
        return complete.head(n)
    # Fall back: drop only rows that are entirely NaN
    partial = df.dropna(how='all')
    return partial.head(n)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Loading Kenya data from CSV...')
    ken = load_country_data('KEN')
    print(f'  KEN shape: {ken.shape}')
    print(f'  Columns: {sorted(ken.columns.tolist())}')
    print(f'  Years: {ken.index.min()} - {ken.index.max()}')
    print(f'  Missing per column:')
    missing = ken.isnull().sum()
    for col, n in missing.items():
        flag = '  ' if n == 0 else '**'
        print(f'    {flag} {col:25s}: {n:2d} missing')

    print('\nPreparing multi-country data...')
    data = prepare_multi_country_data(['KEN'])
    print('\nFirst 3 rows of KEN:')
    print(data['KEN'].head(3).to_string())

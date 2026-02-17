"""
Kenya Economic Data Loader

Loads and processes World Bank economic indicators for Kenya
from API_KEN_DS2_en_csv_v2_14659.csv for use in the SENTINEL dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None

logger = logging.getLogger("sentinel.kenya_data")

# Path to the World Bank dataset
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv"


# Key economic indicators we want to track
KEY_INDICATORS = {
    # GDP indicators
    "GDP (current US$)": "gdp_current",
    "GDP growth (annual %)": "gdp_growth",
    "GDP per capita (current US$)": "gdp_per_capita",
    "GDP, PPP (current international $)": "gdp_ppp",
    
    # Inflation & prices
    "Inflation, consumer prices (annual %)": "inflation",
    "Inflation, GDP deflator (annual %)": "inflation_gdp_deflator",
    "Food price index": "food_price_index",
    
    # Employment
    "Unemployment, total (% of total labor force) (modeled ILO estimate)": "unemployment",
    "Employment to population ratio, 15+, total (%) (modeled ILO estimate)": "employment_ratio",
    
    # Trade
    "Exports of goods and services (% of GDP)": "exports_gdp",
    "Imports of goods and services (% of GDP)": "imports_gdp",
    "Trade (% of GDP)": "trade_gdp",
    "Current account balance (% of GDP)": "current_account",
    
    # Fiscal
    "General government final consumption expenditure (% of GDP)": "govt_consumption",
    "Tax revenue (% of GDP)": "tax_revenue",
    "Central government debt, total (% of GDP)": "govt_debt",
    
    # Monetary
    "Real interest rate (%)": "real_interest_rate",
    "Broad money (% of GDP)": "broad_money",
    "Domestic credit to private sector (% of GDP)": "private_credit",
    
    # Social indicators
    "Population, total": "population",
    "Urban population (% of total population)": "urban_population",
    "School enrollment, primary (% gross)": "school_enrollment",
    "Life expectancy at birth, total (years)": "life_expectancy",
    
    # Infrastructure & services
    "Access to electricity (% of population)": "electricity_access",
    "Individuals using the Internet (% of population)": "internet_users",
    "Mobile cellular subscriptions (per 100 people)": "mobile_subscriptions",
}


@dataclass
class EconomicTimeSeries:
    """Economic indicator time series."""
    indicator: str
    short_name: str
    years: List[int]
    values: List[float]
    unit: str = ""
    
    def get_latest(self, default: float = 0.0) -> float:
        """Get most recent non-null value."""
        for v in reversed(self.values):
            if not pd.isna(v):
                return float(v)
        return default
    
    def get_year(self, year: int, default: float = 0.0) -> float:
        """Get value for specific year."""
        try:
            idx = self.years.index(year)
            v = self.values[idx]
            return float(v) if not pd.isna(v) else default
        except (ValueError, IndexError):
            return default
    
    def get_range(self, start_year: int, end_year: int) -> List[Tuple[int, float]]:
        """Get values for year range."""
        result = []
        for year, value in zip(self.years, self.values):
            if start_year <= year <= end_year and not pd.isna(value):
                result.append((year, float(value)))
        return result


class KenyaEconomicDataLoader:
    """Loader for Kenya economic data from World Bank dataset."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_PATH
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False
        self._time_series: Dict[str, EconomicTimeSeries] = {}
    
    def load(self) -> bool:
        """Load the dataset."""
        if not HAS_PANDAS:
            logger.error("Pandas not available")
            return False
        
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return False
        
        try:
            # World Bank CSVs have 4 metadata rows at the top
            self._df = pd.read_csv(self.data_path, skiprows=4)
            self._loaded = True
            logger.info(f"Loaded {len(self._df)} indicators from {self.data_path.name}")
            
            # Extract key time series
            self._extract_time_series()
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _extract_time_series(self):
        """Extract time series for key indicators."""
        if self._df is None:
            return
        
        # Year columns are everything after 'Indicator Code'
        year_cols = [col for col in self._df.columns if col not in 
                     ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 68']]
        
        for indicator_name, short_name in KEY_INDICATORS.items():
            rows = self._df[self._df['Indicator Name'] == indicator_name]
            
            if rows.empty:
                # Try fuzzy match
                matches = self._df[self._df['Indicator Name'].str.contains(
                    indicator_name.split('(')[0].strip(), case=False, na=False)]
                if not matches.empty:
                    rows = matches.iloc[:1]
                else:
                    logger.warning(f"Indicator not found: {indicator_name}")
                    continue
            
            row = rows.iloc[0]
            years = [int(col) for col in year_cols if col.isdigit()]
            values = [row[str(year)] for year in years]
            
            # Determine unit
            unit = ""
            if "%" in indicator_name:
                unit = "%"
            elif "US$" in indicator_name:
                unit = "USD"
            elif "per 100" in indicator_name:
                unit = "per 100"
            
            self._time_series[short_name] = EconomicTimeSeries(
                indicator=indicator_name,
                short_name=short_name,
                years=years,
                values=values,
                unit=unit,
            )
    
    def get_indicator(self, short_name: str) -> Optional[EconomicTimeSeries]:
        """Get time series for an indicator."""
        return self._time_series.get(short_name)
    
    def get_latest_state(self) -> Dict[str, float]:
        """Get latest values for all key indicators."""
        if not self._loaded:
            self.load()
        
        return {
            name: ts.get_latest()
            for name, ts in self._time_series.items()
        }
    
    def get_historical_trajectory(self, indicators: List[str], start_year: int = 2000) -> pd.DataFrame:
        """Get historical trajectory for multiple indicators."""
        if not self._loaded:
            self.load()
        
        data = {}
        for name in indicators:
            ts = self._time_series.get(name)
            if ts:
                for year, value in ts.get_range(start_year, 2030):
                    if year not in data:
                        data[year] = {}
                    data[year][name] = value
        
        return pd.DataFrame.from_dict(data, orient='index').sort_index()
    
    def get_indicator_summary(self) -> Dict[str, Dict]:
        """Get summary of all loaded indicators."""
        return {
            name: {
                "indicator": ts.indicator,
                "latest": ts.get_latest(),
                "unit": ts.unit,
                "years_available": len([v for v in ts.values if not pd.isna(v)]),
                "data_range": (min(ts.years), max(ts.years)),
            }
            for name, ts in self._time_series.items()
        }


# Global loader instance
_loader: Optional[KenyaEconomicDataLoader] = None


def get_kenya_data_loader() -> KenyaEconomicDataLoader:
    """Get or create global data loader."""
    global _loader
    if _loader is None:
        _loader = KenyaEconomicDataLoader()
        _loader.load()
    return _loader


def get_latest_economic_state() -> Dict[str, float]:
    """Quick access to latest economic indicators."""
    state = get_kenya_data_loader().get_latest_state()
    
    # Try custom KNBS override for cleaner data
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from knbs_data_loader import KnbsDataLoader
        knbs = KnbsDataLoader()
        inf = knbs.get_latest_inflation()
        if inf is not None:
             logger.info(f"Overriding World Bank inflation ({state.get('inflation', 0)}%) with KNBS live data ({inf}%)")
             state['inflation'] = inf
    except Exception as e:
        logger.warning(f"KNBS override failed: {e}")
        
    return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the loader
    loader = KenyaEconomicDataLoader()
    if loader.load():
        print("✓ Data loaded successfully")
        print(f"\nLoaded {len(loader._time_series)} indicators")
        
        # Show GDP
        gdp = loader.get_indicator("gdp_current")
        if gdp:
            print(f"\nGDP (current US$):")
            print(f"  Latest: ${gdp.get_latest() / 1e9:.1f}B ({max([y for y, v in zip(gdp.years, gdp.values) if not pd.isna(v)])})")
        
        # Show inflation
        inflation = loader.get_indicator("inflation")
        if inflation:
            print(f"\nInflation:")
            print(f"  Latest: {inflation.get_latest():.1f}%")
        
        # Show unemployment
        unemployment = loader.get_indicator("unemployment")
        if unemployment:
            print(f"\nUnemployment:")
            print(f"  Latest: {unemployment.get_latest():.1f}%")
        
        print("\n" + "="*50)
        print("Summary of all indicators:")
        summary = loader.get_indicator_summary()
        for name, info in list(summary.items())[:10]:
            print(f"\n{name}:")
            print(f"  Latest value: {info['latest']:.2f} {info['unit']}")
            print(f"  Data points: {info['years_available']}")

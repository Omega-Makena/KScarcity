"""Unified economic dataset utilities for K-Shield causal adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from kshiked.ui.kenya_data_loader import KenyaEconomicDataLoader, KEY_INDICATORS

from .types import TaskWindow, BatchContext


@dataclass
class DatasetSegment:
    context: BatchContext
    data: pd.DataFrame


def load_unified_dataset(
    indicators: Optional[List[str]] = None,
    start_year: int = 1990,
    end_year: Optional[int] = None,
    time_column: str = "year",
    loader: Optional[KenyaEconomicDataLoader] = None,
) -> pd.DataFrame:
    """Load the unified economic dataset with a time column."""
    loader = loader or KenyaEconomicDataLoader()
    if not loader.load():
        raise RuntimeError("Failed to load Kenya economic dataset")

    if indicators is None:
        indicators = list(KEY_INDICATORS.values())

    df = loader.get_historical_trajectory(indicators, start_year=start_year)
    df = df.reset_index().rename(columns={"index": time_column, df.index.name or "index": time_column})

    if end_year is not None:
        df = df[df[time_column] <= end_year]

    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def segment_dataset(
    df: pd.DataFrame,
    time_column: Optional[str],
    windows: Iterable[TaskWindow],
    counties: Iterable[str],
    sectors: Iterable[str],
    county_column: str = "county",
    sector_column: str = "sector",
) -> List[DatasetSegment]:
    """Create dataset segments for windows/counties/sectors if available."""
    windows = list(windows) or [None]
    counties = list(counties) or [None]
    sectors = list(sectors) or [None]

    segments: List[DatasetSegment] = []

    for window in windows:
        df_window = df
        if window and time_column and time_column in df.columns:
            df_window = df_window[(df_window[time_column] >= window.start_year) & (df_window[time_column] <= window.end_year)]

        for county in counties:
            df_county = df_window
            if county and county_column in df_window.columns:
                df_county = df_window[df_window[county_column] == county]

            for sector in sectors:
                df_sector = df_county
                if sector and sector_column in df_county.columns:
                    df_sector = df_county[df_county[sector_column] == sector]

                context = BatchContext(window=window, county=county, sector=sector)
                segments.append(DatasetSegment(context=context, data=df_sector.copy()))

    return segments

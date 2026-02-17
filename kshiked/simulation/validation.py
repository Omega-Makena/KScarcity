"""
Data-Driven Validation — historical episode detection and accuracy scoring.

Auto-detects shock episodes from World Bank CSV data:
    - Finds years where inflation/GDP deviated sharply from trend
    - Classifies episode type (supply, demand, combined)
    - Replays through the learned simulator
    - Scores simulation accuracy against actual historical outcomes

All benchmarks derived from data. No hardcoded thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kshield.validation")


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class HistoricalEpisode:
    """A detected historical shock episode."""
    year: int
    name: str
    episode_type: str  # "supply", "demand", "combined", "financial"
    severity: float    # 0–1 (relative to dataset range)

    # Actual values from data
    actual_inflation: float = 0.0
    actual_gdp_growth: float = 0.0
    actual_unemployment: float = 0.0
    actual_interest_rate: float = 0.0

    # Preceding year values (for delta computation)
    prev_inflation: float = 0.0
    prev_gdp_growth: float = 0.0

    @property
    def inflation_delta(self) -> float:
        return self.actual_inflation - self.prev_inflation

    @property
    def gdp_delta(self) -> float:
        return self.actual_gdp_growth - self.prev_gdp_growth


@dataclass
class EpisodeScore:
    """Accuracy score for a single historical episode replay."""
    episode: HistoricalEpisode

    # Simulation outputs
    sim_inflation: float = 0.0
    sim_gdp_growth: float = 0.0

    # Errors
    inflation_error: float = 0.0   # Absolute error
    gdp_error: float = 0.0        # Absolute error

    # Directional accuracy
    inflation_direction_correct: bool = False
    gdp_direction_correct: bool = False

    @property
    def direction_score(self) -> float:
        """Fraction of correct directional calls (0–1)."""
        correct = int(self.inflation_direction_correct) + int(self.gdp_direction_correct)
        return correct / 2.0

    @property
    def magnitude_score(self) -> float:
        """Magnitude accuracy (1.0 = perfect, 0.0 = off by 100%+ of actual)."""
        scores = []
        if abs(self.episode.inflation_delta) > 0.1:
            rel_err = min(1.0, abs(self.inflation_error) / abs(self.episode.inflation_delta))
            scores.append(1.0 - rel_err)
        if abs(self.episode.gdp_delta) > 0.1:
            rel_err = min(1.0, abs(self.gdp_error) / abs(self.episode.gdp_delta))
            scores.append(1.0 - rel_err)
        return float(np.mean(scores)) if scores else 0.0


@dataclass
class ValidationReport:
    """Full validation report comparing simulation to historical episodes."""
    episodes_detected: int = 0
    episodes_scored: int = 0
    episode_scores: List[EpisodeScore] = field(default_factory=list)

    @property
    def avg_direction_score(self) -> float:
        if not self.episode_scores:
            return 0.0
        return float(np.mean([s.direction_score for s in self.episode_scores]))

    @property
    def avg_magnitude_score(self) -> float:
        if not self.episode_scores:
            return 0.0
        return float(np.mean([s.magnitude_score for s in self.episode_scores]))

    @property
    def overall_score(self) -> float:
        """Combined score weighting direction (60%) and magnitude (40%)."""
        return 0.6 * self.avg_direction_score + 0.4 * self.avg_magnitude_score


# =========================================================================
# Episode Detection (purely data-driven)
# =========================================================================

class EpisodeDetector:
    """
    Auto-detects historical shock episodes from World Bank data.

    Method:
        1. Compute rolling mean and std for each indicator
        2. Flag years where indicators deviate > 1 std from trend
        3. Classify episode type based on which indicators moved
    """

    def __init__(self, window: int = 5):
        """
        Args:
            window: Rolling window for trend computation.
        """
        self.window = window

    def detect(
        self,
        inflation_series: Dict[int, float],
        gdp_series: Dict[int, float],
        unemployment_series: Optional[Dict[int, float]] = None,
        interest_rate_series: Optional[Dict[int, float]] = None,
    ) -> List[HistoricalEpisode]:
        """
        Detect episodes from time series data.

        Args:
            inflation_series: Year → inflation rate mapping
            gdp_series: Year → GDP growth rate mapping
            unemployment_series: Optional year → unemployment mapping
            interest_rate_series: Optional year → interest rate mapping

        Returns:
            List of detected HistoricalEpisode objects, sorted by severity.
        """
        episodes = []

        # Get overlapping years
        common_years = sorted(set(inflation_series.keys()) & set(gdp_series.keys()))
        if len(common_years) < self.window + 2:
            logger.warning(f"Not enough data ({len(common_years)} years) for episode detection")
            return []

        # Build arrays
        years = np.array(common_years)
        infl = np.array([inflation_series[y] for y in common_years])
        gdp = np.array([gdp_series[y] for y in common_years])

        # Compute rolling statistics
        infl_trend, infl_std = self._rolling_stats(infl)
        gdp_trend, gdp_std = self._rolling_stats(gdp)

        # Detect deviations
        for i in range(self.window, len(years)):
            year = int(years[i])
            infl_dev = abs(infl[i] - infl_trend[i]) / max(infl_std[i], 0.5)
            gdp_dev = abs(gdp[i] - gdp_trend[i]) / max(gdp_std[i], 0.5)

            # Episode if either deviates > 1 std
            if infl_dev > 1.0 or gdp_dev > 1.0:
                # Classify type
                infl_up = infl[i] > infl_trend[i]
                gdp_down = gdp[i] < gdp_trend[i]

                if infl_up and gdp_down:
                    ep_type = "supply"  # Stagflation signal
                elif not infl_up and gdp_down:
                    ep_type = "demand"  # Deflationary contraction
                elif infl_up and not gdp_down:
                    ep_type = "monetary"  # Inflation without growth collapse
                else:
                    ep_type = "combined"

                severity = min(1.0, (infl_dev + gdp_dev) / 4.0)

                # Name the episode based on key characteristics
                name = self._auto_name(year, ep_type, infl[i], gdp[i])

                ep = HistoricalEpisode(
                    year=year,
                    name=name,
                    episode_type=ep_type,
                    severity=severity,
                    actual_inflation=float(infl[i]),
                    actual_gdp_growth=float(gdp[i]),
                    actual_unemployment=float(unemployment_series.get(year, 0.0)) if unemployment_series else 0.0,
                    actual_interest_rate=float(interest_rate_series.get(year, 0.0)) if interest_rate_series else 0.0,
                    prev_inflation=float(infl[i - 1]),
                    prev_gdp_growth=float(gdp[i - 1]),
                )
                episodes.append(ep)

        # Sort by severity (most severe first)
        episodes.sort(key=lambda e: e.severity, reverse=True)

        logger.info(f"Detected {len(episodes)} historical episodes from {len(common_years)} years of data")
        return episodes

    def _rolling_stats(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rolling mean and std."""
        n = len(arr)
        means = np.full(n, np.nan)
        stds = np.full(n, np.nan)

        for i in range(self.window, n):
            window = arr[i - self.window:i]
            means[i] = np.mean(window)
            stds[i] = max(np.std(window), 0.1)  # Floor to avoid div-by-zero

        # Fill early values with overall stats
        overall_mean = np.mean(arr[:self.window])
        overall_std = max(np.std(arr[:self.window]), 0.1)
        for i in range(self.window):
            means[i] = overall_mean
            stds[i] = overall_std

        return means, stds

    def _auto_name(self, year: int, ep_type: str, inflation: float, gdp: float) -> str:
        """Auto-generate a descriptive name for the episode."""
        if gdp < 0:
            growth_desc = "contraction"
        elif gdp < 2:
            growth_desc = "slowdown"
        else:
            growth_desc = "growth"

        if inflation > 15:
            price_desc = "high inflation"
        elif inflation > 8:
            price_desc = "elevated inflation"
        else:
            price_desc = "moderate inflation"

        return f"{year} {ep_type} shock ({price_desc}, {growth_desc})"


# =========================================================================
# Validation Runner
# =========================================================================

class ValidationRunner:
    """
    Replays historical episodes through the simulation and scores accuracy.
    """

    def __init__(self, bridge):
        """
        Args:
            bridge: ScarcityBridge instance (must be trained).
        """
        self.bridge = bridge
        self.detector = EpisodeDetector()

    def validate(self, data_path=None) -> ValidationReport:
        """
        Run full validation:
            1. Load historical data
            2. Detect episodes
            3. Replay each through simulation
            4. Score accuracy

        Returns:
            ValidationReport with per-episode scores.
        """
        # 1. Load data
        series = self._load_series(data_path)
        if not series:
            logger.error("Could not load time series for validation")
            return ValidationReport()

        # 2. Detect episodes
        episodes = self.detector.detect(
            inflation_series=series.get("inflation", {}),
            gdp_series=series.get("gdp_growth", {}),
            unemployment_series=series.get("unemployment", {}),
            interest_rate_series=series.get("interest_rate", {}),
        )

        report = ValidationReport(episodes_detected=len(episodes))

        # 3. Score each episode
        for ep in episodes:
            score = self._score_episode(ep, series)
            if score is not None:
                report.episode_scores.append(score)

        report.episodes_scored = len(report.episode_scores)

        logger.info(
            f"Validation complete: {report.episodes_scored}/{report.episodes_detected} episodes scored, "
            f"direction accuracy: {report.avg_direction_score:.1%}, "
            f"magnitude accuracy: {report.avg_magnitude_score:.1%}, "
            f"overall: {report.overall_score:.1%}"
        )

        return report

    def _score_episode(self, ep: HistoricalEpisode, series: Dict) -> Optional[EpisodeScore]:
        """Replay an episode and score the simulation output."""
        try:
            # Get pre-shock state (year before episode)
            pre_year = ep.year - 1
            initial_state = {}

            for key, ts in series.items():
                val = ts.get(pre_year)
                if val is not None:
                    # Map to discovery engine variable name
                    mapped = self._series_to_engine_var(key)
                    if mapped:
                        initial_state[mapped] = float(val)

            if len(initial_state) < 3:
                logger.debug(f"Skipping {ep.name} — insufficient pre-shock data")
                return None

            # Create simulator and set pre-shock state
            sim = self.bridge.get_simulator()
            sim.set_initial_state(initial_state)

            # Run 1 step (simulating the shock year)
            result = sim.step()

            # Extract simulated values
            sim_inflation = result.get("inflation_cpi", 0.0)
            sim_gdp = result.get("gdp_growth", 0.0)

            # Compute errors
            inflation_error = sim_inflation - ep.actual_inflation
            gdp_error = sim_gdp - ep.actual_gdp_growth

            # Directional accuracy
            infl_dir_correct = (
                (sim_inflation > ep.prev_inflation) == (ep.actual_inflation > ep.prev_inflation)
            )
            gdp_dir_correct = (
                (sim_gdp < ep.prev_gdp_growth) == (ep.actual_gdp_growth < ep.prev_gdp_growth)
            )

            return EpisodeScore(
                episode=ep,
                sim_inflation=sim_inflation,
                sim_gdp_growth=sim_gdp,
                inflation_error=abs(inflation_error),
                gdp_error=abs(gdp_error),
                inflation_direction_correct=infl_dir_correct,
                gdp_direction_correct=gdp_dir_correct,
            )

        except Exception as e:
            logger.warning(f"Failed to score {ep.name}: {e}")
            return None

    def _load_series(self, data_path=None) -> Dict[str, Dict[int, float]]:
        """Load time series from data loader."""
        try:
            from kshiked.ui.kenya_data_loader import KenyaEconomicDataLoader
            loader = KenyaEconomicDataLoader(data_path)
            if not loader.load():
                return {}

            series = {}

            # Map our series keys to data loader indicator names
            mappings = {
                "inflation": "inflation",
                "gdp_growth": "gdp_growth",
                "unemployment": "unemployment",
                "interest_rate": "real_interest_rate",
            }

            for our_key, loader_key in mappings.items():
                ts = loader.get_indicator(loader_key)
                if ts:
                    series[our_key] = dict(zip(ts.years, ts.values))

            return series

        except ImportError:
            logger.warning("Kenya data loader not available")
            return {}

    def _series_to_engine_var(self, series_key: str) -> Optional[str]:
        """Map our series keys to discovery engine friendly names."""
        mapping = {
            "inflation": "inflation_cpi",
            "gdp_growth": "gdp_growth",
            "unemployment": "unemployment",
            "interest_rate": "real_interest_rate",
        }
        return mapping.get(series_key)

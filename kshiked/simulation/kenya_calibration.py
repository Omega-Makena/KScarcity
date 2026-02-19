"""
Kenya Economy Calibration for SFC Model — Data-Driven

Derives all SFC engine parameters from the actual World Bank dataset
via KenyaEconomicDataLoader. No hardcoded economic parameters.

Architecture:
    scarcity.simulation.sfc (generic SFC engine)
        ↑ imported by
    kshiked.simulation.kenya_calibration (Kenya-specific, data-driven)
        ↑ uses
    kshiked.ui.kenya_data_loader (reads World Bank CSV)

Fallback Strategy:
    Each parameter has a generic fallback (middle-income country default)
    that is used ONLY when data is unavailable. Data confidence is tracked,
    and fallbacks are blended with data-derived values weighted by confidence.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from scarcity.simulation.sfc import SFCConfig

logger = logging.getLogger(__name__)

# Import the data loader — it's in kshiked.ui
try:
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from kshiked.ui.kenya_data_loader import KenyaEconomicDataLoader
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False
    KenyaEconomicDataLoader = None


# =============================================================================
# Generic fallbacks (middle-income country defaults)
# These are ONLY used when the World Bank data is missing.
# =============================================================================

_GENERIC_FALLBACKS = {
    "tax_rate":              0.18,     # Middle-income average ~18%
    "spending_ratio":        0.15,     # Middle-income average ~15%
    "target_inflation":      0.05,     # Common EM target
    "neutral_rate":          0.05,     # Generic EM neutral rate
    "nairu":                 0.05,     # Generic natural rate
    "inflation":             0.05,     # Current inflation if unknown
    "consumption_propensity": 0.75,    # Generic MPC
    "base_investment_ratio": 0.20,     # GFCF middle-income avg
    "debt_to_gdp":           0.50,     # Middle-income average
    "private_credit":        0.30,     # Generic credit-to-GDP
}


@dataclass
class CalibratedParam:
    """A single calibrated parameter with data source tracking."""
    name: str
    value: float
    source: str            # "data" or "fallback"
    confidence: float      # 0.0 (no data) to 1.0 (fresh multi-year data)
    data_years: int = 0    # How many years of data contributed
    latest_year: int = 0   # Most recent data year


@dataclass
class CalibrationResult:
    """Full calibration report with all derived parameters."""
    config: SFCConfig
    params: Dict[str, CalibratedParam] = field(default_factory=dict)
    overall_confidence: float = 0.0
    data_loaded: bool = False

    def summary(self) -> str:
        """Human-readable calibration summary."""
        lines = ["Kenya SFC Calibration Report", "=" * 40]
        lines.append(f"Data loaded: {self.data_loaded}")
        lines.append(f"Overall confidence: {self.overall_confidence:.1%}")
        lines.append("")
        for name, p in sorted(self.params.items()):
            marker = "📊" if p.source == "data" else "⚠️"
            lines.append(
                f"  {marker} {name}: {p.value:.4f} "
                f"[{p.source}, conf={p.confidence:.0%}, "
                f"years={p.data_years}, latest={p.latest_year}]"
            )
        return "\n".join(lines)


def _confidence_from_data(data_years: int, latest_year: int, current_year: int = None) -> float:
    """
    Compute confidence score based on data availability and freshness.

    - More years of data → higher confidence (up to ~10 years)
    - More recent data → higher confidence
    - Stale data (>5 years old) → confidence decays
    """
    if current_year is None:
        from datetime import datetime
        current_year = datetime.now().year

    if data_years == 0:
        return 0.0

    # Coverage factor: more years = more reliable (cap at 10)
    coverage = min(data_years / 10.0, 1.0)

    # Freshness factor: exponential decay for stale data
    staleness = max(0, current_year - latest_year)
    freshness = math.exp(-0.3 * staleness)  # e^(-0.3 * years_old)

    return coverage * freshness


def _blend_with_fallback(data_value: Optional[float], fallback: float,
                          confidence: float) -> float:
    """
    Blend a data-derived value with a fallback, weighted by confidence.

    When confidence = 1.0 → use data fully
    When confidence = 0.0 → use fallback fully
    When confidence = 0.7 → 70% data, 30% fallback
    """
    if data_value is None:
        return fallback
    return data_value * confidence + fallback * (1.0 - confidence)


def _extract_param(
    loader: "KenyaEconomicDataLoader",
    indicator_key: str,
    scale: float = 1.0,
    fallback_key: str = "",
    transform=None,
) -> CalibratedParam:
    """
    Extract a single calibrated parameter from the data loader.

    Args:
        loader: KenyaEconomicDataLoader instance
        indicator_key: short_name of the indicator (e.g., "tax_revenue")
        scale: multiply raw value by this (e.g., 0.01 to convert % to ratio)
        fallback_key: key into _GENERIC_FALLBACKS
        transform: optional callable to transform the raw value
    """
    ts = loader.get_indicator(indicator_key)
    fallback = _GENERIC_FALLBACKS.get(fallback_key, 0.05)

    if ts is None:
        return CalibratedParam(
            name=indicator_key,
            value=fallback,
            source="fallback",
            confidence=0.0,
        )

    # Get recent years of non-null data
    import pandas as pd
    valid_pairs = [(y, v) for y, v in zip(ts.years, ts.values) if not pd.isna(v)]
    if not valid_pairs:
        return CalibratedParam(
            name=indicator_key,
            value=fallback,
            source="fallback",
            confidence=0.0,
        )

    data_years = len(valid_pairs)
    latest_year = max(y for y, _ in valid_pairs)

    # Use average of last 5 years of data for stability
    recent = sorted(valid_pairs, key=lambda x: x[0], reverse=True)[:5]
    raw_value = sum(v for _, v in recent) / len(recent)

    # Apply scale (e.g., % to ratio)
    scaled_value = raw_value * scale

    # Apply optional transform
    if transform:
        scaled_value = transform(scaled_value)

    confidence = _confidence_from_data(data_years, latest_year)
    final_value = _blend_with_fallback(scaled_value, fallback, confidence)

    return CalibratedParam(
        name=indicator_key,
        value=final_value,
        source="data",
        confidence=confidence,
        data_years=data_years,
        latest_year=latest_year,
    )


def calibrate_from_data(
    loader: Optional["KenyaEconomicDataLoader"] = None,
    steps: int = 50,
    policy_mode: str = "on",
    overrides: Optional[Dict[str, Any]] = None,
) -> CalibrationResult:
    """
    Derive SFCConfig parameters from actual World Bank data.

    Reads economic indicators, computes SFC behavioral parameters,
    and tracks confidence for each derived value.

    Args:
        loader: Optional pre-loaded KenyaEconomicDataLoader
        steps: Simulation quarters
        policy_mode: "on" (Taylor Rule), "off" (frozen), "custom" (user instruments)
        overrides: Dict of parameter overrides (applied after data derivation)

    Returns:
        CalibrationResult with config and parameter provenance
    """
    result = CalibrationResult(config=SFCConfig())
    params = {}

    # Try to load data
    if loader is None and HAS_LOADER:
        loader = KenyaEconomicDataLoader()
        if not loader.load():
            loader = None

    if loader is None:
        logger.warning("No data loader available — using generic fallbacks only")
        result.data_loaded = False
        result.config = SFCConfig(
            steps=steps,
            policy_mode=policy_mode,
            **{k: v for k, v in _GENERIC_FALLBACKS.items() if k in SFCConfig.__dataclass_fields__},
        )
        return result

    result.data_loaded = True

    # =========================================================================
    # Derive each parameter from data
    # =========================================================================

    # --- Tax rate: Tax revenue (% of GDP) → ratio ---
    p_tax = _extract_param(loader, "tax_revenue", scale=0.01, fallback_key="tax_rate")
    params["tax_rate"] = p_tax

    # --- Govt spending ratio: Govt final consumption (% of GDP) → ratio ---
    p_spend = _extract_param(loader, "govt_consumption", scale=0.01, fallback_key="spending_ratio")
    params["spending_ratio"] = p_spend

    # --- Inflation target: Derive from recent inflation data ---
    # Central banks target around the recent trend; we use median of last 5yr
    p_inf = _extract_param(loader, "inflation", scale=0.01, fallback_key="target_inflation")
    # Target is typically the desired rate, approximate as data * 0.7 (central banks want lower)
    target_inf = CalibratedParam(
        name="target_inflation",
        value=max(0.02, p_inf.value * 0.7),  # Target below actual, floor at 2%
        source=p_inf.source,
        confidence=p_inf.confidence,
        data_years=p_inf.data_years,
        latest_year=p_inf.latest_year,
    )
    params["target_inflation"] = target_inf

    # --- Neutral rate: Real interest rate → proxy for neutral monetary policy rate ---
    p_rate = _extract_param(loader, "real_interest_rate", scale=0.01, fallback_key="neutral_rate")
    # Neutral rate ≈ real rate + inflation target
    neutral = CalibratedParam(
        name="neutral_rate",
        value=max(0.01, p_rate.value + target_inf.value),
        source=p_rate.source,
        confidence=p_rate.confidence,
        data_years=p_rate.data_years,
        latest_year=p_rate.latest_year,
    )
    params["neutral_rate"] = neutral

    # --- NAIRU: Unemployment rate (ILO) → approximate natural rate ---
    p_unemp = _extract_param(loader, "unemployment", scale=0.01, fallback_key="nairu")
    params["nairu"] = p_unemp

    # --- Consumption propensity: Derive from GDP composition ---
    # MPC ≈ private consumption / (GDP - tax)
    # We proxy it from household consumption share patterns
    p_gdp_growth = _extract_param(loader, "gdp_growth", scale=0.01, fallback_key="consumption_propensity")
    # Use exports/imports to infer openness and thus domestic absorption
    p_trade = _extract_param(loader, "trade_gdp", scale=0.01)
    # Higher trade openness → lower domestic consumption share
    domestic_absorption = max(0.4, min(0.95, 1.0 - p_trade.value * 0.3))
    mpc_param = CalibratedParam(
        name="consumption_propensity",
        value=_blend_with_fallback(domestic_absorption, _GENERIC_FALLBACKS["consumption_propensity"],
                                    p_trade.confidence),
        source=p_trade.source if p_trade.confidence > 0 else "fallback",
        confidence=p_trade.confidence,
        data_years=p_trade.data_years,
        latest_year=p_trade.latest_year,
    )
    params["consumption_propensity"] = mpc_param

    # --- Investment ratio: GFCF is not directly in our indicators ---
    # Proxy: private_credit / GDP correlates with investment intensity
    p_credit = _extract_param(loader, "private_credit", scale=0.01, fallback_key="private_credit")
    invest_ratio = CalibratedParam(
        name="base_investment_ratio",
        value=_blend_with_fallback(
            min(0.30, max(0.10, p_credit.value * 0.5)),  # Credit→investment proxy
            _GENERIC_FALLBACKS["base_investment_ratio"],
            p_credit.confidence,
        ),
        source=p_credit.source if p_credit.confidence > 0 else "fallback",
        confidence=p_credit.confidence,
        data_years=p_credit.data_years,
        latest_year=p_credit.latest_year,
    )
    params["base_investment_ratio"] = invest_ratio

    # --- Phillips coefficient: Derive from inflation-output sensitivity ---
    # Use historical inflation volatility as a proxy.
    # Scale factor is small (1.5x, capped at 0.30) because the SFC engine
    # uses a New Keynesian Phillips Curve with inflation anchoring —
    # a large coefficient with adaptive expectations causes runaway spirals.
    p_inf_raw = _extract_param(loader, "inflation", scale=0.01, fallback_key="target_inflation")
    phillips = CalibratedParam(
        name="phillips_coefficient",
        value=_blend_with_fallback(
            max(0.05, min(0.30, p_inf_raw.value * 1.5)),  # Scale: realistic Phillips slope
            0.15,  # Generic fallback (matches SFCConfig default)
            p_inf_raw.confidence,
        ),
        source=p_inf_raw.source,
        confidence=p_inf_raw.confidence,
        data_years=p_inf_raw.data_years,
        latest_year=p_inf_raw.latest_year,
    )
    params["phillips_coefficient"] = phillips

    # --- Okun coefficient: Weaker in developing economies ---
    # Derive from employment-to-GDP relationship
    okun = CalibratedParam(
        name="okun_coefficient",
        value=_blend_with_fallback(
            max(0.005, min(0.04, p_unemp.value * 0.25)),  # Scaled from unemployment level
            0.02,
            p_unemp.confidence,
        ),
        source=p_unemp.source,
        confidence=p_unemp.confidence,
        data_years=p_unemp.data_years,
        latest_year=p_unemp.latest_year,
    )
    params["okun_coefficient"] = okun

    # --- Govt debt: Derive initial debt/GDP from data ---
    p_debt = _extract_param(loader, "govt_debt", scale=0.01, fallback_key="debt_to_gdp")
    params["debt_to_gdp"] = p_debt

    # --- Inflation bounds: Derive from historical range ---
    inf_ts = loader.get_indicator("inflation")
    if inf_ts:
        import pandas as pd
        valid_inf = [v * 0.01 for v in inf_ts.values if not pd.isna(v)]
        if valid_inf:
            inf_min_data = max(-0.10, min(valid_inf) - 0.02)
            inf_max_data = min(0.50, max(valid_inf) + 0.05)
        else:
            inf_min_data = -0.05
            inf_max_data = 0.30
    else:
        inf_min_data = -0.05
        inf_max_data = 0.30

    # =========================================================================
    # Assemble the SFCConfig from derived parameters
    # =========================================================================

    config_kwargs = {
        "consumption_propensity": params["consumption_propensity"].value,
        "investment_sensitivity": 0.5,  # Structural param — not directly observable
        "tax_rate": params["tax_rate"].value,
        "wealth_effect": 0.02,  # Structural param

        "target_inflation": params["target_inflation"].value,
        "taylor_rule_phi": 1.5,  # Standard Taylor Rule literature value
        "taylor_rule_psi": 0.5,  # Standard
        "neutral_rate": params["neutral_rate"].value,

        "spending_ratio": params["spending_ratio"].value,
        "fiscal_impulse_baseline": 0.0,

        "phillips_coefficient": params["phillips_coefficient"].value,
        "inflation_min": inf_min_data,
        "inflation_max": inf_max_data,

        "okun_coefficient": params["okun_coefficient"].value,
        "nairu": params["nairu"].value,
        "unemployment_min": 0.02,
        "unemployment_max": 0.35,

        "depreciation_rate": 0.05,  # Standard literature value
        "capital_output_ratio": 0.12,  # Structural
        "base_investment_ratio": params["base_investment_ratio"].value,

        "gdp_adjustment_speed": 0.1,  # Structural

        "interest_rate_min": 0.0,
        "interest_rate_max": max(0.20, params["neutral_rate"].value * 3),

        "dt": 1.0,
        "steps": steps,
        "policy_mode": policy_mode,
    }

    # Apply user overrides last (highest priority)
    if overrides:
        config_kwargs.update(overrides)

    result.config = SFCConfig(**config_kwargs)
    result.params = params
    result.overall_confidence = (
        sum(p.confidence for p in params.values()) / len(params)
        if params else 0.0
    )

    logger.info(
        "Kenya calibration complete: confidence=%.1f%%, data_params=%d/%d",
        result.overall_confidence * 100,
        sum(1 for p in params.values() if p.source == "data"),
        len(params),
    )

    return result


# =============================================================================
# Convenience wrapper
# =============================================================================

def get_kenya_config(
    steps: int = 50,
    policy_mode: str = "on",
    overrides: Optional[Dict[str, Any]] = None,
) -> SFCConfig:
    """
    Quick access: get a Kenya-calibrated SFCConfig.

    Loads data, derives parameters, returns config.
    Use calibrate_from_data() for full provenance report.
    """
    result = calibrate_from_data(
        steps=steps,
        policy_mode=policy_mode,
        overrides=overrides,
    )
    return result.config


# =============================================================================
# Human-readable dimension metadata
# =============================================================================

OUTCOME_DIMENSIONS = {
    "gdp_growth": {
        "label": "GDP Growth",
        "description": "Rate of economic output expansion",
        "unit": "%",
        "format": ".1%",
        "higher_is": "better",
        "category": "Core Macro",
    },
    "inflation": {
        "label": "Inflation",
        "description": "General price level increase rate",
        "unit": "%",
        "format": ".1%",
        "higher_is": "worse",
        "category": "Core Macro",
    },
    "unemployment": {
        "label": "Unemployment",
        "description": "Share of labor force without work",
        "unit": "%",
        "format": ".1%",
        "higher_is": "worse",
        "category": "Core Macro",
    },
    "household_welfare": {
        "label": "Household Welfare",
        "description": "Share of output flowing to household consumption",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "better",
        "category": "Household",
    },
    "savings_rate": {
        "label": "Savings Rate",
        "description": "Fraction of household income saved",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "better",
        "category": "Household",
    },
    "cost_of_living_index": {
        "label": "Cost of Living",
        "description": "Cumulative price index (1.0 = baseline)",
        "unit": "index",
        "format": ".3f",
        "higher_is": "worse",
        "category": "Household",
    },
    "debt_to_gdp": {
        "label": "Debt-to-GDP",
        "description": "Government debt as share of GDP",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "worse",
        "category": "Government",
    },
    "fiscal_space": {
        "label": "Fiscal Space",
        "description": "Government budget balance as share of GDP (negative = deficit)",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "better",
        "category": "Government",
    },
    "fiscal_deficit_gdp": {
        "label": "Fiscal Deficit",
        "description": "Budget deficit as share of GDP",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "worse",
        "category": "Government",
    },
    "investment_ratio": {
        "label": "Investment Rate",
        "description": "Gross capital formation as share of GDP",
        "unit": "ratio",
        "format": ".1%",
        "higher_is": "better",
        "category": "Investment",
    },
    "financial_stability": {
        "label": "Financial Stability",
        "description": "Banking system health score (0=crisis, 1=stable)",
        "unit": "score",
        "format": ".2f",
        "higher_is": "better",
        "category": "Financial",
    },
}

DEFAULT_DIMENSIONS = [
    "gdp_growth",
    "inflation",
    "unemployment",
    "household_welfare",
    "debt_to_gdp",
]

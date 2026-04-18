"""
Canonical sector architecture for multi-sector policy simulation.

Six sectors: economics_finance, healthcare_public_health, environment_climate,
social_demographics, education_labor, geopolitics_security.

Each sector is independently simulatable via SectorState dataclass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Sector identifiers ───────────────────────────────────────────────────────

class SectorID(str, Enum):
    ECONOMICS_FINANCE       = "economics_finance"
    HEALTHCARE_PUBLIC_HEALTH = "healthcare_public_health"
    ENVIRONMENT_CLIMATE     = "environment_climate"
    SOCIAL_DEMOGRAPHICS     = "social_demographics"
    EDUCATION_LABOR         = "education_labor"
    GEOPOLITICS_SECURITY    = "geopolitics_security"


SECTOR_LABELS: Dict[SectorID, str] = {
    SectorID.ECONOMICS_FINANCE:        "Economics & Finance",
    SectorID.HEALTHCARE_PUBLIC_HEALTH: "Healthcare & Public Health",
    SectorID.ENVIRONMENT_CLIMATE:      "Environment & Climate",
    SectorID.SOCIAL_DEMOGRAPHICS:      "Social & Demographics",
    SectorID.EDUCATION_LABOR:          "Education & Labor",
    SectorID.GEOPOLITICS_SECURITY:     "Geopolitics & Security",
}

SECTOR_COLORS: Dict[SectorID, str] = {
    SectorID.ECONOMICS_FINANCE:        "#00ff88",
    SectorID.HEALTHCARE_PUBLIC_HEALTH: "#00aaff",
    SectorID.ENVIRONMENT_CLIMATE:      "#a3e635",
    SectorID.SOCIAL_DEMOGRAPHICS:      "#f5d547",
    SectorID.EDUCATION_LABOR:          "#f97316",
    SectorID.GEOPOLITICS_SECURITY:     "#ff3366",
}

SECTOR_SHORT: Dict[SectorID, str] = {
    SectorID.ECONOMICS_FINANCE:        "Econ",
    SectorID.HEALTHCARE_PUBLIC_HEALTH: "Health",
    SectorID.ENVIRONMENT_CLIMATE:      "Env",
    SectorID.SOCIAL_DEMOGRAPHICS:      "Social",
    SectorID.EDUCATION_LABOR:          "Labor",
    SectorID.GEOPOLITICS_SECURITY:     "Security",
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SubIndicator:
    key: str
    label: str
    unit: str
    description: str
    higher_is_better: bool = True
    typical_range: Tuple[float, float] = (0.0, 1.0)
    weight: float = 1.0   # relative importance in severity calculation


@dataclass
class PolicyConflictWarning:
    sector_id: SectorID
    policy_a: str
    policy_b: str
    indicator: str
    reason: str


@dataclass
class SectorState:
    sector_id: SectorID
    baseline: Dict[str, float]          # indicator values at t=0
    projected: Dict[str, float]         # terminal indicator values
    delta: Dict[str, float]             # projected - baseline per indicator
    severity: float                     # 1–10 composite impact magnitude
    confidence: float                   # 0–100 certainty of projection
    sub_indicators: List[SubIndicator]  # ordered sub-indicator definitions
    direct_effects: List[str]           # indicator keys driven directly
    induced_effects: List[str]          # indicator keys from cross-sector spillover
    spillover_hints: List[Dict]         # [{sector_id, label, reason, magnitude_estimate}]
    model_assumptions: Dict[str, str]   # key → description (auditability)
    timeline: Dict[str, List[float]]    # {indicator_key: [values at each sim step]}

    def delta_direction(self, key: str) -> str:
        d = self.delta.get(key, 0.0)
        if abs(d) < 1e-9:
            return "flat"
        return "up" if d > 0 else "down"

    def is_improving(self, key: str) -> bool:
        ind = next((s for s in self.sub_indicators if s.key == key), None)
        if ind is None:
            return False
        d = self.delta.get(key, 0.0)
        return (d > 0) if ind.higher_is_better else (d < 0)

    def net_impact_score(self) -> float:
        """
        Weighted average of normalized deltas where positive = net improvement.
        Range: -1.0 (all worsened) to +1.0 (all improved).
        """
        total_w, total_score = 0.0, 0.0
        for ind in self.sub_indicators:
            d = self.delta.get(ind.key, 0.0)
            rng = ind.typical_range[1] - ind.typical_range[0]
            if rng < 1e-9:
                continue
            norm_d = d / rng
            signed = norm_d if ind.higher_is_better else -norm_d
            total_score += signed * ind.weight
            total_w += ind.weight
        return total_score / total_w if total_w > 0 else 0.0


# ─── Sub-indicator definitions per sector ────────────────────────────────────

SECTOR_SUB_INDICATORS: Dict[SectorID, List[SubIndicator]] = {
    SectorID.ECONOMICS_FINANCE: [
        SubIndicator("gdp_growth",        "GDP Growth",         "%",     "Real GDP growth rate",                True,  (-0.15, 0.15), 2.0),
        SubIndicator("inflation",         "Inflation",          "%",     "Consumer price inflation",            False, (0.0,  0.30),  1.5),
        SubIndicator("unemployment",      "Unemployment",       "%",     "Labor force unemployed",              False, (0.0,  0.35),  1.5),
        SubIndicator("debt_to_gdp",       "Debt / GDP",         "ratio", "Public debt as fraction of GDP",      False, (0.0,  2.0),   1.0),
        SubIndicator("investment_ratio",  "Investment Ratio",   "%",     "Gross investment / GDP",              True,  (0.0,  0.50),  1.0),
        SubIndicator("financial_stability","Financial Stability","index", "Banking sector health (0–1)",         True,  (0.0,  1.0),   1.2),
        SubIndicator("fiscal_space",      "Fiscal Space",       "index", "Govt capacity to respond",            True,  (-1.0, 1.0),   0.8),
        SubIndicator("household_net_worth","HH Net Worth",      "norm",  "Household balance sheet (normalized)",True,  (-1.0, 2.0),   0.5),
    ],
    SectorID.HEALTHCARE_PUBLIC_HEALTH: [
        SubIndicator("health_capacity",      "Health Capacity",       "index", "Health system output / baseline",     True,  (0.0, 1.5),  2.0),
        SubIndicator("disease_burden",       "Disease Burden",        "index", "Active outbreak severity (0=none)",   False, (0.0, 1.0),  2.0),
        SubIndicator("mortality_risk",       "Excess Mortality Risk", "index", "Normalized excess deaths",            False, (0.0, 1.0),  2.0),
        SubIndicator("vaccination_coverage", "Vaccination Coverage",  "%",     "Population immunized",                True,  (0.0, 1.0),  1.0),
        SubIndicator("health_spending_gdp",  "Health Spending/GDP",   "%",     "Public health expenditure share",     True,  (0.0, 0.15), 1.0),
        SubIndicator("worker_availability",  "Health Worker Index",   "index", "Workforce vs WHO target density",     True,  (0.0, 1.0),  1.5),
    ],
    SectorID.ENVIRONMENT_CLIMATE: [
        SubIndicator("water_access",        "Water Access",       "%",     "Population with safe water",              True,  (0.0, 1.0),  2.0),
        SubIndicator("crop_yield_index",    "Crop Yield Index",   "index", "Agricultural productivity vs baseline",   True,  (0.0, 2.0),  2.0),
        SubIndicator("drought_severity",    "Drought Severity",   "index", "Soil moisture deficit (0=none,1=severe)", False, (0.0, 1.0),  1.5),
        SubIndicator("flood_risk",          "Flood Risk",         "index", "Excess rainfall / flood exposure",        False, (0.0, 1.0),  1.0),
        SubIndicator("food_security_index", "Food Security",      "index", "Population food-secure fraction",         True,  (0.0, 1.0),  2.0),
        SubIndicator("env_degradation",     "Env Degradation",    "index", "Cumulative resource degradation",         False, (0.0, 1.0),  1.0),
    ],
    SectorID.SOCIAL_DEMOGRAPHICS: [
        SubIndicator("household_welfare",   "Household Welfare",  "index", "Composite consumption welfare",           True,  (0.0,  2.0),  2.0),
        SubIndicator("cost_of_living_index","Cost of Living",     "index", "Consumer cost basket (1=baseline)",       False, (0.5,  3.0),  1.5),
        SubIndicator("displacement_rate",   "Displacement Rate",  "%",     "Population displaced",                    False, (0.0,  0.25), 2.0),
        SubIndicator("poverty_headcount",   "Poverty Headcount",  "%",     "Pop below $2.15/day PPP",                 False, (0.0,  1.0),  1.5),
        SubIndicator("inequality_index",    "Inequality (Gini)",  "index", "Income Gini coefficient",                 False, (0.0,  1.0),  1.0),
        SubIndicator("social_cohesion",     "Social Cohesion",    "index", "Community trust and stability",           True,  (0.0,  1.0),  1.0),
    ],
    SectorID.EDUCATION_LABOR: [
        SubIndicator("labor_productivity",  "Labor Productivity", "index", "Output per worker (baseline=1.0)",        True,  (0.0,  2.0),  2.0),
        SubIndicator("employment_rate",     "Employment Rate",    "%",     "Employed / working-age population",       True,  (0.0,  1.0),  2.0),
        SubIndicator("school_attendance",   "School Attendance",  "%",     "School-age children attending",           True,  (0.0,  1.0),  1.5),
        SubIndicator("skills_mismatch",     "Skills Mismatch",    "index", "Labor supply-demand mismatch",            False, (0.0,  1.0),  0.8),
        SubIndicator("human_capital_index", "Human Capital Index","index", "Long-run productivity capacity (WB HCI)", True,  (0.0,  1.0),  1.5),
        SubIndicator("real_wage_growth",    "Real Wage Growth",   "%",     "Inflation-adjusted wage change",          True,  (-0.20,0.20), 1.0),
    ],
    SectorID.GEOPOLITICS_SECURITY: [
        SubIndicator("security_stability",  "Security Stability", "index", "Conflict/crime composite (1=stable)",     True,  (0.0, 1.0),  2.0),
        SubIndicator("conflict_risk",       "Conflict Risk",      "index", "Probability of violence escalation",      False, (0.0, 1.0),  2.0),
        SubIndicator("institutional_trust", "Institutional Trust","index", "Public confidence in government",         True,  (0.0, 1.0),  1.5),
        SubIndicator("trade_disruption",    "Trade Disruption",   "index", "Import/export flow disruption",           False, (0.0, 1.0),  1.0),
        SubIndicator("border_security",     "Border Security",    "index", "Perimeter integrity composite",           True,  (0.0, 1.0),  0.8),
        SubIndicator("cyber_risk",          "Cyber Risk",         "index", "Critical infrastructure vulnerability",   False, (0.0, 1.0),  0.8),
    ],
}


# ─── Cross-sector influence weights ──────────────────────────────────────────
# INFLUENCE_WEIGHTS[from_sector][to_sector] — evidence-based Kenya defaults.
# Positive = amplifying (shock in "from" amplifies impact in "to").
# Self-weights = 1.0; off-diagonal bounded [0, 0.65].

INFLUENCE_WEIGHTS: Dict[SectorID, Dict[SectorID, float]] = {
    SectorID.ECONOMICS_FINANCE: {
        SectorID.ECONOMICS_FINANCE:        1.00,
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 0.45,   # fiscal contraction → health budget cuts
        SectorID.ENVIRONMENT_CLIMATE:      0.20,   # investment drives env exploitation
        SectorID.SOCIAL_DEMOGRAPHICS:      0.65,   # recession → welfare loss, poverty
        SectorID.EDUCATION_LABOR:          0.50,   # GDP → employment, wages
        SectorID.GEOPOLITICS_SECURITY:     0.35,   # fiscal stress → institutional fragility
    },
    SectorID.HEALTHCARE_PUBLIC_HEALTH: {
        SectorID.ECONOMICS_FINANCE:        0.40,   # health crisis → labor supply loss, GDP
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 1.00,
        SectorID.ENVIRONMENT_CLIMATE:      0.15,   # disease vectors linked to water/land
        SectorID.SOCIAL_DEMOGRAPHICS:      0.55,   # disease → displacement, poverty
        SectorID.EDUCATION_LABOR:          0.40,   # illness → school/work absence
        SectorID.GEOPOLITICS_SECURITY:     0.20,   # pandemic → social unrest
    },
    SectorID.ENVIRONMENT_CLIMATE: {
        SectorID.ECONOMICS_FINANCE:        0.35,   # drought → food inflation, GDP shock
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 0.55,   # contamination → disease burden
        SectorID.ENVIRONMENT_CLIMATE:      1.00,
        SectorID.SOCIAL_DEMOGRAPHICS:      0.50,   # displacement from floods/drought
        SectorID.EDUCATION_LABOR:          0.30,   # crop failure → child labor / dropout
        SectorID.GEOPOLITICS_SECURITY:     0.35,   # resource scarcity → conflict
    },
    SectorID.SOCIAL_DEMOGRAPHICS: {
        SectorID.ECONOMICS_FINANCE:        0.30,   # inequality → demand reduction
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 0.40,   # poverty → health vulnerability
        SectorID.ENVIRONMENT_CLIMATE:      0.20,   # overpopulation → resource pressure
        SectorID.SOCIAL_DEMOGRAPHICS:      1.00,
        SectorID.EDUCATION_LABOR:          0.50,   # displacement → school disruption
        SectorID.GEOPOLITICS_SECURITY:     0.45,   # discontent → unrest, conflict
    },
    SectorID.EDUCATION_LABOR: {
        SectorID.ECONOMICS_FINANCE:        0.55,   # human capital → long-run growth
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 0.25,   # educated workers → health outcomes
        SectorID.ENVIRONMENT_CLIMATE:      0.15,   # educated farmers → sustainability
        SectorID.SOCIAL_DEMOGRAPHICS:      0.45,   # employment → poverty reduction
        SectorID.EDUCATION_LABOR:          1.00,
        SectorID.GEOPOLITICS_SECURITY:     0.30,   # youth unemployment → radicalization
    },
    SectorID.GEOPOLITICS_SECURITY: {
        SectorID.ECONOMICS_FINANCE:        0.55,   # conflict → FDI flight, trade collapse
        SectorID.HEALTHCARE_PUBLIC_HEALTH: 0.30,   # conflict → health system attacks
        SectorID.ENVIRONMENT_CLIMATE:      0.25,   # conflict → infrastructure damage
        SectorID.SOCIAL_DEMOGRAPHICS:      0.50,   # violence → displacement, poverty
        SectorID.EDUCATION_LABOR:          0.35,   # insecurity → school closures
        SectorID.GEOPOLITICS_SECURITY:     1.00,
    },
}

ALL_SECTORS: List[SectorID] = list(SectorID)

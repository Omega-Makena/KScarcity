"""
Scenario Templates — Real-World Kenya Economic Scenarios

Pre-built shock & policy combinations calibrated to Kenya's actual
risk landscape. Each scenario translates a real-world event into
SFC engine shock vectors and policy instrument settings.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScenarioTemplate:
    """A named, realistic economic scenario with shocks and suggested policy."""
    id: str
    name: str
    description: str
    category: str  # "Supply", "Demand", "Fiscal", "External", "Combined"

    # Shock magnitudes (direct keys for SFC SHOCK_KEYS)
    shocks: Dict[str, float] = field(default_factory=dict)

    # Shock timing: which quarter does it hit? Default = quarter 5
    shock_onset: int = 5
    # How many quarters does the shock last? 0 = permanent step
    shock_duration: int = 0
    # Shape: "step" (permanent), "pulse" (temporary), "ramp" (gradual)
    shock_shape: str = "step"

    # Suggested policy response (optional preset)
    suggested_policy: Optional[Dict[str, Any]] = None

    # Which outcome dimensions are most relevant for this scenario
    suggested_dimensions: List[str] = field(default_factory=list)

    # Real-world context / narrative for the user
    context: str = ""

    def build_shock_vectors(self, steps: int = 50) -> Dict[str, np.ndarray]:
        """
        Convert this scenario's shocks into vectorized time series
        suitable for SFCConfig.shock_vectors.
        """
        vectors = {}
        for key, magnitude in self.shocks.items():
            vec = np.zeros(steps)

            if self.shock_shape == "step":
                # Permanent level shift from onset onwards
                vec[self.shock_onset:] = magnitude

            elif self.shock_shape == "pulse":
                # Temporary shock for shock_duration quarters
                end = min(self.shock_onset + max(self.shock_duration, 1), steps)
                vec[self.shock_onset:end] = magnitude

            elif self.shock_shape == "ramp":
                # Gradual build-up over shock_duration quarters, then sustain
                ramp_end = min(self.shock_onset + max(self.shock_duration, 1), steps)
                ramp_len = ramp_end - self.shock_onset
                if ramp_len > 0:
                    vec[self.shock_onset:ramp_end] = np.linspace(0, magnitude, ramp_len)
                vec[ramp_end:] = magnitude

            elif self.shock_shape == "decay":
                # Shock hits then decays exponentially
                for t in range(self.shock_onset, steps):
                    decay_factor = np.exp(-0.15 * (t - self.shock_onset))
                    vec[t] = magnitude * decay_factor

            vectors[key] = vec

        return vectors


# =============================================================================
# Policy Response Templates
# =============================================================================

POLICY_TEMPLATES = {
    "do_nothing": {
        "name": "Do Nothing",
        "description": "No policy change — let markets adjust",
        "policy_mode": "off",
        "instruments": {},
    },
    "cbk_tightening": {
        "name": "CBK Tightening",
        "description": "Central Bank raises rates by 2pp to fight inflation",
        "policy_mode": "custom",
        "instruments": {
            "custom_rate": 0.09,  # From 7% to 9%
            "crr": 0.0525,
        },
    },
    "aggressive_tightening": {
        "name": "Aggressive Tightening",
        "description": "Major rate hike + CRR increase to crush inflation",
        "policy_mode": "custom",
        "instruments": {
            "custom_rate": 0.12,  # 12%
            "crr": 0.075,       # CRR to 7.5%
        },
    },
    "fiscal_stimulus": {
        "name": "Fiscal Stimulus",
        "description": "Government increases spending and subsidies to boost demand",
        "policy_mode": "custom",
        "instruments": {
            "custom_spending_ratio": 0.18,  # From 13% to 18%
            "subsidy_rate": 0.02,           # Boost subsidies to 2%
        },
    },
    "austerity": {
        "name": "Austerity / IMF Package",
        "description": "Spending cuts + tax increases to reduce deficit",
        "policy_mode": "custom",
        "instruments": {
            "custom_tax_rate": 0.18,          # Raise taxes to 18%
            "custom_spending_ratio": 0.10,    # Cut spending to 10%
            "subsidy_rate": 0.002,            # Slash subsidies
        },
    },
    "rate_cap_2016": {
        "name": "Kenya 2016 Rate Cap",
        "description": "Interest rate cap at 4pp above CBR (Kenya 2016-2019 experiment)",
        "policy_mode": "custom",
        "instruments": {
            "rate_cap": 0.11,  # Cap at 11%
        },
    },
    "expansionary_mix": {
        "name": "Expansionary Mix",
        "description": "Lower rates + targeted subsidies + moderate spending increase",
        "policy_mode": "custom",
        "instruments": {
            "custom_rate": 0.05,              # Cut to 5%
            "custom_spending_ratio": 0.15,    # Boost to 15%
            "subsidy_rate": 0.015,            # Moderate subsidies
        },
    },
    "price_controls": {
        "name": "Price Controls (Fuel & Food)",
        "description": "Government caps fuel and food prices to control cost of living",
        "policy_mode": "custom",
        "instruments": {
            "price_controls": {"fuel": 1.05, "food": 1.03},
            "subsidy_rate": 0.02,  # Need subsidies to fund the caps
        },
    },
}


# =============================================================================
# Scenario Library
# =============================================================================

SCENARIO_LIBRARY: List[ScenarioTemplate] = [
    # --- SUPPLY SHOCKS ---
    ScenarioTemplate(
        id="oil_crisis",
        name="Oil Price Spike (+30%)",
        description="Global oil prices surge 30%, raising import costs and production expenses across the economy.",
        category="Supply",
        shocks={"supply_shock": 0.08, "fx_shock": 0.05},
        shock_onset=5,
        shock_shape="step",
        suggested_dimensions=["inflation", "cost_of_living_index", "household_welfare", "fiscal_deficit_gdp", "gdp_growth"],
        suggested_policy=POLICY_TEMPLATES["cbk_tightening"],
        context="Kenya imports ~100% of petroleum. A $30/bbl increase flows through to transport, "
                "electricity, and manufacturing costs within 2-3 months. The 2022 oil shock "
                "pushed Kenya's inflation from 6% to 9.6%.",
    ),

    ScenarioTemplate(
        id="drought",
        name="Severe Drought (-20% Agriculture)",
        description="Failed long rains devastate agriculture, cutting output by 20% and displacing rural workers.",
        category="Supply",
        shocks={"supply_shock": 0.12, "demand_shock": -0.05},
        shock_onset=5,
        shock_duration=8,
        shock_shape="pulse",
        suggested_dimensions=["household_welfare", "inflation", "unemployment", "fiscal_space", "savings_rate"],
        suggested_policy=POLICY_TEMPLATES["fiscal_stimulus"],
        context="Agriculture is ~22% of Kenya's GDP and employs ~54% of the workforce. "
                "The 2016-2017 drought cut GDP growth by 1.5pp and spiked food inflation above 20%. "
                "Drought cascades into food insecurity, rural-urban migration, and fiscal pressure "
                "for emergency spending.",
    ),

    ScenarioTemplate(
        id="food_price_surge",
        name="Food Price Surge (+25%)",
        description="Global food prices spike due to conflict or climate, raising import food costs.",
        category="Supply",
        shocks={"supply_shock": 0.06},
        shock_onset=5,
        shock_shape="ramp",
        shock_duration=6,
        suggested_dimensions=["cost_of_living_index", "household_welfare", "inflation", "savings_rate"],
        suggested_policy=POLICY_TEMPLATES["price_controls"],
        context="Kenya imports wheat, rice, and palm oil. The 2021-2022 food price surge pushed "
                "food inflation to 15.8%, disproportionately hitting low-income households.",
    ),

    # --- EXTERNAL SHOCKS ---
    ScenarioTemplate(
        id="kes_depreciation",
        name="Shilling Depreciation (-15%)",
        description="The Kenya Shilling loses 15% against the USD, raising import costs and debt service.",
        category="External",
        shocks={"fx_shock": 0.10},
        shock_onset=5,
        shock_shape="ramp",
        shock_duration=4,
        suggested_dimensions=["inflation", "debt_to_gdp", "cost_of_living_index", "gdp_growth", "financial_stability"],
        suggested_policy=POLICY_TEMPLATES["aggressive_tightening"],
        context="The KES depreciated ~20% against USD in 2023. With ~68% of Kenya's public debt "
                "in foreign currency, depreciation directly inflates the debt stock and import costs. "
                "CBK typically responds with rate hikes and FX interventions.",
    ),

    ScenarioTemplate(
        id="global_recession",
        name="Global Recession",
        description="Major trade partners enter recession, cutting demand for Kenyan exports and reducing remittances.",
        category="External",
        shocks={"demand_shock": -0.10, "fx_shock": 0.03},
        shock_onset=5,
        shock_shape="step",
        suggested_dimensions=["gdp_growth", "unemployment", "fiscal_space", "investment_ratio", "household_welfare"],
        suggested_policy=POLICY_TEMPLATES["expansionary_mix"],
        context="Kenya's exports (tea, flowers, textiles) and diaspora remittances (~$4B/yr) depend on "
                "global demand. The 2008 GFC cut Kenya's GDP growth from 7.0% to 1.5%. Tourism, "
                "a top earner, collapses in global downturns.",
    ),

    ScenarioTemplate(
        id="aid_reduction",
        name="Foreign Aid Reduction (-30%)",
        description="Development partners cut aid by 30%, creating a fiscal financing gap.",
        category="External",
        shocks={"fiscal_shock": -0.04},
        shock_onset=5,
        shock_shape="step",
        suggested_dimensions=["fiscal_space", "debt_to_gdp", "fiscal_deficit_gdp", "unemployment", "investment_ratio"],
        context="Official development assistance to Kenya ~$3.5B/yr (~3% of GDP). Aid cuts force "
                "either spending cuts, tax hikes, or increased borrowing. Health and education "
                "budgets are most vulnerable.",
    ),

    # --- FISCAL SHOCKS ---
    ScenarioTemplate(
        id="debt_crisis",
        name="Sovereign Debt Crisis",
        description="Rising debt costs + credit downgrade forces spending cuts and rate spikes.",
        category="Fiscal",
        shocks={"fiscal_shock": -0.08, "fx_shock": 0.08},
        shock_onset=5,
        shock_shape="ramp",
        shock_duration=6,
        suggested_dimensions=["debt_to_gdp", "fiscal_space", "financial_stability", "gdp_growth", "unemployment"],
        suggested_policy=POLICY_TEMPLATES["austerity"],
        context="Kenya's public debt hit 68% of GDP in 2023. If debt service exceeds 30% of revenue, "
                "the country faces a sustainability cliff. A credit downgrade raises borrowing costs "
                "and triggers capital flight. The 2024 Eurobond repayment ($2B) nearly caused a crisis.",
    ),

    # --- COMBINED SCENARIOS ---
    ScenarioTemplate(
        id="perfect_storm",
        name="Perfect Storm (Drought + Oil + FX)",
        description="Simultaneous drought, oil shock, and currency depreciation — Kenya's worst-case scenario.",
        category="Combined",
        shocks={"supply_shock": 0.15, "demand_shock": -0.05, "fx_shock": 0.10},
        shock_onset=5,
        shock_shape="step",
        suggested_dimensions=["gdp_growth", "inflation", "unemployment", "household_welfare", "debt_to_gdp", "fiscal_space", "financial_stability", "cost_of_living_index"],
        context="This combines Kenya's three most common shocks. The 2011 scenario came close: "
                "post-election violence, drought, and oil spike pushed inflation to 20% and "
                "the KES lost 25% of its value. CBK raised rates to 18% in response.",
    ),

    ScenarioTemplate(
        id="stimulus_boom",
        name="Government Stimulus Boom",
        description="Large fiscal expansion — government doubles down on spending and subsidies.",
        category="Fiscal",
        shocks={"fiscal_shock": 0.06},
        shock_onset=5,
        shock_shape="step",
        suggested_dimensions=["gdp_growth", "debt_to_gdp", "fiscal_deficit_gdp", "inflation", "investment_ratio", "household_welfare"],
        suggested_policy=POLICY_TEMPLATES["fiscal_stimulus"],
        context="Kenya periodically undertakes large infrastructure programs (SGR, expressway, "
                "housing). These boost short-term GDP but raise debt. The question: does the "
                "growth payoff exceed the debt cost?",
    ),
]


def get_scenario_by_id(scenario_id: str) -> Optional[ScenarioTemplate]:
    """Look up a scenario template by its ID."""
    for s in SCENARIO_LIBRARY:
        if s.id == scenario_id:
            return s
    return None


def get_scenarios_by_category(category: str) -> List[ScenarioTemplate]:
    """Get all scenarios in a given category."""
    return [s for s in SCENARIO_LIBRARY if s.category == category]


def get_scenario_categories() -> List[str]:
    """Get unique scenario categories."""
    return sorted(set(s.category for s in SCENARIO_LIBRARY))


def build_custom_scenario(
    name: str,
    shocks: Dict[str, float],
    shock_onset: int = 5,
    shock_duration: int = 0,
    shock_shape: str = "step",
    dimensions: Optional[List[str]] = None,
) -> ScenarioTemplate:
    """
    Build a custom scenario from user inputs.

    Args:
        name: User-provided scenario name
        shocks: Dict of shock_key -> magnitude
        shock_onset: Quarter when shock hits
        shock_duration: How long the shock lasts (0=permanent)
        shock_shape: "step", "pulse", "ramp", "decay"
        dimensions: Which outcome dimensions to track

    Returns:
        ScenarioTemplate ready for simulation
    """
    return ScenarioTemplate(
        id="custom",
        name=name,
        description=f"Custom scenario: {name}",
        category="Custom",
        shocks=shocks,
        shock_onset=shock_onset,
        shock_duration=shock_duration,
        shock_shape=shock_shape,
        suggested_dimensions=dimensions or [
            "gdp_growth", "inflation", "unemployment",
            "household_welfare", "debt_to_gdp",
        ],
    )

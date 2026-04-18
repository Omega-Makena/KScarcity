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


# =============================================================================
# Data-driven registries — fully extensible, nothing hardcoded
# =============================================================================

SHOCK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "demand_shock": {
        "label": "Demand Shock",
        "description": "Aggregate demand contraction or expansion (e.g. consumer spending, exports)",
        "min": -0.30, "max": 0.30, "default": 0.0, "step": 0.01,
        "unit": "ratio",
        "examples": "Tourism collapse (-0.10), Export boom (+0.08), Consumer confidence drop (-0.05)",
    },
    "supply_shock": {
        "label": "Supply Shock",
        "description": "Production capacity disruption (e.g. oil prices, drought, input costs)",
        "min": -0.30, "max": 0.30, "default": 0.0, "step": 0.01,
        "unit": "ratio",
        "examples": "Oil spike (+0.08), Drought (+0.12), Technology boost (-0.05)",
    },
    "fiscal_shock": {
        "label": "Fiscal Shock",
        "description": "Government revenue/spending shock (e.g. aid cuts, windfall, debt crisis)",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio",
        "examples": "Aid cut (-0.04), Stimulus (+0.06), Debt crisis (-0.08)",
    },
    "fx_shock": {
        "label": "FX / External Shock",
        "description": "Exchange rate or capital flow pressure (e.g. depreciation, capital flight)",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio",
        "examples": "KES depreciation (+0.10), Capital inflow (-0.05)",
    },

    # ── HEALTH SECTOR ──────────────────────────────────────────────────
    "cholera_outbreak": {
        "label": "Cholera Outbreak",
        "description": "Cholera/waterborne disease outbreak reduces labour force and consumer spending",
        "min": -0.30, "max": 0.30, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Health",
        "examples": "Moderate outbreak (+0.05), Severe epidemic (+0.15)",
        "sfc_mapping": {"supply_shock": 0.6, "demand_shock": 0.4},
    },
    "health_capacity_collapse": {
        "label": "Health Capacity Collapse",
        "description": "Health facility overload degrades productive capacity and forces emergency fiscal response",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Health",
        "examples": "Partial collapse (+0.06), Full system failure (+0.15)",
        "sfc_mapping": {"supply_shock": 0.7, "fiscal_shock": 0.3},
    },
    "health_worker_obstruction": {
        "label": "Health Worker Obstruction",
        "description": "Aid and health worker access blocked by insecurity or logistics failure",
        "min": -0.15, "max": 0.15, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Health",
        "examples": "Partial obstruction (+0.04), Full blockade (+0.10)",
        "sfc_mapping": {"supply_shock": 1.0},
    },

    # ── WATER & SANITATION SECTOR ──────────────────────────────────────
    "water_contamination": {
        "label": "Water Contamination",
        "description": "Water supply contamination degrades public health and productive capacity",
        "min": -0.25, "max": 0.25, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Water",
        "examples": "Localised contamination (+0.05), Widespread crisis (+0.15)",
        "sfc_mapping": {"supply_shock": 1.0},
    },
    "rainfall_flood": {
        "label": "Rainfall / Flood Event",
        "description": "Extreme rainfall damages infrastructure and disrupts economic activity",
        "min": -0.25, "max": 0.25, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Water",
        "examples": "Seasonal flooding (+0.06), Catastrophic flood (+0.18)",
        "sfc_mapping": {"supply_shock": 0.6, "demand_shock": 0.4},
    },

    # ── TRANSPORT SECTOR ───────────────────────────────────────────────
    "road_closure": {
        "label": "Road Closure / Blockage",
        "description": "Road network disruption cuts supply chains and market access",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Transport",
        "examples": "Highway blockage (+0.05), Network collapse (+0.15)",
        "sfc_mapping": {"supply_shock": 1.0},
    },
    "logistics_breakdown": {
        "label": "Logistics Breakdown",
        "description": "Distribution and logistics failure disrupts both production and delivery",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Transport",
        "examples": "Port congestion (+0.06), Full logistics failure (+0.15)",
        "sfc_mapping": {"supply_shock": 0.6, "demand_shock": 0.4},
    },

    # ── SECURITY SECTOR ────────────────────────────────────────────────
    "security_surge": {
        "label": "Security Incident Surge",
        "description": "Spike in security incidents reduces consumer activity and investor confidence",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Security",
        "examples": "Localised unrest (+0.04), Widespread incidents (+0.12)",
        "sfc_mapping": {"demand_shock": 0.6, "fx_shock": 0.4},
    },
    "civil_unrest": {
        "label": "Civil Unrest / Instability",
        "description": "Broad civil unrest disrupts output, demand, and triggers capital flight",
        "min": -0.25, "max": 0.25, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Security",
        "examples": "Protests (+0.06), Widespread unrest (+0.18)",
        "sfc_mapping": {"demand_shock": 0.4, "supply_shock": 0.3, "fx_shock": 0.3},
    },

    # ── DISPLACEMENT SECTOR ────────────────────────────────────────────
    "mass_displacement": {
        "label": "Mass Displacement",
        "description": "Population displacement disrupts consumption patterns and forces emergency fiscal spending",
        "min": -0.20, "max": 0.20, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Displacement",
        "examples": "County-level IDPs (+0.05), Regional displacement (+0.12)",
        "sfc_mapping": {"demand_shock": 0.5, "fiscal_shock": 0.5},
    },
    "refugee_influx": {
        "label": "Refugee / Cross-Border Influx",
        "description": "External refugee influx pressures fiscal resources and local labour markets",
        "min": -0.15, "max": 0.15, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Displacement",
        "examples": "Moderate influx (+0.04), Large-scale crisis (+0.10)",
        "sfc_mapping": {"fiscal_shock": 0.6, "supply_shock": 0.4},
    },

    # ── FOOD & MARKETS SECTOR ──────────────────────────────────────────
    "market_collapse": {
        "label": "Local Market Collapse",
        "description": "Market activity crash from displacement, security, or supply chain failure",
        "min": -0.25, "max": 0.25, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Food & Markets",
        "examples": "Market disruption (+0.06), Full market shutdown (+0.18)",
        "sfc_mapping": {"demand_shock": 1.0},
    },
    "food_price_spike": {
        "label": "Food Price Spike",
        "description": "Food price surge from supply chain failure, conflict, or climate event",
        "min": -0.25, "max": 0.25, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Food & Markets",
        "examples": "Moderate price rise (+0.06), Severe food inflation (+0.15)",
        "sfc_mapping": {"supply_shock": 1.0},
    },

    # ── COMMUNICATIONS SECTOR ──────────────────────────────────────────
    "misinformation_crisis": {
        "label": "Misinformation Crisis",
        "description": "Misinformation wave causes investor panic and erodes consumer confidence",
        "min": -0.15, "max": 0.15, "default": 0.0, "step": 0.01,
        "unit": "ratio", "sector": "Communications",
        "examples": "Viral rumour (+0.03), Co-ordinated misinfo campaign (+0.10)",
        "sfc_mapping": {"fx_shock": 0.6, "demand_shock": 0.4},
    },
}

POLICY_INSTRUMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "custom_rate": {
        "label": "Central Bank Rate",
        "description": "Policy interest rate set by the central bank",
        "min": 0.01, "max": 0.25, "default": 0.07, "step": 0.0025,
        "unit": "%", "display_scale": 100,
        "category": "Monetary",
    },
    "crr": {
        "label": "Cash Reserve Ratio",
        "description": "Required reserve ratio for commercial banks",
        "min": 0.0, "max": 0.15, "default": 0.0525, "step": 0.0025,
        "unit": "%", "display_scale": 100,
        "category": "Monetary",
    },
    "rate_cap": {
        "label": "Interest Rate Cap",
        "description": "Maximum lending rate ceiling (None = no cap)",
        "min": 0.05, "max": 0.30, "default": 0.11, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Monetary",
    },
    "custom_tax_rate": {
        "label": "Tax Rate",
        "description": "Effective tax rate as share of GDP",
        "min": 0.05, "max": 0.35, "default": 0.156, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Fiscal",
    },
    "custom_spending_ratio": {
        "label": "Govt Spending / GDP",
        "description": "Government consumption expenditure as share of GDP",
        "min": 0.05, "max": 0.35, "default": 0.13, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Fiscal",
    },
    "subsidy_rate": {
        "label": "Subsidies / GDP",
        "description": "Government subsidies as share of GDP",
        "min": 0.0, "max": 0.10, "default": 0.008, "step": 0.001,
        "unit": "%", "display_scale": 100,
        "category": "Fiscal",
    },

    # ── HEALTH POLICY INSTRUMENTS ──────────────────────────────────────
    "health_emergency_spending": {
        "label": "Health Emergency Spending / GDP",
        "description": "Emergency health budget allocation as share of GDP",
        "min": 0.0, "max": 0.08, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Health",
    },
    "vaccination_coverage": {
        "label": "Vaccination / Treatment Coverage",
        "description": "Target vaccination or treatment coverage rate (0-1)",
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
        "unit": "ratio", "display_scale": 100,
        "category": "Health",
    },

    # ── WATER POLICY INSTRUMENTS ───────────────────────────────────────
    "water_infra_spend": {
        "label": "Water Infrastructure Spend / GDP",
        "description": "Emergency water and sanitation infrastructure investment",
        "min": 0.0, "max": 0.05, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Water",
    },

    # ── TRANSPORT POLICY INSTRUMENTS ───────────────────────────────────
    "road_repair_budget": {
        "label": "Road Repair Budget / GDP",
        "description": "Emergency road and bridge repair allocation",
        "min": 0.0, "max": 0.05, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Transport",
    },

    # ── SECURITY POLICY INSTRUMENTS ────────────────────────────────────
    "security_deployment": {
        "label": "Security Deployment Spend / GDP",
        "description": "Security force deployment and peacekeeping budget",
        "min": 0.0, "max": 0.05, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Security",
    },

    # ── SOCIAL PROTECTION INSTRUMENTS ──────────────────────────────────
    "displacement_relief": {
        "label": "IDP Relief Fund / GDP",
        "description": "Emergency IDP and displacement relief allocation",
        "min": 0.0, "max": 0.05, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Social Protection",
    },
    "cash_transfer_rate": {
        "label": "Cash Transfers / GDP",
        "description": "Direct cash transfer programs to affected households",
        "min": 0.0, "max": 0.05, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Social Protection",
    },

    # ── MARKETS POLICY INSTRUMENTS ─────────────────────────────────────
    "price_stabilization": {
        "label": "Price Stabilisation Fund / GDP",
        "description": "Market price stabilisation interventions",
        "min": 0.0, "max": 0.03, "default": 0.0, "step": 0.005,
        "unit": "%", "display_scale": 100,
        "category": "Markets",
    },
    "food_reserve_release": {
        "label": "Strategic Food Reserve Release",
        "description": "National food reserve deployment rate (0-1)",
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1,
        "unit": "ratio", "display_scale": 100,
        "category": "Markets",
    },

    # ── COMMUNICATIONS INSTRUMENTS ─────────────────────────────────────
    "counter_misinfo_spend": {
        "label": "Counter-Misinformation Spend / GDP",
        "description": "Public communications and counter-misinformation budget",
        "min": 0.0, "max": 0.02, "default": 0.0, "step": 0.002,
        "unit": "%", "display_scale": 100,
        "category": "Communications",
    },
}

SHOCK_SHAPES = ["step", "pulse", "ramp", "decay"]


def merge_shock_vectors(
    scenarios: List["ScenarioTemplate"],
    custom_shocks: Optional[List[Dict[str, Any]]] = None,
    steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Merge N scenario shock vectors by additive superposition.

    Each scenario's build_shock_vectors() output is summed element-wise.
    Then custom_shocks (list of dicts with key, magnitude, onset, duration,
    shape) are added on top.

    Args:
        scenarios: List of ScenarioTemplate objects to combine
        custom_shocks: List of custom shock dicts, each with:
            - key: shock type key (e.g. "demand_shock")
            - magnitude: float
            - onset: int (quarter)
            - duration: int (0 = permanent)
            - shape: str ("step", "pulse", "ramp", "decay")
        steps: Simulation length in quarters

    Returns:
        Dict[str, np.ndarray] suitable for SFCConfig.shock_vectors
    """
    merged: Dict[str, np.ndarray] = {}

    # Layer 1: Preset scenarios
    for scenario in scenarios:
        for key, vec in scenario.build_shock_vectors(steps).items():
            if key in merged:
                merged[key] = merged[key] + vec
            else:
                merged[key] = vec.copy()

    # Layer 2: Custom shocks (each is an independent shock event)
    if custom_shocks:
        for cs in custom_shocks:
            key = cs.get("key", "demand_shock")
            magnitude = float(cs.get("magnitude", 0.0))
            onset = int(cs.get("onset", 5))
            duration = int(cs.get("duration", 0))
            shape = cs.get("shape", "step")

            if abs(magnitude) < 1e-9:
                continue

            # Build individual shock vector using same logic as ScenarioTemplate
            temp = ScenarioTemplate(
                id=f"custom_{key}",
                name=f"Custom {key}",
                description="User-defined custom shock",
                category="Custom",
                shocks={key: magnitude},
                shock_onset=onset,
                shock_duration=duration,
                shock_shape=shape,
            )
            for k, v in temp.build_shock_vectors(steps).items():
                if k in merged:
                    merged[k] = merged[k] + v
                else:
                    merged[k] = v.copy()

    return merged


def merge_policy_instruments(
    preset_keys: List[str],
    custom_instruments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Merge N policy presets + custom instruments into a single overrides dict.

    Later presets override earlier ones for the same key.
    Custom instruments override everything.

    Args:
        preset_keys: List of POLICY_TEMPLATES keys to layer
        custom_instruments: List of dicts with 'key' and 'value'

    Returns:
        Dict of config overrides for SFCConfig
    """
    merged: Dict[str, Any] = {}

    # Layer 1: Preset policies (in order)
    for pk in preset_keys:
        tmpl = POLICY_TEMPLATES.get(pk, {})
        instruments = tmpl.get("instruments", {})
        merged.update(instruments)

    # Layer 2: Custom instruments override everything
    if custom_instruments:
        for ci in custom_instruments:
            key = ci.get("key", "")
            value = ci.get("value")
            if key and value is not None:
                merged[key] = value

    return merged


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
    # Exponential decay rate for "decay" shape (per quarter). Calibrated per scenario.
    shock_decay_rate: float = 0.15

    # Suggested policy response (optional preset)
    suggested_policy: Optional[Dict[str, Any]] = None

    # Which outcome dimensions are most relevant for this scenario
    suggested_dimensions: List[str] = field(default_factory=list)

    # Real-world context / narrative for the user
    context: str = ""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ScenarioTemplate):
            return self.id == other.id
        return NotImplemented

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
                # Shock hits then decays exponentially at per-scenario calibrated rate
                for t in range(self.shock_onset, steps):
                    decay_factor = np.exp(-self.shock_decay_rate * (t - self.shock_onset))
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
            "subsidy_rate": 0.03,  # Subsidies to fund the price caps
        },
    },

    # ── SECTOR-SPECIFIC CRISIS POLICY RESPONSES ──────────────────────
    "health_emergency_response": {
        "name": "Health Emergency Response",
        "description": "Surge health spending + vaccination + subsidies for medical supplies",
        "policy_mode": "custom",
        "instruments": {
            "health_emergency_spending": 0.03,  # 3% GDP health surge
            "vaccination_coverage": 0.60,       # 60% coverage target
            "subsidy_rate": 0.015,              # Medical supply subsidies
        },
    },
    "water_crisis_response": {
        "name": "Water Crisis Response",
        "description": "Emergency water infrastructure + treatment subsidies",
        "policy_mode": "custom",
        "instruments": {
            "water_infra_spend": 0.02,     # 2% GDP water emergency
            "subsidy_rate": 0.01,          # Water treatment subsidies
        },
    },
    "transport_emergency": {
        "name": "Transport Emergency",
        "description": "Road repair + transport subsidies + spending boost",
        "policy_mode": "custom",
        "instruments": {
            "road_repair_budget": 0.025,          # 2.5% GDP road repair
            "custom_spending_ratio": 0.15,        # Boost spending to 15%
        },
    },
    "security_stabilization": {
        "name": "Security Stabilisation",
        "description": "Security deployment + peacekeeping + rate hold for stability signal",
        "policy_mode": "custom",
        "instruments": {
            "security_deployment": 0.02,   # 2% GDP security
            "custom_rate": 0.07,           # Hold rate steady
        },
    },
    "displacement_response": {
        "name": "Displacement Response",
        "description": "IDP relief fund + cash transfers + spending increase",
        "policy_mode": "custom",
        "instruments": {
            "displacement_relief": 0.02,          # 2% GDP relief
            "cash_transfer_rate": 0.015,          # 1.5% GDP transfers
            "custom_spending_ratio": 0.16,        # Boost spending to 16%
        },
    },
    "market_intervention": {
        "name": "Market Intervention",
        "description": "Price stabilisation + food reserves + consumer subsidies",
        "policy_mode": "custom",
        "instruments": {
            "price_stabilization": 0.015,  # 1.5% GDP stabilisation
            "food_reserve_release": 0.5,   # Release 50% of reserves
            "subsidy_rate": 0.02,          # Consumer subsidies
        },
    },
    "communications_response": {
        "name": "Communications Response",
        "description": "Counter-misinformation + rate hold for stability signal",
        "policy_mode": "custom",
        "instruments": {
            "counter_misinfo_spend": 0.008,  # 0.8% GDP comms
            "custom_rate": 0.07,             # Hold rate steady
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
        shock_shape="decay",
        shock_decay_rate=0.10,  # ~7 quarters to half-magnitude; oil shocks partly self-correct
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
        shock_decay_rate=0.05,  # Very slow decay; drought impacts persist for 2+ seasons
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
        shock_decay_rate=0.08,  # Moderate persistence; food prices tend to stabilise in 8-10 quarters
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
        shock_decay_rate=0.08,  # Open-economy PPP mean-reversion; KES typically stabilises in ~8 quarters
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
        shock_shape="decay",
        shock_decay_rate=0.05,  # Slow recovery consistent with global cycle (2-3 years)
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
        shock_decay_rate=0.07,  # Fiscal consolidation is slow; markets only recover with credible reforms
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

    # ── CRISIS SECTOR SCENARIOS ──────────────────────────────────────
    ScenarioTemplate(
        id="cholera_crisis",
        name="Cholera Outbreak Crisis",
        description="Multi-county cholera outbreak devastates health system, reduces labour force, "
                    "and collapses consumer demand in affected regions.",
        category="Health",
        shocks={"supply_shock": 0.10, "demand_shock": -0.06},
        shock_onset=5,
        shock_duration=8,
        shock_shape="pulse",
        suggested_dimensions=["gdp_growth", "inflation", "unemployment", "household_welfare",
                              "fiscal_deficit_gdp", "savings_rate"],
        suggested_policy=POLICY_TEMPLATES["health_emergency_response"],
        context="Kenya's cholera outbreaks (2014-15, 2017, 2019) typically affect 3-10 counties, "
                "with case fatality rates of 1.5-3%. Health system overload cascades into worker "
                "absenteeism, market closures, and displacement. The 2017 outbreak caused an "
                "estimated 0.3pp GDP drag in affected counties.",
    ),

    ScenarioTemplate(
        id="water_contamination_crisis",
        name="Water Contamination Crisis",
        description="Widespread water supply contamination degrades public health and productive capacity.",
        category="Water",
        shocks={"supply_shock": 0.08},
        shock_onset=5,
        shock_duration=10,
        shock_shape="ramp",
        suggested_dimensions=["gdp_growth", "inflation", "household_welfare", "cost_of_living_index"],
        suggested_policy=POLICY_TEMPLATES["water_crisis_response"],
        context="40% of Kenyans lack access to clean water. Contamination events hit arid counties "
                "hardest (Turkana, Garissa, Marsabit) where baseline water risk exceeds 70%. "
                "Contamination cascades into cholera, displacement, and market disruption.",
    ),

    ScenarioTemplate(
        id="road_network_collapse",
        name="Road Network Collapse",
        description="Major road closures cut supply chains across multiple counties, disrupting "
                    "production and market access.",
        category="Transport",
        shocks={"supply_shock": 0.12},
        shock_onset=5,
        shock_duration=6,
        shock_shape="pulse",
        suggested_dimensions=["gdp_growth", "inflation", "cost_of_living_index", "household_welfare"],
        suggested_policy=POLICY_TEMPLATES["transport_emergency"],
        context="Kenya's road network carries 93% of passenger and 97% of freight traffic. "
                "The 2023 El Niño floods destroyed 7,000+ km of roads, with reconstruction "
                "costs exceeding KES 50B. Remote counties become isolated when roads fail, "
                "triggering food price spikes and health supply shortages.",
    ),

    ScenarioTemplate(
        id="security_crisis",
        name="Security Crisis",
        description="Widespread security incidents from civil unrest, cross-border tensions, "
                    "or inter-community conflict.",
        category="Security",
        shocks={"demand_shock": -0.08, "supply_shock": 0.05, "fx_shock": 0.06},
        shock_onset=5,
        shock_shape="decay",
        suggested_dimensions=["gdp_growth", "unemployment", "investment_ratio",
                              "financial_stability", "household_welfare"],
        suggested_policy=POLICY_TEMPLATES["security_stabilization"],
        context="Kenya faces periodic security shocks: Westgate (2013), Garissa University (2015), "
                "post-election violence (2007-08, 2017). Tourism drops 30-50%, FDI freezes, "
                "and the KES depreciates. The 2007-08 crisis cut GDP growth from 7% to 1.5%.",
    ),

    ScenarioTemplate(
        id="displacement_crisis",
        name="Mass Displacement Crisis",
        description="Large-scale population displacement from conflict, climate, or disease "
                    "disrupts consumption and forces emergency fiscal response.",
        category="Displacement",
        shocks={"demand_shock": -0.06, "fiscal_shock": -0.04},
        shock_onset=5,
        shock_duration=10,
        shock_shape="ramp",
        suggested_dimensions=["gdp_growth", "unemployment", "household_welfare",
                              "fiscal_deficit_gdp", "debt_to_gdp"],
        suggested_policy=POLICY_TEMPLATES["displacement_response"],
        context="Kenya hosts 550,000+ refugees (Dadaab, Kakuma) and faces internal displacement "
                "from climate and conflict. The 2007-08 violence displaced 600,000. Displacement "
                "disrupts local markets, strains host-community services, and forces emergency "
                "fiscal spending.",
    ),

    ScenarioTemplate(
        id="market_disruption",
        name="Market Disruption Crisis",
        description="Local market collapse from combined displacement, security, and supply chain failure.",
        category="Food & Markets",
        shocks={"demand_shock": -0.10, "supply_shock": 0.06},
        shock_onset=5,
        shock_duration=6,
        shock_shape="pulse",
        suggested_dimensions=["gdp_growth", "inflation", "cost_of_living_index",
                              "household_welfare", "savings_rate"],
        suggested_policy=POLICY_TEMPLATES["market_intervention"],
        context="Kenyan informal markets employ 83% of the workforce. When markets collapse "
                "(from displacement, security, or logistics failure), food prices spike 30-80% "
                "and food insecurity surges. The KIHBS shows bottom quintile households spend "
                "60%+ of income on food, making them most vulnerable.",
    ),

    ScenarioTemplate(
        id="misinformation_wave",
        name="Misinformation Wave",
        description="Co-ordinated misinformation campaign triggers investor panic, currency pressure, "
                    "and consumer confidence collapse.",
        category="Communications",
        shocks={"fx_shock": 0.06, "demand_shock": -0.04},
        shock_onset=5,
        shock_shape="decay",
        suggested_dimensions=["gdp_growth", "inflation", "financial_stability",
                              "household_welfare", "investment_ratio"],
        suggested_policy=POLICY_TEMPLATES["communications_response"],
        context="Misinformation amplifies real crises. During COVID-19, social media rumours "
                "about bank collapses triggered deposit withdrawals. During cholera outbreaks, "
                "false health claims obstruct treatment. Investor sentiment is particularly "
                "sensitive to political misinformation during election cycles.",
    ),

    # ── CRISIS REGIME STRESS TESTS (Item 14) ────────────────────────────
    # These scenarios are designed to push the simulation toward the nonlinear
    # crisis regime thresholds introduced in Item 7:
    #   sudden_stop  → output_gap ≤ −12%
    #   bank_run     → NPL ratio ≥ 20%
    #   debt_crisis  → govt debt/GDP ≥ calibrated threshold (~120% for Kenya)

    ScenarioTemplate(
        id="banking_sector_crisis",
        name="Banking Sector Crisis (NPL Spiral)",
        description=(
            "Severe credit deterioration — non-performing loans surge as collateral "
            "values collapse, credit freezes, and investment collapses. Designed to "
            "stress-test the bank-run regime switch."
        ),
        category="Financial",
        shocks={
            "demand_shock":  -0.18,  # Deep demand collapse → GDP contraction → NPL surge
            "supply_shock":   0.06,  # Cost-push pressure → tighter margins for firms
            "fiscal_shock":  -0.05,  # Fiscal squeeze as tax revenues fall
        },
        shock_onset=4,
        shock_duration=16,
        shock_shape="ramp",          # Gradual build — bank crises unfold over 4–6 quarters
        shock_decay_rate=0.05,
        suggested_dimensions=[
            "financial_stability", "gdp_growth", "unemployment",
            "debt_to_gdp", "investment_ratio",
            "regime_bank_run", "regime_sudden_stop",
            "effective_mpc", "gini",
        ],
        suggested_policy=POLICY_TEMPLATES["expansionary_mix"],
        context=(
            "Kenya's banking sector NPL ratio rose from 9.4% (2019) to 14.8% (2023). "
            "A full banking crisis (NPL ≥ 20%) — triggered by a deep recession, "
            "currency shock, or property market collapse — would freeze credit, "
            "crush investment, and force CBK emergency intervention. "
            "This scenario pushes the BGG financial accelerator toward its crisis regime. "
            "Watch the 'Bank Run Regime' flag in outcomes."
        ),
    ),

    ScenarioTemplate(
        id="sovereign_debt_spiral",
        name="Sovereign Debt Spiral",
        description=(
            "Persistent fiscal deficits compound with rising borrowing costs and "
            "FX depreciation, pushing government debt toward unsustainable levels. "
            "Designed to stress-test the debt-crisis regime switch."
        ),
        category="Fiscal",
        shocks={
            "fiscal_shock": -0.10,   # Revenue collapse + spending pressures
            "fx_shock":      0.08,   # Currency depreciation raises external debt burden
            "demand_shock": -0.05,   # Confidence drag from fiscal uncertainty
        },
        shock_onset=3,
        shock_shape="step",          # Structural: debt accumulates from day one
        suggested_dimensions=[
            "debt_to_gdp", "fiscal_space", "fiscal_deficit_gdp",
            "gdp_growth", "inflation", "financial_stability",
            "regime_debt_crisis", "regime_sudden_stop",
            "q1_share", "gini",
        ],
        suggested_policy=POLICY_TEMPLATES["austerity"],
        context=(
            "Kenya's public debt hit 68% of GDP in 2023, with ~68% in foreign currency. "
            "A prolonged fiscal deterioration — from aid cuts, revenue shortfalls, "
            "and rising debt service costs — can trigger a debt crisis: market access "
            "closes, the IMF is called, and forced austerity compresses growth. "
            "This scenario accumulates deficits over many quarters to push debt/GDP "
            "toward the crisis threshold. Watch the 'Debt Crisis Regime' flag. "
            "Combine with the Austerity policy response to test consolidation paths."
        ),
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

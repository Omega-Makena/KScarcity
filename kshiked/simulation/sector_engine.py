"""
Multi-sector simulation engine.

Post-processes an SFC trajectory into SectorState objects for all six sectors.
Supports three execution modes:

  SINGLE_SECTOR  — deep simulation for one sector; spillover_hints for others
  MULTI_SECTOR   — all selected sectors; cascading + simultaneous + weighted
  FULL_SIMULATION — all six sectors; user-configurable weights; unlimited stacking

Ripple models applied:
  SIMULTANEOUS         — direct shocks hit all sectors at once (t=0)
  CASCADING            — 1st/2nd/3rd-order propagation with geometric decay
  WEIGHTED_INTERDEPENDENCY — influence-weight matrix adjusts terminal projections

Does NOT modify the SFC engine or its trajectory. This is a pure post-processing
layer: SFCEconomy.run() → list[dict] → SectorSimulator.project() → SectorStates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from kshiked.simulation.sector_registry import (
    ALL_SECTORS, INFLUENCE_WEIGHTS, SECTOR_SUB_INDICATORS,
    PolicyConflictWarning, SectorID, SectorState, SubIndicator,
)


# ─── Kenya 2022 sector baselines (World Bank / KNBS) ─────────────────────────

KENYA_BASELINES: Dict[SectorID, Dict[str, float]] = {
    SectorID.ECONOMICS_FINANCE: {
        "gdp_growth":         0.053,
        "inflation":          0.076,
        "unemployment":       0.055,
        "debt_to_gdp":        0.670,
        "investment_ratio":   0.185,
        "financial_stability":0.720,
        "fiscal_space":       0.280,
        "household_net_worth":1.000,
    },
    SectorID.HEALTHCARE_PUBLIC_HEALTH: {
        "health_capacity":      0.720,
        "disease_burden":       0.180,
        "mortality_risk":       0.220,
        "vaccination_coverage": 0.680,
        "health_spending_gdp":  0.044,
        "worker_availability":  0.620,
    },
    SectorID.ENVIRONMENT_CLIMATE: {
        "water_access":         0.620,
        "crop_yield_index":     1.000,
        "drought_severity":     0.220,
        "flood_risk":           0.150,
        "food_security_index":  0.680,
        "env_degradation":      0.350,
    },
    SectorID.SOCIAL_DEMOGRAPHICS: {
        "household_welfare":    1.000,
        "cost_of_living_index": 1.000,
        "displacement_rate":    0.020,
        "poverty_headcount":    0.365,
        "inequality_index":     0.386,
        "social_cohesion":      0.540,
    },
    SectorID.EDUCATION_LABOR: {
        "labor_productivity":   1.000,
        "employment_rate":      0.945,
        "school_attendance":    0.830,
        "skills_mismatch":      0.420,
        "human_capital_index":  0.520,
        "real_wage_growth":     0.015,
    },
    SectorID.GEOPOLITICS_SECURITY: {
        "security_stability":   0.610,
        "conflict_risk":        0.280,
        "institutional_trust":  0.420,
        "trade_disruption":     0.150,
        "border_security":      0.550,
        "cyber_risk":           0.320,
    },
}


# ─── Shock → sector direct-impact coefficients ───────────────────────────────
# Maps shock_id → {SectorID: {indicator: delta_per_unit_magnitude}}
# "Unit magnitude" = 1.0 (i.e. coefficient × actual_magnitude = delta applied).

SHOCK_DIRECT_IMPACTS: Dict[str, Dict[SectorID, Dict[str, float]]] = {
    # ── Macro SFC shocks (primary: Economics) ──────────────────────────────
    "demand_shock": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.80, "household_net_worth": -0.50,
        },
    },
    "supply_shock": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.60, "inflation": +0.50,
        },
        SectorID.ENVIRONMENT_CLIMATE: {
            "crop_yield_index": -0.40, "food_security_index": -0.30,
        },
    },
    "fiscal_shock": {
        SectorID.ECONOMICS_FINANCE: {
            "fiscal_space": -0.60, "debt_to_gdp": +0.30,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_spending_gdp": -0.30, "health_capacity": -0.25,
        },
    },
    "fx_shock": {
        SectorID.ECONOMICS_FINANCE: {
            "inflation": +0.30, "financial_stability": -0.20,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "cost_of_living_index": +0.30,
        },
    },

    # ── Health shocks ───────────────────────────────────────────────────────
    "cholera_outbreak": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "disease_burden":       +0.60,
            "mortality_risk":       +0.40,
            "health_capacity":      -0.35,
            "worker_availability":  -0.20,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate": +0.05, "household_welfare": -0.15,
        },
        SectorID.EDUCATION_LABOR: {
            "school_attendance": -0.10, "employment_rate": -0.05,
        },
    },
    "health_capacity_collapse": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_capacity":     -0.70,
            "worker_availability": -0.50,
            "mortality_risk":      +0.50,
        },
    },
    "health_worker_obstruction": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "worker_availability": -0.60,
            "health_capacity":     -0.40,
            "disease_burden":      +0.15,
        },
    },

    # ── Water / environment shocks ──────────────────────────────────────────
    "water_contamination": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "water_access": -0.55, "food_security_index": -0.20,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "disease_burden": +0.45, "mortality_risk": +0.30,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare": -0.20, "poverty_headcount": +0.10,
        },
    },
    "rainfall_flood": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "flood_risk":           +0.80,
            "crop_yield_index":     -0.30,
            "water_access":         -0.20,
            "env_degradation":      +0.15,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {"displacement_rate": +0.08},
        SectorID.GEOPOLITICS_SECURITY: {"trade_disruption": +0.20},
    },
    "drought": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "drought_severity":     +0.75,
            "crop_yield_index":     -0.45,
            "food_security_index":  -0.40,
            "water_access":         -0.30,
        },
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.30, "inflation": +0.20,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare":  -0.30,
            "poverty_headcount":  +0.12,
            "displacement_rate":  +0.06,
        },
    },

    # ── Transport shocks ────────────────────────────────────────────────────
    "road_closure": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.15, "investment_ratio": -0.10,
        },
        SectorID.EDUCATION_LABOR: {
            "school_attendance": -0.10, "employment_rate": -0.05,
        },
        SectorID.GEOPOLITICS_SECURITY: {"trade_disruption": +0.25},
    },
    "logistics_breakdown": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.20, "inflation": +0.15, "investment_ratio": -0.10,
        },
        SectorID.ENVIRONMENT_CLIMATE: {"food_security_index": -0.15},
        SectorID.GEOPOLITICS_SECURITY: {"trade_disruption": +0.35},
    },

    # ── Security shocks ─────────────────────────────────────────────────────
    "security_surge": {
        SectorID.GEOPOLITICS_SECURITY: {
            "security_stability":  -0.40,
            "conflict_risk":       +0.35,
            "institutional_trust": -0.20,
            "trade_disruption":    +0.15,
        },
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.15, "investment_ratio": -0.20,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate": +0.04, "social_cohesion": -0.20,
        },
    },
    "civil_unrest": {
        SectorID.GEOPOLITICS_SECURITY: {
            "security_stability":  -0.50,
            "conflict_risk":       +0.45,
            "institutional_trust": -0.30,
            "border_security":     -0.15,
        },
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.20, "financial_stability": -0.15,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate":  +0.06,
            "social_cohesion":    -0.35,
            "poverty_headcount":  +0.05,
        },
    },

    # ── Displacement shocks ─────────────────────────────────────────────────
    "mass_displacement": {
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate":  +0.15,
            "household_welfare":  -0.25,
            "social_cohesion":    -0.30,
            "poverty_headcount":  +0.15,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "disease_burden": +0.20, "health_capacity": -0.15,
        },
        SectorID.EDUCATION_LABOR: {
            "school_attendance": -0.20, "skills_mismatch": +0.10,
        },
    },
    "refugee_influx": {
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate":  +0.08,
            "poverty_headcount":  +0.05,
            "inequality_index":   +0.03,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "disease_burden": +0.10, "health_capacity": -0.10,
        },
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": +0.02, "inflation": +0.05,
        },
    },

    # ── Market shocks ───────────────────────────────────────────────────────
    "market_collapse": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth":          -0.35,
            "financial_stability": -0.40,
            "investment_ratio":    -0.25,
            "household_net_worth": -0.35,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare": -0.30, "poverty_headcount": +0.10,
        },
    },
    "food_price_spike": {
        SectorID.ENVIRONMENT_CLIMATE: {"food_security_index": -0.35},
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "cost_of_living_index": +0.25,
            "household_welfare":    -0.20,
            "poverty_headcount":    +0.10,
        },
        SectorID.ECONOMICS_FINANCE: {"inflation": +0.25},
    },

    # ── Misinformation ──────────────────────────────────────────────────────
    "misinformation_crisis": {
        SectorID.GEOPOLITICS_SECURITY: {
            "institutional_trust": -0.35,
            "conflict_risk":       +0.20,
            "border_security":     -0.05,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "vaccination_coverage": -0.15, "disease_burden": +0.10,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "social_cohesion": -0.25, "displacement_rate": +0.02,
        },
    },

    # ── Macro-economic crisis scenarios ─────────────────────────────────────
    "oil_crisis": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.25, "inflation": +0.35, "investment_ratio": -0.15,
        },
        SectorID.GEOPOLITICS_SECURITY: {"trade_disruption": +0.30},
    },
    "kes_depreciation": {
        SectorID.ECONOMICS_FINANCE: {
            "inflation": +0.40, "financial_stability": -0.25, "debt_to_gdp": +0.15,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "cost_of_living_index": +0.35, "poverty_headcount": +0.08,
        },
    },
    "global_recession": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.30, "investment_ratio": -0.25, "financial_stability": -0.20,
        },
        SectorID.GEOPOLITICS_SECURITY: {
            "trade_disruption": +0.30, "institutional_trust": -0.15,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare": -0.20, "poverty_headcount": +0.08,
        },
    },
    "aid_reduction": {
        SectorID.ECONOMICS_FINANCE: {
            "fiscal_space": -0.30, "investment_ratio": -0.10,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_spending_gdp": -0.20, "health_capacity": -0.15,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "poverty_headcount": +0.08, "household_welfare": -0.12,
        },
    },
    "debt_crisis": {
        SectorID.ECONOMICS_FINANCE: {
            "fiscal_space": -0.60, "debt_to_gdp": +0.40, "financial_stability": -0.30,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_spending_gdp": -0.25, "health_capacity": -0.20,
        },
        SectorID.GEOPOLITICS_SECURITY: {"institutional_trust": -0.25},
    },
    "stimulus_boom": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": +0.20, "investment_ratio": +0.15, "fiscal_space": -0.30,
        },
        SectorID.EDUCATION_LABOR: {
            "employment_rate": +0.03, "labor_productivity": +0.05,
        },
    },
    "perfect_storm": {
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.50, "inflation": +0.40, "financial_stability": -0.35,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_capacity": -0.30, "disease_burden": +0.30,
        },
        SectorID.ENVIRONMENT_CLIMATE: {
            "drought_severity": +0.40, "food_security_index": -0.35,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare":  -0.45,
            "poverty_headcount":  +0.20,
            "displacement_rate":  +0.08,
        },
        SectorID.GEOPOLITICS_SECURITY: {
            "security_stability": -0.30, "conflict_risk": +0.25,
        },
    },

    # ── El Niño / climate multi-hazard (added for Kenya context) ────────────
    "el_nino": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "drought_severity":    +0.55,
            "flood_risk":          +0.30,   # El Niño causes both drought (interior) and floods (coast)
            "crop_yield_index":    -0.40,
            "food_security_index": -0.35,
            "water_access":        -0.25,
            "env_degradation":     +0.15,
        },
        SectorID.ECONOMICS_FINANCE: {
            "gdp_growth": -0.35, "inflation": +0.25,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate":  +0.07,
            "poverty_headcount":  +0.10,
            "household_welfare":  -0.25,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "disease_burden":  +0.25,
            "mortality_risk":  +0.15,
        },
        SectorID.GEOPOLITICS_SECURITY: {
            "conflict_risk":    +0.15,  # resource scarcity → herder-farmer conflict
            "trade_disruption": +0.20,
        },
    },
}


# ─── Policy instrument → sector benefit coefficients ─────────────────────────
# Maps policy_key → {SectorID: {indicator: delta_per_unit_magnitude}}

POLICY_SECTOR_BENEFITS: Dict[str, Dict[SectorID, Dict[str, float]]] = {
    "health_emergency_spending": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_capacity":      +0.35,
            "worker_availability":  +0.20,
            "health_spending_gdp":  +0.40,
        },
    },
    "vaccination_coverage": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "vaccination_coverage": +0.50, "disease_burden": -0.20,
        },
    },
    "water_infra_spend": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "water_access": +0.30, "food_security_index": +0.10,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {"disease_burden": -0.15},
    },
    "road_repair_budget": {
        SectorID.EDUCATION_LABOR: {
            "school_attendance": +0.05, "employment_rate": +0.02,
        },
        SectorID.ECONOMICS_FINANCE: {"gdp_growth": +0.05},
        SectorID.GEOPOLITICS_SECURITY: {"trade_disruption": -0.15},
    },
    "security_deployment": {
        SectorID.GEOPOLITICS_SECURITY: {
            "security_stability": +0.30,
            "conflict_risk":      -0.25,
            "border_security":    +0.20,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {"social_cohesion": +0.10},
    },
    "displacement_relief": {
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "displacement_rate":  -0.05,
            "household_welfare":  +0.15,
            "poverty_headcount":  -0.05,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {"disease_burden": -0.10},
    },
    "cash_transfer_rate": {
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "household_welfare":    +0.20,
            "poverty_headcount":    -0.08,
            "cost_of_living_index": -0.05,
        },
        SectorID.ECONOMICS_FINANCE: {"gdp_growth": +0.04},
    },
    "price_stabilization": {
        SectorID.ENVIRONMENT_CLIMATE: {"food_security_index": +0.15},
        SectorID.SOCIAL_DEMOGRAPHICS: {"cost_of_living_index": -0.15},
        SectorID.ECONOMICS_FINANCE:   {"inflation": -0.10},
    },
    "food_reserve_release": {
        SectorID.ENVIRONMENT_CLIMATE: {"food_security_index": +0.20},
        SectorID.SOCIAL_DEMOGRAPHICS: {
            "cost_of_living_index": -0.10, "household_welfare": +0.08,
        },
    },
    "counter_misinfo_spend": {
        SectorID.GEOPOLITICS_SECURITY: {
            "institutional_trust": +0.20, "conflict_risk": -0.10,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {"vaccination_coverage": +0.08},
        SectorID.SOCIAL_DEMOGRAPHICS: {"social_cohesion": +0.15},
    },
    "custom_spending_ratio": {
        SectorID.ECONOMICS_FINANCE: {"fiscal_space": -0.20, "gdp_growth": +0.10},
    },
    "custom_tax_rate": {
        SectorID.ECONOMICS_FINANCE: {"fiscal_space": +0.15, "gdp_growth": -0.05},
    },
    "subsidy_rate": {
        SectorID.ECONOMICS_FINANCE: {
            "inflation": -0.05, "household_net_worth": +0.05,
        },
        SectorID.SOCIAL_DEMOGRAPHICS: {"cost_of_living_index": -0.10},
    },
    "health_emergency_response": {
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {
            "health_capacity":      +0.30,
            "disease_burden":       -0.25,
            "worker_availability":  +0.20,
            "health_spending_gdp":  +0.35,
        },
    },
    "water_crisis_response": {
        SectorID.ENVIRONMENT_CLIMATE: {
            "water_access": +0.35, "food_security_index": +0.15,
        },
        SectorID.HEALTHCARE_PUBLIC_HEALTH: {"disease_burden": -0.20},
        SectorID.SOCIAL_DEMOGRAPHICS: {"household_welfare": +0.10},
    },
}


# ─── Simulation mode enums ────────────────────────────────────────────────────

class SimulationMode(str, Enum):
    SINGLE_SECTOR   = "single_sector"
    MULTI_SECTOR    = "multi_sector"
    FULL_SIMULATION = "full_simulation"


class RippleModel(str, Enum):
    CASCADING              = "cascading"
    SIMULTANEOUS           = "simultaneous"
    WEIGHTED_INTERDEPENDENCY = "weighted_interdependency"


@dataclass
class SectorSimConfig:
    mode: SimulationMode = SimulationMode.FULL_SIMULATION
    selected_sectors: Optional[List[SectorID]] = None      # None = all six
    ripple_models: List[RippleModel] = field(
        default_factory=lambda: list(RippleModel)
    )
    influence_weights: Optional[Dict[SectorID, Dict[SectorID, float]]] = None
    cascade_orders: int = 3


# ─── Main simulator ───────────────────────────────────────────────────────────

class SectorSimulator:
    """
    Post-processes an SFC trajectory into SectorState objects.

    Usage::

        sim = SectorSimulator()
        sector_results = sim.project(
            trajectory=st.session_state["sim_trajectory"],
            shock_ids=["drought", "civil_unrest"],
            policy_keys=["water_infra_spend", "security_deployment"],
        )
        # sector_results: Dict[SectorID, SectorState]
    """

    def __init__(self, config: Optional[SectorSimConfig] = None):
        self.config = config or SectorSimConfig()
        self._weights = self.config.influence_weights or INFLUENCE_WEIGHTS

    # ── Public API ────────────────────────────────────────────────────────────

    def project(
        self,
        trajectory: List[Dict],
        shock_ids: List[str],
        policy_keys: List[str],
        shock_magnitudes: Optional[Dict[str, float]] = None,
        policy_magnitudes: Optional[Dict[str, float]] = None,
    ) -> Dict[SectorID, SectorState]:
        """
        Convert SFC trajectory + shock/policy config into sector states.

        Args:
            trajectory:        list of SFCEconomy frame dicts (must be non-empty)
            shock_ids:         list of scenario/shock IDs active in this run
            policy_keys:       list of active policy instrument keys
            shock_magnitudes:  {shock_id: 0–1 magnitude}; defaults 0.5 if absent
            policy_magnitudes: {policy_key: 0–1 magnitude}; defaults 0.5 if absent

        Returns:
            Dict[SectorID, SectorState] — every selected sector fully populated.
        """
        if not trajectory:
            return {}

        smag = shock_magnitudes or {}
        pmag = policy_magnitudes or {}
        active = self.config.selected_sectors or ALL_SECTORS

        # 1. Compute raw (pre-ripple) deltas from shocks + policies
        raw = self._raw_deltas(shock_ids, policy_keys, smag, pmag)

        # 2. Overwrite Economics indicators with actual SFC macro values
        self._macro_passthrough(trajectory, raw)

        # 3. Apply ripple models to produce final deltas
        final = self._apply_ripples(raw)

        # 4. Classify direct vs induced
        direct_map, induced_map = self._classify_effects(shock_ids, policy_keys, final)

        # 5. Build per-sector step-by-step timelines
        timelines = self._build_timelines(trajectory, final)

        # 6. Construct SectorState for every active sector
        result: Dict[SectorID, SectorState] = {}
        for sid in active:
            result[sid] = self._build_state(
                sid, final, direct_map, induced_map, timelines, active, len(trajectory)
            )

        # 7. Single-sector mode: strip heavy data for non-primary sectors
        if self.config.mode == SimulationMode.SINGLE_SECTOR and active:
            primary = active[0]
            for sid, state in result.items():
                if sid != primary:
                    state.timeline = {}
                    state.model_assumptions = {}

        # 8. Detect and annotate policy conflicts
        for conflict in self._detect_conflicts(policy_keys, final):
            if conflict.sector_id in result:
                key = f"conflict_{conflict.policy_a}_{conflict.policy_b}"
                result[conflict.sector_id].model_assumptions[key] = (
                    f"PolicyConflict: {conflict.policy_a} vs {conflict.policy_b} "
                    f"on {conflict.indicator} — {conflict.reason}"
                )

        return result

    # ── Private: delta computation ────────────────────────────────────────────

    def _raw_deltas(
        self,
        shock_ids: List[str],
        policy_keys: List[str],
        smag: Dict[str, float],
        pmag: Dict[str, float],
    ) -> Dict[SectorID, Dict[str, float]]:
        raw: Dict[SectorID, Dict[str, float]] = {s: {} for s in ALL_SECTORS}

        for shock_id in shock_ids:
            if shock_id not in SHOCK_DIRECT_IMPACTS:
                continue
            mag = smag.get(shock_id, 0.5)
            for sid, impacts in SHOCK_DIRECT_IMPACTS[shock_id].items():
                for ind, coeff in impacts.items():
                    raw[sid][ind] = raw[sid].get(ind, 0.0) + coeff * mag

        for pkey in policy_keys:
            if pkey not in POLICY_SECTOR_BENEFITS:
                continue
            mag = pmag.get(pkey, 0.5)
            for sid, impacts in POLICY_SECTOR_BENEFITS[pkey].items():
                for ind, coeff in impacts.items():
                    raw[sid][ind] = raw[sid].get(ind, 0.0) + coeff * mag

        return raw

    def _macro_passthrough(
        self,
        trajectory: List[Dict],
        raw: Dict[SectorID, Dict[str, float]],
    ) -> None:
        """
        Overwrite Economics sector deltas with ground-truth SFC trajectory values,
        then derive first-order cross-sector signals from macro movements.
        """
        if len(trajectory) < 2:
            return
        t0 = trajectory[0].get("outcomes", {})
        tf = trajectory[-1].get("outcomes", {})
        eco = raw[SectorID.ECONOMICS_FINANCE]

        # Direct SFC override for core macro indicators
        _macro_map = [
            ("gdp_growth",         "gdp_growth"),
            ("inflation",          "inflation"),
            ("unemployment",       "unemployment"),
            ("debt_to_gdp",        "debt_to_gdp"),
            ("investment_ratio",   "investment_ratio"),
            ("financial_stability","financial_stability"),
            ("fiscal_space",       "fiscal_space"),
            ("household_net_worth","household_net_worth"),
        ]
        for sfc_key, sector_key in _macro_map:
            v0 = t0.get(sfc_key)
            vf = tf.get(sfc_key)
            if v0 is not None and vf is not None:
                eco[sector_key] = float(vf) - float(v0)

        # Derived cross-sector first-order signals
        gdp_d   = eco.get("gdp_growth",    0.0)
        infl_d  = eco.get("inflation",     0.0)
        unemp_d = eco.get("unemployment",  0.0)
        fstab_d = eco.get("financial_stability", 0.0)

        soc = raw[SectorID.SOCIAL_DEMOGRAPHICS]
        soc["household_welfare"]    = soc.get("household_welfare",    0.0) + gdp_d  * 0.65
        soc["cost_of_living_index"] = soc.get("cost_of_living_index", 0.0) + infl_d * 1.0
        soc["poverty_headcount"]    = soc.get("poverty_headcount",    0.0) - gdp_d  * 0.30

        lab = raw[SectorID.EDUCATION_LABOR]
        lab["employment_rate"]  = lab.get("employment_rate",  0.0) - unemp_d
        lab["real_wage_growth"] = lab.get("real_wage_growth", 0.0) - unemp_d * 0.30
        lab["labor_productivity"] = lab.get("labor_productivity", 0.0) + gdp_d * 0.40

        geo = raw[SectorID.GEOPOLITICS_SECURITY]
        geo["institutional_trust"] = geo.get("institutional_trust", 0.0) + gdp_d  * 0.20
        geo["conflict_risk"]       = geo.get("conflict_risk",       0.0) - gdp_d  * 0.15
        geo["trade_disruption"]    = geo.get("trade_disruption",    0.0) + infl_d * 0.20

        hlth = raw[SectorID.HEALTHCARE_PUBLIC_HEALTH]
        # Fiscal contraction (gdp drop + fstab drop) → health budget pressure
        hlth["health_capacity"]     = hlth.get("health_capacity",     0.0) + gdp_d  * 0.25
        hlth["health_spending_gdp"] = hlth.get("health_spending_gdp", 0.0) + fstab_d * 0.10

    # ── Private: ripple models ────────────────────────────────────────────────

    def _apply_ripples(
        self,
        raw: Dict[SectorID, Dict[str, float]],
    ) -> Dict[SectorID, Dict[str, float]]:
        final = {s: dict(d) for s, d in raw.items()}

        if RippleModel.WEIGHTED_INTERDEPENDENCY in self.config.ripple_models:
            self._weighted_interdependency(raw, final)

        if RippleModel.CASCADING in self.config.ripple_models:
            self._cascading(raw, final)

        # SIMULTANEOUS is already in raw (all direct impacts at t=0)
        return final

    def _weighted_interdependency(
        self,
        source: Dict[SectorID, Dict[str, float]],
        final:  Dict[SectorID, Dict[str, float]],
    ) -> None:
        """
        delta_j += w[i][j] × net_signed_impact(i) / n_indicators(j) × 0.30
        The 0.30 dampening factor: cross-sector effects are weaker than direct.
        """
        net: Dict[SectorID, float] = {}
        for sid in ALL_SECTORS:
            total = sum(abs(v) for v in source[sid].values())
            sign  = sum(v        for v in source[sid].values())
            net[sid] = math.copysign(total, sign) if total > 0 else 0.0

        for from_sid in ALL_SECTORS:
            ni = net[from_sid]
            if abs(ni) < 1e-9:
                continue
            for to_sid in ALL_SECTORS:
                if to_sid == from_sid:
                    continue
                w = self._weights.get(from_sid, {}).get(to_sid, 0.0)
                if abs(w) < 1e-9:
                    continue
                inds = SECTOR_SUB_INDICATORS.get(to_sid, [])
                if not inds:
                    continue
                per_ind = ni * w * 0.30 / len(inds)
                for ind in inds:
                    final[to_sid][ind.key] = final[to_sid].get(ind.key, 0.0) + per_ind

    def _cascading(
        self,
        raw:   Dict[SectorID, Dict[str, float]],
        final: Dict[SectorID, Dict[str, float]],
    ) -> None:
        """
        1st/2nd/3rd-order cascade with geometric decay 0.5^order.
        Each round propagates the previous round's wave through the influence matrix.
        Labeled in model_assumptions on completion (labels stored externally).
        """
        wave = {s: dict(d) for s, d in raw.items()}

        for order in range(1, self.config.cascade_orders + 1):
            decay = 0.5 ** order
            next_wave: Dict[SectorID, Dict[str, float]] = {s: {} for s in ALL_SECTORS}

            for from_sid in ALL_SECTORS:
                wave_abs  = sum(abs(v) for v in wave[from_sid].values())
                wave_sign = sum(v        for v in wave[from_sid].values())
                net = math.copysign(wave_abs, wave_sign) if wave_abs > 0 else 0.0
                if abs(net) < 1e-9:
                    continue

                for to_sid in ALL_SECTORS:
                    if to_sid == from_sid:
                        continue
                    w = self._weights.get(from_sid, {}).get(to_sid, 0.0)
                    if abs(w) < 1e-9:
                        continue
                    inds = SECTOR_SUB_INDICATORS.get(to_sid, [])
                    if not inds:
                        continue
                    cascade = net * w * decay / len(inds)
                    for ind in inds:
                        next_wave[to_sid][ind.key] = (
                            next_wave[to_sid].get(ind.key, 0.0) + cascade
                        )
                        final[to_sid][ind.key] = (
                            final[to_sid].get(ind.key, 0.0) + cascade
                        )

            wave = next_wave

    # ── Private: classification & timelines ──────────────────────────────────

    def _classify_effects(
        self,
        shock_ids:   List[str],
        policy_keys: List[str],
        final:       Dict[SectorID, Dict[str, float]],
    ) -> Tuple[Dict[SectorID, List[str]], Dict[SectorID, List[str]]]:
        direct_keys: Dict[SectorID, set] = {s: set() for s in ALL_SECTORS}

        for shock_id in shock_ids:
            for sid, impacts in SHOCK_DIRECT_IMPACTS.get(shock_id, {}).items():
                direct_keys[sid].update(impacts.keys())
        for pkey in policy_keys:
            for sid, impacts in POLICY_SECTOR_BENEFITS.get(pkey, {}).items():
                direct_keys[sid].update(impacts.keys())

        # Macro pass-through indicators are also "direct" for Economics
        direct_keys[SectorID.ECONOMICS_FINANCE].update([
            "gdp_growth", "inflation", "unemployment", "debt_to_gdp",
            "investment_ratio", "financial_stability", "fiscal_space",
            "household_net_worth",
        ])
        # And the first-order derived indicators
        direct_keys[SectorID.SOCIAL_DEMOGRAPHICS].update([
            "household_welfare", "cost_of_living_index", "poverty_headcount",
        ])
        direct_keys[SectorID.EDUCATION_LABOR].update([
            "employment_rate", "real_wage_growth", "labor_productivity",
        ])
        direct_keys[SectorID.GEOPOLITICS_SECURITY].update([
            "institutional_trust", "conflict_risk", "trade_disruption",
        ])
        direct_keys[SectorID.HEALTHCARE_PUBLIC_HEALTH].update([
            "health_capacity", "health_spending_gdp",
        ])

        direct:  Dict[SectorID, List[str]] = {}
        induced: Dict[SectorID, List[str]] = {}
        for sid in ALL_SECTORS:
            direct[sid]  = [k for k in final.get(sid, {}) if k in direct_keys[sid]]
            induced[sid] = [k for k in final.get(sid, {}) if k not in direct_keys[sid]]

        return direct, induced

    def _build_timelines(
        self,
        trajectory: List[Dict],
        final:      Dict[SectorID, Dict[str, float]],
    ) -> Dict[SectorID, Dict[str, List[float]]]:
        """
        Build step-by-step indicator series for every sector.
        Economics: read directly from SFC trajectory.
        Others:    exponential approach from baseline toward projected terminal.
        """
        n = len(trajectory)
        timelines: Dict[SectorID, Dict[str, List[float]]] = {}

        for sid in ALL_SECTORS:
            timelines[sid] = {}
            inds     = SECTOR_SUB_INDICATORS.get(sid, [])
            baseline = KENYA_BASELINES.get(sid, {})

            if sid == SectorID.ECONOMICS_FINANCE:
                for ind in inds:
                    series = []
                    for frame in trajectory:
                        val = frame.get("outcomes", {}).get(ind.key)
                        series.append(float(val) if val is not None else baseline.get(ind.key, 0.0))
                    timelines[sid][ind.key] = series

            else:
                onset = max(1, n // 4)
                for ind in inds:
                    base_val = baseline.get(ind.key, 0.0)
                    delta    = final[sid].get(ind.key, 0.0)
                    series   = []
                    for t in range(n):
                        if t < onset:
                            series.append(base_val)
                        else:
                            progress = (t - onset) / max(1, n - onset)
                            series.append(base_val + delta * (1.0 - math.exp(-3.0 * progress)))
                    timelines[sid][ind.key] = series

        return timelines

    # ── Private: state builder ────────────────────────────────────────────────

    def _build_state(
        self,
        sid:         SectorID,
        final:       Dict[SectorID, Dict[str, float]],
        direct_map:  Dict[SectorID, List[str]],
        induced_map: Dict[SectorID, List[str]],
        timelines:   Dict[SectorID, Dict[str, List[float]]],
        active:      List[SectorID],
        n_steps:     int,
    ) -> SectorState:
        inds     = SECTOR_SUB_INDICATORS.get(sid, [])
        baseline = dict(KENYA_BASELINES.get(sid, {}))
        deltas   = final.get(sid, {})

        # Clamp projected values to indicator ranges
        projected: Dict[str, float] = {}
        clamped_delta: Dict[str, float] = {}
        for ind in inds:
            base_val = baseline.get(ind.key, 0.0)
            raw_proj = base_val + deltas.get(ind.key, 0.0)
            lo, hi   = ind.typical_range
            clamped  = max(lo, min(hi, raw_proj))
            projected[ind.key]    = round(clamped, 6)
            clamped_delta[ind.key] = round(clamped - base_val, 6)

        # Severity: RMS of weighted normalized deltas, scaled to 1–10
        sev_num = sev_den = 0.0
        for ind in inds:
            d   = abs(clamped_delta.get(ind.key, 0.0))
            rng = ind.typical_range[1] - ind.typical_range[0]
            if rng < 1e-9:
                continue
            sev_num += (d / rng) * ind.weight
            sev_den += ind.weight
        sev_raw  = sev_num / sev_den if sev_den > 0 else 0.0
        severity = max(1.0, min(10.0, sev_raw * 10.0))

        # Confidence: base 80 %, adjusted for directness ratio and time horizon
        n_direct  = len(direct_map.get(sid, []))
        n_induced = len(induced_map.get(sid, []))
        n_total   = n_direct + n_induced
        dir_ratio = n_direct / n_total if n_total > 0 else 0.5
        time_decay = max(0.0, 1.0 - n_steps / 200.0)
        confidence = min(95.0, max(30.0,
            (0.80 * dir_ratio + 0.50 * (1.0 - dir_ratio)) * 100.0
            * (0.70 + 0.30 * time_decay)
        ))

        # Spillover hints: sectors this one significantly influences
        sev_score = sev_raw
        hints = []
        for to_sid in ALL_SECTORS:
            if to_sid == sid:
                continue
            w  = self._weights.get(sid, {}).get(to_sid, 0.0)
            mg = sev_score * w
            if mg > 0.05:
                hints.append({
                    "sector_id": to_sid.value,
                    "label":     SECTOR_LABELS_SHORT.get(to_sid, to_sid.value),
                    "reason":    f"influence weight {w:.2f} × sector severity {sev_score:.2f}",
                    "magnitude_estimate": round(mg, 3),
                })
        hints.sort(key=lambda x: -x["magnitude_estimate"])

        assumptions = {
            "calibration":     "Kenya 2022 World Bank / KNBS baselines",
            "ripple_models":   ", ".join(r.value for r in self.config.ripple_models),
            "cascade_orders":  str(self.config.cascade_orders),
            "confidence_note": f"directness_ratio={dir_ratio:.2f}, time_decay={time_decay:.2f}",
        }

        return SectorState(
            sector_id=sid,
            baseline=baseline,
            projected=projected,
            delta=clamped_delta,
            severity=round(severity, 1),
            confidence=round(confidence, 1),
            sub_indicators=inds,
            direct_effects=direct_map.get(sid, []),
            induced_effects=induced_map.get(sid, []),
            spillover_hints=hints,
            model_assumptions=assumptions,
            timeline=timelines.get(sid, {}),
        )

    # ── Private: conflict detection ───────────────────────────────────────────

    def _detect_conflicts(
        self,
        policy_keys: List[str],
        final:       Dict[SectorID, Dict[str, float]],
    ) -> List[PolicyConflictWarning]:
        contribs: Dict[SectorID, Dict[str, Dict[str, float]]] = {
            s: {} for s in ALL_SECTORS
        }
        for pkey in policy_keys:
            for sid, impacts in POLICY_SECTOR_BENEFITS.get(pkey, {}).items():
                for ind, coeff in impacts.items():
                    contribs[sid].setdefault(ind, {})[pkey] = coeff

        conflicts = []
        for sid in ALL_SECTORS:
            for ind, pc in contribs[sid].items():
                if len(pc) < 2:
                    continue
                positives = [p for p, v in pc.items() if v > 0]
                negatives = [p for p, v in pc.items() if v < 0]
                if positives and negatives:
                    conflicts.append(PolicyConflictWarning(
                        sector_id=sid,
                        policy_a=positives[0],
                        policy_b=negatives[0],
                        indicator=ind,
                        reason=(
                            f"{positives[0]} → +{pc[positives[0]]:.2f}, "
                            f"{negatives[0]} → {pc[negatives[0]]:.2f}"
                        ),
                    ))
        return conflicts


# Label shorthand (used in spillover_hints)
SECTOR_LABELS_SHORT: Dict[SectorID, str] = {
    SectorID.ECONOMICS_FINANCE:        "Economics",
    SectorID.HEALTHCARE_PUBLIC_HEALTH: "Health",
    SectorID.ENVIRONMENT_CLIMATE:      "Environment",
    SectorID.SOCIAL_DEMOGRAPHICS:      "Social",
    SectorID.EDUCATION_LABOR:          "Labor",
    SectorID.GEOPOLITICS_SECURITY:     "Security",
}


# ─── Scenario-library ID → SHOCK_DIRECT_IMPACTS key mapping ─────────────────
# The SCENARIO_LIBRARY in scenario_templates.py uses descriptive IDs like
# "cholera_crisis"; SHOCK_DIRECT_IMPACTS uses shorter canonical names.
# This mapping bridges the gap so the sector engine always finds direct impacts.

SCENARIO_ID_TO_SHOCK_KEY: Dict[str, str] = {
    # Health shocks
    "cholera_crisis":           "cholera_outbreak",
    "water_contamination_crisis": "water_contamination",
    "health_capacity_crisis":   "health_capacity_collapse",
    # Environment / climate
    "food_price_surge":         "food_price_spike",
    "el_nino_crisis":           "el_nino",
    "rainfall_flood_crisis":    "rainfall_flood",
    "drought_crisis":           "drought",
    # Transport / infrastructure
    "road_network_collapse":    "road_closure",
    "road_collapse":            "road_closure",
    "logistics_collapse":       "logistics_breakdown",
    # Security / displacement
    "security_crisis":          "security_surge",
    "civil_unrest_crisis":      "civil_unrest",
    "displacement_crisis":      "mass_displacement",
    "refugee_crisis":           "refugee_influx",
    # Market / macro
    "market_disruption":        "market_collapse",
    "market_crash":             "market_collapse",
    "misinformation_wave":      "misinformation_crisis",
    "misinformation_crisis_scenario": "misinformation_crisis",
    # These already match SHOCK_DIRECT_IMPACTS keys — listed for completeness
    "oil_crisis":               "oil_crisis",
    "drought":                  "drought",
    "kes_depreciation":         "kes_depreciation",
    "global_recession":         "global_recession",
    "aid_reduction":            "aid_reduction",
    "debt_crisis":              "debt_crisis",
    "stimulus_boom":            "stimulus_boom",
    "perfect_storm":            "perfect_storm",
    "el_nino":                  "el_nino",
}


def _resolve_shock_key(scenario_id: str) -> str:
    """Resolve a scenario library ID to the matching SHOCK_DIRECT_IMPACTS key.

    Falls back to the scenario ID itself (works for exact matches).
    """
    return SCENARIO_ID_TO_SHOCK_KEY.get(scenario_id, scenario_id)


# ─── Helpers for run.py ───────────────────────────────────────────────────────

def extract_shock_info(
    selected_scenarios,
    custom_shocks: list,
    merged_shocks: dict,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract shock IDs and effective magnitudes from scenario config objects
    for passing to SectorSimulator.project().

    Scenario IDs from SCENARIO_LIBRARY are resolved to the canonical
    SHOCK_DIRECT_IMPACTS keys via SCENARIO_ID_TO_SHOCK_KEY so that sector
    direct impacts are always applied correctly.

    Args:
        selected_scenarios: list of ScenarioTemplate objects
        custom_shocks:      list of custom shock dicts
        merged_shocks:      merged shock vector dict (keys → np.ndarray)
    """
    shock_ids: List[str] = []
    shock_magnitudes: Dict[str, float] = {}

    for s in (selected_scenarios or []):
        raw_id = getattr(s, "id", None)
        if not raw_id:
            continue
        sid = _resolve_shock_key(raw_id)
        if sid not in shock_ids:
            shock_ids.append(sid)
        shocks_dict = getattr(s, "shocks", {}) or {}
        if shocks_dict:
            # SFC shock vectors use calibrated magnitudes (typically 0.04–0.15).
            # SHOCK_DIRECT_IMPACTS coefficients are calibrated for magnitude 0.3–1.0,
            # where 1.0 = extreme event. Scale up by summing absolute SFC values and
            # normalising so that a total SFC perturbation of 0.10 ≈ sector magnitude 0.50.
            total_sfc_mag = sum(abs(v) for v in shocks_dict.values())
            sector_magnitude = min(1.0, total_sfc_mag * 5.0)
            shock_magnitudes[sid] = max(0.25, sector_magnitude)
        else:
            shock_magnitudes[sid] = 0.5

    for cs in (custom_shocks or []):
        cid  = _resolve_shock_key(cs.get("shock_type", cs.get("id", "")))
        cmag = abs(cs.get("magnitude", 0.0))
        if cid and cmag > 1e-9:
            if cid not in shock_ids:
                shock_ids.append(cid)
            shock_magnitudes[cid] = max(shock_magnitudes.get(cid, 0.0), cmag)

    return shock_ids, shock_magnitudes

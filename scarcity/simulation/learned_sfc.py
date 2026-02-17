"""
LearnedSFCEconomy — SFC model where relationships come from the discovery engine.

Instead of hardcoded Phillips Curve, Okun's Law, Taylor Rule, etc., this model
uses scarcity's PolicySimulator (backed by 306+ learned hypotheses) to simulate
the economy. Each hypothesis votes on the next value of its target variable.

The parametric SFCEconomy acts as a fallback: the FallbackBlender mixes
learned and parametric predictions based on per-variable confidence.

Output format is trajectory-compatible with the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("scarcity.simulation.learned_sfc")


@dataclass
class LearnedSFCConfig:
    """Configuration for the learned SFC economy."""
    steps: int = 20
    enable_fallback: bool = True
    fallback_weight_override: Optional[float] = None  # None = use per-variable confidence
    initial_state_source: str = "data"  # "data" or "manual"
    manual_initial_state: Optional[Dict[str, float]] = None


class LearnedSFCEconomy:
    """
    SFC model backed by scarcity's learned hypothesis graph.

    Usage:
        bridge = ScarcityBridge()
        bridge.train(data_path)

        from scarcity.simulation.sfc import SFCConfig
        economy = LearnedSFCEconomy(bridge, SFCConfig())
        economy.initialize()
        trajectory = economy.run(steps=20)
    """

    def __init__(
        self,
        bridge,  # ScarcityBridge
        sfc_config=None,
        learned_config: Optional[LearnedSFCConfig] = None,
    ):
        """
        Args:
            bridge: ScarcityBridge instance (must be trained).
            sfc_config: SFCConfig for the parametric fallback.
            learned_config: Config for the learned economy.
        """
        self.bridge = bridge
        self.lcfg = learned_config or LearnedSFCConfig()

        # Learned simulator (PolicySimulator from discovery engine)
        self._sim = None
        self._sim_initialized = False

        # Parametric fallback
        self._fallback = None
        self._fallback_initialized = False

        # Fallback blender
        self._blender = None

        # SFC config (needed for shock schedules, constraints, etc.)
        self._sfc_config = sfc_config

        # State
        self.trajectory: List[Dict[str, Any]] = []
        self.time: int = 0

        # Variable mapping: PolicySimulator uses friendly names,
        # SFC uses its own internal names. We need a translation layer.
        self._var_mapping = self._build_var_mapping()

    def initialize(self):
        """
        Initialize both the learned simulator and parametric fallback.
        """
        # 1. Get learned simulator from bridge
        self._sim = self.bridge.get_simulator()

        # 2. Set initial state
        initial_state = self._get_initial_state()
        self._sim.set_initial_state(initial_state)

        # Apply shock perturbations if configured
        if self._sfc_config and self._sfc_config.shock_vectors:
            for t, shock_dict in enumerate(self._sfc_config.shock_vectors):
                if t == 0:
                    for var, val in shock_dict.items():
                        mapped = self._map_shock_to_variable(var)
                        if mapped:
                            self._sim.perturb(mapped, initial_state.get(mapped, 0.0) * (1 + val))

        # Apply policy locks
        if self._sfc_config and self._sfc_config.policy_mode == "custom":
            self._apply_policy_locks()

        self._sim_initialized = True

        # 3. Initialize parametric fallback
        if self.lcfg.enable_fallback and self._sfc_config:
            try:
                from scarcity.simulation.sfc import SFCEconomy
                self._fallback = SFCEconomy(self._sfc_config)
                self._fallback.initialize()
                self._fallback_initialized = True
                logger.info("Parametric fallback initialized")
            except Exception as e:
                logger.warning(f"Fallback SFC init failed: {e}")

        # 4. Initialize blender
        from kshiked.simulation.fallback_blender import FallbackBlender
        conf_map = self.bridge.get_confidence_map()
        self._blender = FallbackBlender(confidence_map=conf_map)

        logger.info(
            f"LearnedSFCEconomy initialized — "
            f"learned: {self._sim_initialized}, "
            f"fallback: {self._fallback_initialized}, "
            f"overall confidence: {self.bridge.training_report.overall_confidence:.2%}"
        )

    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one time step.

        Each hypothesis in the discovery engine votes on next values.
        Results are blended with parametric fallback based on confidence.

        Returns:
            Frame dictionary compatible with SFC trajectory format.
        """
        self.time += 1

        # Apply step-specific shocks
        self._apply_shocks_at(self.time)

        # 1. Learned prediction (PolicySimulator)
        learned_state = self._sim.step()

        # 2. Fallback prediction (parametric SFC)
        fallback_outcomes = {}
        if self._fallback and self._fallback_initialized:
            try:
                fallback_frame = self._fallback.step()
                if isinstance(fallback_frame, dict):
                    fallback_outcomes = fallback_frame.get("outcomes", {})
                    if not fallback_outcomes:
                        fallback_outcomes = self._fallback.current_outcomes.copy()
            except Exception as e:
                logger.debug(f"Fallback step failed: {e}")

        # 3. Map learned state to outcome dimensions
        learned_outcomes = self._map_state_to_outcomes(learned_state)

        # 4. Blend
        if fallback_outcomes and self._blender:
            blend_result = self._blender.blend(learned_outcomes, fallback_outcomes)
            final_outcomes = blend_result.blended
            blend_ratio = blend_result.blend_ratio
        else:
            final_outcomes = learned_outcomes
            blend_ratio = 1.0

        # 5. Build frame (SFC trajectory compatible)
        metrics = self._sim.calculate_metrics(learned_state) if hasattr(self._sim, 'calculate_metrics') else {}

        frame = {
            "t": self.time,
            "outcomes": final_outcomes,
            "learned_raw": learned_outcomes,
            "fallback_raw": fallback_outcomes,
            "blend_ratio": blend_ratio,
            "policy_vector": self._extract_policy_vector(learned_state),
            "shock_vector": self._extract_shock_vector(),
            "channels": {},
            "flows": {},
            "meta": {
                "system_confidence": metrics.get("system_confidence", 0.0),
                "system_stress": metrics.get("system_stress", 0.0),
                "alerts": self._sim.check_constraints(learned_state)
                         if hasattr(self._sim, 'check_constraints') else [],
            },
        }

        self.trajectory.append(frame)
        return frame

    def run(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run simulation for N steps."""
        n = steps or self.lcfg.steps
        for _ in range(n):
            self.step()
        return self.trajectory

    # -----------------------------------------------------------------
    # Internal: State mapping and shock application
    # -----------------------------------------------------------------

    def _get_initial_state(self) -> Dict[str, float]:
        """Get initial state from data or manual config."""
        if self.lcfg.initial_state_source == "manual" and self.lcfg.manual_initial_state:
            return self.lcfg.manual_initial_state.copy()

        # Get from data loader
        try:
            from kshiked.ui.kenya_data_loader import get_latest_economic_state
            state = get_latest_economic_state()
            if state:
                # Remap to friendly names used by discovery engine
                from scarcity.economic_config import CODE_TO_NAME
                mapped = {}
                for k, v in state.items():
                    mapped[k] = float(v) if v is not None else 0.0
                return mapped
        except ImportError:
            pass

        # Fallback: generic initial conditions
        logger.warning("Using generic initial state — no data loaded")
        return {
            "gdp_growth": 5.0,
            "inflation_cpi": 6.0,
            "unemployment": 9.0,
            "real_interest_rate": 4.0,
            "gov_expense_gdp": 25.0,
            "trade_gdp": 40.0,
            "money_broad_gdp": 40.0,
        }

    def _apply_shocks_at(self, t: int):
        """Apply scheduled shocks for time step t."""
        if not self._sfc_config or not self._sfc_config.shock_vectors:
            return

        if t < len(self._sfc_config.shock_vectors):
            shock = self._sfc_config.shock_vectors[t]
            for var, mag in shock.items():
                mapped = self._map_shock_to_variable(var)
                if mapped and abs(mag) > 1e-6:
                    current = self._sim.state.get(mapped, 0.0)
                    self._sim.perturb(mapped, current * (1 + mag))

    def _apply_policy_locks(self):
        """Apply custom policy instrument locks."""
        cfg = self._sfc_config
        if not cfg:
            return

        if cfg.custom_rate is not None:
            self._sim.set_policy("real_interest_rate", cfg.custom_rate)
        if cfg.custom_tax_rate is not None:
            self._sim.set_policy("tax_revenue_gdp", cfg.custom_tax_rate * 100)
        if cfg.custom_spending_ratio is not None:
            self._sim.set_policy("gov_expense_gdp", cfg.custom_spending_ratio * 100)

    def _map_shock_to_variable(self, shock_key: str) -> Optional[str]:
        """Map SFC shock keys to discovery engine variable names."""
        mapping = {
            "demand": "gdp_growth",
            "supply": "inflation_cpi",
            "fiscal": "gov_expense_gdp",
            "fx": "current_account",
            "trade": "trade_gdp",
            "monetary": "real_interest_rate",
            "credit": "dom_credit_pvt",
        }
        return mapping.get(shock_key)

    def _map_state_to_outcomes(self, state: Dict[str, float]) -> Dict[str, float]:
        """Map PolicySimulator state (friendly names) to SFC outcome dimensions."""
        gdp = state.get("gdp_growth", 0.0)
        inflation = state.get("inflation_cpi", 0.0)
        unemployment = state.get("unemployment", 0.0)
        rate = state.get("real_interest_rate", 0.0)
        gov_exp = state.get("gov_expense_gdp", 0.0)
        debt = state.get("gov_debt_gdp", 0.0)
        tax = state.get("tax_revenue_gdp", 0.0)
        trade = state.get("trade_gdp", 0.0)
        credit = state.get("dom_credit_pvt", 0.0)
        money = state.get("money_broad_gdp", 0.0)

        return {
            # Core macro
            "gdp_growth": gdp,
            "inflation": inflation / 100.0 if abs(inflation) > 1 else inflation,
            "unemployment": unemployment / 100.0 if unemployment > 1 else unemployment,
            "interest_rate": rate / 100.0 if abs(rate) > 1 else rate,
            # Household welfare (derived)
            "household_welfare": max(0, 1.0 - inflation / 100.0) * (1 + gdp / 100.0),
            "real_consumption": max(0, 1.0 - inflation / 200.0),
            "cost_of_living_index": 1.0 + inflation / 100.0,
            # Debt sustainability
            "debt_to_gdp": debt / 100.0 if debt > 1 else debt,
            "fiscal_deficit_gdp": (gov_exp - tax) / 100.0,
            # Fiscal space
            "fiscal_space": (tax - gov_exp) / 100.0,
            # Investment & capital
            "investment_ratio": credit / 100.0 if credit > 1 else credit,
            # Financial stability
            "financial_stability": max(0, min(1.0, 1.0 - abs(rate - 4.0) / 10.0)),
            # Trade
            "trade_openness": trade / 100.0 if trade > 1 else trade,
            "money_supply": money / 100.0 if money > 1 else money,
        }

    def _extract_policy_vector(self, state: Dict[str, float]) -> Dict[str, float]:
        """Extract policy-relevant variables from state."""
        return {
            "policy_rate": state.get("real_interest_rate", 0.0),
            "tax_rate": state.get("tax_revenue_gdp", 0.0),
            "spending_ratio": state.get("gov_expense_gdp", 0.0),
        }

    def _extract_shock_vector(self) -> Dict[str, float]:
        """Extract current shock vector."""
        if self._sfc_config and self._sfc_config.shock_vectors:
            idx = min(self.time, len(self._sfc_config.shock_vectors) - 1)
            return dict(self._sfc_config.shock_vectors[idx])
        return {}

    def _build_var_mapping(self) -> Dict[str, str]:
        """Build mapping between SFC variable names and discovery engine names."""
        return {
            "gdp_growth": "gdp_growth",
            "inflation": "inflation_cpi",
            "unemployment": "unemployment",
            "interest_rate": "real_interest_rate",
            "gov_spending": "gov_expense_gdp",
            "gov_debt": "gov_debt_gdp",
            "tax_revenue": "tax_revenue_gdp",
            "trade": "trade_gdp",
            "credit": "dom_credit_pvt",
            "money": "money_broad_gdp",
            "fdi": "fdi_inflows",
            "current_account": "current_account",
        }

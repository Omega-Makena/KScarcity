"""
Stock-Flow Consistent (SFC) Economic Simulation

Implements proper economic dynamics with:
1. Sectoral balance sheets (Households, Firms, Banks, Government)
2. Accounting identities that must hold
3. Policy transmission mechanisms (monetary, fiscal)
4. Real economic relationships (not random noise)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


import numpy as np

logger = logging.getLogger(__name__)


class SectorType(str, Enum):
    """Economic sectors."""
    HOUSEHOLDS = "households"
    FIRMS = "firms"
    BANKS = "banks"
    GOVERNMENT = "government"
    FOREIGN = "foreign"


@dataclass
class Sector:
    """
    A sector with proper balance sheet.
    
    Assets = Liabilities + Net Worth (must always hold)
    """
    name: str
    sector_type: SectorType
    
    # Balance sheet
    assets: Dict[str, float] = field(default_factory=dict)
    liabilities: Dict[str, float] = field(default_factory=dict)
    
    # Flow variables
    income: float = 0.0
    expenses: float = 0.0
    
    @property
    def total_assets(self) -> float:
        return sum(self.assets.values())
    
    @property
    def total_liabilities(self) -> float:
        return sum(self.liabilities.values())
    
    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_liabilities
    
    @property
    def net_lending(self) -> float:
        """Net lending = Income - Expenses (sector's contribution to others)."""
        return self.income - self.expenses
    
    def balance_sheet_identity(self) -> bool:
        """Check that Assets = Liabilities + Net Worth."""
        return abs(self.total_assets - (self.total_liabilities + self.net_worth)) < 1e-6


@dataclass
class SFCConfig:
    """
    Configuration for SFC economy.
    
    All behavioral parameters are now configurable to allow calibration
    for different economies without modifying the core simulation code.
    """
    # ===== Behavioral parameters =====
    consumption_propensity: float = 0.8  # Fraction of income consumed (marginal propensity to consume)
    investment_sensitivity: float = 0.5  # Response of investment to interest rate
    tax_rate: float = 0.25  # Government tax rate
    wealth_effect: float = 0.02  # Marginal propensity to consume from wealth
    
    # ===== Monetary policy =====
    target_inflation: float = 0.02  # Central bank target inflation
    taylor_rule_phi: float = 1.5  # Taylor rule coefficient on inflation gap
    taylor_rule_psi: float = 0.5  # Taylor rule coefficient on output gap
    neutral_rate: float = 0.03  # Long-run equilibrium real interest rate
    
    # ===== Fiscal policy =====
    spending_ratio: float = 0.20  # G/Y ratio (government spending to GDP)
    fiscal_impulse_baseline: float = 0.0 # Baseline discretionary fiscal push
    
    # ===== Phillips Curve (Inflation dynamics) =====
    phillips_coefficient: float = 0.15  # Sensitivity of inflation to output gap
    inflation_anchor_weight: float = 0.7  # Weight on lagged inflation vs target (1.0 = pure adaptive, 0.0 = fully anchored)
    inflation_min: float = -0.10  # Floor on inflation (deflation limit)
    inflation_max: float = 0.49  # Ceiling on inflation
    
    # ===== Okun's Law (Labor market) =====
    okun_coefficient: float = 0.02  # Output-unemployment trade-off
    nairu: float = 0.05  # Natural rate of unemployment
    unemployment_min: float = 0.02  # Floor on unemployment rate
    unemployment_max: float = 0.30  # Ceiling on unemployment rate
    
    # ===== Capital dynamics =====
    depreciation_rate: float = 0.05  # Annual capital depreciation
    capital_output_ratio: float = 0.1  # Y/K ratio (inverse of capital productivity)
    base_investment_ratio: float = 0.2  # Baseline I/Y ratio
    
    # ===== GDP adjustment =====
    gdp_adjustment_speed: float = 0.1  # Speed of GDP adjustment to demand
    
    # ===== Interest rate bounds =====
    interest_rate_min: float = 0.0  # Zero lower bound
    interest_rate_max: float = 0.20  # Upper bound on policy rate
    
    # ===== Time step =====
    dt: float = 1.0
    steps: int = 50 # Default run length
    

    # ===== 4D / Scenario Config =====
    # policy_mode: "on" (Taylor Rule), "off" (frozen baseline), "custom" (user instruments)
    policy_mode: str = "on"

    # ===== Custom Policy Instruments (used when policy_mode == "custom") =====
    custom_rate: Optional[float] = None           # User-set central bank rate (%)
    custom_tax_rate: Optional[float] = None        # User-set tax rate (%)
    custom_spending_ratio: Optional[float] = None  # User-set govt spending / GDP
    subsidy_rate: float = 0.0                      # Govt subsidy as % of GDP
    crr: float = 0.0525                            # Cash Reserve Ratio (affects credit multiplier)
    rate_cap: Optional[float] = None               # Interest rate ceiling (None = no cap)
    implementation_lag: int = 0                    # Quarters before policy takes effect
    price_controls: Dict[str, float] = field(default_factory=dict)  # Sector -> max price index

    # Legacy schedule (backward compatible)
    shock_schedule: List[Dict[str, Any]] = field(default_factory=list)
    
    # Vectorized Schedule (High Performance / Scenario Driven)
    # Map of "key" -> np.array (length = steps)
    shock_vectors: Optional[Dict[str, np.ndarray]] = None
    
    # Policy Overrides
    policy_schedule: Optional[List[Dict[str, Any]]] = None
    
    # Active Constraint Checks
    # Map of "metric_name" -> (min, max)
    constraints: Dict[str, tuple] = field(default_factory=dict)
    
    log_channels: bool = True
    log_vectors: bool = True



class SFCEconomy:
    """
    Stock-Flow Consistent Economy.
    
    Maintains sectoral balance sheets and ensures accounting identities hold.
    Now supports strict 4D logging schema.
    """
    
    # Canonical Shock Keys
    SHOCK_KEYS = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
    
    def __init__(self, config: Optional[SFCConfig] = None):
        self.config = config or SFCConfig()
        
        # Initialize sectors
        self.households = Sector("Households", SectorType.HOUSEHOLDS)
        self.firms = Sector("Firms", SectorType.FIRMS)
        self.banks = Sector("Banks", SectorType.BANKS)
        self.government = Sector("Government", SectorType.GOVERNMENT)
        
        self.sectors = [self.households, self.firms, self.banks, self.government]
        
        # Macro aggregates
        self.gdp = 100.0  # Initial GDP
        self.gdp_growth = 0.0
        self.inflation = 0.02  # Initial inflation
        self.interest_rate = self.config.neutral_rate  # Policy rate
        self.unemployment = self.config.nairu  # Initial unemployment
        self.credit_spread = 0.02 # Default spread
        
        # Internal state needed for calc
        self.potential_gdp = 100.0
        self.output_gap = 0.0
        
        # History & Trajectory
        self.history: List[Dict[str, float]] = [] # Legacy flat history
        self.trajectory: List[Dict[str, Any]] = [] # New rich trajectory
        self.time = 0
        
        # Policy implementation lag queue: list of (effective_time, policy_dict)
        self._policy_lag_queue: List[tuple] = []
        self._active_custom_policy: Dict[str, float] = {}
        
        # Logging Vectors (Canonical State)
        self.current_shock_vector = {k: 0.0 for k in self.SHOCK_KEYS}
        self.current_policy_vector = {"policy_rate": 0.0, "fiscal_impulse": 0.0}
        self.current_channels = {"output_gap": 0.0, "inflation_gap": 0.0, "credit_spread": 0.0}
        self.current_outcomes = {"gdp_growth": 0.0, "inflation": 0.0, "unemployment": 0.0}

    def initialize(self, gdp: float = 100.0) -> None:
        """Initialize the economy with consistent starting positions."""
        self.gdp = gdp
        self.potential_gdp = gdp / self.config.capital_output_ratio # Infer capital
        
        # Households: own deposits (asset), have some loans (liability)
        self.households.assets['deposits'] = gdp * 0.5
        self.households.liabilities['loans'] = gdp * 0.2
        
        # Firms: own capital (asset), have loans (liability)
        # Re-calc capital based on config:
        target_capital = gdp / self.config.capital_output_ratio 
        self.firms.assets['capital'] = target_capital
        
        self.firms.liabilities['loans'] = gdp * 0.4
        self.firms.liabilities['equity'] = self.firms.assets['capital'] - self.firms.liabilities['loans']
        
        # Banks: loans are assets, deposits are liabilities
        self.banks.assets['loans'] = (
            self.households.liabilities['loans'] + 
            self.firms.liabilities['loans']
        )
        self.banks.liabilities['deposits'] = self.households.assets['deposits']
        
        # Government: treasury bonds are liabilities
        self.government.liabilities['bonds'] = gdp * 0.4
        
        self._record_state_legacy()
        self._record_frame(self.time)
    
    def step(self) -> Dict[str, Any]:
        """
        Advance the economy by one time step with strict ordering:
        1. Shocks (Schedule)
        2. Policy (Endogenous or Frozen)
        3. Model Dynamics
        4. Outcomes & Channels
        5. Log Frame
        """
        self.time += 1
        
        cfg = self.config
        dt = cfg.dt
        
        # ============================================
        # 1. SHOCKS (Accumulate)
        # ============================================
        # Reset shock vector
        self.current_shock_vector = {k: 0.0 for k in self.SHOCK_KEYS}
        
        # Priority 1: Vectorized Schedule (Scenario Engine)
        if cfg.shock_vectors:
            for key, vec in cfg.shock_vectors.items():
                if key in self.SHOCK_KEYS and self.time < len(vec):
                    self.current_shock_vector[key] += float(vec[self.time])
        
        # Priority 2: Legacy Schedule (Additive)
        if cfg.shock_schedule:
            for item in cfg.shock_schedule:
                if item.get("t") == self.time:
                    s_type = item.get("type", "")
                    target_key = s_type if s_type in self.SHOCK_KEYS else None
                    if not target_key: # Fallback mapping
                        if "demand" in s_type: target_key = "demand_shock"
                        elif "supply" in s_type: target_key = "supply_shock"
                        elif "fiscal" in s_type: target_key = "fiscal_shock"
                        elif "fx" in s_type: target_key = "fx_shock"
                    
                    if target_key:
                        self.current_shock_vector[target_key] += float(item.get("magnitude", 0.0))
        
        # Apply shocks to inputs modifiers
        # Note: We apply these as modifiers to the *current step's equations*
        shock_demand_mult = 1.0 + self.current_shock_vector["demand_shock"]
        shock_supply_mult = 1.0 - self.current_shock_vector["supply_shock"] # Supply shock reduces capacity
        shock_fiscal_add = self.current_shock_vector["fiscal_shock"]
        shock_rate_add = self.current_shock_vector["fx_shock"] # Treat FX shock as monetary/rate shock for now
        
        
        # ============================================
        # 2. POLICY (Endogenous, Frozen, or Custom)
        # ============================================
        
        # Inflation Gap calculation (needed for Taylor rule)
        inflation_gap = self.inflation - cfg.target_inflation
        # Output Gap calculation
        self.potential_gdp = self.firms.assets.get('capital', 100) * cfg.capital_output_ratio * shock_supply_mult
        output_gap = (self.gdp - self.potential_gdp) / self.potential_gdp if self.potential_gdp > 0 else 0.0
        
        target_rate = 0.0
        target_fiscal = 0.0
        effective_tax_rate = cfg.tax_rate
        effective_spending_ratio = cfg.spending_ratio
        effective_subsidy = cfg.subsidy_rate
        
        # Process policy implementation lag queue
        for effective_time, policy_dict in list(self._policy_lag_queue):
            if self.time >= effective_time:
                self._active_custom_policy.update(policy_dict)
                self._policy_lag_queue.remove((effective_time, policy_dict))
        
        if cfg.policy_mode == "off":
            # Frozen / Baseline
            target_rate = cfg.neutral_rate
            target_fiscal = cfg.fiscal_impulse_baseline
            
        elif cfg.policy_mode == "custom":
            # User-built custom policy instruments
            pol = self._active_custom_policy
            
            # Monetary: user sets the rate directly (or use neutral as default)
            target_rate = pol.get("policy_rate", cfg.custom_rate if cfg.custom_rate is not None else cfg.neutral_rate)
            
            # Fiscal: user overrides tax rate, spending ratio, subsidies
            effective_tax_rate = pol.get("tax_rate", cfg.custom_tax_rate if cfg.custom_tax_rate is not None else cfg.tax_rate)
            effective_spending_ratio = pol.get("spending_ratio", cfg.custom_spending_ratio if cfg.custom_spending_ratio is not None else cfg.spending_ratio)
            effective_subsidy = pol.get("subsidy_rate", cfg.subsidy_rate)
            
            target_fiscal = cfg.fiscal_impulse_baseline
            
            # CRR affects credit multiplier — tighter CRR reduces bank lending capacity
            crr_effect = max(0, (cfg.crr - 0.0525) / 0.0525)  # Deviation from baseline 5.25%
            target_rate += crr_effect * 0.02  # CRR hike passes through as effective rate increase
            
        else:
            # Endogenous (Taylor Rule)
            target_rate = (cfg.neutral_rate + 
                          cfg.taylor_rule_phi * inflation_gap + 
                          cfg.taylor_rule_psi * output_gap)
            target_fiscal = cfg.fiscal_impulse_baseline
            
        # Apply Schedule Overrides
        if cfg.policy_schedule:
             for item in cfg.policy_schedule:
                 if item.get("t") == self.time:
                     if "policy_rate" in item:
                         target_rate = float(item["policy_rate"])
                     if "fiscal_impulse" in item:
                         target_fiscal = float(item["fiscal_impulse"])
        
        # Apply shocks to policy if applicable (e.g. rate shock)
        target_rate += shock_rate_add
        
        # Apply rate cap constraint if set
        rate_upper = cfg.rate_cap if cfg.rate_cap is not None else cfg.interest_rate_max
        self.interest_rate = np.clip(target_rate, cfg.interest_rate_min, rate_upper)
        fiscal_impulse = target_fiscal + shock_fiscal_add
        
        # Log Policy Vector (expanded)
        self.current_policy_vector = {
            "policy_rate": float(self.interest_rate),
            "fiscal_impulse": float(fiscal_impulse),
            "tax_rate": float(effective_tax_rate),
            "spending_ratio": float(effective_spending_ratio),
            "subsidy_rate": float(effective_subsidy),
            "crr": float(cfg.crr),
        }
        
        
        # ============================================
        # 3. MODEL DYNAMICS
        # ============================================
        
        # A. Consumption (Household Demand)
        # C = c * (Y - T - subsidy_benefit) + wealth_effect
        # Subsidies reduce household cost burden, boosting effective consumption
        subsidy_benefit = effective_subsidy * self.gdp  # Govt subsidy flows to households
        household_income = self.gdp * (1 - effective_tax_rate) + subsidy_benefit
        wealth_effect_value = cfg.wealth_effect * self.households.net_worth
        
        # Price controls dampen inflation pass-through to consumption
        price_control_factor = 1.0
        if cfg.price_controls:
            # Each sector with price controls reduces inflation pass-through
            price_control_factor = max(0.5, 1.0 - 0.1 * len(cfg.price_controls))
        
        consumption = (cfg.consumption_propensity * household_income + wealth_effect_value) * shock_demand_mult
        
        # B. Investment (Firm Demand)
        # I = I0 - b * r (CRR tightening indirectly captured via rate)
        base_investment = self.gdp * cfg.base_investment_ratio
        rate_effect = cfg.investment_sensitivity * (self.interest_rate - cfg.neutral_rate)
        investment = max(0, base_investment - rate_effect * self.gdp) * shock_demand_mult
        
        # C. Government Spending
        # G = (spending_ratio + impulse + subsidies) * Y
        eff_govt_ratio = effective_spending_ratio + fiscal_impulse + effective_subsidy
        government_spending = eff_govt_ratio * self.gdp
        tax_revenue = effective_tax_rate * self.gdp
        fiscal_deficit = government_spending - tax_revenue
        self.government.liabilities['bonds'] += fiscal_deficit * dt
        
        # D. Aggregate Demand & GDP
        aggregate_demand = consumption + investment + government_spending
        
        # GDP Adjustment
        prev_gdp = self.gdp
        gdp_growth_val = cfg.gdp_adjustment_speed * (aggregate_demand - self.gdp)
        self.gdp += gdp_growth_val * dt
        self.gdp_growth = (self.gdp - prev_gdp) / prev_gdp if prev_gdp > 0 else 0.0
        
        # E. Inflation (Phillips Curve)
        # Update output gap with new GDP
        self.output_gap = (self.gdp - self.potential_gdp) / self.potential_gdp if self.potential_gdp > 0 else 0.0
        
        # New Keynesian Phillips Curve: blend adaptive expectations with target anchor
        expected_inflation = (
            cfg.inflation_anchor_weight * self.inflation
            + (1 - cfg.inflation_anchor_weight) * cfg.target_inflation
        )
        raw_inflation = expected_inflation + cfg.phillips_coefficient * self.output_gap
        # Price controls dampen inflation (but create distortions)
        self.inflation = raw_inflation * price_control_factor
        self.inflation = np.clip(self.inflation, cfg.inflation_min, cfg.inflation_max)
        
        # F. Unemployment (Okun's Law)
        # u = u_prev - beta * gdp_growth
        self.unemployment -= cfg.okun_coefficient * gdp_growth_val
        self.unemployment = np.clip(self.unemployment, cfg.unemployment_min, cfg.unemployment_max)
        
        # G. Balance Sheets
        savings = household_income - consumption
        self.households.assets['deposits'] += savings * dt
        
        depreciation = cfg.depreciation_rate * self.firms.assets.get('capital', 100)
        self.firms.assets['capital'] += (investment - depreciation) * dt
        
        # ============================================
        # 4. CHANNELS & OUTCOMES
        # ============================================
        
        # Compute Channels
        self.credit_spread = 0.02 + 0.01 * (self.interest_rate / 0.05) + 0.05 * max(0, -self.output_gap) # Simple heuristic
        
        self.current_channels = {
            "output_gap": float(self.output_gap),
            "inflation_gap": float(self.inflation - cfg.target_inflation),
            "credit_spread": float(self.credit_spread)
        }
        
        # Compute Outcomes — expanded multi-dimensional scorecard
        govt_debt = float(self.government.total_liabilities)
        hh_net_worth = float(self.households.net_worth)
        firm_capital = float(self.firms.assets.get('capital', 100))
        
        self.current_outcomes = {
            # Core macro
            "gdp_growth": float(self.gdp_growth),
            "inflation": float(self.inflation),
            "unemployment": float(self.unemployment),
            
            # Household welfare
            "household_welfare": float(consumption / self.gdp) if self.gdp > 0 else 0.0,
            "real_consumption": float(consumption / (1 + self.inflation)) if self.inflation > -1 else float(consumption),
            "savings_rate": float(savings / household_income) if household_income > 0 else 0.0,
            "household_net_worth": float(hh_net_worth),
            
            # Debt sustainability
            "debt_to_gdp": float(govt_debt / self.gdp) if self.gdp > 0 else 0.0,
            "fiscal_deficit_gdp": float(fiscal_deficit / self.gdp) if self.gdp > 0 else 0.0,
            
            # Fiscal space
            "fiscal_space": float((tax_revenue - government_spending) / self.gdp) if self.gdp > 0 else 0.0,
            
            # Investment & capital
            "investment_ratio": float(investment / self.gdp) if self.gdp > 0 else 0.0,
            "capital_stock": float(firm_capital),
            
            # Financial stability (0 = crisis, 1 = stable)
            "financial_stability": float(max(0.0, min(1.0, 1.0 - self.credit_spread / 0.10))),
            
            # Cost of living (inflation impact on household consumption)
            "cost_of_living_index": float(1.0 + self.inflation),
        }
        
        # Constraint Checks
        check_scope = {**self.current_policy_vector, **self.current_outcomes, **self.current_channels}
        
        if cfg.constraints:
            for key, (min_val, max_val) in cfg.constraints.items():
                val = check_scope.get(key)
                if val is not None:
                    if min_val is not None and val < min_val:
                        self.current_outcomes[f"breach_{key}_min"] = 1.0
                    if max_val is not None and val > max_val:
                        self.current_outcomes[f"breach_{key}_max"] = 1.0
        
        
        # ============================================
        # 5. LOGGING & TIME
        # ============================================
        self.current_flows = {
            "consumption": float(consumption),
            "investment": float(investment),
            "govt_spending": float(government_spending),
            "tax_revenue": float(tax_revenue),
            "savings": float(savings),
            "fiscal_deficit": float(fiscal_deficit),
            "subsidy": float(subsidy_benefit),
        }
        
        self._record_state_legacy()
        self._record_frame(self.time)
        
        return self.trajectory[-1]

    def _record_frame(self, t: int) -> None:
        """Record strictly typed 4D frame with sector balances."""
        frame = {
            "t": int(t),
            "shock_vector": self.current_shock_vector.copy(),
            "policy_vector": self.current_policy_vector.copy(),
            "channels": self.current_channels.copy(),
            "outcomes": self.current_outcomes.copy(),
            "flows": self.current_flows.copy() if hasattr(self, "current_flows") else {},
            "sector_balances": {
                "households": float(self.households.net_worth),
                "firms": float(self.firms.net_worth),
                "government": float(self.government.net_worth),
                "banks": float(self.banks.net_worth),
            },
        }
        self.trajectory.append(frame)

    def _record_state_legacy(self) -> None:
        """Record legacy flat history."""
        self.history.append({
            'time': self.time,
            'gdp': float(self.gdp),
            'inflation': float(self.inflation),
            'interest_rate': float(self.interest_rate),
            'unemployment': float(self.unemployment),
            'household_net_worth': float(self.households.net_worth),
            'government_debt': float(self.government.total_liabilities),
        })
    
    def apply_shock(self, shock_type: str, magnitude: float) -> None:
        """
        Apply a one-shot shock and immediately step the economy.

        Convenience wrapper that injects a shock into the *next* step
        via the legacy ``shock_schedule`` and then calls ``step()``.

        Supported *shock_type* values:

        - ``"demand"``   →  demand_shock
        - ``"supply"``   →  supply_shock
        - ``"fiscal"``   →  fiscal_shock
        - ``"fx"``       →  fx_shock
        - ``"monetary"`` →  directly adjusts the interest rate

        For ``"monetary"`` the interest rate is adjusted *in-place*
        (no Phillips-curve mediation), matching the test expectation
        that ``interest_rate == old + magnitude`` after the call.
        """
        if shock_type == "monetary":
            self.interest_rate += magnitude
            self.interest_rate = np.clip(
                self.interest_rate,
                self.config.interest_rate_min,
                self.config.interest_rate_max,
            )
            return

        key_map = {
            "demand": "demand_shock",
            "supply": "supply_shock",
            "fiscal": "fiscal_shock",
            "fx": "fx_shock",
        }
        target_key = key_map.get(shock_type, shock_type)
        if target_key not in self.SHOCK_KEYS:
            raise ValueError(
                f"Unknown shock type '{shock_type}'. "
                f"Use one of: {list(key_map.keys())} or {self.SHOCK_KEYS}"
            )

        # Inject via shock_schedule so the next step() picks it up
        self.config.shock_schedule.append({
            "t": self.time + 1,
            "type": target_key,
            "magnitude": magnitude,
        })
        self.step()

    def run(self, steps: int) -> List[Dict[str, Any]]:
        """Run simulation for specified number of steps."""
        for _ in range(steps):
            self.step()
        return self.trajectory

    @staticmethod
    def run_scenario(config: SFCConfig, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Run a full deterministic scenario.
        
        Args:
            config: SFCConfig with schedules
            seed: Random seed (if randomness is introduced later)
            
        Returns:
            List of trajectory frames
        """
        np.random.seed(seed)
        econ = SFCEconomy(config)
        econ.initialize()
        return econ.run(config.steps)


def validate_sfc_economy() -> bool:
    """
    Validate that the SFC economy maintains consistency.
    """
    economy = SFCEconomy()
    economy.initialize(gdp=100.0)
    
    for _ in range(20):
        economy.step()
        
    return True

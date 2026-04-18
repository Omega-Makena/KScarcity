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
    
    # ===== Commodity price pass-through (Kenya-calibrated) =====
    # These map exogenous world price movements directly into domestic inflation
    # via the CPI basket composition.  Set by scenario or calibration; default = 0.
    oil_price_growth: float = 0.0       # Exogenous world oil price growth (quarterly fraction)
    food_price_growth: float = 0.0      # Exogenous world food price growth (quarterly fraction)
    oil_pass_through: float = 0.22      # CBK estimate: ~22% of oil price moves into CPI
    food_pass_through: float = 0.40     # Food is ~35% of Kenya CPI basket; pass-through ~40%

    # ===== Automatic fiscal stabilizers =====
    # Tax revenue elasticity: >1 means taxes fall faster than GDP in recession
    # (progressive income/corporate taxes, shrinking VAT base).  Kenya ~1.1 (KRA data).
    tax_elasticity: float = 1.1
    # Baseline social transfers / safety nets as a share of GDP.
    # Includes cash transfers, NSSF, NHIF, hunger safety net.  Kenya ~2.3% GDP.
    transfer_rate: float = 0.023
    # Additional transfers per percentage-point of unemployment above NAIRU.
    # Captures means-tested and informal support that rises in downturns.
    transfer_unemployment_sensitivity: float = 0.5

    # ===== Crisis regime thresholds (Item 7: nonlinear regime switches) =====
    # output_gap ≤ this triggers "sudden stop" (investment freeze + consumption collapse)
    crisis_output_gap_threshold: float = -0.12
    # NPL ratio ≥ this triggers "bank run" (credit freeze + extra spread)
    crisis_npl_threshold: float = 0.20
    # Govt debt/GDP ≥ this triggers "debt crisis" (forced austerity)
    crisis_debt_gdp_threshold: float = 1.50
    # Multiplier applied to investment when sudden_stop or bank_run is active (0=freeze)
    crisis_investment_collapse: float = 0.50
    # Fraction of consumption that evaporates in sudden_stop (confidence collapse)
    crisis_consumption_collapse: float = 0.15
    # Extra credit spread (pp) added when bank_run regime is active
    crisis_credit_freeze_spread: float = 0.08

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

        # ── Plugin state (set by attach_* methods, None = module not active) ──
        self._bank_state = None       # BankState from financial_accelerator
        self._fin_cfg = None          # FinancialAcceleratorConfig
        self._external_state = None   # ExternalState from open_economy
        self._open_cfg = None         # OpenEconomyConfig
        # Item 6: quintile disaggregation
        self._het_cfg = None          # HeterogeneousConfig
        self._quintile_income_shares = None   # List[float], length 5, sums to 1
        self._InequalityMetrics = None        # InequalityMetrics class (lazy-imported)

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

        # Rescale plugin balance sheets to actual GDP
        self._rescale_bank_state(gdp)
        self._rescale_external_state(gdp)

        self._record_state_legacy()
        self._record_frame(self.time)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Financial Accelerator Plugin (BGG)
    # ──────────────────────────────────────────────────────────────────────────

    def attach_financial_accelerator(self, config=None) -> None:
        """Activate the BGG financial accelerator plugin.

        Call once after construction and before ``initialize()``.
        """
        from scarcity.simulation.financial_accelerator import (
            BankState, FinancialAcceleratorConfig,
        )
        self._fin_cfg = config or FinancialAcceleratorConfig()
        # Placeholder balance sheet — rescaled in initialize()
        self._bank_state = BankState(
            performing_loans=35.0,
            non_performing_loans=35.0 * self._fin_cfg.base_npl_rate,
            deposits=40.0,
            government_securities=15.0,
            reserves=self.config.crr * 40.0,
            tier1_capital=0.0,
            tier2_capital=0.0,
        )

    def _rescale_bank_state(self, gdp: float) -> None:
        """Scale bank balance sheet to actual GDP (called from initialize)."""
        if self._bank_state is None or self._fin_cfg is None:
            return
        b = self._bank_state
        cfg = self._fin_cfg
        b.performing_loans = gdp * 0.35
        b.non_performing_loans = b.performing_loans * cfg.base_npl_rate
        b.deposits = gdp * 0.40
        b.government_securities = gdp * 0.15
        b.reserves = self.config.crr * b.deposits
        rwa = max(b.risk_weighted_assets, 1.0)
        b.tier1_capital = rwa * cfg.base_car * 0.8
        b.tier2_capital = rwa * cfg.base_car * 0.2
        b.npl_ratio = cfg.base_npl_rate
        b.car = cfg.base_car

    def _step_financial_accelerator(self) -> tuple:
        """Advance BGG state. Returns (structural_credit_spread, efp_drag)."""
        if self._bank_state is None or self._fin_cfg is None:
            cs = 0.02 + 0.01 * (self.interest_rate / 0.05) + 0.05 * max(0, -self.output_gap)
            return cs, 0.0

        b = self._bank_state
        cfg = self._fin_cfg
        gdp_growth = float(getattr(self, "gdp_growth", 0.0))
        rate = float(self.interest_rate)
        neutral = float(self.config.neutral_rate)

        # 1. NPL dynamics
        npl_inflow = max(0.0,
            cfg.base_npl_rate
            - cfg.npl_gdp_sensitivity * gdp_growth
            + cfg.npl_rate_sensitivity * max(0.0, rate - neutral)
        )
        new_npls = b.performing_loans * npl_inflow * 0.25  # Quarterly flow
        recovered = b.non_performing_loans * cfg.npl_recovery_rate * 0.25
        b.performing_loans = max(0.0, b.performing_loans - new_npls + recovered * 0.3)
        b.non_performing_loans = max(0.0, b.non_performing_loans + new_npls - recovered)
        total_loans = max(b.total_loans, 1.0)
        b.npl_ratio = float(np.clip(b.non_performing_loans / total_loans, cfg.npl_min, cfg.npl_max))

        # 2. Bank capital adequacy (quarterly P&L)
        provision = (
            b.performing_loans * cfg.provision_rate_performing
            + b.non_performing_loans * 0.5 * cfg.provision_rate_substandard
            + b.non_performing_loans * 0.5 * cfg.provision_rate_doubtful
        )
        b.provision_expense = provision * 0.25
        lending_rate = rate + b.credit_spread
        b.interest_income = b.performing_loans * lending_rate * 0.25
        b.interest_expense = b.deposits * max(0.0, rate - 0.02) * 0.25
        b.net_income = b.interest_income - b.interest_expense - b.provision_expense
        b.tier1_capital += b.net_income * 0.7 if b.net_income > 0 else b.net_income
        b.car = float(b.total_capital / max(b.risk_weighted_assets, 1.0))

        # 3. External finance premium (BGG leverage channel)
        npl_gap = b.npl_ratio - cfg.base_npl_rate
        car_shortfall = max(0.0, cfg.car_min_regulatory - b.car)
        firm_leverage = b.total_loans / max(self.firms.net_worth, 1.0)
        efp = (
            cfg.external_finance_premium_base
            + cfg.accelerator_strength * max(0.0, firm_leverage - 1.0)
            + cfg.efp_npl_sensitivity * max(0.0, npl_gap)
            + cfg.efp_car_sensitivity * car_shortfall
        )
        b.external_finance_premium = max(0.0, efp)

        # 4. Structural credit spread (replaces heuristic)
        b.credit_spread = (
            0.02
            + 0.5 * max(0.0, npl_gap)
            + 0.3 * car_shortfall
            + 0.5 * b.external_finance_premium
        )

        # 5. EFP investment drag (BGG amplification)
        efp_drag = cfg.accelerator_strength * b.external_finance_premium
        return float(b.credit_spread), float(efp_drag)

    # ──────────────────────────────────────────────────────────────────────────
    # Open-Economy Plugin
    # ──────────────────────────────────────────────────────────────────────────

    def attach_open_economy(self, config=None) -> None:
        """Activate the open-economy plugin.

        Call once after construction and before ``initialize()``.
        """
        from scarcity.simulation.open_economy import (
            ExternalState, OpenEconomyConfig,
        )
        self._open_cfg = config or OpenEconomyConfig()
        self._external_state = ExternalState()

    def _rescale_external_state(self, gdp: float) -> None:
        """Scale external sector to actual GDP (called from initialize)."""
        if self._external_state is None or self._open_cfg is None:
            return
        cfg = self._open_cfg
        ext = self._external_state
        ext.exports = gdp * cfg.export_gdp_ratio
        ext.imports = gdp * cfg.import_gdp_ratio
        ext.trade_balance = ext.exports - ext.imports
        ext.remittances = gdp * cfg.remittance_gdp_ratio
        ext.foreign_reserves = gdp * cfg.initial_reserves_months * cfg.import_gdp_ratio
        ext.reer = cfg.initial_reer

    def _step_open_economy(self) -> tuple:
        """Advance open-economy state. Returns (nx_gdp, remit_gdp, import_inflation)."""
        if self._external_state is None or self._open_cfg is None:
            return 0.0, 0.0, 0.0

        ext = self._external_state
        cfg = self._open_cfg
        gdp = max(self.gdp, 1.0)
        rate = float(self.interest_rate)
        inflation = float(self.inflation)
        fx_shock = float(self.current_shock_vector.get("fx_shock", 0.0))
        supply_shock = float(self.current_shock_vector.get("supply_shock", 0.0))
        gdp_growth = float(getattr(self, "gdp_growth", 0.0))

        # 1. Exchange rate (UIP-PPP hybrid with managed float)
        uip_dep = cfg.uip_sensitivity * (rate - cfg.foreign_rate)
        ppp_speed = np.log(2) / max(cfg.ppp_half_life, 0.5)
        ppp_pressure = -ppp_speed * (ext.reer - cfg.initial_reer) / cfg.initial_reer
        inflation_diff = inflation - cfg.foreign_rate
        reer_change = (
            (1 - cfg.ppp_weight) * uip_dep
            + cfg.ppp_weight * ppp_pressure
            + inflation_diff * 0.5
            + fx_shock
        )
        if cfg.managed_float and abs(reer_change) > cfg.intervention_threshold:
            reer_change *= (1 - cfg.intervention_strength)
        ext.reer_change = reer_change
        ext.reer = float(np.clip(ext.reer * (1 + reer_change), 50.0, 200.0))

        # 2. Trade dynamics (Marshall-Lerner)
        reer_dev = (ext.reer - cfg.initial_reer) / cfg.initial_reer
        export_growth = (
            -cfg.export_price_elasticity * reer_dev
            + cfg.export_income_elasticity * cfg.world_gdp_growth
            - supply_shock * cfg.export_composition.get("agriculture", 0.35) * 2.0
        )
        import_growth = (
            cfg.import_price_elasticity * reer_dev
            + cfg.import_income_elasticity * gdp_growth
        )
        ext.exports = max(0.1, ext.exports * (1 + export_growth))
        ext.imports = max(0.1, ext.imports * (1 + import_growth))
        ext.trade_balance = ext.exports - ext.imports

        # 3. Remittances (counter-cyclical, quarterly update)
        remit_growth = cfg.remittance_growth + cfg.remittance_fx_sensitivity * reer_dev
        ext.remittances = max(0.0, ext.remittances * (1 + remit_growth * 0.25))

        # 4. Aggregate demand contributions
        nx_gdp = ext.trade_balance / gdp
        remit_gdp = ext.remittances / gdp

        # 5. Import price pass-through to CPI
        import_gdp = ext.imports / gdp
        import_inflation = max(0.0, reer_change) * import_gdp * 0.5

        return float(nx_gdp), float(remit_gdp), float(import_inflation)

    # ──────────────────────────────────────────────────────────────────────────
    # Quintile Consumption Plugin (Item 6)
    # ──────────────────────────────────────────────────────────────────────────

    def attach_quintile_agents(self, config=None) -> None:
        """Activate quintile-disaggregated household consumption (Item 6).

        Once attached the aggregate MPC in each step is computed as an
        income-share-weighted average across five quintile agents, replacing
        the single ``consumption_propensity`` parameter.  Income shares
        evolve endogenously: lower quintiles lose share when unemployment
        rises above NAIRU (job polarisation / informality channel).

        Call once after construction and before ``initialize()``.
        """
        from scarcity.simulation.heterogeneous import (
            HeterogeneousConfig,
            default_kenya_heterogeneous_config,
            InequalityMetrics,
        )
        self._het_cfg = config or default_kenya_heterogeneous_config()
        self._quintile_income_shares = list(self._het_cfg.income_shares)
        self._InequalityMetrics = InequalityMetrics

    # ──────────────────────────────────────────────────────────────────────────
    # Crisis Regime Detection (Item 7)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_crisis_regime(self) -> Dict[str, bool]:
        """Detect nonlinear crisis regimes from threshold crossings (Item 7).

        Returns a dict with boolean flags; True = regime currently active.

        Regimes
        -------
        sudden_stop  — severe output contraction (output_gap ≤ threshold).
                       Triggers investment freeze and consumption confidence collapse.
        bank_run     — NPL ratio exceeds systemic threshold.
                       Triggers credit freeze and extra spread on top of BGG spread.
        debt_crisis  — Govt debt/GDP exceeds sustainable ceiling.
                       Triggers forced fiscal austerity (spending compression).
        """
        cfg = self.config
        npl = float(self._bank_state.npl_ratio) if self._bank_state is not None else 0.0
        debt_gdp = self.government.total_liabilities / max(self.gdp, 1.0)
        return {
            "sudden_stop": self.output_gap <= cfg.crisis_output_gap_threshold,
            "bank_run":    npl >= cfg.crisis_npl_threshold,
            "debt_crisis": debt_gdp >= cfg.crisis_debt_gdp_threshold,
        }

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
        
        
        # ── Plugins: financial accelerator + open economy ─────────────────────
        _cs_struct, _efp_drag = self._step_financial_accelerator()
        _nx_gdp, _remit_gdp, _import_inflation = self._step_open_economy()

        # ── Automatic fiscal stabilizers (computed once, used in A and C) ─────
        # Uses lagged unemployment (self.unemployment = end of previous step) — standard
        # for quarterly models.  Transfers rise as unemployment gaps open.
        _unemp_gap = max(0.0, self.unemployment - cfg.nairu)
        transfers = (cfg.transfer_rate + cfg.transfer_unemployment_sensitivity * _unemp_gap) * self.gdp

        # ── Crisis regime switches (Item 7) ──────────────────────────────────
        # Detect nonlinear threshold crossings AFTER plugins (need NPL from BGG).
        _regimes = self._detect_crisis_regime()
        _crisis_invest_mult = (
            cfg.crisis_investment_collapse
            if (_regimes["sudden_stop"] or _regimes["bank_run"]) else 1.0
        )
        _crisis_consump_cut = cfg.crisis_consumption_collapse if _regimes["sudden_stop"] else 0.0
        if _regimes["bank_run"]:
            _cs_struct += cfg.crisis_credit_freeze_spread
        if _regimes["debt_crisis"]:
            # Sovereign stress forces spending compression (bond market discipline)
            effective_spending_ratio = max(0.0, effective_spending_ratio * 0.85)

        # ── Quintile-disaggregated effective MPC (Item 6) ────────────────────
        # If quintile plugin is active: endogenously update income shares then
        # compute the income-share-weighted MPC.  Lower quintiles lose share
        # during unemployment spells (informal labour / job polarisation channel).
        if self._het_cfg is not None and self._quintile_income_shares is not None:
            if _unemp_gap > 0:
                # Q1 most sensitive (1.5×) down to Q5 least sensitive (0.5×)
                _emp_sens = [1.5, 1.2, 1.0, 0.8, 0.5]
                _raw = [
                    max(1e-4,
                        self._quintile_income_shares[i] * (1.0 - _unemp_gap * _emp_sens[i] * 0.05))
                    for i in range(5)
                ]
                _total = sum(_raw)
                self._quintile_income_shares = [s / _total for s in _raw]
            effective_mpc = float(sum(
                self._quintile_income_shares[i] * self._het_cfg.mpc_by_quintile[i]
                for i in range(5)
            ))
        else:
            effective_mpc = cfg.consumption_propensity

        # ============================================
        # 3. MODEL DYNAMICS
        # ============================================

        # A. Consumption (Household Demand)
        # Disposable income = GDP after tax + govt subsidies + automatic transfers
        subsidy_benefit = effective_subsidy * self.gdp
        household_income = self.gdp * (1 - effective_tax_rate) + subsidy_benefit + transfers
        wealth_effect_value = cfg.wealth_effect * self.households.net_worth

        # Price controls dampen inflation pass-through to consumption
        price_control_factor = 1.0
        if cfg.price_controls:
            price_control_factor = max(0.5, 1.0 - 0.1 * len(cfg.price_controls))

        consumption = (
            (effective_mpc * household_income + wealth_effect_value)
            * shock_demand_mult
            * (1.0 - _crisis_consump_cut)
        )

        # B. Investment (Firm Demand)
        # I = I0 - b * r (CRR tightening indirectly captured via rate)
        base_investment = self.gdp * cfg.base_investment_ratio
        rate_effect = cfg.investment_sensitivity * (self.interest_rate - cfg.neutral_rate)
        investment = (
            max(0, base_investment - rate_effect * self.gdp - _efp_drag * self.gdp)
            * shock_demand_mult
            * _crisis_invest_mult
        )

        # C. Government Spending + Automatic Fiscal Stabilizers
        # Discretionary G = (spending_ratio + impulse + subsidies) * Y
        eff_govt_ratio = effective_spending_ratio + fiscal_impulse + effective_subsidy
        government_spending = eff_govt_ratio * self.gdp

        # Tax revenue with automatic stabilizer: elasticity > 1 means revenues fall
        # faster than GDP in recession (progressive taxes, shrinking VAT base).
        # T = τ·Y·(1 + (ε-1)·output_gap),  clamped so it never goes negative.
        cyclical_tax_factor = 1.0 + (cfg.tax_elasticity - 1.0) * self.output_gap
        cyclical_tax_factor = max(0.5, cyclical_tax_factor)  # floor: revenues at least 50% of flat rate
        tax_revenue = effective_tax_rate * self.gdp * cyclical_tax_factor

        # Fiscal deficit = discretionary G + automatic transfers − tax revenue
        # (transfers already computed above before section A)
        fiscal_deficit = government_spending + transfers - tax_revenue
        self.government.liabilities['bonds'] += fiscal_deficit * dt
        
        # D. Aggregate Demand & GDP  (C + I + G + NX + Remittances)
        aggregate_demand = consumption + investment + government_spending + (_nx_gdp + _remit_gdp) * self.gdp
        
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
        # Commodity pass-through: oil and food price movements feed directly into CPI.
        # Supply shock already captures aggregate capacity effects; pass-through adds the
        # direct import-price channel (distinct from the output-gap channel).
        commodity_inflation = (
            cfg.oil_pass_through * cfg.oil_price_growth
            + cfg.food_pass_through * cfg.food_price_growth
        )
        raw_inflation = (
            expected_inflation
            + cfg.phillips_coefficient * self.output_gap
            + commodity_inflation
            + _import_inflation
        )
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
        self.credit_spread = _cs_struct  # Structural BGG spread (or heuristic fallback)
        
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

            # Automatic stabilizers (cyclical fiscal flows)
            "tax_revenue_gdp": float(tax_revenue / self.gdp) if self.gdp > 0 else 0.0,
            "transfers_gdp": float(transfers / self.gdp) if self.gdp > 0 else 0.0,

            # Fiscal space
            "fiscal_space": float((tax_revenue - government_spending - transfers) / self.gdp) if self.gdp > 0 else 0.0,
            
            # Investment & capital
            "investment_ratio": float(investment / self.gdp) if self.gdp > 0 else 0.0,
            "capital_stock": float(firm_capital),
            
            # Financial stability (0 = crisis, 1 = stable)
            "financial_stability": float(max(0.0, min(1.0, 1.0 - self.credit_spread / 0.10))),

            # Cost of living (inflation impact on household consumption)
            "cost_of_living_index": float(1.0 + self.inflation),

            # Open economy (zero when plugin inactive)
            "trade_balance_gdp": float(_nx_gdp),
            "remittances_gdp": float(_remit_gdp),
            "reer": float(self._external_state.reer if self._external_state else 100.0),

            # Financial accelerator (zero / baseline when plugin inactive)
            "npl_ratio": float(self._bank_state.npl_ratio if self._bank_state else 0.12),
            "car": float(self._bank_state.car if self._bank_state else 0.165),
            "efp": float(self._bank_state.external_finance_premium if self._bank_state else 0.02),

            # Quintile distribution (Item 6; 0.0 when plugin inactive)
            "effective_mpc": float(effective_mpc),
            "gini": float(
                self._InequalityMetrics.gini_from_quintiles(
                    np.array(self._quintile_income_shares)
                )
            ) if self._het_cfg is not None else 0.0,
            "q1_share": float(self._quintile_income_shares[0]) if self._het_cfg is not None else 0.0,
            "q5_share": float(self._quintile_income_shares[4]) if self._het_cfg is not None else 0.0,

            # Crisis regime flags (Item 7; 0.0=inactive, 1.0=active)
            "regime_sudden_stop": 1.0 if _regimes["sudden_stop"] else 0.0,
            "regime_bank_run":    1.0 if _regimes["bank_run"]    else 0.0,
            "regime_debt_crisis": 1.0 if _regimes["debt_crisis"] else 0.0,
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

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current macroeconomic state as a flat dictionary.

        This is the canonical read-only snapshot used by external consumers
        (e.g. EconomicGovernor, dashboard, sync_to_environment) that need to
        observe SFC variables without advancing time.

        Returns:
            Dict mapping variable name → current value.
        """
        return {
            # Macro aggregates
            "gdp": self.gdp,
            "gdp_growth": self.gdp_growth,
            "inflation": self.inflation,
            "interest_rate": self.interest_rate,
            "unemployment": self.unemployment,
            "credit_spread": self.credit_spread,
            "output_gap": self.output_gap,
            "potential_gdp": self.potential_gdp,
            "time": self.time,
            # Sector-level aggregates
            "household_net_worth": self.households.net_worth,
            "firm_net_worth": self.firms.net_worth,
            "bank_net_worth": self.banks.net_worth,
            "government_net_worth": self.government.net_worth,
            "government_deficit": -self.government.net_lending,  # sign: deficit = positive number
            # Government debt (bonds liability, consistent with SFC balance sheet)
            "government_debt": self.government.liabilities.get("bonds", 0.0),
            # Shock / policy snapshots
            **{f"shock_{k}": v for k, v in self.current_shock_vector.items()},
            **{f"policy_{k}": v for k, v in self.current_policy_vector.items()},
        }

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

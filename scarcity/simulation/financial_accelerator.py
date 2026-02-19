"""
Financial Accelerator Module for SFC Models.

Implements Bernanke-Gertler-Gilchrist (BGG) financial accelerator dynamics
within the SFC framework.  Extends the existing bank sector (sfc.py) with:

1. Non-Performing Loan (NPL) dynamics with endogenous default rates
2. Collateral/Loan-to-Value (LTV) constraints on credit
3. Bank capital adequacy (CAR) and Basel-style risk weights
4. Credit cycle amplification (financial accelerator mechanism)
5. Endogenous credit spread driven by bank health and NPLs

Builds on:
- SFCEconomy bank sector balance sheet (sfc.py)
- credit_spread heuristic (sfc.py) — replaced with structural model
- CRR and rate_cap policy instruments (sfc.py → SFCConfig)
- Scenario templates' CRR/rate_cap channels (scenario_templates.py)

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scarcity.simulation.sfc import SFCConfig, SFCEconomy, Sector, SectorType

logger = logging.getLogger("scarcity.simulation.financial_accelerator")


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class FinancialAcceleratorConfig:
    """
    Configuration for the financial accelerator mechanism.
    
    Parameter defaults calibrated to Kenya's banking sector
    (CBK Financial Stability Reports).
    """
    
    # --- NPL Dynamics ---
    base_npl_rate: float = 0.12         # Baseline NPL ratio (Kenya ~12%)
    npl_gdp_sensitivity: float = 0.8    # How much GDP contraction raises NPLs
    npl_rate_sensitivity: float = 0.3   # How much rate hikes raise NPLs
    npl_recovery_rate: float = 0.15     # Annual NPL recovery/write-off rate
    npl_max: float = 0.50              # Maximum NPL ratio
    npl_min: float = 0.02             # Minimum NPL ratio
    
    # --- Collateral / LTV ---
    base_ltv: float = 0.70              # Baseline Loan-to-Value ratio
    ltv_procyclical: float = 0.1        # LTV increases in booms
    collateral_haircut_crisis: float = 0.3  # Collateral value drop in crisis
    
    # --- Bank Capital ---
    base_car: float = 0.165            # Capital Adequacy Ratio (Kenya ~16.5%)
    car_min_regulatory: float = 0.145   # CBK minimum CAR (14.5%)
    car_target: float = 0.18           # Bank target CAR (buffer above minimum)
    risk_weight_performing: float = 0.75  # Risk weight on performing loans
    risk_weight_npl: float = 1.50       # Risk weight on NPLs (150%)
    
    # --- Credit Supply ---
    credit_growth_base: float = 0.08    # Baseline annual credit growth
    credit_car_sensitivity: float = 2.0  # How much low CAR constrains credit
    credit_npl_sensitivity: float = 1.5  # How much high NPLs constrain credit
    credit_max_growth: float = 0.25     # Maximum annual credit growth
    credit_min_growth: float = -0.10    # Maximum annual credit contraction
    
    # --- Financial Accelerator ---
    accelerator_strength: float = 0.3    # BGG accelerator coefficient
    net_worth_sensitivity: float = 0.5   # Sensitivity of borrowing to net worth
    external_finance_premium_base: float = 0.02  # Base EFP
    efp_npl_sensitivity: float = 0.5     # How NPLs widen EFP
    efp_car_sensitivity: float = 0.3     # How low CAR widens EFP
    
    # --- Provisioning ---
    provision_rate_performing: float = 0.01   # General provision on performing
    provision_rate_watch: float = 0.05        # Provision on watch-listed
    provision_rate_substandard: float = 0.25  # Provision on substandard
    provision_rate_doubtful: float = 0.50     # Provision on doubtful
    provision_rate_loss: float = 1.00         # Provision on loss
    
    # --- Interbank ---
    interbank_rate_spread: float = 0.005  # Spread above policy rate


# =========================================================================
# Bank State
# =========================================================================

@dataclass
class BankState:
    """
    Detailed banking sector state.
    
    Extends the simple Sector balance sheet in sfc.py with
    banking-specific variables.
    """
    # Assets
    performing_loans: float = 60.0
    non_performing_loans: float = 8.0
    government_securities: float = 20.0
    reserves: float = 10.0    # Cash + CBK deposits
    other_assets: float = 5.0
    
    # Liabilities
    deposits: float = 75.0
    borrowings: float = 10.0   # Interbank + CBK borrowing
    other_liabilities: float = 5.0
    
    # Capital
    tier1_capital: float = 10.0
    tier2_capital: float = 3.0
    
    # Flow variables
    interest_income: float = 0.0
    interest_expense: float = 0.0
    provision_expense: float = 0.0
    net_income: float = 0.0
    
    # Ratios (computed)
    npl_ratio: float = 0.12
    car: float = 0.165
    credit_spread: float = 0.02
    external_finance_premium: float = 0.02
    
    @property
    def total_loans(self) -> float:
        return self.performing_loans + self.non_performing_loans
    
    @property
    def total_assets(self) -> float:
        return (self.performing_loans + self.non_performing_loans +
                self.government_securities + self.reserves + self.other_assets)
    
    @property
    def total_liabilities(self) -> float:
        return self.deposits + self.borrowings + self.other_liabilities
    
    @property
    def total_capital(self) -> float:
        return self.tier1_capital + self.tier2_capital
    
    @property
    def risk_weighted_assets(self) -> float:
        """Risk-weighted assets for CAR computation."""
        rwa = (self.performing_loans * 0.75 +
               self.non_performing_loans * 1.50 +
               self.government_securities * 0.0 +  # Zero risk weight
               self.other_assets * 1.0)
        return max(rwa, 1.0)
    
    @property 
    def leverage_ratio(self) -> float:
        return self.total_capital / max(self.total_assets, 1.0)


# =========================================================================
# Financial Accelerator Engine
# =========================================================================

class FinancialAccelerator:
    """
    Financial accelerator mechanism for the SFC economy.
    
    Implements the BGG financial accelerator where:
    1. Firm net worth determines borrowing capacity (collateral constraint)
    2. External finance premium (EFP) depends on leverage
    3. Bank health (NPLs, CAR) amplifies credit conditions
    4. Credit cycles endogenously amplify real economy fluctuations
    
    Wraps SFCEconomy and adds financial sector dynamics that feed back
    into credit_spread, investment, and GDP.
    """
    
    def __init__(
        self,
        sfc_config: Optional[SFCConfig] = None,
        fin_config: Optional[FinancialAcceleratorConfig] = None,
    ):
        self.sfc_config = sfc_config or SFCConfig()
        self.fin_cfg = fin_config or FinancialAcceleratorConfig()
        
        # Base SFC economy
        self.economy = SFCEconomy(self.sfc_config)
        
        # Bank state
        self.bank = BankState()
        
        # Firm net worth (collateral for BGG)
        self.firm_net_worth = 0.0
        
        # Financial trajectory
        self.financial_trajectory: List[Dict[str, float]] = []
        self.time = 0
    
    def initialize(self, gdp: float = 100.0):
        """Initialize the financial accelerator."""
        self.economy.initialize(gdp)
        
        # Initialize bank balance sheet relative to GDP
        self.bank.performing_loans = gdp * 0.35  # dom_credit_pvt ≈ 35% GDP (Kenya)
        self.bank.non_performing_loans = self.bank.performing_loans * self.fin_cfg.base_npl_rate
        self.bank.deposits = gdp * 0.40           # Broad money ≈ 40% GDP
        self.bank.government_securities = gdp * 0.15
        self.bank.reserves = gdp * self.sfc_config.crr  # CRR × deposits
        self.bank.tier1_capital = self.bank.risk_weighted_assets * self.fin_cfg.base_car * 0.8
        self.bank.tier2_capital = self.bank.risk_weighted_assets * self.fin_cfg.base_car * 0.2
        
        # Firm net worth
        self.firm_net_worth = self.economy.firms.net_worth
        
        self._record_financial_state()
    
    def step(self) -> Dict[str, Any]:
        """
        Step the financial accelerator:
        1. Update NPL dynamics
        2. Compute bank capital adequacy
        3. Determine credit supply (pro-cyclical lending)
        4. Compute external finance premium (BGG)
        5. Feed credit conditions into SFC economy
        6. Run base SFC step
        7. Record financial state
        """
        self.time += 1
        cfg = self.fin_cfg
        
        # ========================
        # 1. NPL Dynamics
        # ========================
        gdp_growth = self.economy.gdp_growth
        rate = self.economy.interest_rate
        
        # NPL inflow: driven by GDP contraction and high rates
        npl_inflow_rate = (
            cfg.base_npl_rate
            - cfg.npl_gdp_sensitivity * gdp_growth  # Contraction raises NPLs
            + cfg.npl_rate_sensitivity * max(0, rate - self.sfc_config.neutral_rate)
        )
        npl_inflow_rate = max(0, npl_inflow_rate)
        
        # New NPLs from performing loans
        new_npls = self.bank.performing_loans * npl_inflow_rate * 0.1  # Quarterly flow
        
        # NPL recovery/write-off
        recovered = self.bank.non_performing_loans * cfg.npl_recovery_rate
        
        # Update
        self.bank.performing_loans -= new_npls
        self.bank.performing_loans += recovered * 0.3  # 30% of recovery returns to performing
        self.bank.non_performing_loans += new_npls - recovered
        self.bank.non_performing_loans = max(0, self.bank.non_performing_loans)
        
        # NPL ratio
        total_loans = max(self.bank.total_loans, 1.0)
        self.bank.npl_ratio = self.bank.non_performing_loans / total_loans
        self.bank.npl_ratio = np.clip(self.bank.npl_ratio, cfg.npl_min, cfg.npl_max)
        
        # ========================
        # 2. Bank Capital Adequacy
        # ========================
        # Provisions eat into capital
        provision = (
            self.bank.performing_loans * cfg.provision_rate_performing +
            self.bank.non_performing_loans * 0.5 * cfg.provision_rate_substandard +
            self.bank.non_performing_loans * 0.5 * cfg.provision_rate_doubtful
        )
        self.bank.provision_expense = provision * 0.1  # Quarterly provisioning
        
        # Net interest income
        lending_rate = rate + self.bank.credit_spread
        self.bank.interest_income = self.bank.performing_loans * lending_rate * 0.25
        self.bank.interest_expense = self.bank.deposits * (rate - 0.02) * 0.25  # Deposit rate
        
        # Net income
        self.bank.net_income = (
            self.bank.interest_income 
            - self.bank.interest_expense 
            - self.bank.provision_expense
        )
        
        # Retained earnings add to capital
        if self.bank.net_income > 0:
            self.bank.tier1_capital += self.bank.net_income * 0.7  # 70% retained
        else:
            self.bank.tier1_capital += self.bank.net_income  # Losses deplete capital
        
        # CAR
        rwa = max(self.bank.risk_weighted_assets, 1.0)
        self.bank.car = self.bank.total_capital / rwa
        
        # ========================
        # 3. Credit Supply
        # ========================
        # Credit growth depends on CAR buffer and NPL level
        car_buffer = self.bank.car - cfg.car_min_regulatory
        npl_gap = self.bank.npl_ratio - cfg.base_npl_rate
        
        credit_growth = (
            cfg.credit_growth_base
            + cfg.credit_car_sensitivity * car_buffer  # High CAR → more lending
            - cfg.credit_npl_sensitivity * max(0, npl_gap)  # High NPLs → less lending
        )
        credit_growth = np.clip(credit_growth, cfg.credit_min_growth, cfg.credit_max_growth)
        
        # CRR effect: higher CRR reduces loanable funds
        crr_effect = max(0, self.sfc_config.crr - 0.0525) / 0.0525
        credit_growth -= crr_effect * 0.05
        
        # Apply credit growth
        self.bank.performing_loans *= (1 + credit_growth * 0.25)  # Quarterly
        
        # ========================
        # 4. External Finance Premium (BGG)
        # ========================
        # EFP = base + f(leverage, NPLs, CAR)
        self.firm_net_worth = self.economy.firms.net_worth
        firm_leverage = self.bank.total_loans / max(self.firm_net_worth, 1.0)
        
        efp = (
            cfg.external_finance_premium_base
            + cfg.accelerator_strength * max(0, firm_leverage - 1.0)
            + cfg.efp_npl_sensitivity * max(0, npl_gap)
            + cfg.efp_car_sensitivity * max(0, cfg.car_min_regulatory - self.bank.car)
        )
        self.bank.external_finance_premium = max(0, efp)
        
        # ========================
        # 5. Credit Spread (structural)
        # ========================
        # Replace the heuristic credit_spread in base SFC with structural model
        self.bank.credit_spread = (
            0.02  # Base spread
            + 0.5 * max(0, self.bank.npl_ratio - cfg.base_npl_rate)
            + 0.3 * max(0, cfg.car_min_regulatory - self.bank.car)
            + self.bank.external_finance_premium * 0.5
        )
        
        # Feed credit conditions into SFC economy
        self.economy.credit_spread = self.bank.credit_spread
        
        # Financial accelerator: EFP reduces effective investment
        # This creates the amplification channel
        investment_drag = cfg.accelerator_strength * self.bank.external_finance_premium
        
        # Apply as a demand shock modifier  
        if self.economy.config.shock_vectors is None:
            self.economy.config.shock_vectors = {}
        
        # Temporary demand shock from financial conditions
        shock_key = "demand_shock"
        if shock_key not in self.economy.config.shock_vectors:
            self.economy.config.shock_vectors[shock_key] = np.zeros(self.sfc_config.steps)
        
        if self.time < len(self.economy.config.shock_vectors.get(shock_key, [])):
            self.economy.config.shock_vectors[shock_key][self.time] -= investment_drag
        
        # ========================
        # 6. Base SFC Step
        # ========================
        agg_frame = self.economy.step()
        
        # Update deposits from household savings
        self.bank.deposits = self.economy.households.assets.get('deposits', self.bank.deposits)
        
        # ========================
        # 7. Record
        # ========================
        self._record_financial_state()
        
        # Augmented frame
        augmented = dict(agg_frame) if isinstance(agg_frame, dict) else {}
        augmented["financial"] = self._current_financial_state()
        
        return augmented
    
    def run(self, steps: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Run financial accelerator simulation."""
        for _ in range(steps):
            self.step()
        return self.economy.trajectory, self.financial_trajectory
    
    def stress_test(
        self, 
        npl_shock: float = 0.0,
        rate_shock: float = 0.0,
        deposit_outflow: float = 0.0,
    ) -> Dict[str, float]:
        """
        Bank stress test: apply shocks and compute post-shock CAR.
        
        Args:
            npl_shock: Additional NPL ratio (e.g., 0.10 for +10pp)
            rate_shock: Interest rate shock
            deposit_outflow: Fraction of deposits withdrawn
            
        Returns:
            Post-shock bank health metrics.
        """
        # Clone bank state
        post_npl = self.bank.npl_ratio + npl_shock
        post_npl = np.clip(post_npl, self.fin_cfg.npl_min, self.fin_cfg.npl_max)
        
        # Increased provisions
        additional_npls = self.bank.total_loans * npl_shock
        additional_provision = additional_npls * 0.50  # 50% provision rate
        
        # Capital impact
        post_capital = self.bank.total_capital - additional_provision
        
        # Rate shock impacts NII
        nii_impact = -self.bank.total_loans * rate_shock * 0.02  # Duration effect
        post_capital += nii_impact
        
        # Deposit outflow
        post_deposits = self.bank.deposits * (1 - deposit_outflow)
        liquidity_stress = max(0, self.bank.reserves - post_deposits * self.sfc_config.crr)
        
        # Post-shock CAR
        post_rwa = (
            self.bank.performing_loans * (1 - npl_shock) * self.fin_cfg.risk_weight_performing +
            (self.bank.non_performing_loans + additional_npls) * self.fin_cfg.risk_weight_npl +
            self.bank.government_securities * 0.0 +
            self.bank.other_assets * 1.0
        )
        post_car = post_capital / max(post_rwa, 1.0)
        
        return {
            "post_npl_ratio": float(post_npl),
            "post_car": float(post_car),
            "capital_shortfall": float(max(0, self.fin_cfg.car_min_regulatory - post_car) * post_rwa),
            "breaches_minimum": post_car < self.fin_cfg.car_min_regulatory,
            "liquidity_surplus": float(liquidity_stress),
            "post_capital": float(post_capital),
        }
    
    def _current_financial_state(self) -> Dict[str, float]:
        """Current financial sector state."""
        return {
            "npl_ratio": self.bank.npl_ratio,
            "car": self.bank.car,
            "credit_spread": self.bank.credit_spread,
            "efp": self.bank.external_finance_premium,
            "performing_loans": self.bank.performing_loans,
            "non_performing_loans": self.bank.non_performing_loans,
            "total_capital": self.bank.total_capital,
            "deposits": self.bank.deposits,
            "net_income": self.bank.net_income,
            "leverage_ratio": self.bank.leverage_ratio,
            "credit_to_gdp": self.bank.total_loans / max(self.economy.gdp, 1.0),
        }
    
    def _record_financial_state(self):
        """Record financial state."""
        state = self._current_financial_state()
        state["t"] = self.time
        self.financial_trajectory.append(state)

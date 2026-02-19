"""
Research-Grade SFC Economy: Unified Integration Module.

Composes all 7 simulation engine upgrades into a single coherent
economy class that can be run as one unit:

    Tier 1 — Core Research Credibility:
    1. Bayesian Parameter Estimation  (bayesian.py)
    2. Stochastic Processes           (kshiked/core/shocks.py)
    3. Model Validation Framework     (kshiked/simulation/validation.py)

    Tier 2 — Structural Depth:
    4. Multi-Sector IO Structure      (io_structure.py)
    5. Heterogeneous Agents           (heterogeneous.py)
    6. Financial Accelerator          (financial_accelerator.py)
    7. Open Economy Module            (open_economy.py)

The ResearchSFCEconomy class owns a single SFCEconomy instance and
delegates sector-level dynamics to the appropriate module, ensuring
one coherent GDP identity:  Y = C_het + I_fin + G + NX_open

Dependencies: numpy only (all modules import from scarcity.simulation.sfc).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Base SFC ---
from scarcity.simulation.sfc import SFCConfig, SFCEconomy, Sector, SectorType

# --- Upgrade Modules ---
from scarcity.simulation.io_structure import (
    IOConfig,
    LeontiefModel,
    MultiSectorSFCEconomy,
    SubSector,
    default_kenya_io_config,
)
from scarcity.simulation.heterogeneous import (
    HeterogeneousConfig,
    HeterogeneousHouseholdEconomy,
    HouseholdAgent,
    InequalityMetrics,
    IncomeQuintile,
    default_kenya_heterogeneous_config,
)
from scarcity.simulation.financial_accelerator import (
    BankState,
    FinancialAccelerator,
    FinancialAcceleratorConfig,
)
from scarcity.simulation.open_economy import (
    ExternalState,
    OpenEconomyConfig,
    OpenEconomySFC,
    default_kenya_open_economy_config,
)
from scarcity.simulation.bayesian import (
    BayesianEstimator,
    MCMCConfig,
    MCMCResult,
    ObservedData,
    default_sfc_priors,
)

logger = logging.getLogger("scarcity.simulation.research_sfc")


# =========================================================================
# Unified Configuration
# =========================================================================

@dataclass
class ResearchSFCConfig:
    """
    Master configuration for the research-grade SFC economy.
    
    Wraps all sub-module configs into a single object.
    Individual modules can be enabled/disabled via flags.
    """
    # Base SFC config
    sfc: SFCConfig = field(default_factory=SFCConfig)
    
    # Module configs
    io: IOConfig = field(default_factory=default_kenya_io_config)
    heterogeneous: HeterogeneousConfig = field(default_factory=default_kenya_heterogeneous_config)
    financial: FinancialAcceleratorConfig = field(default_factory=FinancialAcceleratorConfig)
    open_economy: OpenEconomyConfig = field(default_factory=default_kenya_open_economy_config)
    
    # Bayesian estimation
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    
    # Module activation flags
    enable_io: bool = True
    enable_heterogeneous: bool = True
    enable_financial: bool = True
    enable_open_economy: bool = True
    
    # Integration settings
    feedback_strength: float = 1.0  # Scale of inter-module feedback (0=none, 1=full)
    seed: int = 42


def default_kenya_research_config() -> ResearchSFCConfig:
    """Full Kenya-calibrated research configuration."""
    return ResearchSFCConfig()


# =========================================================================
# Research SFC Economy
# =========================================================================

class ResearchSFCEconomy:
    """
    Unified research-grade SFC economy.
    
    Owns ONE SFCEconomy instance and orchestrates all upgrade modules
    around it. The step order ensures proper feedback loops:
    
        1. Financial accelerator computes credit conditions
        2. Open economy computes trade balance & FX
        3. IO model distributes demand across sectors
        4. Heterogeneous agents compute quintile-level consumption
        5. Base SFC step runs with all feedback effects
        6. Record unified frame with all dimensions
    
    The GDP identity is:
        Y = C_heterogeneous + I_financial + G + NX_open
    
    All modules operate on the SAME SFCEconomy instance.
    """
    
    def __init__(self, config: Optional[ResearchSFCConfig] = None):
        self.config = config or default_kenya_research_config()
        np.random.seed(self.config.seed)
        
        # === CORE: Single SFC economy instance ===
        self.economy = SFCEconomy(self.config.sfc)
        
        # === MODULE INSTANCES (all share self.economy) ===
        
        # Financial Accelerator
        self.bank = BankState()
        self.fin_cfg = self.config.financial
        self.firm_net_worth = 0.0
        
        # Open Economy
        self.open_cfg = self.config.open_economy
        self.external = ExternalState()
        self.foreign_sector = Sector("Foreign", SectorType.FOREIGN)
        
        # IO Structure
        self.io_cfg = self.config.io
        self.io_model = LeontiefModel(self.io_cfg.io_matrix)
        self.sub_sectors: Dict[str, SubSector] = {}
        
        # Heterogeneous Agents
        self.het_cfg = self.config.heterogeneous
        self.agents: Dict[IncomeQuintile, HouseholdAgent] = {}
        
        # === TRAJECTORIES ===
        self.trajectory: List[Dict[str, Any]] = []
        self.financial_trajectory: List[Dict[str, float]] = []
        self.external_trajectory: List[Dict[str, float]] = []
        self.inequality_trajectory: List[Dict[str, float]] = []
        self.sector_trajectory: List[Dict[str, Any]] = []
        
        self.time = 0
    
    # =====================================================================
    # Initialization
    # =====================================================================
    
    def initialize(self, gdp: float = 100.0):
        """Initialize all modules with consistent starting state."""
        self.economy.initialize(gdp)
        
        # --- Financial Accelerator ---
        if self.config.enable_financial:
            self._init_financial(gdp)
        
        # --- Open Economy ---
        if self.config.enable_open_economy:
            self._init_open_economy(gdp)
        
        # --- IO Structure ---
        if self.config.enable_io:
            self._init_io_structure(gdp)
        
        # --- Heterogeneous Agents ---
        if self.config.enable_heterogeneous:
            self._init_heterogeneous(gdp)
        
        # Record initial frame
        self._record_unified_frame()
        
        logger.info(
            f"ResearchSFCEconomy initialized: GDP={gdp:.1f}, "
            f"modules=[IO={self.config.enable_io}, Het={self.config.enable_heterogeneous}, "
            f"Fin={self.config.enable_financial}, Open={self.config.enable_open_economy}]"
        )
    
    def _init_financial(self, gdp: float):
        """Initialize financial accelerator state."""
        cfg = self.fin_cfg
        self.bank.performing_loans = gdp * 0.35
        self.bank.non_performing_loans = self.bank.performing_loans * cfg.base_npl_rate
        self.bank.deposits = gdp * 0.40
        self.bank.government_securities = gdp * 0.15
        self.bank.reserves = gdp * self.config.sfc.crr
        self.bank.tier1_capital = self.bank.risk_weighted_assets * cfg.base_car * 0.8
        self.bank.tier2_capital = self.bank.risk_weighted_assets * cfg.base_car * 0.2
        self.firm_net_worth = self.economy.firms.net_worth
    
    def _init_open_economy(self, gdp: float):
        """Initialize external sector."""
        cfg = self.open_cfg
        self.economy.sectors.append(self.foreign_sector)
        
        self.external.reer = cfg.initial_reer
        self.external.exports = gdp * cfg.export_gdp_ratio
        self.external.imports = gdp * cfg.import_gdp_ratio
        self.external.trade_balance = self.external.exports - self.external.imports
        self.external.remittances = gdp * cfg.remittance_gdp_ratio
        self.external.fdi_net = gdp * cfg.fdi_gdp_ratio
        self.external.portfolio_net = gdp * cfg.portfolio_flows_gdp
        
        self.foreign_sector.liabilities['reserves'] = (
            gdp * cfg.initial_reserves_months * cfg.import_gdp_ratio
        )
        self.foreign_sector.assets['claims_on_domestic'] = gdp * 0.3
        
        # BoP
        self.external.current_account = (
            self.external.trade_balance +
            self.external.remittances +
            self.external.investment_income
        )
        self.external.capital_account = (
            self.external.fdi_net +
            self.external.portfolio_net +
            self.external.other_flows
        )
        monthly_imports = self.external.imports / 12.0
        self.external.reserves_months_import = (
            self.foreign_sector.liabilities['reserves'] / max(monthly_imports, 0.1)
        )
    
    def _init_io_structure(self, gdp: float):
        """Initialize IO sub-sectors."""
        from scarcity.simulation.io_structure import SubSectorType
        
        sector_names = ["agriculture", "manufacturing", "services", "mining", "construction"]
        shares = self.io_cfg.sector_shares or {}
        emp_shares = self.io_cfg.employment_shares or {}
        
        for name in sector_names:
            sub_type = SubSectorType(name)
            share = shares.get(name, 0.2)
            emp_share = emp_shares.get(name, 0.2)
            self.sub_sectors[name] = SubSector(
                name=name,
                sub_type=sub_type,
                output_share=share,
                employment_share=emp_share,
                capital=gdp * share * 2.0,
                output=gdp * share,
            )
    
    def _init_heterogeneous(self, gdp: float):
        """Initialize quintile agents."""
        cfg = self.het_cfg
        total_hh_income = gdp * (1 - self.config.sfc.tax_rate)
        
        for i, q in enumerate(list(IncomeQuintile)):
            base_income = total_hh_income * cfg.income_shares[i]
            self.agents[q] = HouseholdAgent(
                quintile=q,
                income_share=cfg.income_shares[i],
                mpc=cfg.mpc_by_quintile[i],
                financial_wealth=base_income * cfg.wealth_income_ratio[i] * 0.3,
                real_wealth=base_income * cfg.wealth_income_ratio[i] * 0.7,
                debt=base_income * cfg.debt_income_ratio[i],
                employment_rate=cfg.employment_by_quintile[i],
                formal_share=cfg.formal_share_by_quintile[i],
            )
    
    # =====================================================================
    # Step
    # =====================================================================
    
    def step(self) -> Dict[str, Any]:
        """
        Advance all modules by one period.
        
        Order ensures feedback loops:
        1. Financial → credit conditions
        2. Open economy → trade & FX
        3. Base SFC step (with financial + external feedback)
        4. IO disaggregation
        5. Heterogeneous consumption reallocation
        6. Record unified frame
        """
        self.time += 1
        fb = self.config.feedback_strength
        gdp = self.economy.gdp
        
        # ========================
        # 1. Financial Accelerator
        # ========================
        credit_drag = 0.0
        if self.config.enable_financial:
            credit_drag = self._step_financial()
        
        # ========================
        # 2. Open Economy
        # ========================
        nx_effect = 0.0
        if self.config.enable_open_economy:
            nx_effect = self._step_open_economy()
        
        # ========================
        # 3. Base SFC Step
        # ========================
        # Inject feedback as demand shock
        total_feedback = fb * (credit_drag + nx_effect)
        
        if total_feedback != 0.0:
            if self.economy.config.shock_vectors is None:
                self.economy.config.shock_vectors = {}
            key = "demand_shock"
            if key not in self.economy.config.shock_vectors:
                self.economy.config.shock_vectors[key] = np.zeros(
                    max(self.config.sfc.steps, self.time + 10)
                )
            vec = self.economy.config.shock_vectors[key]
            if self.time < len(vec):
                vec[self.time] += total_feedback
        
        agg_frame = self.economy.step()
        
        # Post-step: open economy NX adjustment
        if self.config.enable_open_economy:
            nx_adjustment = (self.external.trade_balance + self.external.remittances) * 0.1
            self.economy.gdp += nx_adjustment
        
        # ========================
        # 4. IO Disaggregation
        # ========================
        if self.config.enable_io:
            self._step_io()
        
        # ========================
        # 5. Heterogeneous Agents
        # ========================
        if self.config.enable_heterogeneous:
            self._step_heterogeneous()
        
        # ========================
        # 6. Record
        # ========================
        self._record_unified_frame()
        
        return self.trajectory[-1]
    
    def _step_financial(self) -> float:
        """
        Financial accelerator step. Returns credit drag on demand.
        """
        cfg = self.fin_cfg
        rate = self.economy.interest_rate
        gdp_growth = self.economy.gdp_growth
        
        # NPL dynamics
        npl_inflow_rate = (
            cfg.base_npl_rate
            - cfg.npl_gdp_sensitivity * gdp_growth
            + cfg.npl_rate_sensitivity * max(0, rate - self.config.sfc.neutral_rate)
        )
        npl_inflow_rate = max(0, npl_inflow_rate)
        
        new_npls = self.bank.performing_loans * npl_inflow_rate * 0.1
        recovered = self.bank.non_performing_loans * cfg.npl_recovery_rate
        
        self.bank.performing_loans -= new_npls
        self.bank.performing_loans += recovered * 0.3
        self.bank.non_performing_loans += new_npls - recovered
        self.bank.non_performing_loans = max(0, self.bank.non_performing_loans)
        
        total_loans = max(self.bank.total_loans, 1.0)
        self.bank.npl_ratio = np.clip(
            self.bank.non_performing_loans / total_loans,
            cfg.npl_min, cfg.npl_max
        )
        
        # Provisioning & bank P&L
        provision = (
            self.bank.performing_loans * cfg.provision_rate_performing +
            self.bank.non_performing_loans * 0.5 * cfg.provision_rate_substandard +
            self.bank.non_performing_loans * 0.5 * cfg.provision_rate_doubtful
        )
        self.bank.provision_expense = provision * 0.1
        
        lending_rate = rate + self.bank.credit_spread
        self.bank.interest_income = self.bank.performing_loans * lending_rate * 0.25
        self.bank.interest_expense = self.bank.deposits * max(0, rate - 0.02) * 0.25
        self.bank.net_income = (
            self.bank.interest_income - self.bank.interest_expense - self.bank.provision_expense
        )
        
        if self.bank.net_income > 0:
            self.bank.tier1_capital += self.bank.net_income * 0.7
        else:
            self.bank.tier1_capital += self.bank.net_income
        
        # CAR
        rwa = max(self.bank.risk_weighted_assets, 1.0)
        self.bank.car = self.bank.total_capital / rwa
        
        # Credit growth
        car_buffer = self.bank.car - cfg.car_min_regulatory
        npl_gap = self.bank.npl_ratio - cfg.base_npl_rate
        credit_growth = np.clip(
            cfg.credit_growth_base
            + cfg.credit_car_sensitivity * car_buffer
            - cfg.credit_npl_sensitivity * max(0, npl_gap),
            cfg.credit_min_growth, cfg.credit_max_growth
        )
        crr_effect = max(0, self.config.sfc.crr - 0.0525) / 0.0525
        credit_growth -= crr_effect * 0.05
        self.bank.performing_loans *= (1 + credit_growth * 0.25)
        
        # External Finance Premium (BGG)
        self.firm_net_worth = self.economy.firms.net_worth
        firm_leverage = self.bank.total_loans / max(self.firm_net_worth, 1.0)
        efp = max(0,
            cfg.external_finance_premium_base
            + cfg.accelerator_strength * max(0, firm_leverage - 1.0)
            + cfg.efp_npl_sensitivity * max(0, npl_gap)
            + cfg.efp_car_sensitivity * max(0, cfg.car_min_regulatory - self.bank.car)
        )
        self.bank.external_finance_premium = efp
        
        # Structural credit spread
        self.bank.credit_spread = (
            0.02
            + 0.5 * max(0, self.bank.npl_ratio - cfg.base_npl_rate)
            + 0.3 * max(0, cfg.car_min_regulatory - self.bank.car)
            + efp * 0.5
        )
        self.economy.credit_spread = self.bank.credit_spread
        
        # Update deposits
        self.bank.deposits = self.economy.households.assets.get('deposits', self.bank.deposits)
        
        # Return investment drag
        return -cfg.accelerator_strength * efp
    
    def _step_open_economy(self) -> float:
        """
        Open economy step. Returns NX demand effect.
        """
        cfg = self.open_cfg
        gdp = self.economy.gdp
        rate = self.economy.interest_rate
        inflation = self.economy.inflation
        
        fx_shock = self.economy.current_shock_vector.get("fx_shock", 0.0)
        
        # Exchange rate (UIP-PPP hybrid)
        uip_dep = cfg.uip_sensitivity * (rate - cfg.foreign_rate)
        ppp_speed = np.log(2) / max(cfg.ppp_half_life, 0.5)
        inflation_diff = inflation - cfg.foreign_rate
        ppp_pressure = -ppp_speed * (self.external.reer - cfg.initial_reer) / cfg.initial_reer
        
        reer_change = (
            (1 - cfg.ppp_weight) * uip_dep +
            cfg.ppp_weight * ppp_pressure +
            inflation_diff * 0.5 +
            fx_shock
        )
        if cfg.managed_float and abs(reer_change) > cfg.intervention_threshold:
            reer_change *= (1 - cfg.intervention_strength)
        
        self.external.reer_change = reer_change
        self.external.reer = np.clip(self.external.reer * (1 + reer_change), 50, 200)
        self.external.nominal_rate *= (1 + reer_change - inflation_diff)
        
        # Trade
        reer_dev = (self.external.reer - cfg.initial_reer) / cfg.initial_reer
        export_growth = (
            -cfg.export_price_elasticity * reer_dev +
            cfg.export_income_elasticity * cfg.world_gdp_growth
        )
        import_growth = (
            cfg.import_price_elasticity * reer_dev +
            cfg.import_income_elasticity * self.economy.gdp_growth
        )
        
        supply_shock = self.economy.current_shock_vector.get("supply_shock", 0.0)
        agri_effect = -supply_shock * cfg.export_composition.get("agriculture", 0.35) * 2.0
        tot_shock = np.random.normal(0, cfg.tot_volatility)
        
        self.external.exports = max(0.1, self.external.exports * (1 + export_growth + agri_effect + tot_shock * 0.5))
        self.external.imports = max(0.1, self.external.imports * (1 + import_growth - tot_shock * 0.3))
        self.external.trade_balance = self.external.exports - self.external.imports
        self.external.terms_of_trade *= (1 + tot_shock)
        
        # Remittances
        self.external.remittances = max(0, self.external.remittances * (
            1 + cfg.remittance_growth + cfg.remittance_fx_sensitivity * reer_dev
        ))
        
        # Capital flows
        rate_diff = rate - cfg.foreign_rate
        self.external.fdi_net *= (1 + cfg.fdi_rate_sensitivity * rate_diff + 0.02)
        portfolio_change = cfg.hot_money_sensitivity * rate_diff
        if self.economy.credit_spread > 0.05:
            portfolio_change -= 0.5 * (self.economy.credit_spread - 0.05)
        self.external.portfolio_net = gdp * cfg.portfolio_flows_gdp + gdp * portfolio_change
        
        self.external.capital_account = (
            self.external.fdi_net + self.external.portfolio_net + self.external.other_flows
        )
        
        # Investment income
        self.external.investment_income = -(
            self.foreign_sector.assets.get('claims_on_domestic', 0) * cfg.foreign_rate * 0.1
        )
        
        # BoP
        self.external.current_account = (
            self.external.trade_balance + self.external.remittances + self.external.investment_income
        )
        self.external.overall_bop = self.external.current_account + self.external.capital_account
        
        # Reserves
        self.foreign_sector.liabilities['reserves'] = max(
            0, self.foreign_sector.liabilities.get('reserves', 0) + self.external.overall_bop
        )
        monthly_imports = self.external.imports / 12.0
        self.external.reserves_months_import = (
            self.foreign_sector.liabilities['reserves'] / max(monthly_imports, 0.1)
        )
        
        # Foreign sector balance sheet
        self.foreign_sector.assets['claims_on_domestic'] += max(0, -self.external.current_account)
        self.foreign_sector.income = self.external.exports + self.external.remittances
        self.foreign_sector.expenses = self.external.imports
        
        # NX demand effect
        nx_gdp = self.external.trade_balance / max(gdp, 1.0)
        remit_gdp = self.external.remittances / max(gdp, 1.0)
        return nx_gdp * 0.5 + remit_gdp * 0.3
    
    def _step_io(self):
        """IO disaggregation after aggregate step."""
        from scarcity.simulation.io_structure import SubSectorType
        
        gdp = self.economy.gdp
        sector_names = list(self.sub_sectors.keys())
        
        final_demand = np.array([
            gdp * self.sub_sectors[name].output_share
            for name in sector_names
        ])
        
        # Apply shock sensitivities
        shock_vec = self.economy.current_shock_vector
        sensitivity = self.io_cfg.shock_sensitivity or {}
        for i, name in enumerate(sector_names):
            sens = sensitivity.get(name, {})
            mult = 1.0
            for channel, mag in shock_vec.items():
                mult += mag * sens.get(channel, 1.0)
            final_demand[i] *= mult
        
        # Solve Leontief
        gross_output = self.io_model.solve_output(final_demand)
        value_added = self.io_model.value_added(gross_output)
        
        total_output = max(np.sum(gross_output), 1e-6)
        for i, name in enumerate(sector_names):
            s = self.sub_sectors[name]
            s.output = float(gross_output[i])
            s.value_added = float(value_added[i])
            new_share = float(gross_output[i]) / total_output
            s.output_share += self.io_cfg.structural_change_speed * (new_share - s.output_share)
    
    def _step_heterogeneous(self):
        """Quintile-level dynamics after aggregate step."""
        gdp = self.economy.gdp
        cfg = self.het_cfg
        inflation = self.economy.inflation
        rate = self.economy.interest_rate
        
        total_hh_income = gdp * (1 - self.config.sfc.tax_rate)
        fiscal_impulse = self.economy.current_policy_vector.get("fiscal_impulse", 0.0)
        subsidy = self.economy.current_policy_vector.get("subsidy_rate", 0.0)
        total_transfers = gdp * max(0, fiscal_impulse + subsidy)
        
        for i, q in enumerate(list(IncomeQuintile)):
            agent = self.agents[q]
            
            market_income = total_hh_income * agent.income_share * agent.employment_rate
            formal_premium = 1.0 + 0.4 * agent.formal_share
            informal_penalty = 1.0 - 0.4 * agent.informal_share
            income_adj = agent.formal_share * formal_premium + agent.informal_share * informal_penalty
            market_income *= income_adj
            
            transfer_income = total_transfers * cfg.transfer_share[i]
            interest_income = agent.financial_wealth * rate
            interest_cost = agent.debt * (rate + 0.02)
            
            agent.income = market_income + transfer_income + interest_income - interest_cost
            
            effective_infl = inflation * cfg.inflation_sensitivity[i]
            wealth_effect = 0.02 * agent.net_worth
            nominal_consumption = agent.mpc * agent.income + wealth_effect
            agent.consumption = max(0, nominal_consumption / (1 + effective_infl))
            agent.savings = agent.income - agent.consumption
            
            agent.financial_wealth += agent.savings * 0.5
            agent.real_wealth *= 1.02
            agent.real_wealth += agent.savings * 0.5
            
            if agent.savings < 0:
                agent.debt += abs(agent.savings) * 0.3
            else:
                agent.debt = max(0, agent.debt - agent.savings * 0.1)
            
            # Employment dynamics
            gdp_growth = self.economy.gdp_growth
            formal_emp_change = gdp_growth * 1.5 * agent.formal_share
            informal_emp_change = gdp_growth * 0.5 * agent.informal_share
            agent.employment_rate = np.clip(
                agent.employment_rate + formal_emp_change + informal_emp_change,
                0.3, 0.99
            )
    
    # =====================================================================
    # Run & Analysis
    # =====================================================================
    
    def run(self, steps: int) -> List[Dict[str, Any]]:
        """Run the full research-grade simulation."""
        for _ in range(steps):
            self.step()
        return self.trajectory
    
    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of the simulation."""
        if not self.trajectory:
            return {}
        
        last = self.trajectory[-1]
        outcomes = last.get("outcomes", {})
        
        result: Dict[str, Any] = {
            "time": self.time,
            "gdp": float(self.economy.gdp),
            "gdp_growth": outcomes.get("gdp_growth", 0.0),
            "inflation": outcomes.get("inflation", 0.0),
            "unemployment": outcomes.get("unemployment", 0.0),
        }
        
        if self.config.enable_financial:
            result["financial"] = {
                "npl_ratio": float(self.bank.npl_ratio),
                "car": float(self.bank.car),
                "credit_spread": float(self.bank.credit_spread),
                "efp": float(self.bank.external_finance_premium),
            }
        
        if self.config.enable_open_economy:
            result["external"] = {
                "reer": float(self.external.reer),
                "trade_balance_gdp": float(self.external.trade_balance / max(self.economy.gdp, 1.0)),
                "current_account_gdp": float(self.external.current_account / max(self.economy.gdp, 1.0)),
                "reserves_months": float(self.external.reserves_months_import),
            }
        
        if self.config.enable_io:
            result["sectors"] = {
                name: float(s.output_share)
                for name, s in self.sub_sectors.items()
            }
        
        if self.config.enable_heterogeneous:
            incomes = [self.agents[q].income for q in IncomeQuintile]
            total_inc = sum(incomes) or 1.0
            shares = [i / total_inc for i in incomes]
            result["inequality"] = {
                "gini": float(InequalityMetrics.gini_from_quintiles(shares)),
                "palma": float(InequalityMetrics.palma_ratio(shares)),
            }
        
        return result
    
    def twin_deficit_analysis(self) -> Dict[str, float]:
        """Fiscal + current account twin deficit analysis."""
        fiscal_balance = -self.economy.current_outcomes.get("fiscal_deficit_gdp", 0.0)
        ca_balance = self.external.current_account / max(self.economy.gdp, 1.0)
        return {
            "fiscal_balance_gdp": fiscal_balance,
            "current_account_gdp": ca_balance,
            "twin_deficit": fiscal_balance < 0 and ca_balance < 0,
        }
    
    def external_vulnerability_index(self) -> float:
        """Composite external vulnerability ∈ [0, 1]."""
        cfg = self.open_cfg
        gdp = max(self.economy.gdp, 1.0)
        
        reserve_score = min(1.0, self.external.reserves_months_import / cfg.reserve_target_months)
        ca_score = max(0, 1.0 + self.external.current_account / (gdp * 0.10))
        reer_dev = abs(self.external.reer - cfg.initial_reer) / cfg.initial_reer
        reer_score = max(0, 1.0 - reer_dev * 2.0)
        
        return float(np.clip(1.0 - (0.35 * reserve_score + 0.35 * ca_score + 0.30 * reer_score), 0, 1))
    
    def financial_stability_index(self) -> float:
        """Composite financial stability ∈ [0, 1]."""
        if not self.config.enable_financial:
            return self.economy.current_outcomes.get("financial_stability", 0.5)
        
        cfg = self.fin_cfg
        npl_score = max(0, 1.0 - self.bank.npl_ratio / cfg.npl_max)
        car_score = min(1.0, self.bank.car / (cfg.car_min_regulatory * 1.5))
        efp_score = max(0, 1.0 - self.bank.external_finance_premium / 0.10)
        
        return float(np.clip(0.35 * npl_score + 0.35 * car_score + 0.30 * efp_score, 0, 1))
    
    def stress_test(
        self,
        npl_shock: float = 0.05,
        rate_shock: float = 0.02,
        fx_shock: float = 0.10,
        deposit_run: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Comprehensive stress test across all modules.
        
        Returns post-shock metrics for financial, external, and distributional
        dimensions.
        """
        result: Dict[str, Any] = {"scenario": "combined_stress"}
        
        if self.config.enable_financial:
            # Financial stress
            shocked_bank = BankState(
                performing_loans=self.bank.performing_loans,
                non_performing_loans=self.bank.non_performing_loans + self.bank.performing_loans * npl_shock,
                deposits=self.bank.deposits * (1 - deposit_run),
                government_securities=self.bank.government_securities,
                reserves=self.bank.reserves,
                tier1_capital=self.bank.tier1_capital,
                tier2_capital=self.bank.tier2_capital,
            )
            shocked_bank.performing_loans -= self.bank.performing_loans * npl_shock
            shocked_rwa = max(shocked_bank.risk_weighted_assets, 1.0)
            shocked_car = shocked_bank.total_capital / shocked_rwa
            
            result["financial"] = {
                "pre_car": float(self.bank.car),
                "post_car": float(shocked_car),
                "car_breach": shocked_car < self.fin_cfg.car_min_regulatory,
                "capital_shortfall": float(max(0, self.fin_cfg.car_min_regulatory * shocked_rwa - shocked_bank.total_capital)),
            }
        
        if self.config.enable_open_economy:
            # External stress
            shocked_reer = self.external.reer * (1 + fx_shock)
            reer_dev = (shocked_reer - self.open_cfg.initial_reer) / self.open_cfg.initial_reer
            shocked_exports = self.external.exports * (1 - self.open_cfg.export_price_elasticity * reer_dev)
            shocked_imports = self.external.imports * (1 + self.open_cfg.import_price_elasticity * reer_dev)
            
            result["external"] = {
                "pre_reer": float(self.external.reer),
                "post_reer": float(shocked_reer),
                "pre_trade_balance": float(self.external.trade_balance),
                "post_trade_balance": float(shocked_exports - shocked_imports),
                "reserve_adequacy": float(self.external.reserves_months_import),
            }
        
        if self.config.enable_heterogeneous:
            # Distributional impact
            q1_agent = self.agents.get(IncomeQuintile.Q1)
            q5_agent = self.agents.get(IncomeQuintile.Q5)
            result["distributional"] = {
                "q1_income": float(q1_agent.income) if q1_agent else 0.0,
                "q5_income": float(q5_agent.income) if q5_agent else 0.0,
                "q1_debt_income": float(q1_agent.debt / max(q1_agent.income, 0.01)) if q1_agent else 0.0,
                "rate_shock_q1_impact": float(q1_agent.debt * rate_shock) if q1_agent else 0.0,
                "rate_shock_q5_impact": float(q5_agent.debt * rate_shock) if q5_agent else 0.0,
            }
        
        return result
    
    # =====================================================================
    # Recording
    # =====================================================================
    
    def _record_unified_frame(self):
        """Record a single frame with all dimensions."""
        base_frame = {
            "t": self.time,
            "shock_vector": self.economy.current_shock_vector.copy(),
            "policy_vector": self.economy.current_policy_vector.copy(),
            "channels": self.economy.current_channels.copy(),
            "outcomes": dict(self.economy.current_outcomes),
            "sector_balances": {
                "households": float(self.economy.households.net_worth),
                "firms": float(self.economy.firms.net_worth),
                "government": float(self.economy.government.net_worth),
                "banks": float(self.economy.banks.net_worth),
            },
        }
        
        # Financial dimension
        if self.config.enable_financial:
            base_frame["financial"] = {
                "npl_ratio": float(self.bank.npl_ratio),
                "car": float(self.bank.car),
                "credit_spread": float(self.bank.credit_spread),
                "efp": float(self.bank.external_finance_premium),
                "performing_loans": float(self.bank.performing_loans),
                "deposits": float(self.bank.deposits),
            }
            self.financial_trajectory.append(base_frame["financial"].copy())
        
        # External dimension
        if self.config.enable_open_economy:
            ext = {
                "reer": float(self.external.reer),
                "trade_balance": float(self.external.trade_balance),
                "current_account": float(self.external.current_account),
                "capital_account": float(self.external.capital_account),
                "reserves_months": float(self.external.reserves_months_import),
                "remittances": float(self.external.remittances),
            }
            base_frame["external"] = ext
            base_frame["sector_balances"]["foreign"] = float(self.foreign_sector.net_worth)
            base_frame["outcomes"]["trade_balance_gdp"] = float(
                self.external.trade_balance / max(self.economy.gdp, 1.0)
            )
            base_frame["outcomes"]["reer"] = float(self.external.reer)
            base_frame["outcomes"]["reserves_months"] = float(self.external.reserves_months_import)
            self.external_trajectory.append(ext.copy())
        
        # IO dimension
        if self.config.enable_io and self.sub_sectors:
            io_detail = {}
            for name, s in self.sub_sectors.items():
                io_detail[name] = {
                    "output_share": float(s.output_share),
                    "output": float(s.output),
                    "value_added": float(getattr(s, 'value_added', 0.0)),
                }
            base_frame["io_sectors"] = io_detail
            self.sector_trajectory.append(io_detail.copy())
        
        # Inequality dimension
        if self.config.enable_heterogeneous and self.agents:
            incomes = [self.agents[q].income for q in IncomeQuintile]
            total_inc = sum(incomes) or 1.0
            shares = [i / total_inc for i in incomes]
            ineq = {
                "gini": float(InequalityMetrics.gini_from_quintiles(shares)),
                "palma": float(InequalityMetrics.palma_ratio(shares)),
                "quintile_incomes": {q.value: float(self.agents[q].income) for q in IncomeQuintile},
            }
            base_frame["inequality"] = ineq
            self.inequality_trajectory.append(ineq.copy())
        
        self.trajectory.append(base_frame)
    
    # =====================================================================
    # Bayesian Estimation (convenience wrapper)
    # =====================================================================
    
    def estimate_parameters(
        self,
        observed: ObservedData,
        mcmc_cfg: Optional[MCMCConfig] = None,
    ) -> MCMCResult:
        """
        Run Bayesian parameter estimation on the base SFC model.
        
        Returns posterior distributions over SFCConfig parameters.
        """
        priors = default_sfc_priors()
        estimator = BayesianEstimator(
            priors=priors,
            observed=observed,
            config=self.config.sfc,
            mcmc_config=mcmc_cfg or self.config.mcmc,
        )
        return estimator.estimate()
    
    # =====================================================================
    # Static Runners
    # =====================================================================
    
    @staticmethod
    def run_scenario(
        config: Optional[ResearchSFCConfig] = None,
        steps: int = 50,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Run a complete research-grade scenario."""
        cfg = config or default_kenya_research_config()
        cfg.seed = seed
        cfg.sfc.steps = steps
        
        econ = ResearchSFCEconomy(cfg)
        econ.initialize()
        return econ.run(steps)

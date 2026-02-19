"""
Heterogeneous Agents Module for SFC Models.

Extends scarcity's agent framework (simulation/agents.py) and KShield's
terrain quintile analysis (ui/kshield/terrain.py) with:

1. Income quintile distribution (Q1-Q5) with differentiated consumption behavior
2. Formal/Informal labor market segmentation
3. Gini coefficient tracking and inequality dynamics
4. Heterogeneous consumption propensities (poor consume more of income)
5. Distributional impact analysis for policy shocks

Builds on:
- NodeAgent with regime field (agents.py) — regime maps to income quintile
- Sector balance sheets (sfc.py) — household sector disaggregated
- Terrain quintile analysis (kshiked/ui/kshield/terrain.py)

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from scarcity.simulation.sfc import SFCConfig, SFCEconomy, Sector, SectorType

logger = logging.getLogger("scarcity.simulation.heterogeneous")


# =========================================================================
# Agent Types
# =========================================================================

class IncomeQuintile(str, Enum):
    """Income distribution quintiles."""
    Q1 = "q1_bottom_20"
    Q2 = "q2_lower_20"
    Q3 = "q3_middle_20"
    Q4 = "q4_upper_20"
    Q5 = "q5_top_20"


class LaborType(str, Enum):
    """Labor market segmentation."""
    FORMAL = "formal"
    INFORMAL = "informal"


@dataclass
class HouseholdAgent:
    """
    A heterogeneous household agent within a quintile.
    
    Builds on NodeAgent's regime concept — here regime maps to quintile.
    Each quintile has different:
    - Consumption propensity (MPC)
    - Income level
    - Wealth stock
    - Labor market participation
    """
    quintile: IncomeQuintile
    
    # Income share of total household income (sums to 1 across quintiles)
    income_share: float = 0.2
    
    # Marginal Propensity to Consume (higher for lower quintiles)
    mpc: float = 0.8
    
    # Financial assets (deposits, savings)
    financial_wealth: float = 10.0
    
    # Real assets (property, land)
    real_wealth: float = 20.0
    
    # Debt (loans outstanding)
    debt: float = 5.0
    
    # Employment
    employment_rate: float = 0.90  # fraction employed
    formal_share: float = 0.50    # fraction in formal sector
    
    # Current period flows
    income: float = 0.0
    consumption: float = 0.0
    savings: float = 0.0
    
    @property
    def net_worth(self) -> float:
        return self.financial_wealth + self.real_wealth - self.debt
    
    @property
    def informal_share(self) -> float:
        return 1.0 - self.formal_share


@dataclass
class HeterogeneousConfig:
    """Configuration for the heterogeneous agents module."""
    
    # Kenya-calibrated income shares (World Bank / KNBS)
    # Q1 (bottom 20%) through Q5 (top 20%)
    income_shares: Tuple[float, ...] = (0.04, 0.08, 0.12, 0.20, 0.56)
    
    # MPC by quintile (Keynesian: poor have higher MPC)
    mpc_by_quintile: Tuple[float, ...] = (0.95, 0.90, 0.85, 0.75, 0.60)
    
    # Formal sector share by quintile (lower quintiles more informal)
    formal_share_by_quintile: Tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85)
    
    # Employment rates by quintile
    employment_by_quintile: Tuple[float, ...] = (0.75, 0.82, 0.88, 0.93, 0.97)
    
    # Wealth-to-income ratios by quintile
    wealth_income_ratio: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 12.0)
    
    # Debt-to-income ratios
    debt_income_ratio: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.4)
    
    # Inflation pass-through by quintile (food/energy weight)
    # Lower quintiles feel inflation more (higher food share in consumption)
    inflation_sensitivity: Tuple[float, ...] = (1.5, 1.3, 1.0, 0.8, 0.6)
    
    # Interest rate sensitivity (access to credit)
    rate_sensitivity: Tuple[float, ...] = (0.2, 0.4, 0.8, 1.0, 1.2)
    
    # Fiscal transfer share (govt transfers mostly reach lower quintiles)
    transfer_share: Tuple[float, ...] = (0.35, 0.25, 0.20, 0.12, 0.08)


# =========================================================================
# Default Kenya Configuration
# =========================================================================

def default_kenya_heterogeneous_config() -> HeterogeneousConfig:
    """Kenya-calibrated heterogeneous agent configuration."""
    return HeterogeneousConfig()


# =========================================================================
# Inequality Metrics
# =========================================================================

class InequalityMetrics:
    """
    Compute inequality metrics from quintile distributions.
    
    Builds on the quintile analysis from kshiked/ui/kshield/terrain.py.
    """
    
    @staticmethod
    def gini_from_quintiles(income_shares: np.ndarray) -> float:
        """
        Compute Gini coefficient from quintile income shares.
        
        Uses the trapezoidal approximation of the Lorenz curve.
        
        Args:
            income_shares: Array of 5 income shares (Q1 through Q5), summing to 1.
            
        Returns:
            Gini coefficient ∈ [0, 1].
        """
        shares = np.asarray(income_shares, dtype=np.float64)
        if shares.sum() < 1e-10:
            return 0.0
        
        # Normalize
        shares = shares / shares.sum()
        
        # Cumulative income shares (Lorenz curve ordinates)
        lorenz = np.cumsum(shares)
        lorenz = np.insert(lorenz, 0, 0.0)  # (0,0) origin
        
        # Population quintiles
        pop = np.linspace(0, 1, len(lorenz))
        
        # Perfect equality line area = 0.5
        # Area under Lorenz curve (trapezoidal)
        lorenz_area = np.trapz(lorenz, pop)
        
        gini = 1.0 - 2.0 * lorenz_area
        return float(np.clip(gini, 0.0, 1.0))
    
    @staticmethod
    def palma_ratio(income_shares: np.ndarray) -> float:
        """
        Palma ratio: income share of top 10% / bottom 40%.
        
        Approximated from quintiles: Q5/2 (top 10%) over Q1+Q2 (bottom 40%).
        """
        if len(income_shares) < 5:
            return 0.0
        bottom_40 = income_shares[0] + income_shares[1]
        top_10 = income_shares[4] / 2.0  # Top half of Q5 ≈ top 10%
        return float(top_10 / max(bottom_40, 1e-6))
    
    @staticmethod
    def theil_index(income_shares: np.ndarray, n_quintiles: int = 5) -> float:
        """
        Theil index (GE(1)) from quintile income shares.
        
        T = Σ_i (y_i / μ) · ln(y_i / μ)
        where y_i is per-quintile income and μ is mean income.
        """
        shares = np.asarray(income_shares, dtype=np.float64)
        if shares.sum() < 1e-10:
            return 0.0
        
        # Per-quintile normalized income (relative to equal share)
        equal_share = 1.0 / n_quintiles
        ratios = shares / equal_share
        
        # Theil index
        theil = 0.0
        for r in ratios:
            if r > 0:
                theil += r * np.log(r) / n_quintiles
        
        return float(max(0.0, theil))


# =========================================================================
# Heterogeneous Household Economy
# =========================================================================

class HeterogeneousHouseholdEconomy:
    """
    Disaggregated household sector with quintile-level dynamics.
    
    Wraps the base SFCEconomy and replaces the single-agent household
    sector with 5 heterogeneous quintile agents. Each agent has:
    - Differentiated MPC
    - Differentiated inflation sensitivity
    - Formal/informal labor split
    - Independent wealth accumulation
    
    Policy impacts (transfers, tax changes, rate changes) are distributed
    unevenly across quintiles for realistic distributional analysis.
    """
    
    QUINTILES = list(IncomeQuintile)
    
    def __init__(
        self,
        sfc_config: Optional[SFCConfig] = None,
        het_config: Optional[HeterogeneousConfig] = None,
    ):
        self.sfc_config = sfc_config or SFCConfig()
        self.het_cfg = het_config or default_kenya_heterogeneous_config()
        
        # Base SFC economy
        self.economy = SFCEconomy(self.sfc_config)
        
        # Quintile agents
        self.agents: Dict[IncomeQuintile, HouseholdAgent] = {}
        self._initialize_agents()
        
        # Inequality trajectory
        self.inequality_trajectory: List[Dict[str, float]] = []
        self.time = 0
    
    def _initialize_agents(self):
        """Create quintile agents from config."""
        for i, q in enumerate(self.QUINTILES):
            base_income = 100.0 * self.het_cfg.income_shares[i] / 0.2  # Per-quintile average
            
            self.agents[q] = HouseholdAgent(
                quintile=q,
                income_share=self.het_cfg.income_shares[i],
                mpc=self.het_cfg.mpc_by_quintile[i],
                financial_wealth=base_income * self.het_cfg.wealth_income_ratio[i] * 0.3,
                real_wealth=base_income * self.het_cfg.wealth_income_ratio[i] * 0.7,
                debt=base_income * self.het_cfg.debt_income_ratio[i],
                employment_rate=self.het_cfg.employment_by_quintile[i],
                formal_share=self.het_cfg.formal_share_by_quintile[i],
            )
    
    def initialize(self, gdp: float = 100.0):
        """Initialize aggregate and heterogeneous economies."""
        self.economy.initialize(gdp)
        
        # Set initial incomes
        total_hh_income = gdp * (1 - self.sfc_config.tax_rate)
        for i, q in enumerate(self.QUINTILES):
            self.agents[q].income = total_hh_income * self.het_cfg.income_shares[i]
        
        self._record_inequality()
    
    def step(self) -> Dict[str, Any]:
        """
        Step with heterogeneous agent dynamics:
        1. Run base SFC aggregate step
        2. Distribute income across quintiles
        3. Compute quintile-specific consumption
        4. Update wealth and inequality
        """
        self.time += 1
        
        # 1. Base SFC step
        agg_frame = self.economy.step()
        agg_gdp = self.economy.gdp
        inflation = self.economy.inflation
        interest_rate = self.economy.interest_rate
        
        # 2. Total household income
        total_hh_income = agg_gdp * (1 - self.sfc_config.tax_rate)
        
        # Government transfers (distributed per transfer_share)
        fiscal_impulse = self.economy.current_policy_vector.get("fiscal_impulse", 0.0)
        subsidy = self.economy.current_policy_vector.get("subsidy_rate", 0.0)
        total_transfers = agg_gdp * max(0, fiscal_impulse + subsidy)
        
        # 3. Quintile-level dynamics
        total_consumption = 0.0
        total_savings = 0.0
        
        for i, q in enumerate(self.QUINTILES):
            agent = self.agents[q]
            
            # Market income (employment-weighted)
            market_income = total_hh_income * agent.income_share * agent.employment_rate
            
            # Formal vs informal income differential
            # Informal earns ~60% of formal wage
            formal_premium = 1.0 + 0.4 * agent.formal_share
            informal_penalty = 1.0 - 0.4 * agent.informal_share
            income_adj = (agent.formal_share * formal_premium +
                         agent.informal_share * informal_penalty)
            market_income *= income_adj
            
            # Transfers
            transfer_income = total_transfers * self.het_cfg.transfer_share[i]
            
            # Interest income / cost
            interest_income = agent.financial_wealth * interest_rate
            interest_cost = agent.debt * (interest_rate + 0.02)  # Spread over policy rate
            
            agent.income = market_income + transfer_income + interest_income - interest_cost
            
            # Quintile-specific effective inflation
            # Lower quintiles face higher effective inflation (food/energy weight)
            effective_inflation = inflation * self.het_cfg.inflation_sensitivity[i]
            
            # Consumption (MPC × income + wealth effect, deflated by inflation)
            wealth_effect = 0.02 * agent.net_worth
            nominal_consumption = agent.mpc * agent.income + wealth_effect
            real_consumption = nominal_consumption / (1 + effective_inflation)
            agent.consumption = max(0, real_consumption)
            
            # Savings
            agent.savings = agent.income - agent.consumption
            
            # Update wealth
            agent.financial_wealth += agent.savings * 0.5  # Half to financial
            agent.real_wealth *= (1 + 0.02)  # Real asset appreciation
            agent.real_wealth += agent.savings * 0.5  # Half to real
            
            # Debt dynamics (lower quintiles borrow more when stressed)
            if agent.savings < 0:
                agent.debt += abs(agent.savings) * 0.3
            else:
                agent.debt = max(0, agent.debt - agent.savings * 0.1)
            
            # Employment dynamics
            # Unemployment shock hits lower quintiles harder
            unemp_change = self.economy.unemployment - self.sfc_config.nairu
            emp_sensitivity = 1.5 - i * 0.2  # Q1 more sensitive
            agent.employment_rate -= unemp_change * emp_sensitivity * 0.1
            agent.employment_rate = np.clip(agent.employment_rate, 0.3, 0.99)
            
            total_consumption += agent.consumption
            total_savings += agent.savings
        
        # 4. Update income shares (endogenous inequality dynamics)
        total_income = sum(a.income for a in self.agents.values())
        if total_income > 0:
            for q, agent in self.agents.items():
                agent.income_share = agent.income / total_income
        
        # 5. Record inequality metrics
        self._record_inequality()
        
        # 6. Augmented frame
        augmented = dict(agg_frame) if isinstance(agg_frame, dict) else {}
        augmented["quintile_detail"] = self._current_quintile_detail()
        augmented["inequality"] = self.inequality_trajectory[-1]
        
        return augmented
    
    def run(self, steps: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Run heterogeneous simulation."""
        for _ in range(steps):
            self.step()
        return self.economy.trajectory, self.inequality_trajectory
    
    def _current_quintile_detail(self) -> Dict[str, Dict[str, float]]:
        """Current state of all quintile agents."""
        detail = {}
        for q, agent in self.agents.items():
            detail[q.value] = {
                "income": agent.income,
                "income_share": agent.income_share,
                "consumption": agent.consumption,
                "savings": agent.savings,
                "net_worth": agent.net_worth,
                "employment_rate": agent.employment_rate,
                "formal_share": agent.formal_share,
                "mpc": agent.mpc,
                "debt": agent.debt,
            }
        return detail
    
    def _record_inequality(self):
        """Record inequality metrics."""
        shares = np.array([self.agents[q].income_share for q in self.QUINTILES])
        
        record = {
            "t": self.time,
            "gini": InequalityMetrics.gini_from_quintiles(shares),
            "palma": InequalityMetrics.palma_ratio(shares),
            "theil": InequalityMetrics.theil_index(shares),
            "q1_share": float(shares[0]),
            "q5_share": float(shares[4]),
            "q5_q1_ratio": float(shares[4] / max(shares[0], 1e-6)),
            "avg_mpc": float(np.mean([a.mpc for a in self.agents.values()])),
            "total_debt": float(sum(a.debt for a in self.agents.values())),
            "informal_employment": float(np.mean([
                a.informal_share * a.employment_rate 
                for a in self.agents.values()
            ])),
        }
        self.inequality_trajectory.append(record)
    
    def distributional_impact(
        self, 
        policy_variable: str, 
        policy_change: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute distributional impact of a policy change across quintiles.
        
        Args:
            policy_variable: "tax_rate", "transfer", "interest_rate", "subsidy"
            policy_change: Magnitude of policy change (e.g., +0.02 for 2pp increase)
        
        Returns:
            Dict mapping quintile → impact metrics (income_change, welfare_change).
        """
        impacts = {}
        
        for i, q in enumerate(self.QUINTILES):
            agent = self.agents[q]
            
            if policy_variable == "tax_rate":
                # Tax increase hurts all, proportional to income
                income_change = -agent.income * policy_change
                welfare_change = income_change * agent.mpc
                
            elif policy_variable == "transfer":
                # Transfer increase, distributed by transfer_share
                income_change = policy_change * self.het_cfg.transfer_share[i]
                welfare_change = income_change * agent.mpc
                
            elif policy_variable == "interest_rate":
                # Rate increase: helps savers (Q5), hurts borrowers (Q1)
                savings_effect = agent.financial_wealth * policy_change
                debt_effect = -agent.debt * policy_change
                income_change = savings_effect + debt_effect
                welfare_change = income_change * self.het_cfg.rate_sensitivity[i]
                
            elif policy_variable == "subsidy":
                # Subsidy distributed by transfer_share (targets poor)
                income_change = policy_change * self.het_cfg.transfer_share[i]
                welfare_change = income_change
                
            else:
                income_change = 0.0
                welfare_change = 0.0
            
            impacts[q.value] = {
                "income_change": income_change,
                "welfare_change": welfare_change,
                "consumption_change": income_change * agent.mpc,
                "relative_impact": income_change / max(agent.income, 1e-6),
            }
        
        return impacts

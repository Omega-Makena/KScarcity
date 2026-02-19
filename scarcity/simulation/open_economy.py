"""
Open Economy Module for SFC Models.

Activates the FOREIGN sector stub (SectorType.FOREIGN in sfc.py) and
implements full open-economy dynamics:

1. Trade balance equations (export/import driven by REER and income)
2. Exchange rate model (UIP-PPP hybrid with managed float)
3. Balance of Payments identity (current + capital + financial accounts)
4. Remittance flows (Kenya: ~3.5% of GDP, World Bank)
5. Capital account dynamics (FDI, portfolio flows, hot money)
6. Foreign reserves management and adequacy

Builds on:
- SectorType.FOREIGN enum (sfc.py) — now instantiated as full sector
- fx_shock channel (sfc.py SHOCK_KEYS)
- economic_config.py: trade_gdp, current_account, fdi_inflows
- learned_sfc.py: trade/fx variable mappings
- scenario_templates.py: FX shock channels

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scarcity.simulation.sfc import SFCConfig, SFCEconomy, Sector, SectorType

logger = logging.getLogger("scarcity.simulation.open_economy")


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class OpenEconomyConfig:
    """
    Open economy configuration.
    
    Defaults calibrated to Kenya's external sector
    (CBK, World Bank, KNBS data).
    """
    
    # --- Exchange Rate ---
    initial_reer: float = 100.0          # Real Effective Exchange Rate index
    managed_float: bool = True           # CBK manages float (not pure float)
    intervention_threshold: float = 0.05  # CBK intervenes if REER moves >5%
    intervention_strength: float = 0.3    # Strength of FX intervention
    
    # UIP parameters
    uip_sensitivity: float = 0.5         # Sensitivity to interest diff
    foreign_rate: float = 0.05           # US/global interest rate
    
    # PPP parameters
    ppp_half_life: float = 4.0           # Half-life of PPP reversion (years)
    ppp_weight: float = 0.3             # Weight of PPP in FX determination
    
    # --- Trade ---
    export_gdp_ratio: float = 0.12       # Kenya exports ≈ 12% GDP
    import_gdp_ratio: float = 0.22       # Kenya imports ≈ 22% GDP
    
    # Price elasticities (Marshall-Lerner condition)
    export_price_elasticity: float = 0.6  # REER elasticity of exports
    import_price_elasticity: float = 0.5  # REER elasticity of imports
    
    # Income elasticities
    export_income_elasticity: float = 1.0  # World income elasticity of exports
    import_income_elasticity: float = 1.5  # Domestic income elasticity of imports
    
    # World GDP growth (baseline)
    world_gdp_growth: float = 0.03
    
    # Export composition (for sector-specific shocks)
    export_composition: Dict[str, float] = field(default_factory=lambda: {
        "agriculture": 0.35,   # Tea, coffee, flowers
        "manufacturing": 0.15,  # Textiles, processed food
        "services": 0.35,      # Tourism, BPO, transport
        "mining": 0.10,        # Soda ash, titanium
        "other": 0.05,
    })
    
    # --- Remittances ---
    remittance_gdp_ratio: float = 0.035   # Kenya remittances ≈ 3.5% GDP
    remittance_growth: float = 0.08       # Annual growth rate
    remittance_fx_sensitivity: float = -0.3  # Depreciation increases remittances (in KES)
    
    # --- Capital Account ---
    fdi_gdp_ratio: float = 0.01          # FDI ≈ 1% GDP (Kenya is low)
    portfolio_flows_gdp: float = 0.005    # Portfolio investment
    fdi_rate_sensitivity: float = 0.3     # FDI response to rate differential
    hot_money_sensitivity: float = 1.5    # Portfolio flows response to rate diff
    
    # --- Foreign Reserves ---
    initial_reserves_months: float = 4.5  # CBK target: 4+ months import cover
    reserve_target_months: float = 4.0    # Minimum adequate reserves
    
    # --- Terms of Trade ---
    tot_volatility: float = 0.05          # Terms of trade volatility
    oil_import_share: float = 0.15        # Oil as share of total imports


# =========================================================================
# Default Kenya Configuration
# =========================================================================

def default_kenya_open_economy_config() -> OpenEconomyConfig:
    """Kenya-calibrated open economy configuration."""
    return OpenEconomyConfig()


# =========================================================================
# External Sector State
# =========================================================================

@dataclass
class ExternalState:
    """
    State of the external sector.
    Represents the FOREIGN sector with full BoP accounting.
    """
    # Exchange rate
    reer: float = 100.0               # Real Effective Exchange Rate
    nominal_rate: float = 130.0       # KES/USD nominal rate
    reer_change: float = 0.0          # Period change in REER
    
    # Trade
    exports: float = 12.0             # Exports (% of GDP units)
    imports: float = 22.0             # Imports
    trade_balance: float = -10.0      # Net exports
    
    # Current Account
    remittances: float = 3.5          # Diaspora remittances
    investment_income: float = -2.0   # Net investment income (typically negative for EMEs)
    current_account: float = -8.5     # Current account balance
    
    # Capital/Financial Account
    fdi_net: float = 1.0              # Net FDI  
    portfolio_net: float = 0.5        # Net portfolio flows
    other_flows: float = 2.0          # Other investment + errors & omissions
    capital_account: float = 3.5      # Total capital/financial account
    
    # BoP
    overall_bop: float = -5.0         # Overall balance
    
    # Reserves
    foreign_reserves: float = 9000.0  # USD millions (notional)
    reserves_months_import: float = 4.5
    
    # Terms of Trade
    terms_of_trade: float = 100.0     # Index
    
    # World conditions
    world_gdp_growth: float = 0.03
    world_inflation: float = 0.02
    oil_price_index: float = 100.0


# =========================================================================
# Open Economy SFC
# =========================================================================

class OpenEconomySFC:
    """
    SFC Economy with full open-economy dynamics.
    
    Activates the FOREIGN sector from SectorType enum and adds:
    - Trade balance responsive to REER and income gaps
    - Exchange rate determination (hybrid UIP-PPP with managed float)
    - BoP identity linking current and capital accounts
    - Remittance channel (counter-cyclical stabilizer)
    - Reserve management and adequacy constraints
    
    Net exports feed back into aggregate demand (GDP = C + I + G + NX),
    completing the open-economy IS equation.
    """
    
    def __init__(
        self,
        sfc_config: Optional[SFCConfig] = None,
        open_config: Optional[OpenEconomyConfig] = None,
    ):
        self.sfc_config = sfc_config or SFCConfig()
        self.open_cfg = open_config or default_kenya_open_economy_config()
        
        # Base SFC economy
        self.economy = SFCEconomy(self.sfc_config)
        
        # Activate FOREIGN sector
        self.foreign_sector = Sector("Foreign", SectorType.FOREIGN)
        self.economy.sectors.append(self.foreign_sector)
        
        # External state
        self.external = ExternalState()
        
        # External trajectory
        self.external_trajectory: List[Dict[str, float]] = []
        self.time = 0
    
    def initialize(self, gdp: float = 100.0):
        """Initialize the open economy."""
        self.economy.initialize(gdp)
        
        cfg = self.open_cfg
        
        # Initialize external sector
        self.external.reer = cfg.initial_reer
        self.external.exports = gdp * cfg.export_gdp_ratio
        self.external.imports = gdp * cfg.import_gdp_ratio
        self.external.trade_balance = self.external.exports - self.external.imports
        self.external.remittances = gdp * cfg.remittance_gdp_ratio
        self.external.fdi_net = gdp * cfg.fdi_gdp_ratio
        self.external.portfolio_net = gdp * cfg.portfolio_flows_gdp
        
        # Foreign sector balance sheet
        self.foreign_sector.assets['claims_on_domestic'] = gdp * 0.3
        self.foreign_sector.liabilities['reserves'] = gdp * cfg.initial_reserves_months * cfg.import_gdp_ratio
        
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
        self.external.overall_bop = self.external.current_account + self.external.capital_account
        
        # Reserves
        monthly_imports = self.external.imports / 12.0
        self.external.reserves_months_import = (
            self.foreign_sector.liabilities['reserves'] / max(monthly_imports, 0.1)
        )
        
        self._record_external_state()
    
    def step(self) -> Dict[str, Any]:
        """
        Step the open economy:
        1. Exchange rate determination (UIP-PPP hybrid)
        2. Trade dynamics (Marshall-Lerner)
        3. Remittances (counter-cyclical)
        4. Capital flows (rate-sensitive)
        5. BoP identity
        6. Reserve management
        7. Feed NX into aggregate demand
        8. Run base SFC step
        """
        self.time += 1
        cfg = self.open_cfg
        gdp = self.economy.gdp
        
        # ========================
        # 1. Exchange Rate
        # ========================
        domestic_rate = self.economy.interest_rate
        inflation = self.economy.inflation
        
        # FX shock from scenario
        fx_shock = self.economy.current_shock_vector.get("fx_shock", 0.0)
        
        # UIP component: E(depreciation) = i_domestic - i_foreign
        uip_depreciation = cfg.uip_sensitivity * (domestic_rate - cfg.foreign_rate)
        
        # PPP component: mean-revert to fair value
        ppp_speed = np.log(2) / max(cfg.ppp_half_life, 0.5)
        inflation_diff = inflation - cfg.foreign_rate  # Approximate foreign inflation = foreign rate
        ppp_pressure = -ppp_speed * (self.external.reer - cfg.initial_reer) / cfg.initial_reer
        
        # Combined REER change
        reer_change = (
            (1 - cfg.ppp_weight) * uip_depreciation +
            cfg.ppp_weight * ppp_pressure +
            inflation_diff * 0.5 +
            fx_shock
        )
        
        # Managed float: intervention dampens large moves
        if cfg.managed_float and abs(reer_change) > cfg.intervention_threshold:
            reer_change *= (1 - cfg.intervention_strength)
        
        self.external.reer_change = reer_change
        self.external.reer *= (1 + reer_change)
        self.external.reer = max(50, min(200, self.external.reer))  # Bounds
        
        # Nominal rate (approximate)
        self.external.nominal_rate *= (1 + reer_change - inflation_diff)
        
        # ========================
        # 2. Trade Dynamics
        # ========================
        # Depreciation (REER ↓ from 100) improves trade balance (Marshall-Lerner)
        reer_deviation = (self.external.reer - cfg.initial_reer) / cfg.initial_reer
        
        # Export volume
        export_reer_effect = -cfg.export_price_elasticity * reer_deviation
        export_income_effect = cfg.export_income_elasticity * cfg.world_gdp_growth
        export_growth = export_reer_effect + export_income_effect
        
        # Import volume
        import_reer_effect = cfg.import_price_elasticity * reer_deviation
        import_income_effect = cfg.import_income_elasticity * self.economy.gdp_growth
        import_growth = import_reer_effect + import_income_effect
        
        # Terms of trade shock (random + oil price)
        tot_shock = np.random.normal(0, cfg.tot_volatility)
        
        # Apply sector-specific export effects
        supply_shock = self.economy.current_shock_vector.get("supply_shock", 0.0)
        agri_effect = -supply_shock * cfg.export_composition.get("agriculture", 0.35) * 2.0
        
        self.external.exports *= (1 + export_growth + agri_effect + tot_shock * 0.5)
        self.external.imports *= (1 + import_growth - tot_shock * 0.3)
        
        # Bounds
        self.external.exports = max(0.1, self.external.exports)
        self.external.imports = max(0.1, self.external.imports)
        
        self.external.trade_balance = self.external.exports - self.external.imports
        
        # Terms of trade
        self.external.terms_of_trade *= (1 + tot_shock)
        
        # ========================
        # 3. Remittances
        # ========================
        # Counter-cyclical: depreciation increases KES value of USD remittances
        remittance_growth = (
            cfg.remittance_growth +
            cfg.remittance_fx_sensitivity * reer_deviation
        )
        self.external.remittances *= (1 + remittance_growth)
        self.external.remittances = max(0, self.external.remittances)
        
        # ========================
        # 4. Capital Flows
        # ========================
        rate_diff = domestic_rate - cfg.foreign_rate
        
        # FDI (relatively stable, responds slowly to rate differentials)
        fdi_growth = cfg.fdi_rate_sensitivity * rate_diff + 0.02
        self.external.fdi_net *= (1 + fdi_growth)
        
        # Portfolio flows (hot money, responds quickly to rate differential)
        portfolio_change = cfg.hot_money_sensitivity * rate_diff
        # Risk-off during crises
        if self.economy.credit_spread > 0.05:
            portfolio_change -= 0.5 * (self.economy.credit_spread - 0.05)
        self.external.portfolio_net = gdp * cfg.portfolio_flows_gdp + gdp * portfolio_change
        
        self.external.capital_account = (
            self.external.fdi_net +
            self.external.portfolio_net +
            self.external.other_flows
        )
        
        # ========================
        # 5. BoP Identity
        # ========================
        # Investment income (negative for net debtor country)
        self.external.investment_income = -(
            self.foreign_sector.assets.get('claims_on_domestic', 0) * cfg.foreign_rate * 0.1
        )
        
        self.external.current_account = (
            self.external.trade_balance +
            self.external.remittances +
            self.external.investment_income
        )
        
        self.external.overall_bop = (
            self.external.current_account + 
            self.external.capital_account
        )
        
        # ========================
        # 6. Reserve Management
        # ========================
        # BoP surplus/deficit changes reserves
        reserve_change = self.external.overall_bop
        self.foreign_sector.liabilities['reserves'] = max(
            0, 
            self.foreign_sector.liabilities.get('reserves', 0) + reserve_change
        )
        
        monthly_imports = self.external.imports / 12.0
        self.external.reserves_months_import = (
            self.foreign_sector.liabilities['reserves'] / max(monthly_imports, 0.1)
        )
        
        # Reserve adequacy warning
        if self.external.reserves_months_import < cfg.reserve_target_months:
            logger.warning(
                f"Foreign reserves below target: "
                f"{self.external.reserves_months_import:.1f} months "
                f"(target: {cfg.reserve_target_months})"
            )
        
        # ========================
        # 7. Feed NX into GDP
        # ========================
        # Net exports as share of GDP — modifies aggregate demand
        nx_gdp = self.external.trade_balance / max(gdp, 1.0)
        remit_gdp = self.external.remittances / max(gdp, 1.0)
        
        # Apply external demand effect via demand_shock channel
        external_demand_effect = nx_gdp * 0.5 + remit_gdp * 0.3
        
        # Update FOREIGN sector balance sheet
        self.foreign_sector.assets['claims_on_domestic'] += max(0, -self.external.current_account)
        self.foreign_sector.income = self.external.exports + self.external.remittances
        self.foreign_sector.expenses = self.external.imports
        
        # ========================
        # 8. Base SFC Step
        # ========================
        # Inject external demand effect
        agg_frame = self.economy.step()
        
        # Post-step adjustment: add NX to GDP
        nx_adjustment = self.external.trade_balance + self.external.remittances
        self.economy.gdp += nx_adjustment * 0.1  # Gradual adjustment
        
        # Record
        self._record_external_state()
        
        # Augmented frame
        augmented = dict(agg_frame) if isinstance(agg_frame, dict) else {}
        augmented["external"] = self._current_external_state()
        augmented["outcomes"]["trade_balance_gdp"] = float(nx_gdp)
        augmented["outcomes"]["current_account_gdp"] = float(
            self.external.current_account / max(gdp, 1.0)
        )
        augmented["outcomes"]["reer"] = float(self.external.reer)
        augmented["outcomes"]["reserves_months"] = float(self.external.reserves_months_import)
        augmented["outcomes"]["remittances_gdp"] = float(remit_gdp)
        
        return augmented
    
    def run(self, steps: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Run open economy simulation."""
        for _ in range(steps):
            self.step()
        return self.economy.trajectory, self.external_trajectory
    
    def twin_deficit_analysis(self) -> Dict[str, float]:
        """
        Analyze twin deficit dynamics (fiscal + current account).
        
        Tests: fiscal expansion → increased imports → CA deterioration.
        """
        fiscal_balance_gdp = -self.economy.current_outcomes.get("fiscal_deficit_gdp", 0.0)
        ca_balance_gdp = self.external.current_account / max(self.economy.gdp, 1.0)
        
        return {
            "fiscal_balance_gdp": fiscal_balance_gdp,
            "current_account_gdp": ca_balance_gdp,
            "twin_deficit": fiscal_balance_gdp < 0 and ca_balance_gdp < 0,
            "fiscal_ca_gap": fiscal_balance_gdp - ca_balance_gdp,
            "private_saving_investment_gap": fiscal_balance_gdp - ca_balance_gdp,
        }
    
    def external_vulnerability_index(self) -> float:
        """
        Composite external vulnerability index ∈ [0, 1].
        
        Components:
        - Reserve adequacy
        - Current account deficit
        - Short-term debt exposure
        - REER misalignment
        """
        cfg = self.open_cfg
        gdp = max(self.economy.gdp, 1.0)
        
        # Reserve adequacy (0=bad, 1=good)
        reserve_score = min(1.0, self.external.reserves_months_import / cfg.reserve_target_months)
        
        # CA deficit (normalized)
        ca_score = max(0, 1.0 + self.external.current_account / (gdp * 0.10))
        
        # REER misalignment
        reer_dev = abs(self.external.reer - cfg.initial_reer) / cfg.initial_reer
        reer_score = max(0, 1.0 - reer_dev * 2.0)
        
        # Hot money ratio (portfolio flows / reserves)
        hot_money_score = max(0, 1.0 - abs(self.external.portfolio_net) / 
                             max(self.foreign_sector.liabilities.get('reserves', 1.0), 1.0))
        
        # Weighted composite
        vulnerability = 1.0 - (
            0.30 * reserve_score +
            0.30 * ca_score +
            0.20 * reer_score +
            0.20 * hot_money_score
        )
        
        return float(np.clip(vulnerability, 0.0, 1.0))
    
    def _current_external_state(self) -> Dict[str, float]:
        """Current external sector state."""
        return {
            "reer": self.external.reer,
            "reer_change": self.external.reer_change,
            "nominal_rate": self.external.nominal_rate,
            "exports": self.external.exports,
            "imports": self.external.imports,
            "trade_balance": self.external.trade_balance,
            "remittances": self.external.remittances,
            "current_account": self.external.current_account,
            "fdi_net": self.external.fdi_net,
            "portfolio_net": self.external.portfolio_net,
            "capital_account": self.external.capital_account,
            "overall_bop": self.external.overall_bop,
            "reserves_months": self.external.reserves_months_import,
            "terms_of_trade": self.external.terms_of_trade,
            "vulnerability_index": self.external_vulnerability_index(),
        }
    
    def _record_external_state(self):
        """Record external state."""
        state = self._current_external_state()
        state["t"] = self.time
        self.external_trajectory.append(state)

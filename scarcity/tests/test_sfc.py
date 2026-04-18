"""
Test: Stock-Flow Consistent Economic Simulation

Validates SFC economy maintains consistency and produces sensible dynamics.
"""

import numpy as np
import pytest
from scarcity.simulation.sfc import (
    SFCEconomy,
    SFCConfig,
    Sector,
    SectorType,
    validate_sfc_economy,
)


class TestSectorBalanceSheet:
    def test_balance_sheet_identity(self):
        """Assets = Liabilities + Net Worth."""
        sector = Sector("Test", SectorType.HOUSEHOLDS)
        sector.assets['deposits'] = 100
        sector.liabilities['loans'] = 30
        
        # Net Worth should be 70
        assert sector.net_worth == 70
        assert sector.balance_sheet_identity()
    
    def test_net_lending(self):
        """Net lending = Income - Expenses."""
        sector = Sector("Test", SectorType.FIRMS)
        sector.income = 100
        sector.expenses = 80
        
        assert sector.net_lending == 20


class TestSFCEconomyInitialization:
    def test_initialize_creates_consistent_state(self):
        """Initialization should create internally consistent state."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        assert economy.gdp == 100
        assert economy.households.total_assets > 0
        assert economy.firms.assets.get('capital', 0) > 0
        
        # Banks' loans should equal borrowers' loans
        bank_loans = economy.banks.assets.get('loans', 0)
        household_loans = economy.households.liabilities.get('loans', 0)
        firm_loans = economy.firms.liabilities.get('loans', 0)
        assert abs(bank_loans - (household_loans + firm_loans)) < 1e-6


class TestSFCEconomyDynamics:
    def test_gdp_follows_demand(self):
        """GDP should adjust toward aggregate demand."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        # Record initial GDP
        initial_gdp = economy.gdp
        
        # Run simulation
        for _ in range(10):
            economy.step()
        
        # GDP should have evolved
        assert economy.gdp != initial_gdp
        print(f"GDP: {initial_gdp:.2f} → {economy.gdp:.2f}")
    
    def test_taylor_rule_responds_to_inflation(self):
        """Interest rate should be higher with higher inflation."""
        # Scenario 1: Low inflation
        econ_low = SFCEconomy()
        econ_low.initialize(gdp=100)
        econ_low.inflation = 0.02  # 2%
        econ_low.step()
        rate_low = econ_low.interest_rate
        
        # Scenario 2: High inflation  
        econ_high = SFCEconomy()
        econ_high.initialize(gdp=100)
        econ_high.inflation = 0.15  # 15%
        econ_high.step()
        rate_high = econ_high.interest_rate
        
        # Rate should be higher with higher inflation
        print(f"Low inflation (2%) → Rate: {rate_low:.3f}")
        print(f"High inflation (15%) → Rate: {rate_high:.3f}")
        assert rate_high >= rate_low, f"Rate should increase with inflation"
    
    def test_unemployment_follows_okun(self):
        """Unemployment falls with GDP growth and rises with GDP contraction (Okun's Law direction).

        The original test set economy.gdp = 90 and hoped GDP would grow, but since
        consumption / investment / spending all scale with current GDP, aggregate demand
        ≈ GDP and there is no restoring force.  The fix: apply an explicit demand shock
        to create genuine excess demand, then verify the directional response.
        """
        # --- Positive demand → GDP grows → unemployment falls ---
        cfg_boom = SFCConfig(
            shock_vectors={"demand_shock": np.full(20, 0.08)},  # 8% persistent demand boost
        )
        econ_boom = SFCEconomy(cfg_boom)
        econ_boom.initialize(gdp=100)
        u0_boom = econ_boom.unemployment
        for _ in range(10):
            econ_boom.step()
        assert econ_boom.gdp > 100, "Demand boom should raise GDP"
        assert econ_boom.unemployment < u0_boom, (
            f"Unemployment should fall in a boom: {u0_boom:.4f} → {econ_boom.unemployment:.4f}"
        )

        # --- Negative demand → GDP contracts → unemployment rises ---
        cfg_bust = SFCConfig(
            shock_vectors={"demand_shock": np.full(20, -0.10)},  # 10% persistent demand collapse
        )
        econ_bust = SFCEconomy(cfg_bust)
        econ_bust.initialize(gdp=100)
        u0_bust = econ_bust.unemployment
        for _ in range(10):
            econ_bust.step()
        assert econ_bust.gdp < 100, "Demand bust should contract GDP"
        assert econ_bust.unemployment > u0_bust, (
            f"Unemployment should rise in a bust: {u0_bust:.4f} → {econ_bust.unemployment:.4f}"
        )
    
    def test_fiscal_deficit_increases_debt(self):
        """Government deficit should increase debt."""
        config = SFCConfig(spending_ratio=0.30, tax_rate=0.20)  # Deficit
        economy = SFCEconomy(config)
        economy.initialize(gdp=100)
        
        initial_debt = economy.government.total_liabilities
        
        for _ in range(5):
            economy.step()
        
        # Debt should have increased
        assert economy.government.total_liabilities > initial_debt


class TestSFCShocks:
    def test_demand_shock(self):
        """Demand shock should increase GDP."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        initial_gdp = economy.gdp
        economy.apply_shock('demand', 0.10)  # 10% boost
        
        assert economy.gdp > initial_gdp
    
    def test_monetary_shock(self):
        """Monetary shock should change interest rate."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        initial_rate = economy.interest_rate
        economy.apply_shock('monetary', 0.02)  # +2pp
        
        assert economy.interest_rate == initial_rate + 0.02


class TestSFCConsistency:
    def test_validate_sfc_economy(self):
        """Full validation should pass."""
        result = validate_sfc_economy()
        assert result, "SFC economy validation failed"
    
    def test_history_recorded(self):
        """History should be recorded at each step."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        economy.run(10)
        
        assert len(economy.history) == 11  # Initial + 10 steps
        assert all('gdp' in h for h in economy.history)
    
    def test_no_explosive_behavior(self):
        """Economy should not explode over reasonable horizon."""
        economy = SFCEconomy()
        economy.initialize(gdp=100)
        
        for _ in range(50):
            economy.step()
        
        # GDP should remain reasonable
        assert 10 < economy.gdp < 10000, f"GDP exploded: {economy.gdp}"
        assert -0.5 < economy.inflation < 0.5, f"Inflation exploded: {economy.inflation}"

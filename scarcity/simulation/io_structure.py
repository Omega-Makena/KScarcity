"""
Multi-Sector Input-Output Structure for SFC Models.

Extends scarcity's SFCEconomy (simulation/sfc.py) with:

1. Leontief Input-Output matrix for inter-sector flows
2. Sub-sector disaggregation (Agriculture, Manufacturing, Services, Mining)
3. Sector-specific shocks and productivity dynamics
4. Value-added chain tracking and structural change analysis

Builds on:
- SectorType enum (sfc.py) — existing FOREIGN stub now activated
- Sector class balance sheets (sfc.py)
- Scenario templates (kshiked/simulation/scenario_templates.py) sector references
- Economic config (scarcity/economic_config.py) for trade/industry data

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from scarcity.simulation.sfc import SectorType, Sector, SFCConfig, SFCEconomy

logger = logging.getLogger("scarcity.simulation.io_structure")


# =========================================================================
# Sub-Sector Definitions (Kenya-calibrated defaults)
# =========================================================================

class SubSectorType(str, Enum):
    """Production sub-sectors for IO disaggregation."""
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    MINING = "mining"
    CONSTRUCTION = "construction"


@dataclass
class SubSector:
    """
    A production sub-sector with output, employment, and IO linkages.
    
    Extends the Sector balance sheet concept with production-specific fields.
    """
    name: str
    sub_type: SubSectorType
    
    # Output share of GDP (must sum to ~1 across sub-sectors)
    output_share: float = 0.25
    
    # Employment share of labor force
    employment_share: float = 0.25
    
    # Total Factor Productivity (index, 1.0 = baseline)
    tfp: float = 1.0
    
    # Sector-specific capital stock
    capital: float = 25.0
    
    # Depreciation rate
    depreciation: float = 0.05
    
    # Output (current period)
    output: float = 0.0
    
    # Value added
    value_added: float = 0.0
    
    # Export share (fraction of output exported)
    export_share: float = 0.1
    
    # Import dependency (fraction of intermediate inputs imported)
    import_dependency: float = 0.2


@dataclass
class IOConfig:
    """Configuration for the IO structure."""
    # Number of sub-sectors
    n_sectors: int = 5
    
    # Input-Output coefficient matrix (technical coefficients, A matrix)
    # A[i,j] = fraction of sector j's output used as input by sector i
    # Default: Kenya-calibrated approximate coefficients
    io_matrix: Optional[np.ndarray] = None
    
    # Sector GDP shares (default: Kenya structure)
    sector_shares: Optional[Dict[str, float]] = None
    
    # Sector employment shares
    employment_shares: Optional[Dict[str, float]] = None
    
    # Sector-specific shock sensitivities
    shock_sensitivity: Optional[Dict[str, Dict[str, float]]] = None
    
    # Structural change parameters
    structural_change_speed: float = 0.01  # How fast sectoral shares shift


# =========================================================================
# Default Kenya IO Configuration
# =========================================================================

def default_kenya_io_config() -> IOConfig:
    """
    Kenya-calibrated IO configuration.
    
    GDP composition (approximate, from KNBS):
    - Agriculture: 22%
    - Manufacturing: 8%
    - Services: 53%
    - Mining: 4%
    - Construction: 13%
    
    IO coefficients derived from Kenya's make-use tables.
    """
    cfg = IOConfig()
    
    cfg.sector_shares = {
        "agriculture": 0.22,
        "manufacturing": 0.08,
        "services": 0.53,
        "mining": 0.04,
        "construction": 0.13,
    }
    
    cfg.employment_shares = {
        "agriculture": 0.40,
        "manufacturing": 0.07,
        "services": 0.42,
        "mining": 0.01,
        "construction": 0.10,
    }
    
    # Technical coefficient matrix A (5×5)
    # Rows: consuming sector, Cols: supplying sector
    # [agriculture, manufacturing, services, mining, construction]
    cfg.io_matrix = np.array([
        [0.10, 0.05, 0.03, 0.01, 0.02],  # agriculture uses...
        [0.15, 0.12, 0.08, 0.05, 0.03],  # manufacturing uses...
        [0.05, 0.08, 0.10, 0.02, 0.05],  # services uses...
        [0.02, 0.03, 0.05, 0.08, 0.04],  # mining uses...
        [0.04, 0.10, 0.06, 0.08, 0.05],  # construction uses...
    ], dtype=np.float64)
    
    # Shock sensitivity: how each sector responds to canonical shock channels
    cfg.shock_sensitivity = {
        "agriculture": {"supply_shock": 1.5, "demand_shock": 0.3, "fx_shock": 0.8, "fiscal_shock": 0.5},
        "manufacturing": {"supply_shock": 0.8, "demand_shock": 1.0, "fx_shock": 1.2, "fiscal_shock": 0.3},
        "services": {"supply_shock": 0.3, "demand_shock": 1.2, "fx_shock": 0.5, "fiscal_shock": 0.8},
        "mining": {"supply_shock": 0.5, "demand_shock": 0.5, "fx_shock": 1.5, "fiscal_shock": 0.2},
        "construction": {"supply_shock": 0.4, "demand_shock": 1.3, "fx_shock": 0.6, "fiscal_shock": 1.5},
    }
    
    return cfg


# =========================================================================
# Leontief Solver
# =========================================================================

class LeontiefModel:
    """
    Leontief Input-Output model.
    
    Given technical coefficient matrix A and final demand vector d,
    solves: X = (I - A)^{-1} · d   (Leontief inverse)
    
    where X is gross output and d is final demand.
    """
    
    def __init__(self, A: np.ndarray):
        """
        Args:
            A: Technical coefficient matrix (n × n).
               A[i,j] = amount of sector j's output needed per unit of sector i's output.
        """
        self.A = A.copy()
        self.n = A.shape[0]
        self._leontief_inverse: Optional[np.ndarray] = None
        self._compute_inverse()
    
    def _compute_inverse(self):
        """Compute Leontief inverse (I - A)^{-1}."""
        I = np.eye(self.n)
        try:
            self._leontief_inverse = np.linalg.inv(I - self.A)
        except np.linalg.LinAlgError:
            logger.warning("IO matrix (I-A) is singular; using pseudo-inverse")
            self._leontief_inverse = np.linalg.pinv(I - self.A)
    
    @property
    def leontief_inverse(self) -> np.ndarray:
        return self._leontief_inverse
    
    def solve_output(self, final_demand: np.ndarray) -> np.ndarray:
        """
        Solve for gross output given final demand.
        
        X = L · d  where L = (I-A)^{-1}
        """
        return self._leontief_inverse @ final_demand
    
    def value_added(self, gross_output: np.ndarray) -> np.ndarray:
        """
        Compute value added per sector.
        
        VA_i = X_i - Σ_j A[i,j] · X_j
        """
        intermediate = self.A @ gross_output
        return gross_output - intermediate
    
    def output_multipliers(self) -> np.ndarray:
        """Column sums of Leontief inverse — total output effect per unit of final demand."""
        return self._leontief_inverse.sum(axis=0)
    
    def backward_linkages(self) -> np.ndarray:
        """
        Backward linkages: how much a sector's demand pulls from all sectors.
        = column sums of A (direct) or Leontief inverse (total).
        """
        return self._leontief_inverse.sum(axis=0)
    
    def forward_linkages(self) -> np.ndarray:
        """
        Forward linkages: how much a sector's supply pushes to all sectors.
        = row sums of Leontief inverse.
        """
        return self._leontief_inverse.sum(axis=1)
    
    def shock_propagation(
        self, sector_index: int, shock_magnitude: float
    ) -> np.ndarray:
        """
        Compute sectoral output effects of a demand shock to one sector.
        
        Args:
            sector_index: Index of the shocked sector.
            shock_magnitude: Size of the final demand shock.
        
        Returns:
            Array of output changes per sector.
        """
        demand_shock = np.zeros(self.n)
        demand_shock[sector_index] = shock_magnitude
        return self._leontief_inverse @ demand_shock


# =========================================================================
# Multi-Sector SFC Economy
# =========================================================================

class MultiSectorSFCEconomy:
    """
    SFC Economy with Input-Output disaggregation.
    
    Wraps the base SFCEconomy and adds:
    - Sub-sector production dynamics
    - Inter-sector flows via Leontief model
    - Sector-specific shock propagation
    - Structural change tracking
    
    The base SFCEconomy handles aggregate macro dynamics (inflation, interest rates,
    fiscal policy).  This layer adds production-side disaggregation.
    """
    
    SECTOR_NAMES = ["agriculture", "manufacturing", "services", "mining", "construction"]
    
    def __init__(
        self,
        sfc_config: Optional[SFCConfig] = None,
        io_config: Optional[IOConfig] = None,
    ):
        self.sfc_config = sfc_config or SFCConfig()
        self.io_cfg = io_config or default_kenya_io_config()
        
        # Base SFC economy (aggregate dynamics)
        self.economy = SFCEconomy(self.sfc_config)
        
        # IO Model
        self.io_model = LeontiefModel(self.io_cfg.io_matrix)
        
        # Sub-sectors
        self.sub_sectors: Dict[str, SubSector] = {}
        self._initialize_sub_sectors()
        
        # Trajectory (augmented with sector detail)
        self.sector_trajectory: List[Dict[str, Any]] = []
        self.time = 0
    
    def _initialize_sub_sectors(self):
        """Create sub-sectors from IO config."""
        shares = self.io_cfg.sector_shares or {}
        emp_shares = self.io_cfg.employment_shares or {}
        
        for i, name in enumerate(self.SECTOR_NAMES):
            sub_type = SubSectorType(name)
            share = shares.get(name, 1.0 / len(self.SECTOR_NAMES))
            emp_share = emp_shares.get(name, 1.0 / len(self.SECTOR_NAMES))
            
            self.sub_sectors[name] = SubSector(
                name=name,
                sub_type=sub_type,
                output_share=share,
                employment_share=emp_share,
                capital=self.economy.gdp * share * 2.0,  # K/Y ≈ 2
            )
    
    def initialize(self, gdp: float = 100.0):
        """Initialize both aggregate and sectoral economies."""
        self.economy.initialize(gdp)
        
        # Set initial sector outputs
        for name, sector in self.sub_sectors.items():
            sector.output = gdp * sector.output_share
        
        self._record_sector_frame()
    
    def step(self) -> Dict[str, Any]:
        """
        Step with IO disaggregation:
        1. Run base SFC aggregate step
        2. Distribute aggregate demand across sectors via IO model
        3. Apply sector-specific shocks
        4. Compute value added and structural change
        5. Feed back to aggregate
        """
        self.time += 1
        
        # 1. Base SFC step (aggregate dynamics)
        agg_frame = self.economy.step()
        agg_gdp = self.economy.gdp
        
        # 2. Compute sectoral final demand
        # Distribute aggregate demand according to current shares
        final_demand = np.array([
            agg_gdp * self.sub_sectors[name].output_share
            for name in self.SECTOR_NAMES
        ])
        
        # 3. Apply sector-specific shock sensitivities
        shock_vec = self.economy.current_shock_vector
        sensitivity = self.io_cfg.shock_sensitivity or {}
        
        for i, name in enumerate(self.SECTOR_NAMES):
            sens = sensitivity.get(name, {})
            shock_mult = 1.0
            for channel, magnitude in shock_vec.items():
                channel_sens = sens.get(channel, 1.0)
                shock_mult += magnitude * channel_sens
            
            # TFP shock effect
            self.sub_sectors[name].tfp *= (1.0 + 0.01 * (shock_mult - 1.0))
            final_demand[i] *= shock_mult
        
        # 4. Solve Leontief for gross output
        gross_output = self.io_model.solve_output(final_demand)
        value_added = self.io_model.value_added(gross_output)
        
        # 5. Update sub-sectors
        total_output = max(np.sum(gross_output), 1e-6)
        for i, name in enumerate(self.SECTOR_NAMES):
            sector = self.sub_sectors[name]
            sector.output = float(gross_output[i])
            sector.value_added = float(value_added[i])
            
            # Update output share (structural change)
            new_share = float(gross_output[i]) / total_output
            sector.output_share += self.io_cfg.structural_change_speed * (new_share - sector.output_share)
            
            # Capital accumulation
            investment = sector.output * 0.15  # Rough I/Y for sector
            sector.capital += (investment - sector.depreciation * sector.capital)
        
        # 6. Normalize shares
        total_share = sum(s.output_share for s in self.sub_sectors.values())
        if total_share > 0:
            for s in self.sub_sectors.values():
                s.output_share /= total_share
        
        self._record_sector_frame()
        
        # Return augmented frame
        augmented = dict(agg_frame) if isinstance(agg_frame, dict) else {}
        augmented["sector_detail"] = self._current_sector_detail()
        augmented["io_multipliers"] = self.io_model.output_multipliers().tolist()
        
        return augmented
    
    def run(self, steps: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run multi-sector simulation.
        
        Returns:
            Tuple of (aggregate_trajectory, sector_trajectory)
        """
        for _ in range(steps):
            self.step()
        return self.economy.trajectory, self.sector_trajectory
    
    def _current_sector_detail(self) -> Dict[str, Dict[str, float]]:
        """Current state of all sub-sectors."""
        detail = {}
        for name, sector in self.sub_sectors.items():
            detail[name] = {
                "output": sector.output,
                "output_share": sector.output_share,
                "value_added": sector.value_added,
                "employment_share": sector.employment_share,
                "tfp": sector.tfp,
                "capital": sector.capital,
                "export_share": sector.export_share,
            }
        return detail
    
    def _record_sector_frame(self):
        """Record sectoral state."""
        frame = {
            "t": self.time,
            "sectors": self._current_sector_detail(),
            "backward_linkages": self.io_model.backward_linkages().tolist(),
            "forward_linkages": self.io_model.forward_linkages().tolist(),
            "herfindahl": self._herfindahl_index(),
        }
        self.sector_trajectory.append(frame)
    
    def _herfindahl_index(self) -> float:
        """
        Herfindahl-Hirschman Index of sectoral concentration.
        Lower = more diversified economy.  Range: [1/n, 1].
        """
        shares = [s.output_share for s in self.sub_sectors.values()]
        return float(sum(s ** 2 for s in shares))
    
    def structural_change_index(self) -> float:
        """
        Lilien index of structural change (std of sectoral growth rates).
        Higher = more structural transformation.
        """
        if len(self.sector_trajectory) < 2:
            return 0.0
        
        prev = self.sector_trajectory[-2]["sectors"]
        curr = self.sector_trajectory[-1]["sectors"]
        
        growth_rates = []
        for name in self.SECTOR_NAMES:
            prev_out = prev.get(name, {}).get("output", 1.0)
            curr_out = curr.get(name, {}).get("output", 1.0)
            if prev_out > 0:
                growth_rates.append((curr_out - prev_out) / prev_out)
        
        return float(np.std(growth_rates)) if growth_rates else 0.0

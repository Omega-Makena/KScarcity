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
    """Production and crisis sub-sectors for IO disaggregation."""
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    MINING = "mining"
    CONSTRUCTION = "construction"
    # Crisis / public sectors
    HEALTH = "health"
    WATER = "water"
    TRANSPORT = "transport"
    SECURITY = "security"


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
    Kenya 2019-vintage IO configuration with crisis sub-sectors.

    GDP composition from KNBS Statistical Abstract 2020
    (GDP at basic prices, 2019 — Table 10.14) and IMF Kenya 2020 Article IV.
    Shares normalised to sum to 1 across the nine modelled sub-sectors.

    Sector shares (2019 vintage):
    - Agriculture, forestry & fishing: 22.8%
    - Manufacturing:                    7.6%
    - Services (trade, finance, ICT,
      real estate, professional, etc.):  49.0%
    - Mining & quarrying:                0.5%  (Kenya is not a major minerals economy)
    - Construction:                      7.1%
    - Health & social work:              2.4%
    - Water & waste management:          0.9%
    - Transport & storage:               5.8%
    - Public admin & defense (security): 3.7%  (includes internal security + peacekeeping)

    Employment shares from KNBS 2020 Labour Force Report (Q4 2019).

    IO technical coefficients updated to reflect Kenya's 2017 Supply-Use
    Table (KNBS, the latest publicly available table as at 2019; applied
    as a proxy for 2019 given modest structural change over 2017-2019).
    Key data source: KNBS "Supply and Use Tables, 2017" and UNCTAD/UNIDO
    regional IO estimates for East Africa (2019).

    Column-sum check: no column exceeds 0.55 (Hawkins-Simon condition
    ensures the Leontief inverse is positive and well-defined).
    """
    cfg = IOConfig()
    cfg.n_sectors = 9

    # ── GDP shares (2019 vintage, normalised) ────────────────────────────────
    cfg.sector_shares = {
        "agriculture":   0.228,
        "manufacturing": 0.076,
        "services":      0.490,
        "mining":        0.005,
        "construction":  0.071,
        "health":        0.024,
        "water":         0.009,
        "transport":     0.058,
        "security":      0.037,
    }

    # ── Employment shares (2019 vintage, KNBS Labour Force Report Q4 2019) ──
    cfg.employment_shares = {
        "agriculture":   0.330,
        "manufacturing": 0.050,
        "services":      0.370,
        "mining":        0.003,
        "construction":  0.090,
        "health":        0.030,
        "water":         0.005,
        "transport":     0.070,
        "security":      0.052,
    }

    # ── Technical coefficient matrix A (9×9) — 2017 SUT proxy for 2019 ──────
    # Rows: sector consuming inputs, Cols: sector supplying them.
    # A[i,j] = value of sector i's output used per unit of sector j's output.
    # Source: KNBS Supply-Use Tables 2017; UNCTAD East Africa IO 2019 estimates.
    #
    # Sector order:
    # [agri, mfg, serv, mine, const, health, water, transport, security]
    #
    # Key empirically-grounded linkages (2019 vintage updates vs. prior):
    #  Agriculture  → uses more services inputs (mobile banking, agri-tech)
    #  Manufacturing → agri inputs slightly lower (more imported intermediates)
    #  Services     → largest downstream user of agri & transport
    #  Mining       → very low domestic linkages (enclave character, small sector)
    #  Construction → leading user of manufacturing & transport
    #  Health       → services-intensive (pharma distribution, ICT diagnostics)
    #  Water        → construction-intensive (NRW rehab, LAPSSET connections)
    #  Transport    → manufacturing & construction dominant inputs
    #  Security     → mostly services (logistics, comms) and some manufacturing
    cfg.io_matrix = np.array([
        # agri    mfg    serv   mine   const  health water  trans  secur
        [0.12,   0.04,  0.04,  0.01,  0.01,  0.01,  0.05,  0.03,  0.00],  # agriculture
        [0.12,   0.14,  0.07,  0.04,  0.04,  0.03,  0.01,  0.03,  0.01],  # manufacturing
        [0.07,   0.09,  0.12,  0.03,  0.06,  0.05,  0.02,  0.04,  0.02],  # services
        [0.01,   0.02,  0.02,  0.05,  0.02,  0.00,  0.01,  0.01,  0.00],  # mining
        [0.03,   0.09,  0.05,  0.06,  0.04,  0.01,  0.03,  0.04,  0.01],  # construction
        [0.02,   0.05,  0.09,  0.01,  0.02,  0.06,  0.05,  0.05,  0.01],  # health
        [0.01,   0.02,  0.03,  0.02,  0.09,  0.01,  0.04,  0.02,  0.00],  # water
        [0.03,   0.07,  0.06,  0.02,  0.11,  0.02,  0.01,  0.07,  0.01],  # transport
        [0.01,   0.02,  0.07,  0.01,  0.02,  0.01,  0.01,  0.05,  0.04],  # security
    ], dtype=np.float64)

    # ── Shock sensitivities (updated for 2019 structure) ─────────────────────
    # Agriculture is now more FX-sensitive (horticulture export dominance).
    # Services less demand-sensitive due to M-PESA / digital resilience.
    # Mining near-zero fiscal sensitivity (enclave, minimal fiscal linkage).
    cfg.shock_sensitivity = {
        "agriculture":   {"supply_shock": 1.6, "demand_shock": 0.3, "fx_shock": 1.0, "fiscal_shock": 0.5},
        "manufacturing": {"supply_shock": 0.9, "demand_shock": 1.0, "fx_shock": 1.3, "fiscal_shock": 0.4},
        "services":      {"supply_shock": 0.3, "demand_shock": 1.1, "fx_shock": 0.4, "fiscal_shock": 0.7},
        "mining":        {"supply_shock": 0.4, "demand_shock": 0.3, "fx_shock": 1.4, "fiscal_shock": 0.1},
        "construction":  {"supply_shock": 0.5, "demand_shock": 1.4, "fx_shock": 0.6, "fiscal_shock": 1.6},
        "health":        {"supply_shock": 1.7, "demand_shock": 0.7, "fx_shock": 0.3, "fiscal_shock": 1.3},
        "water":         {"supply_shock": 1.4, "demand_shock": 0.4, "fx_shock": 0.2, "fiscal_shock": 1.1},
        "transport":     {"supply_shock": 1.2, "demand_shock": 0.9, "fx_shock": 0.8, "fiscal_shock": 1.0},
        "security":      {"supply_shock": 0.5, "demand_shock": 0.8, "fx_shock": 0.6, "fiscal_shock": 1.4},
    }

    return cfg


# =========================================================================
# KNBS → SFC 4-Sector Concordance and Aggregation
# =========================================================================

# Order matches the io_matrix rows/cols in default_kenya_io_config().
_KNBS_ORDER: list[str] = [
    "agriculture", "manufacturing", "services", "mining",
    "construction", "health", "water", "transport", "security",
]

# Maps each KNBS sub-sector → one of the three formal SFC production sectors.
# The Informal sector is not represented in the KNBS Supply-Use Table and
# must be handled separately by the caller.
KNBS_TO_SFC_SECTOR: dict[str, str] = {
    "agriculture":   "agriculture",
    "manufacturing": "manufacturing",
    "mining":        "manufacturing",   # enclave-like; aggregated with industry
    "construction":  "manufacturing",   # capital-goods adjacent
    "water":         "manufacturing",   # utility / infrastructure
    "services":      "services",
    "health":        "services",
    "transport":     "services",
    "security":      "services",
}

_SFC_FORMAL_SECTORS: list[str] = ["agriculture", "manufacturing", "services"]


def aggregate_io_to_sfc_sectors(io_cfg: "IOConfig | None" = None) -> dict:
    """Aggregate the 9-sector KNBS IO matrix into a 3-sector (agri/mfg/serv) block.

    Standard IO aggregation formula:
        A_agg[I, J] = Σ_{i∈I} Σ_{j∈J} A[i,j] · x_j / X_J

    where x_j is the GDP share of KNBS sub-sector j and X_J is the total GDP
    share of the SFC super-sector J.  GDP shares serve as gross-output weights
    (consistent with the KNBS 2019 vintage data in default_kenya_io_config()).

    Returns a dict with:
      ``io_block``       — dict[str, dict[str, float]] 3×3 technical coefficients
      ``import_content`` — dict[str, float] weighted import content per sector

    The INFORMAL sector is absent from the KNBS table; the caller is responsible
    for supplying informal row/column entries (typically from field estimates).
    """
    cfg = io_cfg or default_kenya_io_config()
    A = cfg.io_matrix          # 9×9 numpy array
    shares: dict[str, float] = cfg.sector_shares or {}

    # Build index groups: SFC sector → list of KNBS column/row indices
    groups: dict[str, list[int]] = {s: [] for s in _SFC_FORMAL_SECTORS}
    for idx, name in enumerate(_KNBS_ORDER):
        sfc = KNBS_TO_SFC_SECTOR.get(name)
        if sfc:
            groups[sfc].append(idx)

    # Aggregate GDP shares per SFC super-sector
    group_shares: dict[str, float] = {
        sfc: sum(shares.get(_KNBS_ORDER[i], 0.0) for i in idxs)
        for sfc, idxs in groups.items()
    }

    # Compute aggregated technical coefficients
    io_block: dict[str, dict[str, float]] = {}
    for sfc_row in _SFC_FORMAL_SECTORS:
        io_block[sfc_row] = {}
        for sfc_col in _SFC_FORMAL_SECTORS:
            X_J = max(group_shares[sfc_col], 1e-12)
            total = sum(
                float(A[i, j]) * shares.get(_KNBS_ORDER[j], 0.0) / X_J
                for i in groups[sfc_row]
                for j in groups[sfc_col]
            )
            io_block[sfc_row][sfc_col] = round(total, 4)

    # Derive import content from sector-specific FX shock sensitivity.
    # Rationale: higher exchange-rate pass-through ≈ higher import dependency.
    # Calibration anchor: manufacturing FX sensitivity ~1.3 maps to ~0.31 import content.
    # Scale: import_content ≈ 0.24 × avg_fx_sensitivity (clamped to [0.05, 0.50]).
    sens_map: dict[str, dict[str, float]] = cfg.shock_sensitivity or {}
    import_content: dict[str, float] = {}
    for sfc, idxs in groups.items():
        total_share = max(group_shares[sfc], 1e-12)
        avg_fx = sum(
            shares.get(_KNBS_ORDER[i], 0.0)
            * sens_map.get(_KNBS_ORDER[i], {}).get("fx_shock", 1.0)
            for i in idxs
        ) / total_share
        import_content[sfc] = round(min(0.50, max(0.05, 0.24 * avg_fx)), 3)

    return {"io_block": io_block, "import_content": import_content}


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
    
    SECTOR_NAMES = [
        "agriculture", "manufacturing", "services", "mining", "construction",
        "health", "water", "transport", "security",
    ]
    
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

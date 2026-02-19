"""
ScarcityBridge — K-SHIELD's window into the entire scarcity architecture.

Connects K-SHIELD to ALL scarcity subsystems:
    - Discovery Engine  (learn economic relationships from data)
    - PolicySimulator    (propagate shocks through learned graph)
    - Meta-Learning      (Reptile optimizer, global priors)
    - Meta-Integrative   (governance, safety checks)
    - Governor           (resource stability)
    - Causal Engine      (DoWhy/EconML verification)

Design principle:
    scarcity provides the learning infrastructure.
    kshiked provides Kenya-specific data + fallbacks.
    Fallbacks fade as learned confidence rises.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kshield.bridge")


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class TrainingReport:
    """Result of training the discovery engine on historical data."""
    years_fed: int = 0
    hypotheses_created: int = 0
    hypotheses_active: int = 0
    top_relationships: List[Dict[str, Any]] = field(default_factory=list)
    confidence_map: Dict[str, float] = field(default_factory=dict)
    training_time_ms: float = 0.0
    variables_seen: List[str] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        if not self.confidence_map:
            return 0.0
        return float(np.mean(list(self.confidence_map.values())))


# =========================================================================
# ScarcityBridge
# =========================================================================

class ScarcityBridge:
    """
    K-SHIELD's universal adapter to the scarcity architecture.
    
    Usage:
        bridge = ScarcityBridge()
        report = bridge.train(data_path)         # Feed historical data
        sim = bridge.get_simulator()               # Get learned simulator
        rels = bridge.get_top_relationships(10)     # Inspect what was learned
        conf = bridge.get_confidence_map()          # Per-variable confidence
    """

    def __init__(self):
        self._discovery = None     # EconomicDiscoveryEngine
        self._meta_agent = None    # MetaLearningAgent
        self._supervisor = None    # MetaSupervisor
        self._governor = None      # DynamicResourceGovernor
        self._bus = None           # EventBus

        self._trained = False
        self._training_report: Optional[TrainingReport] = None
        self._confidence_map: Dict[str, float] = {}

        self._init_subsystems()

    def _init_subsystems(self):
        """Initialize all accessible scarcity subsystems."""
        # Runtime (event bus)
        try:
            from scarcity.runtime import EventBus, get_bus
            self._bus = get_bus()
            logger.info("EventBus connected")
        except ImportError:
            logger.warning("scarcity.runtime not available — running without event bus")

        # Tier 0-1: Discovery Engine
        try:
            from scarcity.engine.economic_engine import EconomicDiscoveryEngine
            self._discovery = EconomicDiscoveryEngine()
            logger.info(f"Discovery engine initialized — "
                        f"{len(self._discovery.core.hypotheses.population)} hypotheses pre-loaded")
        except ImportError as e:
            logger.warning(f"Discovery engine not available: {e}")

        # Tier 4: Meta-Learning Agent
        try:
            from scarcity.meta.meta_learning import MetaLearningAgent, MetaLearningConfig
            if self._bus:
                self._meta_agent = MetaLearningAgent(self._bus, MetaLearningConfig())
                logger.info("Meta-learning agent connected")
        except ImportError:
            logger.info("Meta-learning not available (optional)")

        # Tier 5: Meta-Integrative Supervisor
        try:
            from scarcity.meta.integrative_meta import MetaSupervisor
            if self._bus:
                self._supervisor = MetaSupervisor(bus=self._bus)
                logger.info("Meta-supervisor connected")
        except ImportError:
            logger.info("Meta-supervisor not available (optional)")

        # Governor
        try:
            from scarcity.governor.drg_core import DynamicResourceGovernor, DRGConfig
            if self._bus:
                self._governor = DynamicResourceGovernor(DRGConfig(), self._bus)
                logger.info("Resource governor connected")
        except ImportError:
            logger.info("Governor not available (optional)")

    # -----------------------------------------------------------------
    # Training: Feed historical data → learn relationships
    # -----------------------------------------------------------------

    def train(self, data_path: Optional[Path] = None) -> TrainingReport:
        """
        Train the discovery engine on historical World Bank data.

        Reads the CSV, transposes rows into annual observations (one row per year),
        and feeds each through the discovery engine. All 306+ pairwise hypotheses
        are updated and scored against real data.

        Args:
            data_path: Path to World Bank CSV. Defaults to the project data path.

        Returns:
            TrainingReport with hypothesis counts, confidence map, and top relationships.
        """
        if not self._discovery:
            raise RuntimeError("Discovery engine not available — cannot train")

        import pandas as pd
        from scarcity.economic_config import ECONOMIC_VARIABLES

        # Resolve data path
        if data_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        t0 = time.time()

        # Load CSV
        df = pd.read_csv(data_path, skiprows=4)

        # Build indicator code → row mapping
        indicator_codes = set(ECONOMIC_VARIABLES.values())
        code_rows = {}
        for _, row in df.iterrows():
            code = str(row.get("Indicator Code", ""))
            if code in indicator_codes:
                code_rows[code] = row

        # Year columns
        year_cols = [c for c in df.columns if c.isdigit() or (isinstance(c, str) and c.strip().isdigit())]
        year_cols_sorted = sorted(year_cols, key=lambda x: int(str(x).strip()))

        # Feed one row per year (transpose: year → {code: value, code: value, ...})
        years_fed = 0
        for year_col in year_cols_sorted:
            year_int = int(str(year_col).strip())
            if year_int < 1980:  # Skip very sparse early years
                continue

            annual_row = {}
            valid_count = 0
            for code, series_row in code_rows.items():
                val = series_row.get(year_col)
                if pd.notna(val):
                    try:
                        annual_row[code] = float(val)
                        valid_count += 1
                    except (ValueError, TypeError):
                        pass

            if valid_count >= 3:  # Need at least 3 valid indicators to learn
                self._discovery.process_row_raw(annual_row)
                years_fed += 1

        self._trained = True
        training_time = (time.time() - t0) * 1000

        # Build confidence map
        self._confidence_map = self._compute_confidence_map()

        # Build report
        pool = self._discovery.core.hypotheses
        active_count = sum(
            1 for h in pool.population.values()
            if hasattr(h, 'meta') and hasattr(h.meta, 'state')
            and h.meta.state.value in ("active", "tentative")
        )

        top_rels = self._get_top_relationships_raw(20)

        self._training_report = TrainingReport(
            years_fed=years_fed,
            hypotheses_created=len(pool.population),
            hypotheses_active=active_count,
            top_relationships=top_rels,
            confidence_map=dict(self._confidence_map),
            training_time_ms=training_time,
            variables_seen=list(set(
                var for h in pool.population.values()
                for var in h.variables
            )),
        )

        logger.info(
            f"Training complete: {years_fed} years, "
            f"{len(pool.population)} hypotheses, "
            f"overall confidence {self._training_report.overall_confidence:.2%}, "
            f"{training_time:.0f}ms"
        )

        return self._training_report

    # -----------------------------------------------------------------
    # Simulator: Get a PolicySimulator backed by learned relationships
    # -----------------------------------------------------------------

    def get_simulator(self):
        """
        Get a PolicySimulator initialized with learned hypotheses.

        The simulator propagates shocks through the discovered relationship
        graph — each hypothesis votes on the next value of its target variable.

        Returns:
            PolicySimulator instance
        """
        if not self._discovery:
            raise RuntimeError("Discovery engine not available")
        if not self._trained:
            logger.warning("get_simulator() called before train() — predictions may be poor")

        return self._discovery.get_simulation_handle()

    # -----------------------------------------------------------------
    # Inspection: What did scarcity learn?
    # -----------------------------------------------------------------

    def get_top_relationships(self, k: int = 20) -> List[Dict[str, Any]]:
        """Get the top-k strongest discovered relationships."""
        return self._get_top_relationships_raw(k)

    def get_confidence_map(self) -> Dict[str, float]:
        """
        Per-variable confidence (0.0–1.0).

        Higher confidence = more data, stronger hypotheses → fallback fades.
        """
        return dict(self._confidence_map)

    def get_knowledge_graph(self) -> List[Dict[str, Any]]:
        """Get the full knowledge graph as a list of edges."""
        if not self._discovery:
            return []
        return self._discovery.core.get_knowledge_graph()

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def training_report(self) -> Optional[TrainingReport]:
        return self._training_report

    @property
    def discovery_engine(self):
        """Direct access to the EconomicDiscoveryEngine."""
        return self._discovery

    @property
    def meta_agent(self):
        """Direct access to MetaLearningAgent (may be None)."""
        return self._meta_agent

    @property
    def governor(self):
        """Direct access to DynamicResourceGovernor (may be None)."""
        return self._governor

    @property
    def bus(self):
        """Shared event bus."""
        return self._bus

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_top_relationships_raw(self, k: int) -> List[Dict[str, Any]]:
        """Extract top-k hypotheses as dicts."""
        if not self._discovery:
            return []

        hyps = self._discovery.core.hypotheses.get_strongest(k)
        results = []
        for h in hyps:
            d = h.to_dict() if hasattr(h, 'to_dict') else {}
            d["variables"] = list(h.variables)
            d["confidence"] = float(h.confidence)
            d["rel_type"] = h.rel_type.value if hasattr(h.rel_type, 'value') else str(h.rel_type)
            results.append(d)
        return results

    def _compute_confidence_map(self) -> Dict[str, float]:
        """
        Compute per-variable confidence based on how many strong hypotheses
        involve that variable.

        Confidence = (number of active hypotheses involving variable with
        confidence > 0.3) / total hypotheses involving variable.
        """
        if not self._discovery:
            return {}

        from scarcity.economic_config import CODE_TO_NAME
        pool = self._discovery.core.hypotheses

        var_total: Dict[str, int] = {}
        var_strong: Dict[str, int] = {}

        for h in pool.population.values():
            for var in h.variables:
                var_total[var] = var_total.get(var, 0) + 1
                if h.confidence > 0.3:
                    var_strong[var] = var_strong.get(var, 0) + 1

        conf = {}
        for var in var_total:
            total = var_total[var]
            strong = var_strong.get(var, 0)
            # Scale to 0-1, with diminishing returns
            raw = strong / max(total, 1)
            conf[var] = min(1.0, raw * 1.5)  # Boost slightly, cap at 1.0

        return conf

"""
Price Aggregator for KShield Pulse

Computes inflation indices from e-commerce price data:
- Daily price indices by category
- Week-over-week price changes
- Maps to Economic Satisfaction Index (ESI)

Usage:
    aggregator = PriceAggregator(database)
    
    # Get inflation by category
    indices = await aggregator.compute_indices(days=7)
    
    # Get ESI contribution
    esi_score = await aggregator.compute_esi()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from .base import ResourceDomain, PriceData

logger = logging.getLogger("kshield.pulse.ecommerce.aggregator")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PriceIndex:
    """Computed price index for a category/domain."""
    domain: ResourceDomain
    category: str
    current_avg: float
    previous_avg: float
    change_percent: float
    sample_size: int
    period_start: datetime
    period_end: datetime
    source: str  # jiji, jumia, kilimall, or "combined"
    
    @property
    def is_inflationary(self) -> bool:
        """Check if prices increased."""
        return self.change_percent > 0
    
    @property
    def severity(self) -> str:
        """Categorize price change severity."""
        pct = abs(self.change_percent)
        if pct < 2:
            return "stable"
        elif pct < 5:
            return "moderate"
        elif pct < 10:
            return "significant"
        else:
            return "severe"


@dataclass
class EconomicSatisfactionScore:
    """
    Economic Satisfaction Index (ESI) contribution from price data.
    
    ESI measures how well citizens' basic needs are being met.
    Higher prices = lower satisfaction = higher threat potential.
    """
    # Overall score (0-1, lower is worse)
    esi_score: float
    
    # Component scores by domain
    food_score: float = 1.0
    fuel_score: float = 1.0
    housing_score: float = 1.0
    transport_score: float = 1.0
    healthcare_score: float = 1.0
    
    # Price change context
    food_change_pct: float = 0.0
    fuel_change_pct: float = 0.0
    housing_change_pct: float = 0.0
    
    # Confidence
    sample_size: int = 0
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_threat_modifier(self) -> float:
        """
        Convert ESI to threat level modifier.
        
        Low ESI → Higher threat potential from grievances
        """
        # Invert: low ESI = high modifier
        return 1.0 - self.esi_score


# =============================================================================
# Aggregator
# =============================================================================

class PriceAggregator:
    """
    Aggregates price data into economic indices.
    
    Computes:
    1. Category-level price indices
    2. Domain-level aggregates (FOOD, FUEL, etc.)
    3. Overall ESI contribution
    """
    
    # Weight each domain contributes to ESI
    DOMAIN_WEIGHTS = {
        ResourceDomain.FOOD: 0.35,      # Food has highest impact
        ResourceDomain.FUEL: 0.25,      # Fuel affects everything
        ResourceDomain.HOUSING: 0.20,   # Housing costs
        ResourceDomain.TRANSPORT: 0.10, # Transport costs
        ResourceDomain.HEALTHCARE: 0.05,# Healthcare
        ResourceDomain.GENERAL: 0.05,   # Other
    }
    
    # Thresholds for scoring
    SEVERE_INCREASE_PCT = 10.0    # >10% = severe
    MODERATE_INCREASE_PCT = 5.0   # >5% = moderate
    
    def __init__(self, database):
        """
        Initialize aggregator.
        
        Args:
            database: Database instance (pulse.db.Database)
        """
        self.db = database
    
    async def compute_indices(
        self,
        days: int = 7,
        source: Optional[str] = None,
    ) -> List[PriceIndex]:
        """
        Compute price indices for all categories.
        
        Args:
            days: Number of days to analyze.
            source: Specific source, or None for all.
            
        Returns:
            List of PriceIndex by category.
        """
        from ...db.models import PriceSnapshot
        from sqlalchemy import select, func
        
        indices = []
        now = datetime.utcnow()
        cutoff = now - timedelta(days=days)
        midpoint = now - timedelta(days=days // 2)
        
        async with self.db.session() as session:
            # Get unique categories
            stmt = select(
                PriceSnapshot.source,
                func.min(PriceSnapshot.product_name).label('category'),
            ).where(
                PriceSnapshot.scraped_at >= cutoff
            )
            
            if source:
                stmt = stmt.where(PriceSnapshot.source == source)
            
            stmt = stmt.group_by(PriceSnapshot.source)
            
            result = await session.execute(stmt)
            sources = result.all()
            
            # Compute indices per source
            for src, _ in sources:
                index = await self._compute_source_index(
                    session, src, cutoff, midpoint, now
                )
                if index:
                    indices.append(index)
        
        return indices
    
    async def _compute_source_index(
        self,
        session,
        source: str,
        start: datetime,
        midpoint: datetime,
        end: datetime,
    ) -> Optional[PriceIndex]:
        """Compute index for a single source."""
        from ...db.models import PriceSnapshot
        from sqlalchemy import select, func
        
        try:
            # Previous period average
            prev_stmt = select(
                func.avg(PriceSnapshot.price_kes)
            ).where(
                PriceSnapshot.source == source,
                PriceSnapshot.scraped_at >= start,
                PriceSnapshot.scraped_at < midpoint,
            )
            prev_result = await session.execute(prev_stmt)
            prev_avg = prev_result.scalar() or 0
            
            # Current period average
            curr_stmt = select(
                func.avg(PriceSnapshot.price_kes),
                func.count(),
            ).where(
                PriceSnapshot.source == source,
                PriceSnapshot.scraped_at >= midpoint,
                PriceSnapshot.scraped_at <= end,
            )
            curr_result = await session.execute(curr_stmt)
            row = curr_result.one()
            curr_avg = row[0] or 0
            sample_size = row[1] or 0
            
            if prev_avg == 0 or sample_size == 0:
                return None
            
            change_pct = ((curr_avg - prev_avg) / prev_avg) * 100
            
            return PriceIndex(
                domain=ResourceDomain.GENERAL,
                category=source,
                current_avg=curr_avg,
                previous_avg=prev_avg,
                change_percent=change_pct,
                sample_size=sample_size,
                period_start=start,
                period_end=end,
                source=source,
            )
            
        except Exception as e:
            logger.warning(f"Failed to compute index for {source}: {e}")
            return None
    
    async def compute_domain_indices(
        self,
        days: int = 7,
    ) -> Dict[ResourceDomain, PriceIndex]:
        """
        Compute price indices by economic domain.
        
        Aggregates all sources into domain-level indices.
        """
        # For now, return placeholder
        # Full implementation would:
        # 1. Map each product to its domain via category
        # 2. Aggregate prices within domain
        # 3. Compute domain-level indices
        
        return {
            ResourceDomain.FOOD: PriceIndex(
                domain=ResourceDomain.FOOD,
                category="food",
                current_avg=0,
                previous_avg=0,
                change_percent=0,
                sample_size=0,
                period_start=datetime.utcnow() - timedelta(days=days),
                period_end=datetime.utcnow(),
                source="combined",
            ),
        }
    
    async def compute_esi(
        self,
        days: int = 7,
    ) -> EconomicSatisfactionScore:
        """
        Compute Economic Satisfaction Index from price data.
        
        ESI formula:
            For each domain:
                score = 1.0 - (price_change_pct / threshold)
                score = max(0, min(1, score))
            
            ESI = weighted_average(domain_scores)
        
        Args:
            days: Analysis period in days.
            
        Returns:
            EconomicSatisfactionScore
        """
        indices = await self.compute_indices(days=days)
        
        if not indices:
            # No data, return neutral score
            return EconomicSatisfactionScore(
                esi_score=0.5,
                sample_size=0,
            )
        
        # Aggregate by domain (simplified - just use overall for now)
        total_change = sum(idx.change_percent for idx in indices) / len(indices)
        total_samples = sum(idx.sample_size for idx in indices)
        
        # Convert price change to satisfaction score
        # Higher prices = lower satisfaction
        # -10% to +10% price change maps to 1.0 to 0.0 score
        raw_score = 1.0 - (total_change / self.SEVERE_INCREASE_PCT)
        esi_score = max(0.0, min(1.0, raw_score))
        
        return EconomicSatisfactionScore(
            esi_score=esi_score,
            food_change_pct=total_change,  # Simplified
            sample_size=total_samples,
            computed_at=datetime.utcnow(),
        )
    
    async def get_inflation_report(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Generate human-readable inflation report.
        
        Returns:
            Report dictionary with indices and ESI.
        """
        indices = await self.compute_indices(days=days)
        esi = await self.compute_esi(days=days)
        
        return {
            "period_days": days,
            "computed_at": datetime.utcnow().isoformat(),
            "esi_score": esi.esi_score,
            "esi_interpretation": self._interpret_esi(esi.esi_score),
            "threat_modifier": esi.to_threat_modifier(),
            "indices": [
                {
                    "source": idx.source,
                    "change_percent": round(idx.change_percent, 2),
                    "severity": idx.severity,
                    "sample_size": idx.sample_size,
                }
                for idx in indices
            ],
            "total_samples": sum(idx.sample_size for idx in indices),
        }
    
    def _interpret_esi(self, score: float) -> str:
        """Get human-readable ESI interpretation."""
        if score >= 0.8:
            return "Economic conditions stable - low grievance potential"
        elif score >= 0.6:
            return "Moderate economic stress - monitor for grievance signals"
        elif score >= 0.4:
            return "Significant economic stress - elevated threat potential"
        else:
            return "Severe economic stress - high threat potential from economic grievances"


# =============================================================================
# Factory Function
# =============================================================================

def create_price_aggregator(database) -> PriceAggregator:
    """Create price aggregator instance."""
    return PriceAggregator(database)

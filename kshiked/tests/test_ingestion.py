"""
Tests for KShield Pulse Data Ingestion Pipeline

Tests:
- Database models and operations
- Scraper initialization
- LLM provider
- Pipeline integration
- Kenya filters
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch


# =============================================================================
# Database Tests
# =============================================================================

class TestDatabaseModels:
    """Test database models."""
    
    def test_social_post_model(self):
        """Test SocialPost model creation."""
        from kshiked.pulse.db.models import SocialPost
        
        post = SocialPost(
            platform="twitter",
            platform_id="123456",
            text="Test post about Kenya",
            author_id="user123",
            author_username="testuser",
            posted_at=datetime.utcnow(),
            scraped_at=datetime.utcnow(),
        )
        
        assert post.platform == "twitter"
        assert post.text == "Test post about Kenya"
        assert post.author_username == "testuser"
    
    def test_price_snapshot_model(self):
        """Test PriceSnapshot model creation."""
        from kshiked.pulse.db.models import PriceSnapshot
        
        snapshot = PriceSnapshot(
            source="jumia",
            product_url="https://jumia.co.ke/product/123",
            product_id="123",
            product_name="Unga Flour 2kg",
            price_kes=350.0,
            original_price_kes=400.0,
            discount_percent=12.5,
            in_stock=True,
        )
        
        assert snapshot.source == "jumia"
        assert snapshot.price_kes == 350.0
        assert snapshot.discount_percent == 12.5
    
    def test_llm_analysis_model(self):
        """Test LLMAnalysis model creation."""
        from kshiked.pulse.db.models import LLMAnalysis
        
        analysis = LLMAnalysis(
            post_id=1,
            tier="tier_3",
            confidence=0.85,
            reasoning="Mobilization language detected",
            intent_score=0.7,
            capability_score=0.5,
            model_name="gemini-1.5-flash",
        )
        
        assert analysis.tier == "tier_3"
        assert analysis.confidence == 0.85


# =============================================================================
# Scraper Tests
# =============================================================================

class TestScraperBase:
    """Test scraper base classes."""
    
    def test_platform_enum(self):
        """Test Platform enum."""
        from kshiked.pulse.scrapers.base import Platform
        
        assert Platform.TWITTER.value == "twitter"
        assert Platform.REDDIT.value == "reddit"
        assert Platform.TELEGRAM.value == "telegram"
    
    def test_scraper_result(self):
        """Test ScraperResult dataclass."""
        from kshiked.pulse.scrapers.base import ScraperResult, Platform
        
        result = ScraperResult(
            platform=Platform.TWITTER,
            platform_id="12345",
            text="Maandamano tomorrow in Nairobi!",
            author_id="user1",
            author_username="activist",
            posted_at=datetime.utcnow(),
            scraped_at=datetime.utcnow(),
        )
        
        assert result.platform == Platform.TWITTER
        assert "Maandamano" in result.text
        
        # Test to_social_post conversion
        post = result.to_social_post()
        assert post.platform == "twitter"
        assert post.text == result.text


class TestKenyaFilters:
    """Test Kenya keyword filters."""
    
    def test_political_keywords(self):
        """Test political keywords list."""
        from kshiked.pulse.filters import KENYA_POLITICAL
        
        assert "ruto" in KENYA_POLITICAL
        assert "raila" in KENYA_POLITICAL
        assert len(KENYA_POLITICAL) > 20
    
    def test_threat_signals(self):
        """Test threat signal keywords."""
        from kshiked.pulse.filters import KENYA_THREAT_SIGNALS
        
        assert "maandamano" in KENYA_THREAT_SIGNALS
        assert "rise up" in KENYA_THREAT_SIGNALS
    
    def test_is_threat_related(self):
        """Test threat detection function."""
        from kshiked.pulse.filters import is_threat_related
        
        assert is_threat_related("Join the maandamano tomorrow!")
        assert is_threat_related("Rise up against the government!")
        assert not is_threat_related("Beautiful weather in Nairobi today")
    
    def test_contains_kenya_keyword(self):
        """Test Kenya keyword detection."""
        from kshiked.pulse.filters import contains_kenya_keyword
        
        assert contains_kenya_keyword("President Ruto addressed parliament")
        assert contains_kenya_keyword("Fuel prices in Nairobi rising")
        assert not contains_kenya_keyword("Random text about nothing")
    
    def test_get_matched_keywords(self):
        """Test keyword extraction."""
        from kshiked.pulse.filters import get_matched_keywords
        
        text = "Ruto and Raila discussed unga prices in Nairobi"
        matches = get_matched_keywords(text)
        
        assert "ruto" in matches
        assert "raila" in matches
        assert "nairobi" in matches


# =============================================================================
# LLM Tests
# =============================================================================

class TestLLMBase:
    """Test LLM base classes."""
    
    def test_threat_tier_enum(self):
        """Test ThreatTier enum."""
        from kshiked.pulse.llm import ThreatTier
        
        assert ThreatTier.TIER_1.value == "tier_1"
        assert ThreatTier.TIER_1.severity == 1
        assert ThreatTier.TIER_5.severity == 5
        assert ThreatTier.TIER_0.severity == 0
    
    def test_role_type_enum(self):
        """Test RoleType enum."""
        from kshiked.pulse.llm import RoleType
        
        assert RoleType.MOBILIZER.value == "mobilizer"
        assert RoleType.IDEOLOGUE.value == "ideologue"
    
    def test_threat_classification(self):
        """Test ThreatClassification dataclass."""
        from kshiked.pulse.llm.base import ThreatClassification, ThreatTier
        
        classification = ThreatClassification(
            tier=ThreatTier.TIER_3,
            confidence=0.85,
            reasoning="Mobilization language detected",
            matched_signals=["rise up", "tomorrow"],
        )
        
        assert classification.tier == ThreatTier.TIER_3
        assert classification.is_threat == True
        assert classification.is_critical == False
        
        # Test critical threat
        critical = ThreatClassification(
            tier=ThreatTier.TIER_1,
            confidence=0.95,
        )
        assert critical.is_critical == True


# =============================================================================
# E-Commerce Tests
# =============================================================================

class TestEcommerceBase:
    """Test e-commerce base classes."""
    
    def test_resource_domain_enum(self):
        """Test ResourceDomain enum."""
        from kshiked.pulse.scrapers.ecommerce import ResourceDomain
        
        assert ResourceDomain.FOOD.value == "food"
        assert ResourceDomain.FUEL.value == "fuel"
        assert ResourceDomain.HOUSING.value == "housing"
    
    def test_price_data(self):
        """Test PriceData dataclass."""
        from kshiked.pulse.scrapers.ecommerce import PriceData, ResourceDomain
        
        price = PriceData(
            source="jumia",
            product_url="https://jumia.co.ke/123",
            product_id="123",
            product_name="Maize Flour 2kg",
            category="groceries",
            price_kes=350.0,
            original_price_kes=400.0,
            discount_percent=12.5,
            economic_domain=ResourceDomain.FOOD,
        )
        
        assert price.price_kes == 350.0
        assert price.economic_domain == ResourceDomain.FOOD


class TestPriceAggregator:
    """Test price aggregator."""
    
    def test_esi_score(self):
        """Test ESI score dataclass."""
        from kshiked.pulse.scrapers.ecommerce import EconomicSatisfactionScore
        
        esi = EconomicSatisfactionScore(
            esi_score=0.7,
            food_change_pct=5.0,
            sample_size=100,
        )
        
        assert esi.esi_score == 0.7
        assert esi.to_threat_modifier() == 0.3  # 1 - 0.7


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test component integration."""
    
    def test_imports(self):
        """Test all major imports work."""
        # Database
        from kshiked.pulse.db import Database, DatabaseConfig
        from kshiked.pulse.db.models import Base, SocialPost, Author, PriceSnapshot
        
        # Scrapers
        from kshiked.pulse.scrapers.base import BaseScraper, ScraperResult, Platform
        
        # E-commerce
        from kshiked.pulse.scrapers.ecommerce import (
            JijiScraper, JumiaScraper, KilimallScraper, PriceAggregator
        )
        
        # LLM
        from kshiked.pulse.llm import (
            LLMProvider, ThreatClassification, GeminiProvider, ThreatTier
        )
        
        # Filters
        from kshiked.pulse.filters import (
            KENYA_POLITICAL, KENYA_ECONOMIC, is_threat_related
        )
        
        # Ingestion
        from kshiked.pulse.ingestion import (
            IngestionOrchestrator, IngestionConfig, IngestionScheduler
        )
        
        # All imports successful
        assert True
    
    def test_pulse_sensor_integration(self):
        """Test PulseSensor still works."""
        from kshiked.pulse import PulseSensor
        
        sensor = PulseSensor()
        
        # Process text
        detections = sensor.process_text(
            "We are suffering! Rise up against high fuel prices!",
            {"platform": "twitter"}
        )
        
        # Should detect signals
        assert len(sensor._detectors) == 15  # 15 default detectors


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

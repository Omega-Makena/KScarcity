"""
Database Models for KShield Pulse

SQLAlchemy 2.0 models for storing:
- Social media posts from all platforms
- Processed signal detections
- LLM analysis results
- Author profiles for network analysis
- Graph edges for relationship tracking
- E-commerce price data for inflation monitoring

Design Principles:
- All models use UUID primary keys for distributed systems compatibility
- Timestamps are stored in UTC
- JSON fields for flexible metadata storage
- Indexes optimized for time-series queries
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    String, Text, Float, Integer, Boolean, DateTime, JSON, ForeignKey,
    Index, UniqueConstraint, CheckConstraint, Enum as SQLEnum,
    func, event,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    validates,
)


# =============================================================================
# Enums
# =============================================================================

class Platform(str, Enum):
    """Social media platforms."""
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"


class ThreatTier(str, Enum):
    """Threat severity tiers from KShield taxonomy."""
    TIER_0 = "tier_0"  # Protected speech
    TIER_1 = "tier_1"  # Existential threats
    TIER_2 = "tier_2"  # Severe stability threats
    TIER_3 = "tier_3"  # High-risk destabilization
    TIER_4 = "tier_4"  # Emerging threats
    TIER_5 = "tier_5"  # Non-threat


class RoleType(str, Enum):
    """Actor roles in threat networks."""
    IDEOLOGUE = "ideologue"       # Produces justification narratives
    MOBILIZER = "mobilizer"       # Calls for action, coordinates
    AMPLIFIER = "amplifier"       # High-volume resharing
    BROKER = "broker"             # Connects communities
    LEGITIMIZER = "legitimizer"   # Adds authority cues
    GATEKEEPER = "gatekeeper"     # Controls channels
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Types of network relationships."""
    REPLY = "reply"
    RETWEET = "retweet"
    QUOTE = "quote"
    MENTION = "mention"
    COOCCURRENCE = "cooccurrence"
    SAME_NARRATIVE = "same_narrative"


class ResourceDomain(str, Enum):
    """Economic resource domains for price tracking."""
    FOOD = "food"
    FUEL = "fuel"
    HOUSING = "housing"
    TRANSPORT = "transport"
    HEALTHCARE = "healthcare"
    GENERAL = "general"


# =============================================================================
# Base Class
# =============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Social Media Models
# =============================================================================

class Author(Base):
    """
    Social media account profile.
    
    Tracks authors across platforms for network analysis.
    Used to identify roles (Mobilizer, Broker, etc.) and
    build relationship graphs.
    """
    __tablename__ = "authors"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    platform: Mapped[str] = mapped_column(SQLEnum(Platform), nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(255))
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Profile metadata
    follower_count: Mapped[Optional[int]] = mapped_column(Integer)
    following_count: Mapped[Optional[int]] = mapped_column(Integer)
    account_created: Mapped[Optional[datetime]] = mapped_column(DateTime)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(255))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # KShield analysis
    role_classification: Mapped[Optional[str]] = mapped_column(SQLEnum(RoleType))
    suspicion_score: Mapped[float] = mapped_column(Float, default=0.0)
    post_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    first_seen: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    posts: Mapped[List["SocialPost"]] = relationship(back_populates="author")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('platform', 'platform_id', name='uq_author_platform_id'),
        Index('ix_author_platform', 'platform'),
        Index('ix_author_suspicion', 'suspicion_score'),
        Index('ix_author_role', 'role_classification'),
    )


class SocialPost(Base):
    """
    Raw social media post from any platform.
    
    Unified representation of posts/tweets/messages across platforms.
    Stores original data plus normalized fields for analysis.
    """
    __tablename__ = "social_posts"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    platform: Mapped[str] = mapped_column(SQLEnum(Platform), nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Author reference
    author_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("authors.id"), nullable=True
    )
    author: Mapped[Optional["Author"]] = relationship(back_populates="posts")
    
    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")
    
    # Engagement metrics
    likes: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    replies: Mapped[int] = mapped_column(Integer, default=0)
    views: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Location (Kenya-focused)
    geo_location: Mapped[Optional[str]] = mapped_column(String(255))
    mentioned_locations: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Context
    hashtags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    mentions: Mapped[Optional[List[str]]] = mapped_column(JSON)
    urls: Mapped[Optional[List[str]]] = mapped_column(JSON)
    media_urls: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Reply/thread context
    reply_to_id: Mapped[Optional[str]] = mapped_column(String(255))
    conversation_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Timestamps
    posted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Raw data for debugging
    raw_data: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Processing status
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    signals: Mapped[List["ProcessedSignal"]] = relationship(back_populates="post")
    llm_analyses: Mapped[List["LLMAnalysis"]] = relationship(back_populates="post")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('platform', 'platform_id', name='uq_post_platform_id'),
        Index('ix_post_platform', 'platform'),
        Index('ix_post_posted_at', 'posted_at'),
        Index('ix_post_scraped_at', 'scraped_at'),
        Index('ix_post_processed', 'processed'),
        Index('ix_post_author', 'author_id'),
        Index('ix_post_conversation', 'conversation_id'),
    )
    
    @validates('text')
    def validate_text(self, key: str, value: str) -> str:
        """Ensure text is not empty."""
        if not value or not value.strip():
            raise ValueError("Post text cannot be empty")
        return value.strip()


class ProcessedSignal(Base):
    """
    Signal detection results from Pulse engine.
    
    Links a detected signal (from the 15-signal taxonomy) to
    a specific social media post, with intensity and confidence scores.
    """
    __tablename__ = "processed_signals"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Link to source post
    post_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("social_posts.id"), nullable=False
    )
    post: Mapped["SocialPost"] = relationship(back_populates="signals")
    
    # Signal identification (from mapper.py SignalID)
    signal_id: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_category: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Scores
    intensity: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Context
    matched_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    emotion_scores: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    
    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_signal_post', 'post_id'),
        Index('ix_signal_id', 'signal_id'),
        Index('ix_signal_intensity', 'intensity'),
        Index('ix_signal_detected_at', 'detected_at'),
        CheckConstraint('intensity >= 0 AND intensity <= 1', name='ck_signal_intensity'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='ck_signal_confidence'),
    )


class LLMAnalysis(Base):
    """
    LLM (Gemini) classification results.
    
    Stores threat tier classification, role identification,
    and narrative analysis from the LLM.
    """
    __tablename__ = "llm_analyses"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Link to source post
    post_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("social_posts.id"), nullable=False
    )
    post: Mapped["SocialPost"] = relationship(back_populates="llm_analyses")
    
    # Model info
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Threat classification
    threat_tier: Mapped[str] = mapped_column(SQLEnum(ThreatTier), nullable=False)
    threat_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    threat_reasoning: Mapped[Optional[str]] = mapped_column(Text)
    
    # Risk scores (from your documentation)
    base_risk: Mapped[Optional[float]] = mapped_column(Float)
    intent_score: Mapped[Optional[float]] = mapped_column(Float)
    capability_score: Mapped[Optional[float]] = mapped_column(Float)
    specificity_score: Mapped[Optional[float]] = mapped_column(Float)
    reach_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Role classification (if author)
    role_classification: Mapped[Optional[str]] = mapped_column(SQLEnum(RoleType))
    role_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    # Narrative analysis
    narrative_type: Mapped[Optional[str]] = mapped_column(String(100))
    narrative_maturity: Mapped[Optional[str]] = mapped_column(String(20))  # Rumor/Narrative/Campaign
    
    # Raw response for debugging
    raw_response: Mapped[Optional[Dict]] = mapped_column(JSON)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Timestamps
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_llm_post', 'post_id'),
        Index('ix_llm_threat_tier', 'threat_tier'),
        Index('ix_llm_model', 'model_name'),
        Index('ix_llm_analyzed_at', 'analyzed_at'),
    )


class NetworkEdge(Base):
    """
    Graph edge for network analysis.
    
    Represents relationships between authors for detecting
    coordinated behavior and identifying network roles.
    """
    __tablename__ = "network_edges"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Nodes (authors)
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("authors.id"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("authors.id"), nullable=False
    )
    
    # Edge properties
    edge_type: Mapped[str] = mapped_column(SQLEnum(EdgeType), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Context
    post_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("social_posts.id")
    )
    
    # Timestamps
    first_interaction: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    last_interaction: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    interaction_count: Mapped[int] = mapped_column(Integer, default=1)
    
    # Indexes
    __table_args__ = (
        Index('ix_edge_source', 'source_id'),
        Index('ix_edge_target', 'target_id'),
        Index('ix_edge_type', 'edge_type'),
        UniqueConstraint(
            'source_id', 'target_id', 'edge_type', 
            name='uq_edge_source_target_type'
        ),
    )


# =============================================================================
# E-Commerce Models (Inflation Monitoring)
# =============================================================================

class ProductCategory(Base):
    """
    Product category mapping to economic domains.
    
    Maps e-commerce categories (e.g., "Food & Grocery")
    to KShield's ResourceDomain for inflation analysis.
    """
    __tablename__ = "product_categories"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Source info
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # jiji, jumia, kilimall
    category_name: Mapped[str] = mapped_column(String(255), nullable=False)
    subcategory_name: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Mapping to economic domain
    economic_domain: Mapped[str] = mapped_column(
        SQLEnum(ResourceDomain), nullable=False
    )
    
    # Weighting for inflation calculation
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint(
            'source', 'category_name', 'subcategory_name',
            name='uq_category_source'
        ),
        Index('ix_category_domain', 'economic_domain'),
    )


class PriceSnapshot(Base):
    """
    E-commerce price data point.
    
    Tracks product prices over time from Jiji, Kilimall, Jumia
    for computing inflation indices.
    """
    __tablename__ = "price_snapshots"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Source
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # jiji, jumia, kilimall
    product_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    product_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Product info
    product_name: Mapped[str] = mapped_column(String(500), nullable=False)
    category_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("product_categories.id")
    )
    
    # Pricing (in KES)
    price_kes: Mapped[float] = mapped_column(Float, nullable=False)
    original_price_kes: Mapped[Optional[float]] = mapped_column(Float)  # Before discount
    discount_percent: Mapped[Optional[float]] = mapped_column(Float)
    
    # Availability
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
    stock_quantity: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Seller info (for tracking market dynamics)
    seller_name: Mapped[Optional[str]] = mapped_column(String(255))
    seller_rating: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_price_source', 'source'),
        Index('ix_price_product', 'product_id'),
        Index('ix_price_category', 'category_id'),
        Index('ix_price_scraped_at', 'scraped_at'),
        Index('ix_price_kes', 'price_kes'),
    )


# =============================================================================
# Materialized Views / Aggregates (for future use)
# =============================================================================

class InflationIndex(Base):
    """
    Computed price index for economic domain.
    
    Daily/weekly aggregated price indices computed from
    PriceSnapshot data for ESI (Economic Satisfaction Index).
    """
    __tablename__ = "inflation_indices"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    # Time period
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)  # daily, weekly
    
    # Domain
    economic_domain: Mapped[str] = mapped_column(
        SQLEnum(ResourceDomain), nullable=False
    )
    source: Mapped[Optional[str]] = mapped_column(String(50))  # null = all sources
    
    # Index values
    price_index: Mapped[float] = mapped_column(Float, nullable=False)  # 100 = baseline
    change_percent: Mapped[float] = mapped_column(Float, default=0.0)  # vs previous period
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Statistics
    min_price: Mapped[Optional[float]] = mapped_column(Float)
    max_price: Mapped[Optional[float]] = mapped_column(Float)
    avg_price: Mapped[Optional[float]] = mapped_column(Float)
    std_dev: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Indexes
    __table_args__ = (
        UniqueConstraint(
            'date', 'period_type', 'economic_domain', 'source',
            name='uq_inflation_index'
        ),
        Index('ix_inflation_date', 'date'),
        Index('ix_inflation_domain', 'economic_domain'),
    )

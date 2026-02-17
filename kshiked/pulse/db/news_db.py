import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.signals import KShieldSignal

logger = logging.getLogger("kshield.pulse.news_db")

DB_PATH = Path("data/news_db.sqlite")

class NewsDatabase:
    """Persistent SQLite storage for news articles."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE,
                        title TEXT,
                        source TEXT,
                        category TEXT,
                        published_at TEXT,
                        description TEXT,
                        content TEXT,
                        image_url TEXT,
                        author TEXT,
                        fetched_at TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS deep_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_url TEXT,
                        timestamp TEXT,
                        base_risk REAL,
                        adjusted_risk REAL,
                        threat_tier TEXT,
                        threat_category TEXT,
                        economic_strain TEXT,
                        social_fracture TEXT,
                        threat_data TEXT,   -- Full JSON of ThreatSignal
                        context_data TEXT,  -- Full JSON of ContextAnalysis
                        indices_data TEXT,  -- Full JSON of AdvancedIndices
                        tta TEXT,           -- V3 Time-to-Action
                        role TEXT,          -- V3 Author Role
                        resilience_data TEXT, -- V3 Resilience JSON
                        status TEXT,
                        FOREIGN KEY(article_url) REFERENCES news_articles(url)
                    )
                """)
                
                # Migration for V3 columns (Idempotent)
                v3_columns = [
                    ("tta", "TEXT"),
                    ("role", "TEXT"),
                    ("resilience_data", "TEXT")
                ]
                for col_name, col_type in v3_columns:
                    try:
                        cursor.execute(f"ALTER TABLE deep_signals ADD COLUMN {col_name} {col_type}")
                        logger.info(f"Migrated DB: Added column {col_name}")
                    except sqlite3.OperationalError:
                        # Column likely exists
                        pass
                        
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON news_articles(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_published ON news_articles(published_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_status ON deep_signals(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjusted_risk ON deep_signals(adjusted_risk)")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize news database: {e}")

    def add_signal(self, signal: 'KShieldSignal'):
        """Store a KShield V3.0 signal."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize component objects to JSON
                threat_json = json.dumps(asdict(signal.threat)) if signal.threat else "{}"
                context_json = json.dumps(asdict(signal.context)) if signal.context else "{}"
                indices_json = json.dumps(asdict(signal.indices)) if signal.indices else "{}"
                resilience_json = json.dumps(asdict(signal.resilience)) if signal.resilience else "{}"
                
                cursor.execute("""
                    INSERT INTO deep_signals 
                    (article_url, timestamp, base_risk, adjusted_risk, threat_tier, threat_category, 
                     economic_strain, social_fracture, threat_data, context_data, indices_data, 
                     tta, role, resilience_data, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.source_id,
                    signal.timestamp.isoformat(),
                    signal.base_risk,
                    signal.adjusted_risk,
                    signal.threat.tier.value if signal.threat else "UNKNOWN",
                    signal.threat.category.value if signal.threat else "UNKNOWN",
                    signal.context.economic_strain.value if signal.context else "UNKNOWN",
                    signal.context.social_fracture.value if signal.context else "UNKNOWN",
                    threat_json,
                    context_json,
                    indices_json,
                    signal.tta.value if signal.tta else "UNKNOWN",
                    signal.role.value if signal.role else "UNKNOWN",
                    resilience_json,
                    signal.status
                ))
                conn.commit()
                logger.info(f"Stored V3 Signal for {signal.source_id} (Risk: {signal.adjusted_risk:.1f}, Status: {signal.status})")
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")

    def add_articles(self, category: str, articles: List[Dict]):
        """
        Add new articles to the database.
        Ignores duplicates based on URL.
        """
        if not articles:
            return

        fetched_at = datetime.utcnow().isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                inserted = 0
                for art in articles:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO news_articles 
                            (url, title, source, category, published_at, description, content, image_url, author, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            art.get("url"),
                            art.get("title"),
                            art.get("source"),
                            category,
                            art.get("published_at"),
                            art.get("description"),
                            art.get("content"),
                            art.get("image_url"),
                            art.get("author"),
                            fetched_at
                        ))
                        if cursor.rowcount > 0:
                            inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert article {art.get('title')}: {e}")
                
                conn.commit()
                if inserted > 0:
                    logger.info(f"Archived {inserted} new articles for {category}")
                    
        except Exception as e:
            logger.error(f"Database error adding articles: {e}")

    def get_history(self, category: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve historical articles."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if category:
                    cursor.execute("""
                        SELECT * FROM news_articles 
                        WHERE category = ? 
                        ORDER BY published_at DESC LIMIT ?
                    """, (category, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM news_articles 
                        ORDER BY published_at DESC LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return []

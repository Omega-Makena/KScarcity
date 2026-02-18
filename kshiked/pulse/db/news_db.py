import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..llm.signals import KShieldSignal

logger = logging.getLogger("kshield.pulse.news_db")

DB_PATH = Path("data/news_db.sqlite")
RAW_HTML_DIR = Path("data/news_content/raw_html")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NewsDatabase:
    """Persistent SQLite storage for news articles and extracted full content."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema and migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
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
                    """
                )
                cursor.execute(
                    """
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
                        threat_data TEXT,
                        context_data TEXT,
                        indices_data TEXT,
                        tta TEXT,
                        role TEXT,
                        resilience_data TEXT,
                        status TEXT,
                        FOREIGN KEY(article_url) REFERENCES news_articles(url)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS news_content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_url TEXT UNIQUE,
                        canonical_url TEXT,
                        raw_html TEXT,
                        extracted_text TEXT,
                        extraction_method TEXT,
                        content_hash TEXT,
                        extracted_at TEXT,
                        status TEXT,
                        error_reason TEXT,
                        http_status INTEGER,
                        storage_path TEXT,
                        dedup_of_url TEXT,
                        FOREIGN KEY(article_url) REFERENCES news_articles(url)
                    )
                    """
                )

                # Migration for V3 columns (idempotent)
                for col_name, col_type in [
                    ("tta", "TEXT"),
                    ("role", "TEXT"),
                    ("resilience_data", "TEXT"),
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE deep_signals ADD COLUMN {col_name} {col_type}")
                        logger.info("Migrated DB: Added column %s", col_name)
                    except sqlite3.OperationalError:
                        pass

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON news_articles(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_published ON news_articles(published_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_status ON deep_signals(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjusted_risk ON deep_signals(adjusted_risk)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_status ON news_content(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_canonical ON news_content(canonical_url)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON news_content(content_hash)")
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize news database: %s", e)

    def add_signal(self, signal: "KShieldSignal"):
        """Store a KShield V3.0 signal."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                threat_json = json.dumps(asdict(signal.threat)) if signal.threat else "{}"
                context_json = json.dumps(asdict(signal.context)) if signal.context else "{}"
                indices_json = json.dumps(asdict(signal.indices)) if signal.indices else "{}"
                resilience_json = json.dumps(asdict(signal.resilience)) if signal.resilience else "{}"

                cursor.execute(
                    """
                    INSERT INTO deep_signals
                    (article_url, timestamp, base_risk, adjusted_risk, threat_tier, threat_category,
                     economic_strain, social_fracture, threat_data, context_data, indices_data,
                     tta, role, resilience_data, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                        signal.status,
                    ),
                )
                conn.commit()
                logger.info(
                    "Stored V3 Signal for %s (Risk: %.1f, Status: %s)",
                    signal.source_id,
                    signal.adjusted_risk,
                    signal.status,
                )
        except Exception as e:
            logger.error("Failed to store signal: %s", e)

    def add_articles(self, category: str, articles: List[Dict]):
        """Add new articles to the database (dedup by URL)."""
        if not articles:
            return

        fetched_at = _utc_now_iso()

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                inserted = 0
                for art in articles:
                    try:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO news_articles
                            (url, title, source, category, published_at, description, content, image_url, author, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                art.get("url"),
                                art.get("title"),
                                art.get("source"),
                                category,
                                art.get("published_at") or art.get("publishedAt"),
                                art.get("description"),
                                art.get("content"),
                                art.get("image_url") or art.get("urlToImage"),
                                art.get("author"),
                                fetched_at,
                            ),
                        )
                        if cursor.rowcount > 0:
                            inserted += 1
                    except Exception as e:
                        logger.warning("Failed to insert article %s: %s", art.get("title"), e)

                conn.commit()
                if inserted > 0:
                    logger.info("Archived %s new articles for %s", inserted, category)

        except Exception as e:
            logger.error("Database error adding articles: %s", e)

    def _persist_raw_html(self, content_hash: str, html: str) -> str:
        if not html:
            return ""
        filename = RAW_HTML_DIR / f"{content_hash}.html"
        try:
            filename.write_text(html, encoding="utf-8")
            return str(filename)
        except Exception as exc:
            logger.warning("Failed to persist raw html %s: %s", filename, exc)
            return ""

    def upsert_content_extraction(self, article_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Insert/update extracted article content with dedup hints."""
        if not article_url:
            return {}

        canonical_url = str(payload.get("canonical_url") or article_url)
        extracted_text = str(payload.get("extracted_text") or "")
        extraction_method = str(payload.get("extraction_method") or "")
        content_hash = str(payload.get("content_hash") or "")
        status = str(payload.get("status") or "error")
        error_reason = str(payload.get("error_reason") or "")
        http_status = int(payload.get("http_status") or 0)
        raw_html = str(payload.get("raw_html") or "")
        extracted_at = _utc_now_iso()

        storage_path = ""
        if raw_html and content_hash:
            storage_path = self._persist_raw_html(content_hash, raw_html)

        dedup_of_url = None

        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Resolve dedup by canonical URL/content hash
                if canonical_url:
                    row = cursor.execute(
                        "SELECT article_url FROM news_content WHERE canonical_url = ? AND article_url != ? LIMIT 1",
                        (canonical_url, article_url),
                    ).fetchone()
                    if row:
                        dedup_of_url = row["article_url"]
                if not dedup_of_url and content_hash:
                    row = cursor.execute(
                        "SELECT article_url FROM news_content WHERE content_hash = ? AND article_url != ? LIMIT 1",
                        (content_hash, article_url),
                    ).fetchone()
                    if row:
                        dedup_of_url = row["article_url"]

                cursor.execute(
                    """
                    INSERT INTO news_content (
                        article_url, canonical_url, raw_html, extracted_text,
                        extraction_method, content_hash, extracted_at,
                        status, error_reason, http_status, storage_path, dedup_of_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_url) DO UPDATE SET
                        canonical_url=excluded.canonical_url,
                        raw_html=excluded.raw_html,
                        extracted_text=excluded.extracted_text,
                        extraction_method=excluded.extraction_method,
                        content_hash=excluded.content_hash,
                        extracted_at=excluded.extracted_at,
                        status=excluded.status,
                        error_reason=excluded.error_reason,
                        http_status=excluded.http_status,
                        storage_path=excluded.storage_path,
                        dedup_of_url=excluded.dedup_of_url
                    """,
                    (
                        article_url,
                        canonical_url,
                        raw_html[:100000] if raw_html else "",
                        extracted_text,
                        extraction_method,
                        content_hash,
                        extracted_at,
                        status,
                        error_reason,
                        http_status,
                        storage_path,
                        dedup_of_url,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Failed to upsert content extraction for %s: %s", article_url, exc)

        return self.get_content_record(article_url) or {}

    def get_content_record(self, article_url: str) -> Optional[Dict[str, Any]]:
        if not article_url:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        c.id,
                        c.article_url,
                        c.canonical_url,
                        c.extracted_text,
                        c.extraction_method,
                        c.content_hash,
                        c.extracted_at,
                        c.status,
                        c.error_reason,
                        c.http_status,
                        c.storage_path,
                        c.dedup_of_url,
                        a.id AS article_id,
                        a.category,
                        a.title,
                        a.source,
                        a.published_at,
                        a.fetched_at,
                        a.url
                    FROM news_content c
                    LEFT JOIN news_articles a ON a.url = c.article_url
                    WHERE c.article_url = ?
                    LIMIT 1
                    """,
                    (article_url,),
                ).fetchone()
            return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to get content record for %s: %s", article_url, exc)
            return None

    def get_recent_articles_with_content(
        self,
        category: Optional[str] = None,
        limit: int = 300,
        ok_only: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                base_query = """
                    SELECT
                        a.id AS article_id,
                        a.url,
                        a.title,
                        a.source,
                        a.category,
                        a.published_at,
                        a.description,
                        a.content,
                        a.fetched_at,
                        c.id AS content_record_id,
                        c.canonical_url,
                        c.extracted_text,
                        c.extraction_method,
                        c.content_hash,
                        c.extracted_at,
                        c.status AS extraction_status,
                        c.error_reason,
                        c.http_status,
                        c.storage_path,
                        c.dedup_of_url
                    FROM news_articles a
                    LEFT JOIN news_content c ON a.url = c.article_url
                """
                filters: List[str] = []
                params: List[Any] = []
                if category:
                    filters.append("a.category = ?")
                    params.append(category)
                if ok_only:
                    filters.append("c.status = 'ok'")

                if filters:
                    base_query += " WHERE " + " AND ".join(filters)

                base_query += " ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ?"
                params.append(int(limit))

                rows = conn.execute(base_query, tuple(params)).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Failed to fetch enriched article history: %s", exc)
            return []

    def get_history(self, category: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve historical articles."""
        try:
            with self._connect() as conn:
                if category:
                    rows = conn.execute(
                        """
                        SELECT * FROM news_articles
                        WHERE category = ?
                        ORDER BY published_at DESC
                        LIMIT ?
                        """,
                        (category, int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM news_articles
                        ORDER BY published_at DESC
                        LIMIT ?
                        """,
                        (int(limit),),
                    ).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to fetch history: %s", e)
            return []

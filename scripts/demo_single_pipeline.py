from kshiked.pulse.news import NewsIngestor
import logging
import sys

# Configure logging to show us what's happening
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def demo_single_fetch():
    print("--- Demo: Fetching Single Pipeline ---")
    
    ingestor = NewsIngestor()
    
    # Fetch ONLY Technology
    # This proves we can target a specific pipeline without running others
    print("\n1. Triggering 'technology' pipeline only...")
    articles = ingestor.fetch_pipeline("technology", force=True)
    
    print(f"\nResult: Fetched {len(articles)} articles for Technology.")
    if articles:
        print(f"Sample: {articles[0]['title']}")
        
    print("\nCheck 'data/news_cache/technology.json' for independent update.")
    
    # Verify DB
    from kshiked.pulse.db.news_db import NewsDatabase
    db = NewsDatabase()
    # Explicit DB Test
    print("\n2. Testing DB insertion explicitly...")
    db.add_articles("test_category", [{"title": "DB Test Article", "url": "http://test.com/db", "source": "Test", "published_at": "2023-01-01T00:00:00Z"}])
    
    rows = db.get_history("test_category", limit=5)
    print(f"DB Verification: Found {len(rows)} test articles in SQLite.")
    if rows:
        print(f"Latest DB Entry: {rows[0]['title']}")

if __name__ == "__main__":
    demo_single_fetch()

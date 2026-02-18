"""
Script to verify the News Pipeline.
Checks:
- All categories return results.
- Domains are within the whitelist (for search categories).
- Caching works (second run should accept cache).
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from kshiked.pulse.news import get_news_ingestor, TRUSTED_DOMAINS
from kshiked.pulse.config import NewsAPIConfig

def main():
    print("--- Verifying News Pipeline ---")
    
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    ingestor = get_news_ingestor()
    
    # 1. Fetch All
    print("\n1. Fetching All Categories (this may take time)...")
    results = ingestor.fetch_all(force=True)
    
    total_articles = 0
    extracted_articles = 0
    categories_found = []
    
    for category, articles in results.items():
        count =_len = len(articles)
        print(f"   [{category.upper()}] Found {count} articles")
            if count > 0:
                total_articles += count
                categories_found.append(category)
            
            # Print first article
            first = articles[0]
            print(f"      HEADLINE: {first['title']}")
            print(f"      SOURCE: {first['source']}")
            
            # Check Domain Whitelist for non-native categories
            # (Business/Tech etc use native which might return other sources, but Search ones use whitelist)
                if category in ["politics", "economics", "agriculture"]:
                    if any(chk in first['url'] for chk in TRUSTED_DOMAINS):
                        print("      [PASS] Source in whitelist")
                    else:
                        print(f"      [WARN] Source not in whitelist logic? URL: {first['url']}")

                # Verify extraction metadata
                with_extracted = [a for a in articles if a.get("extracted_text")]
                extracted_articles += len(with_extracted)
                print(f"      Extraction coverage: {len(with_extracted)}/{len(articles)}")

    print(f"\nTotal Articles: {total_articles}")
    print(f"With extracted_text: {extracted_articles}")
    print(f"Categories with Content: {len(categories_found)}/{len(results)}")
    
    # 2. Test Cache
    print("\n2. Testing Cache (Immediate Second Fetch)...")
    start_time = os.times().elapsed
    results_cached = ingestor.fetch_all(force=False)
    end_time = os.times().elapsed
    
    if (end_time - start_time) < 1.0:
        print(f"   [PASS] Cache hit! Took {end_time - start_time:.4f}s")
    else:
        print(f"   [WARN] Slow response, might have re-fetched? Took {end_time - start_time:.4f}s")

if __name__ == "__main__":
    main()

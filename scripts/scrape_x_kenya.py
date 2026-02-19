"""
Run the Kenya X/Twitter scraper.

Usage:
    # One-time scrape (uses saved cookies or env vars for auth)
    python scripts/scrape_x_kenya.py

    # First run — provide credentials
    python scripts/scrape_x_kenya.py --user myhandle --pw mypass --email me@mail.com

    # Custom queries
    python scripts/scrape_x_kenya.py --queries "Kenya budget" "Ruto economy"

    # Scrape specific accounts
    python scripts/scrape_x_kenya.py --timeline KenyaRedCross NationAfrica

    # Rotate proxies + sessions from file and resume
    python scripts/scrape_x_kenya.py \\
      --session-config config/x_sessions.json \\
      --checkpoint data/pulse/x_scraper_checkpoint.json --resume \\
      --conservative-mode --detection-cooldown-hours 24

Environment variables (alternative to CLI args):
    X_USERNAME  – X/Twitter username or handle
    X_PASSWORD  – X/Twitter password
    X_EMAIL     – Email associated with the account
    X_PROXIES   – Comma-separated proxy URLs (optional)
    X_SESSION_COOKIES – Comma-separated cookie paths (optional)
    X_SESSION_CONFIG  – Path to JSON session config (optional)
    X_CHECKPOINT_PATH – Checkpoint path (optional)
    X_RESUME_CHECKPOINT – Set to 1/true/yes to auto-resume
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from kshiked.pulse.scrapers.x_web_scraper import _main

if __name__ == "__main__":
    asyncio.run(_main())

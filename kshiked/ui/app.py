"""
K-Scarcity Dashboard — App Router

Redirects everything to the SENTINEL Command Center v2.0
"""

import sys
from pathlib import Path

# Ensure ui/ is on the path for relative imports
UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR))

from sentinel_dashboard import main

if __name__ == "__main__":
    main()

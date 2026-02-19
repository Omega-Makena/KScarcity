"""
SENTINEL Command Center Dashboard v2.0

Thin wrapper -- all logic lives in the sentinel/ subpackage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ui/ directory is on sys.path for sibling imports
_UI_DIR = Path(__file__).resolve().parent
if str(_UI_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_DIR))

# Ensure project root is on sys.path for package imports
_PROJECT_ROOT = _UI_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel import render_sentinel_dashboard, main  # noqa: E402

__all__ = ["render_sentinel_dashboard", "main"]

if __name__ == "__main__":
    main()

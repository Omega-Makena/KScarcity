"""Shared imports and constants used by all sentinel submodules."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Ensure ui/ directory is on sys.path for sibling imports (theme, data_connector, etc.)
_UI_DIR = Path(__file__).resolve().parent.parent
if str(_UI_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_DIR))

# Ensure project root is on sys.path for package imports (kshiked, scarcity, etc.)
_PROJECT_ROOT = _UI_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import streamlit as st
    import streamlit.components.v1 as components
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None
    components = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None

from theme import DARK_THEME, LIGHT_THEME, THREAT_LEVELS, generate_css, get_plotly_theme
from kshiked.ui.connector import get_dashboard_data, DashboardData

logger = logging.getLogger("sentinel.dashboard")

# Kenya county coordinates for map
KENYA_COUNTIES = {
    "Nairobi": {"lat": -1.2921, "lon": 36.8219},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    "Kisumu": {"lat": -0.0917, "lon": 34.7680},
    "Nakuru": {"lat": -0.3031, "lon": 36.0666},
    "Eldoret": {"lat": 0.5143, "lon": 35.2698},
    "Garissa": {"lat": -0.4533, "lon": 39.6460},
    "Meru": {"lat": 0.0515, "lon": 37.6493},
    "Nyeri": {"lat": -0.4246, "lon": 36.9514},
    "Machakos": {"lat": -1.5177, "lon": 37.2634},
    "Kisii": {"lat": -0.6698, "lon": 34.7660},
    "Turkana": {"lat": 3.1167, "lon": 35.5667},
    "Wajir": {"lat": 1.7500, "lon": 40.0667},
    "Mandera": {"lat": 3.9167, "lon": 41.8500},
    "Kakamega": {"lat": 0.2833, "lon": 34.7500},
    "Bungoma": {"lat": 0.5667, "lon": 34.5500},
}

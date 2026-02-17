"""
Home Landing Page

The main SENTINEL landing page with hero section and 4 top-level cards:
K-SHIELD, K-PULSE, K-COLLAB, K-EDUCATION.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.landing import render_landing


# Top-level cards — each leads to a module with its own auth gate + landing
TOP_CARDS = [
    (
        "K-SHIELD",
        "Run large-scale economic simulations across sectors. "
        "Test policy scenarios, model shocks, and evaluate cascading risk "
        "using adaptive ABM agents.",
        "KSHIELD",
    ),
    (
        "K-PULSE",
        "Continuously ingest and analyze live signals to detect anomalies. "
        "Monitor behavioral shifts and generate early warning intelligence "
        "in real-time.",
        "KPULSE",
    ),
    (
        "K-COLLAB",
        "Enable organizations to collaboratively train models and generate "
        "insights using federated learning and secure aggregation.",
        "KCOLLAB",
    ),
    (
        "K-EDUCATION",
        "Translate complex security intelligence into clear public knowledge "
        "through explainable analytics and accessible awareness dashboards.",
        "KEDUCATION",
    ),
]


def render(theme):
    """Render the SENTINEL home landing page."""
    render_landing(
        theme=theme,
        title="WELCOME TO SENTINEL",
        subtitle="The Autonomous Economic Defense & Simulation Platform",
        tagline="— Powered by Scarcity —",
        cards=TOP_CARDS,
    )

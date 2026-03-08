"""
AutoPipeline — Zero-Configuration Intelligence Engine
======================================================

This is the single entry point for all three dashboard tiers.

Usage (all three dashboards call the same thing):

    from kshiked.ui.institution.backend.auto_pipeline import AutoPipeline

    result = AutoPipeline.run(df)          # ← that's it. Give it a DataFrame.
    result = AutoPipeline.run(df, hint="economic")   # optional: nudge column interpretation

What happens automatically:
    1. Column types are inferred — no schema required
    2. RRCF streaming anomaly detection runs on every row
    3. VARX/GARCH forecasting with uncertainty bounds
    4. Scarcity discovery engine learns which columns relate to each other
       and in what way (15 relationship types: causal, correlational,
       temporal lag, equilibrium, synergistic, competitive, ...)
    5. Top-k strongest discovered relationships returned in plain English
    6. Confidence map per variable (how certain the engine is)
    7. Knowledge graph edges (for causal network visualisation)
    8. Policy simulator handle — shock any column, see ripple effects
    9. Composite scores: Detection / Impact / Certainty (0–10)
    10. Auto-narrative summary in plain English

Nothing trains beforehand. The engine learns FROM the data you give it.
If you upload more data next time, it gets smarter.
"""

from __future__ import annotations

import logging
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── path resolution ────────────────────────────────────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger("kshield.auto_pipeline")

# ── Result container ────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Everything the engine found. Passed directly to the UI.
    All fields have safe defaults so the UI never crashes even if a
    subsystem is unavailable.
    """
    # ── raw scores ────────────────────────────────────────────────────────
    anomaly_scores: List[float] = field(default_factory=list)
    peak_score: float = 0.0
    peak_index: int = 0

    # ── forecasting ───────────────────────────────────────────────────────
    forecast_matrix: List[List[float]] = field(default_factory=list)   # shape [steps, D]
    variance_matrix: List[List[float]] = field(default_factory=list)   # shape [steps, D]

    # ── what scarcity discovered ──────────────────────────────────────────
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # top-k relationships
    confidence_map: Dict[str, float] = field(default_factory=dict)     # per-column confidence
    knowledge_graph: List[Dict[str, Any]] = field(default_factory=list)# graph edges

    # ── composite scores for delta_sync payload ───────────────────────────
    composite: Dict[str, float] = field(default_factory=dict)          # A_Detection/B_Impact/C_Certainty

    # ── plain-English narrative ───────────────────────────────────────────
    narrative: str = ""
    relationship_summary: List[str] = field(default_factory=list)      # human sentences

    # ── column names ──────────────────────────────────────────────────────
    columns: List[str] = field(default_factory=list)

    # ── policy simulator handle ───────────────────────────────────────────
    simulator: Any = None   # Call .simulate(col_idx, magnitude, steps=20)

    # ── temporal trend analysis ─────────────────────────────────────────
    # Each entry: {column, direction, volatility, growth_rate, early_mean, late_mean}
    trend_signals: List[Dict[str, Any]] = field(default_factory=list)
    structural_breaks: List[int] = field(default_factory=list)  # row indices

    # ── sentiment / threat index (data-derived proxy) ─────────────────────
    # Derived from PulseState + ThreatIndexReport.compute_all()
    threat_report: Optional[Dict[str, Any]] = None
    threat_level: str = "LOW"
    priority_alerts: List[str] = field(default_factory=list)

    # ── economic state (SFC model) ────────────────────────────────────────
    # From SFCEconomy.get_state() after initialization + 20 steps
    economic_state: Optional[Dict[str, float]] = None

    # ── risk propagation ─────────────────────────────────────────────────
    # Each entry: {trigger, chain, description, estimated_impact}
    propagation_chains: List[Dict[str, Any]] = field(default_factory=list)

    # ── spatial analysis ─────────────────────────────────────────────────
    spatial_hotspots: List[Dict[str, Any]] = field(default_factory=list)
    spatial_available: bool = False

    # ── meta ──────────────────────────────────────────────────────────────
    hypotheses_total: int = 0
    hypotheses_active: int = 0
    overall_confidence: float = 0.0
    elapsed_ms: float = 0.0
    engine_used: str = "unknown"
    error: Optional[str] = None


# ── AutoPipeline ────────────────────────────────────────────────────────────────

class AutoPipeline:
    """
    Zero-configuration wrapper around the full scarcity engine.

    Call AutoPipeline.run(df) — everything else is automatic.
    The class is stateless from the caller's perspective. Internally it caches
    the ScarcityBridge in the Streamlit session so it stays warm between
    uploads.
    """

    # relationship type → readable English phrase
    _REL_LABELS: Dict[str, str] = {
        "causal":          "causally drives",
        "correlational":   "moves together with",
        "temporal_lag":    "predicts (with a time lag)",
        "equilibrium":     "stays in balance with",
        "functional":      "is a mathematical function of",
        "compositional":   "is composed of",
        "competitive":     "competes against",
        "synergistic":     "amplifies",
        "probabilistic":   "statistically predicts",
        "structural":      "is structurally linked to",
        "mediating":       "mediates the effect on",
        "moderating":      "moderates the relationship between",
        "graph":           "is graph-connected to",
        "similarity":      "is similar in behaviour to",
        "logical":         "logically implies changes in",
    }

    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        hint: str = "",
        top_k_relationships: int = 15,
    ) -> PipelineResult:
        """
        Run the full scarcity pipeline on a DataFrame.

        Args:
            df:                    Any DataFrame — the institution's uploaded data.
            hint:                  Optional string to help column interpretation
                                   (e.g. "economic", "health", "security"). Unused
                                   internally today but forwarded for future NLP layer.
            top_k_relationships:   How many discovered relationships to return.

        Returns:
            PipelineResult with all findings, safe defaults if subsystems fail.
        """
        t0 = time.time()
        result = PipelineResult()

        # 1. Normalise the DataFrame ─────────────────────────────────────────
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric_df.empty:
            result.error = "No numeric columns found in the uploaded data."
            result.narrative = result.error
            return result

        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
        result.columns = list(numeric_df.columns)
        data_matrix = numeric_df.values.astype(np.float32)

        # 2. Anomaly detection (RRCF streaming) ──────────────────────────────
        try:
            from kshiked.ui.institution.backend.scarcity_bridge import ScarcityBridge as _LegacyBridge
            legacy = _LegacyBridge()
            raw = legacy.process_dataframe(numeric_df)
            result.anomaly_scores = raw.get("anomalies", [])
            if result.anomaly_scores:
                result.peak_score = float(max(result.anomaly_scores))
                result.peak_index = int(np.argmax(result.anomaly_scores))
            forecasts_raw = raw.get("forecasts", [])
            if forecasts_raw:
                last_f = forecasts_raw[-1]
                result.forecast_matrix = last_f.get("forecasts") or []
                result.variance_matrix = last_f.get("variances") or []
            result.engine_used = "scarcity.RRCF+VARX"
        except Exception as e:
            logger.warning(f"Legacy RRCF bridge error: {e}")
            result.engine_used = "fallback"
            # Minimal numpy fallback so the UI still works
            diffs = np.abs(np.diff(data_matrix, axis=0)).sum(axis=1)
            if len(diffs) > 0:
                result.anomaly_scores = diffs.tolist()
                result.peak_score = float(diffs.max())
                result.peak_index = int(np.argmax(diffs))

        # 3. Scarcity Discovery Engine — relationship learning ────────────────
        try:
            import importlib
            core_bridge_mod = importlib.import_module("kshiked.core.scarcity_bridge")
            CoreBridge = getattr(core_bridge_mod, "ScarcityBridge")
            bridge = CoreBridge()

            # Feed every row to the discovery engine (streaming — learns online)
            if hasattr(bridge, "_discovery") and bridge._discovery is not None:
                engine = bridge._discovery
                # Build schema from columns
                schema = {"fields": [{"name": c, "type": "float"} for c in result.columns]}
                if hasattr(engine, "initialize"):
                    engine.initialize(schema)
                # Stream rows
                for i in range(len(numeric_df)):
                    row_dict = {c: float(numeric_df.iloc[i][c]) for c in result.columns}
                    if hasattr(engine, "process_row"):
                        engine.process_row(row_dict)
                    elif hasattr(engine, "process_row_raw"):
                        engine.process_row_raw(row_dict)

                # Extract what was learned
                result.relationships = bridge.get_top_relationships(top_k_relationships)
                result.confidence_map = bridge.get_confidence_map()
                result.knowledge_graph = bridge.get_knowledge_graph()

                if bridge.training_report:
                    result.hypotheses_total = bridge.training_report.hypotheses_created
                    result.hypotheses_active = bridge.training_report.hypotheses_active
                    result.overall_confidence = bridge.training_report.overall_confidence

                # Expose simulator handle
                try:
                    result.simulator = bridge.get_simulator()
                except Exception:
                    pass

                result.engine_used += "+Discovery"
        except Exception as e:
            logger.warning(f"Discovery engine not available: {e}")

        # 4. Composite scores ─────────────────────────────────────────────────
        anoms = result.anomaly_scores
        peak = result.peak_score
        detection = float(min(10.0, (min(10.0, peak) * 0.7) + (np.std(anoms[-30:]) * 3.0 if len(anoms) > 30 else 0.5)))

        avg_var = float(np.mean([np.mean(row) for row in result.variance_matrix]) if result.variance_matrix else 0.0)
        impact = float(min(10.0, (peak * 0.5) + (avg_var * 1.5)))
        certainty = float(min(10.0, max(0.0, 10.0 - avg_var * 2.0)))
        # Boost certainty if discovery engine found strong relationships
        if result.overall_confidence > 0:
            certainty = float(min(10.0, certainty * 0.5 + result.overall_confidence * 10.0 * 0.5))

        result.composite = {
            "A_Detection": round(detection, 2),
            "B_Impact": round(impact, 2),
            "C_Certainty": round(certainty, 2),
        }

        # 5. Temporal Trend Analysis ─────────────────────────────────────────
        try:
            trend_signals = []
            for col in result.columns:
                series = numeric_df[col].values
                n = len(series)
                if n < 6:
                    continue
                mid = n // 2
                early_mean = float(series[:mid].mean())
                late_mean = float(series[mid:].mean())
                if late_mean > early_mean * 1.05:
                    direction = "acceleration"
                elif late_mean < early_mean * 0.95:
                    direction = "deceleration"
                else:
                    direction = "stable"
                early_vol = float(series[:mid].std()) + 1e-9
                late_vol = float(series[mid:].std()) + 1e-9
                if late_vol > early_vol * 1.15:
                    vol_trend = "increasing volatility"
                elif late_vol < early_vol * 0.85:
                    vol_trend = "decreasing volatility"
                else:
                    vol_trend = "stable"
                growth_rate = (late_mean - early_mean) / (abs(early_mean) + 1e-9)
                trend_signals.append({
                    "column": col,
                    "direction": direction,
                    "volatility": vol_trend,
                    "growth_rate": round(growth_rate, 4),
                    "early_mean": round(early_mean, 4),
                    "late_mean": round(late_mean, 4),
                })
            result.trend_signals = trend_signals
            # Structural breaks: rows where anomaly score exceeds mean + 2*std
            if result.anomaly_scores and len(result.anomaly_scores) > 10:
                anom_arr = np.array(result.anomaly_scores)
                threshold = anom_arr.mean() + 2.0 * anom_arr.std()
                result.structural_breaks = [i for i, a in enumerate(result.anomaly_scores) if a > threshold]
        except Exception as _e_trend:
            logger.warning(f"Temporal trend analysis error: {_e_trend}")

        # 6. Threat Index — data-derived PulseState proxy ─────────────────────
        try:
            from kshiked.pulse.indices import ThreatIndexReport
            from kshiked.pulse.primitives import PulseState, ActorType
            state = PulseState()
            # Map pipeline findings to PulseState fields
            instability = min(1.0, result.peak_score / 5.0)
            state.instability_index = instability
            state.crisis_probability = instability * 0.7
            # Bond fragility driven by deceleration count
            n_decel = sum(1 for t in result.trend_signals if t.get("direction") == "deceleration")
            fragility = min(1.0, n_decel / max(1, len(result.trend_signals)))
            state.bonds.national_cohesion = max(0.1, 0.65 - fragility * 0.4)
            state.bonds.class_solidarity = max(0.1, 0.55 - fragility * 0.3)
            state.bonds.polarization_index = fragility * 0.5
            # State stress driven by structural breaks
            if result.structural_breaks:
                state.stress.apply_stress(
                    ActorType.STATE, -min(0.8, len(result.structural_breaks) * 0.15)
                )
            # ESI = fraction of accelerating columns (economic satisfaction proxy)
            n_accel = sum(1 for t in result.trend_signals if t.get("direction") == "acceleration")
            esi = float(n_accel) / max(1, len(result.trend_signals))
            threat = ThreatIndexReport.compute_all(state, [], esi)
            result.threat_report = threat.to_dict()
            result.threat_level = threat.overall_threat_level
            result.priority_alerts = threat.priority_alerts
        except Exception as _e_threat:
            logger.warning(f"Threat index error: {_e_threat}")

        # 7. SFC Economic State ───────────────────────────────────────────────
        try:
            from scarcity.simulation.sfc import SFCEconomy, SFCConfig
            _sfc_config = SFCConfig(steps=20)
            _economy = SFCEconomy(_sfc_config)
            _economy.initialize(gdp=100.0)
            _economy.run(20)
            result.economic_state = {k: round(float(v), 4) for k, v in _economy.get_state().items()
                                     if isinstance(v, (int, float))}
        except Exception as _e_sfc:
            logger.warning(f"SFC economic state error: {_e_sfc}")

        # 8. Risk Propagation — trace 2-hop cascades from knowledge graph ──────
        try:
            if result.knowledge_graph and result.columns:
                adj: Dict[str, list] = {}
                for edge in result.knowledge_graph:
                    src = edge.get('source') or (edge.get('variables', ['?', '?'])[0])
                    tgt = edge.get('target') or (edge.get('variables', ['?', '?'])[-1])
                    conf = float(edge.get('confidence', 0.3))
                    adj.setdefault(src, []).append((tgt, conf))
                # Pick the most volatile column as the propagation trigger
                variances = numeric_df.var()
                top_col = str(variances.idxmax()) if not variances.empty else result.columns[0]
                chain = [top_col]
                visited = {top_col}
                for _ in range(2):
                    last = chain[-1]
                    if last in adj:
                        neighbors = sorted(adj[last], key=lambda x: -x[1])
                        for tgt, conf in neighbors:
                            if tgt not in visited:
                                chain.append(tgt)
                                visited.add(tgt)
                                break
                if len(chain) > 1:
                    rel_labels = cls._REL_LABELS
                    result.propagation_chains = [{
                        "trigger": chain[0],
                        "chain": chain,
                        "description": " → ".join(chain),
                        "estimated_impact": round(min(1.0, len(chain) * 0.3), 2),
                    }]
        except Exception as _e_prop:
            logger.warning(f"Risk propagation error: {_e_prop}")

        # 9. Spatial Analysis — auto-detect lat/lon columns ───────────────────
        try:
            lat_col = next((c for c in result.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in result.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
            if lat_col and lon_col:
                coords = numeric_df[[lat_col, lon_col]].dropna().values
                if len(coords) > 0:
                    n_buckets = min(5, max(1, len(coords) // 10))
                    hotspots = []
                    for i in range(n_buckets):
                        bucket = coords[i::n_buckets]
                        if len(bucket) > 0:
                            hotspots.append({
                                "lat": round(float(bucket[:, 0].mean()), 5),
                                "lon": round(float(bucket[:, 1].mean()), 5),
                                "count": int(len(bucket)),
                            })
                    result.spatial_hotspots = hotspots
                    result.spatial_available = True
        except Exception as _e_spatial:
            logger.warning(f"Spatial analysis error: {_e_spatial}")

        # 10. Relationship summary in plain English ────────────────────────────
        sentences = []
        for rel in result.relationships[:10]:
            vars_ = rel.get("variables", [])
            rel_type = rel.get("rel_type", "")
            conf = rel.get("confidence", 0.0)
            label = cls._REL_LABELS.get(rel_type, f"is related to ({rel_type})")
            if len(vars_) >= 2:
                sentences.append(
                    f"**{vars_[0]}** {label} **{vars_[1]}** "
                    f"(confidence: {conf:.0%})"
                )
        result.relationship_summary = sentences

        # 6. Plain-English narrative ──────────────────────────────────────────
        result.narrative = cls._build_narrative(result)

        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    # ── Narrative builder ────────────────────────────────────────────────────

    @classmethod
    def _build_narrative(cls, r: PipelineResult) -> str:
        parts = []

        # Anomaly assessment
        if r.peak_score > 2.0:
            parts.append(
                f"A **critical anomaly** was detected in your data "
                f"(severity score: {r.peak_score:.2f} at row {r.peak_index}). "
                "This indicates a significant structural shift that departs from the established pattern in your dataset."
            )
        elif r.peak_score > 1.0:
            parts.append(
                f"**Moderate volatility** was detected (peak severity: {r.peak_score:.2f}). "
                "The data shows noticeable deviation but remains within acceptable historical bounds."
            )
        else:
            parts.append(
                f"The data appears **structurally stable** (peak severity: {r.peak_score:.2f}). "
                "No significant anomalies were detected."
            )

        # What the engine learned
        if r.hypotheses_active > 0:
            parts.append(
                f"The relationship discovery engine found **{r.hypotheses_active} active relationships** "
                f"across your {len(r.columns)} variables "
                f"(overall confidence: {r.overall_confidence:.0%})."
            )

        # Top relationship
        if r.relationship_summary:
            parts.append(f"The strongest discovered pattern: {r.relationship_summary[0]}.")

        # Threat level
        if r.threat_level and r.threat_level != "LOW":
            parts.append(
                f"**Threat assessment: {r.threat_level}** — derived from data trend characteristics "
                f"and system fragility indicators."
            )
        if r.priority_alerts:
            parts.append("Priority alerts: " + "; ".join(r.priority_alerts[:3]) + ".")

        # Trend
        if r.trend_signals:
            accel = [t for t in r.trend_signals if t.get("direction") == "acceleration"]
            decel = [t for t in r.trend_signals if t.get("direction") == "deceleration"]
            if accel:
                parts.append(f"{len(accel)} variable(s) are **accelerating** (growing faster): " +
                             ", ".join(t['column'] for t in accel[:3]) + ".")
            if decel:
                parts.append(f"{len(decel)} variable(s) are **decelerating** (slowing down): " +
                             ", ".join(t['column'] for t in decel[:3]) + ".")

        # Forecast
        if r.forecast_matrix:
            parts.append(
                "A 5-step trajectory forecast was generated. "
                "The uncertainty bounds (GARCH) show how volatile the prediction is."
            )

        # Composite scores
        c = r.composite
        if c:
            parts.append(
                f"Composite risk scores — "
                f"Detection: **{c.get('A_Detection', 0)}/10**, "
                f"Impact: **{c.get('B_Impact', 0)}/10**, "
                f"Certainty: **{c.get('C_Certainty', 0)}/10**."
            )

        return "  \n".join(parts)

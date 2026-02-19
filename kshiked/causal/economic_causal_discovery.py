"""
Economic causal discovery runner for KShield.

Uses Scarcity's OnlineDiscoveryEngine (Granger-style causal hypothesis)
to learn directional relationships from the World Bank Kenya dataset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover - handled at runtime
    HAS_PANDAS = False
    pd = None

logger = logging.getLogger("kshield.causal")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "API_KEN_DS2_en_csv_v2_14659.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "economic_causal_graph.json"


DEFAULT_DISPLAY_NAMES: Dict[str, str] = {
    "gdp_current": "GDP",
    "gdp_growth": "GDP Growth",
    "gdp_per_capita": "GDP Per Capita",
    "inflation": "Inflation",
    "inflation_gdp_deflator": "GDP Deflator Inflation",
    "food_price_index": "Food Prices",
    "unemployment": "Unemployment",
    "employment_ratio": "Employment",
    "exports_gdp": "Exports",
    "imports_gdp": "Imports",
    "trade_gdp": "Trade",
    "current_account": "Current Account",
    "govt_consumption": "Gov Spending",
    "tax_revenue": "Taxes",
    "govt_debt": "Public Debt",
    "real_interest_rate": "Interest Rate",
    "broad_money": "Money Supply",
    "private_credit": "Credit Supply",
    "population": "Population",
    "urban_population": "Urban Population",
    "electricity_access": "Electricity Access",
    "internet_users": "Internet Users",
    "mobile_subscriptions": "Mobile Subscriptions",
}


@dataclass
class EconomicCausalConfig:
    data_path: Path = DEFAULT_DATA_PATH
    indicators: Optional[List[str]] = None
    start_year: int = 1990
    min_coverage: float = 0.6
    max_indicators: int = 40
    use_display_names: bool = True
    explore_interval: int = 10
    use_causal: bool = True
    min_confidence: float = 0.7
    top_k: int = 50


@dataclass
class EconomicCausalState:
    engine: Any = None
    config: Optional[EconomicCausalConfig] = None
    cached_results: Optional[Dict[str, Any]] = None
    last_signature: Optional[Tuple[float, int, Tuple[str, ...]]] = None
    last_years: Optional[List[int]] = None


_STATE: Optional[EconomicCausalState] = None


def _get_state() -> EconomicCausalState:
    global _STATE
    if _STATE is None:
        _STATE = EconomicCausalState()
    return _STATE


def _config_signature(config: EconomicCausalConfig) -> Tuple:
    return (
        tuple(config.indicators) if config.indicators else None,
        config.start_year,
        config.min_coverage,
        config.max_indicators,
        config.use_display_names,
        config.explore_interval,
        config.use_causal,
        config.min_confidence,
        config.top_k,
        str(config.data_path),
    )


def load_economic_dataframe(
    config: EconomicCausalConfig,
) -> "pd.DataFrame":
    """Load and filter the Kenya macro dataset as a wide time series."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required to load the economic dataset")

    try:
        from kshiked.ui.kenya_data_loader import KenyaEconomicDataLoader, KEY_INDICATORS
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("kenya_data_loader not available") from exc

    loader = KenyaEconomicDataLoader(config.data_path)
    if not loader.load():
        raise RuntimeError(f"Failed to load data from {config.data_path}")

    if config.indicators:
        indicators = list(config.indicators)
    else:
        indicators = list(KEY_INDICATORS.values())

    df = loader.get_historical_trajectory(indicators, start_year=config.start_year)
    df = df.sort_index()

    if config.use_display_names:
        df = df.rename(columns={k: DEFAULT_DISPLAY_NAMES.get(k, k) for k in df.columns})

    if config.min_coverage > 0:
        coverage = df.notna().mean().sort_values(ascending=False)
        keep = coverage[coverage >= config.min_coverage]
        if config.max_indicators:
            keep = keep.head(config.max_indicators)
        df = df[keep.index]

    return df


def _iter_rows(df: "pd.DataFrame") -> Iterable[Dict[str, float]]:
    """Yield per-row dicts without NaNs for streaming ingestion."""
    for _, row in df.iterrows():
        row_dict = {}
        for k, v in row.items():
            if pd.isna(v):
                continue
            row_dict[str(k)] = float(v)
        if row_dict:
            yield row_dict


def run_economic_causal_discovery(
    config: Optional[EconomicCausalConfig] = None,
) -> Dict[str, Any]:
    """Run Scarcity causal discovery and return a graph payload."""
    if config is None:
        config = EconomicCausalConfig()

    df = load_economic_dataframe(config)
    if df.empty:
        raise RuntimeError("No data available after filtering")

    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    engine = OnlineDiscoveryEngine(explore_interval=config.explore_interval)
    schema = {"fields": [{"name": str(c), "type": "float"} for c in df.columns]}
    engine.initialize_v2(schema, use_causal=config.use_causal)

    for row_dict in _iter_rows(df):
        engine.process_row(row_dict)

    graph = engine.get_knowledge_graph()
    edges = extract_causal_edges(
        graph,
        min_confidence=config.min_confidence,
        top_k=config.top_k,
    )

    nodes, links = build_force_graph(edges)
    return {
        "nodes": nodes,
        "links": links,
        "edges": edges,
        "columns": list(df.columns),
    }


def extract_causal_edges(
    graph: List[Dict[str, Any]],
    min_confidence: float = 0.7,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Filter Scarcity graph results down to causal edges with direction."""
    edges: List[Dict[str, Any]] = []
    for item in graph:
        if item.get("type") != "causal":
            continue
        metrics = item.get("metrics", {}) or {}
        conf = float(metrics.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        direction = int(metrics.get("direction", 1) or 1)
        source = item.get("source")
        target = item.get("target")
        variables = item.get("variables", []) or []

        if source and target:
            cause, effect = (source, target) if direction == 1 else (target, source)
        elif len(variables) >= 2:
            cause, effect = (variables[0], variables[1]) if direction != -1 else (variables[1], variables[0])
        else:
            continue

        gain_fwd = float(metrics.get("gain_forward", 0.0))
        gain_bwd = float(metrics.get("gain_backward", 0.0))
        strength = max(gain_fwd, gain_bwd)
        lag = int(metrics.get("lag", 2))

        edges.append({
            "cause": str(cause),
            "effect": str(effect),
            "confidence": conf,
            "strength": strength,
            "lag": lag,
        })

    edges.sort(key=lambda e: e["confidence"], reverse=True)
    return edges[:top_k]


def _infer_group(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["gdp", "consumption", "investment", "production"]):
        return "Real"
    if any(k in n for k in ["inflation", "price", "energy", "deflator"]):
        return "Price"
    if any(k in n for k in ["tax", "debt", "deficit", "spending"]):
        return "Fiscal"
    if any(k in n for k in ["money", "interest", "credit"]):
        return "Monetary"
    if any(k in n for k in ["export", "import", "trade", "current account"]):
        return "External"
    if any(k in n for k in ["employment", "unemployment", "wage"]):
        return "Labor"
    if any(k in n for k in ["population", "urban", "internet", "mobile", "electricity"]):
        return "Social"
    return "Other"


def build_force_graph(
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert causal edges to nodes/links for the 3D force graph UI."""
    nodes: Dict[str, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []

    for edge in edges:
        cause = edge["cause"]
        effect = edge["effect"]
        conf = float(edge.get("confidence", 0.0))
        strength = float(edge.get("strength", 0.0))

        if cause not in nodes:
            nodes[cause] = {"id": cause, "group": _infer_group(cause), "val": 1}
        if effect not in nodes:
            nodes[effect] = {"id": effect, "group": _infer_group(effect), "val": 1}

        nodes[cause]["val"] = nodes[cause].get("val", 1) + 1
        nodes[effect]["val"] = nodes[effect].get("val", 1) + 1

        links.append({
            "source": cause,
            "target": effect,
            "value": strength,
            "width": 1 + conf * 3,
            "color": "#f59e0b",
        })

    return list(nodes.values()), links


def build_granger_results_from_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert causal edges into a Granger-style results list for the UI."""
    results: List[Dict[str, Any]] = []
    for edge in edges:
        conf = float(edge.get("confidence", 0.0))
        strength = float(edge.get("strength", 0.0))
        results.append({
            "cause": edge.get("cause"),
            "effect": edge.get("effect"),
            "lag": int(edge.get("lag", 2)),
            "f_stat": strength * 100.0,
            "p_value": max(0.0, min(1.0, 1.0 - conf)),
            "significant": conf >= 0.7,
            "strength": strength,
            "confidence": conf,
        })
    return results


def export_graph_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_dynamic_causal_discovery(
    config: Optional[EconomicCausalConfig] = None,
    force_full: bool = False,
) -> Dict[str, Any]:
    """
    Run discovery dynamically:
    - If data unchanged: return cached results.
    - If data appended with new years: incrementally update engine.
    - If schema/config changed: full retrain.
    """
    if config is None:
        config = EconomicCausalConfig()

    state = _get_state()
    df = load_economic_dataframe(config)
    if df.empty:
        raise RuntimeError("No data available after filtering")

    source_path = config.data_path
    signature = (source_path.stat().st_mtime, source_path.stat().st_size, tuple(df.columns))
    cfg_sig = _config_signature(config)

    source_changed = (
        state.last_signature is None
        or state.last_signature[0] != signature[0]
        or state.last_signature[1] != signature[1]
        or state.last_signature[2] != signature[2]
    )

    # Return cached results when nothing changed.
    if not force_full and state.cached_results and state.last_signature == signature and state.config:
        if _config_signature(state.config) == cfg_sig:
            return state.cached_results

    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    full_retrain = (
        force_full
        or state.engine is None
        or state.config is None
        or _config_signature(state.config) != cfg_sig
        or state.last_signature is None
        or state.last_signature[2] != signature[2]
    )

    if full_retrain:
        engine = OnlineDiscoveryEngine(explore_interval=config.explore_interval)
        schema = {"fields": [{"name": str(c), "type": "float"} for c in df.columns]}
        engine.initialize_v2(schema, use_causal=config.use_causal)
        for row_dict in _iter_rows(df):
            engine.process_row(row_dict)
        state.engine = engine
        state.last_years = list(df.index)
    else:
        # Incremental update for newly appended years
        engine = state.engine
        prev_years = set(state.last_years or [])
        new_rows = df.loc[[y for y in df.index if y not in prev_years]]
        if new_rows.empty and source_changed:
            # Data changed without new years (edits/backfills) -> full retrain.
            engine = OnlineDiscoveryEngine(explore_interval=config.explore_interval)
            schema = {"fields": [{"name": str(c), "type": "float"} for c in df.columns]}
            engine.initialize_v2(schema, use_causal=config.use_causal)
            for row_dict in _iter_rows(df):
                engine.process_row(row_dict)
            state.engine = engine
            state.last_years = list(df.index)
        elif not new_rows.empty:
            for row_dict in _iter_rows(new_rows):
                engine.process_row(row_dict)
            state.last_years = list(df.index)

    graph = engine.get_knowledge_graph()
    edges = extract_causal_edges(
        graph,
        min_confidence=config.min_confidence,
        top_k=config.top_k,
    )
    nodes, links = build_force_graph(edges)
    results = {
        "nodes": nodes,
        "links": links,
        "edges": edges,
        "columns": list(df.columns),
    }

    state.cached_results = results
    state.last_signature = signature
    state.config = config
    return results


def get_cached_or_run(
    config: Optional[EconomicCausalConfig] = None,
    path: Path = DEFAULT_OUTPUT_PATH,
    force: bool = False,
) -> Dict[str, Any]:
    """Load cached results when available; otherwise run and cache."""
    source_path = (config.data_path if config else DEFAULT_DATA_PATH)
    source_mtime: Optional[float] = None
    if source_path.exists():
        source_mtime = source_path.stat().st_mtime

    if not force and path.exists():
        try:
            cache_mtime = path.stat().st_mtime
            if source_mtime is None or source_mtime <= cache_mtime:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    results = run_dynamic_causal_discovery(config, force_full=force)
    if source_mtime is not None:
        results["source_mtime"] = source_mtime
    export_graph_json(results, path)
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    results = get_cached_or_run()
    print(f"Wrote {len(results['edges'])} edges to {DEFAULT_OUTPUT_PATH}")

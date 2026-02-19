"""Surface MPIE runtime status and data entry to the dashboard."""

from typing import Any, Dict, List
import io

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.dependencies import get_engine_runner
from app.engine import EngineRunner
from app.schemas.mpie import EngineSimulationFrame, EngineStatus

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]


router = APIRouter()


@router.get("/status", response_model=EngineStatus)
async def engine_status(runner: EngineRunner = Depends(get_engine_runner)) -> EngineStatus:
    """Return the latest MPIE metrics and resource hints."""

    if runner is None:
        raise HTTPException(status_code=503, detail="Engine runner not initialised.")
    snapshot = await runner.snapshot()
    return EngineStatus(**snapshot)


@router.get("/simulation", response_model=EngineSimulationFrame)
async def engine_simulation_frame(runner: EngineRunner = Depends(get_engine_runner)) -> EngineSimulationFrame:
    """Return the latest rendered simulation frame."""

    if runner is None:
        raise HTTPException(status_code=503, detail="Engine runner not initialised.")
    payload = await runner.simulation_snapshot()
    return EngineSimulationFrame(**payload)


def _select_feature_matrix(df: "pd.DataFrame") -> np.ndarray:
    """
    Convert an arbitrary tabular dataset into a numeric feature matrix.

    Strategy:
    - Keep only numeric columns.
    - Drop columns that are entirely NaN.
    - Drop rows that become all-NaN across the selected features.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError("No numeric columns found in uploaded dataset.")
    numeric_df = numeric_df.dropna(axis=0, how="all")
    if numeric_df.empty:
        raise ValueError("No non-empty rows remain after cleaning uploaded dataset.")
    return numeric_df.to_numpy(dtype=np.float32)


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    runner: EngineRunner = Depends(get_engine_runner),
) -> Dict[str, Any]:
    """
    Upload a dataset (Excel or CSV) and stream it through the engine.

    This allows the UI to supply a plain spreadsheet while the backend handles
    feature extraction and windowing.
    """
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine runner not initialised.")
    if pd is None:
        raise HTTPException(
            status_code=500,
            detail="pandas is required for dataset uploads but is not installed on the backend.",
        )

    contents = await file.read()
    filename = file.filename or ""
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))  # type: ignore[arg-type]
        else:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="ignore")))  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {exc}") from exc

    try:
        features = _select_feature_matrix(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = await runner.ingest_feature_matrix(features)
    return {
        "filename": filename,
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "windows_ingested": int(stats.get("windows_ingested", 0)),
    }


class SyntheticConfig(BaseModel):
    rows: int = 512
    features: int = 16
    sectors: int = 4


@router.post("/generate-synthetic")
async def generate_synthetic(
    config: SyntheticConfig,
    runner: EngineRunner = Depends(get_engine_runner),
) -> Dict[str, Any]:
    """
    Generate a synthetic dataset on the fly and stream it through the engine.

    Creates a small number of latent sector signals and maps them onto feature
    groups so the hypergraph and simulation have visible structure.
    """
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine runner not initialised.")

    rows = max(1, config.rows)
    feats = max(1, config.features)
    sectors = max(1, min(config.sectors, feats))

    rng = np.random.default_rng()
    latent = rng.standard_normal(size=(rows, sectors)).astype(np.float32)
    noise = 0.35 * rng.standard_normal(size=(rows, feats)).astype(np.float32)

    features = np.empty((rows, feats), dtype=np.float32)
    for j in range(feats):
        k = j % sectors
        features[:, j] = latent[:, k] + noise[:, j]

    stats = await runner.ingest_feature_matrix(features)
    return {
        "filename": "synthetic",
        "rows": rows,
        "features": feats,
        "windows_ingested": int(stats.get("windows_ingested", 0)),
        "sectors": sectors,
    }


@router.get("/insights")
async def engine_insights(runner: EngineRunner = Depends(get_engine_runner)) -> Dict[str, Any]:
    """
    Return high-level forecasting and classification insights derived from the engine.

    - Forecasting: trends and simple next-step projections for key metrics.
    - Classification: top hubs, strongest links, and domain breakdown from the hypergraph store.
    """
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine runner not initialised.")

    snapshot = await runner.snapshot()
    history: List[Dict[str, Any]] = snapshot.get("history", [])

    # Forecasting insights: build simple trends for a few core metrics.
    metric_keys = ["accept_rate", "stability_avg", "gain_p50", "gain_p90"]
    forecasting: List[Dict[str, Any]] = []
    for key in metric_keys:
        series: List[Dict[str, Any]] = []
        for entry in reversed(history):
            values = entry.get("values", {})
            if key in values:
                ts = entry.get("timestamp")
                series.append({"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts, "value": values[key]})
        if not series:
            continue
        forecast_next = series[-1]["value"]
        if len(series) >= 2 and isinstance(series[-1]["value"], (int, float)) and isinstance(
            series[-2]["value"], (int, float)
        ):
            last = float(series[-1]["value"])
            prev = float(series[-2]["value"])
            # Simple one-step linear extrapolation.
            forecast_next = last + (last - prev)
        forecasting.append({"name": key, "history": series, "forecast_next": forecast_next})

    # Classification insights: derive from the hypergraph store, if available.
    top_nodes: List[Dict[str, Any]] = []
    top_edges: List[Dict[str, Any]] = []
    domains: List[Dict[str, Any]] = []

    store = getattr(getattr(runner, "orchestrator", None), "store", None)
    if store is not None:
        store_snapshot = store.snapshot()
        nodes = store_snapshot.get("nodes", {})
        edges = store_snapshot.get("edges", {})

        # Compute node degrees and aggregate weights.
        node_stats: Dict[int, Dict[str, float]] = {}
        for raw_id, data in nodes.items():
            try:
                nid = int(raw_id)
            except (TypeError, ValueError):
                nid = raw_id  # type: ignore[assignment]
            node_stats[nid] = {
                "degree": 0.0,
                "weight_sum": 0.0,
            }

        for key_str, payload in edges.items():
            try:
                # keys are "(src, dst)"
                inner = key_str.strip()[1:-1]
                src_s, dst_s = inner.split(",")
                src_id = int(src_s.strip())
                dst_id = int(dst_s.strip())
            except Exception:
                continue
            weight = float(payload.get("weight", 0.0))
            for nid in (src_id, dst_id):
                stats = node_stats.get(nid)
                if stats is None:
                    continue
                stats["degree"] += 1.0
                stats["weight_sum"] += abs(weight)

        # Top nodes by degree and weight_sum.
        scored_nodes: List[Dict[str, Any]] = []
        for raw_id, data in nodes.items():
            try:
                nid = int(raw_id)
            except (TypeError, ValueError):
                nid = raw_id  # type: ignore[assignment]
            stats = node_stats.get(nid)
            if not stats:
                continue
            name = data.get("name", str(raw_id))
            domain = int(data.get("domain", 0))
            scored_nodes.append(
                {
                    "name": name,
                    "domain": domain,
                    "degree": stats["degree"],
                    "weight_sum": stats["weight_sum"],
                }
            )
        scored_nodes.sort(key=lambda x: (x["degree"], x["weight_sum"]), reverse=True)
        top_nodes = scored_nodes[:8]

        # Top edges by |weight| * stability.
        scored_edges: List[Dict[str, Any]] = []
        for key_str, payload in edges.items():
            try:
                inner = key_str.strip()[1:-1]
                src_s, dst_s = inner.split(",")
                src_id = int(src_s.strip())
                dst_id = int(dst_s.strip())
            except Exception:
                continue
            src_data = nodes.get(src_id, {})
            dst_data = nodes.get(dst_id, {})
            src_name = src_data.get("name", str(src_id))
            dst_name = dst_data.get("name", str(dst_id))
            weight = float(payload.get("weight", 0.0))
            stability = float(payload.get("stability", 0.0))
            score = abs(weight) * max(stability, 0.0)
            scored_edges.append(
                {
                    "source": src_name,
                    "target": dst_name,
                    "weight": weight,
                    "stability": stability,
                    "score": score,
                }
            )
        scored_edges.sort(key=lambda x: x["score"], reverse=True)
        top_edges = scored_edges[:8]

        # Domain breakdown.
        domain_counts: Dict[int, int] = {}
        for data in nodes.values():
            domain = int(data.get("domain", 0))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        domains = [{"domain": k, "nodes": v} for k, v in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)]

    return {
        "forecasting": forecasting,
        "classification": {
            "top_nodes": top_nodes,
            "top_edges": top_edges,
            "domains": domains,
        },
    }

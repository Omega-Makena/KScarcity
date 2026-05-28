"""
Chronos zero-shot baseline for the v2 benchmark.

Chronos-T5 (Amazon, 2024) is a pretrained language model for time-series
forecasting. It requires NO training data, NO feature engineering, and NO
graph — it reads historical values and outputs probabilistic forecasts.

This is the correct null hypothesis for Scarcity: if a model that has
implicitly memorised macroeconomic dynamics from global pretraining data
outperforms Scarcity's graph-conditioned forecasts, then the online
discovery adds no value beyond what large-scale pretraining already knows.

Model sizes:
  chronos-t5-tiny   — fastest, weakest
  chronos-t5-small  — good balance (default here)
  chronos-t5-base   — stronger but slower
  chronos-t5-large  — strongest, requires more VRAM

Install:
  pip install chronos-forecasting

Usage in rolling-origin evaluation:
  At each cutoff, feed the historical series (up to and including cutoff).
  Chronos predicts h steps ahead.
  We extract the median (p50) forecast.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import — Chronos is optional
# ---------------------------------------------------------------------------

_CHRONOS_AVAILABLE: Optional[bool] = None
_pipeline_cache: Dict[str, object] = {}


def _try_import() -> bool:
    global _CHRONOS_AVAILABLE
    if _CHRONOS_AVAILABLE is not None:
        return _CHRONOS_AVAILABLE
    try:
        import chronos  # noqa: F401
        import torch    # noqa: F401
        _CHRONOS_AVAILABLE = True
    except ImportError:
        _CHRONOS_AVAILABLE = False
    return _CHRONOS_AVAILABLE


def is_available() -> bool:
    return _try_import()


def get_pipeline(model_name: str = 'amazon/chronos-t5-small'):
    """Load (or return cached) Chronos pipeline."""
    if not _try_import():
        raise ImportError(
            "chronos-forecasting not installed. Run: pip install chronos-forecasting"
        )
    if model_name not in _pipeline_cache:
        from chronos import ChronosPipeline
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _pipeline_cache[model_name] = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        print(f"  [Chronos] Loaded {model_name} on {device}", flush=True)
    return _pipeline_cache[model_name]


# ---------------------------------------------------------------------------
# Rolling-origin forecast
# ---------------------------------------------------------------------------

def chronos_forecast(
    series: np.ndarray,
    horizons: List[int],
    model_name: str = 'amazon/chronos-t5-small',
    n_samples: int = 20,
) -> Dict[int, float]:
    """
    Given a historical series (1-D array up to the cutoff), return median
    Chronos forecasts at each requested horizon.

    Args:
        series:     1-D array of historical values (NaN-filled gaps allowed).
        horizons:   List of forecast steps (e.g. [1, 3, 5, 10]).
        model_name: HuggingFace model ID for Chronos.
        n_samples:  Number of sample paths to draw for the median.

    Returns:
        Dict mapping horizon → median forecast (NaN if unavailable).
    """
    if not _try_import():
        return {h: np.nan for h in horizons}

    if len(series) < 4:
        return {h: np.nan for h in horizons}

    try:
        import torch
        pipeline = get_pipeline(model_name)
        max_h = max(horizons)

        # Replace NaN with linear interpolation for Chronos context
        s = _fill_series(series)
        context = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # (1, T)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            forecast = pipeline.predict(
                context,
                prediction_length=max_h,
                num_samples=n_samples,
                limit_prediction_length=False,
            )  # shape: (1, n_samples, max_h)

        # Median across samples — move to CPU before numpy conversion
        median = np.quantile(forecast[0].cpu().float().numpy(), 0.5, axis=0)  # (max_h,)

        return {h: float(median[h - 1]) for h in horizons}

    except Exception as exc:
        warnings.warn(f"Chronos forecast failed: {exc}")
        return {h: np.nan for h in horizons}


def _fill_series(series: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values; edge NaNs filled with nearest valid."""
    s = series.astype(float).copy()
    nans = np.isnan(s)
    if not nans.any():
        return s
    idx = np.arange(len(s))
    valid = ~nans
    if valid.sum() < 2:
        s[nans] = np.nanmean(s) if valid.any() else 0.0
        return s
    s[nans] = np.interp(idx[nans], idx[valid], s[valid])
    return s

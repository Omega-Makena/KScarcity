"""Time-series validation utilities for the causal pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from scarcity.causal.graph import parse_dot_edges, validate_temporal_edges
from scarcity.causal.specs import EstimandSpec, EstimandType, TimeSeriesPolicy


class TimeSeriesValidationError(RuntimeError):
    """Raised when strict time-series validation fails."""


@dataclass
class TemporalDiagnostics:
    policy: str
    time_column: Optional[str]
    lag: Optional[int]
    mediator_lag: Optional[int]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    temporal_edge_violations: List[Dict[str, Any]] = field(default_factory=list)
    valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "time_column": self.time_column,
            "lag": self.lag,
            "mediator_lag": self.mediator_lag,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "temporal_edge_violations": list(self.temporal_edge_violations),
            "valid": self.valid,
        }


def _append_issue(diag: TemporalDiagnostics, message: str, strict: bool) -> None:
    if strict:
        diag.issues.append(message)
    else:
        diag.warnings.append(message)


def validate_time_series(
    data: pd.DataFrame,
    spec: EstimandSpec,
    dot_text: Optional[str],
    policy: TimeSeriesPolicy,
) -> TemporalDiagnostics:
    if isinstance(policy, str):
        policy = TimeSeriesPolicy(policy)
    strict = policy == TimeSeriesPolicy.STRICT
    diagnostics = TemporalDiagnostics(
        policy=policy.value,
        time_column=spec.time_column,
        lag=spec.lag,
        mediator_lag=spec.mediator_lag,
    )

    if policy == TimeSeriesPolicy.NONE:
        return diagnostics

    if not spec.time_column:
        _append_issue(diagnostics, "time_column is required for time-series policy.", strict)
    elif spec.time_column not in data.columns:
        _append_issue(diagnostics, f"time_column '{spec.time_column}' not found in data.", strict)
    else:
        series = data[spec.time_column]
        if series.isna().any():
            _append_issue(diagnostics, "time_column contains NaNs.", strict)
        try:
            ordered = series.is_monotonic_increasing
        except Exception:
            ordered = False
        if not ordered:
            _append_issue(diagnostics, "time_column is not monotonic increasing.", strict)

        if spec.panel_keys:
            for key in spec.panel_keys:
                if key not in data.columns:
                    _append_issue(diagnostics, f"panel key '{key}' missing from data.", strict)
        else:
            if series.duplicated().any():
                _append_issue(diagnostics, "time_column contains duplicates without panel_keys.", strict)

    if spec.lag is None:
        _append_issue(diagnostics, "lag is required for time-series policy.", strict)
    elif spec.lag <= 0:
        _append_issue(diagnostics, "lag must be positive for time-series policy.", strict)
    elif len(data) <= spec.lag:
        _append_issue(diagnostics, "data length is insufficient for requested lag.", strict)

    if spec.type in (EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE):
        if spec.mediator_lag is None:
            _append_issue(diagnostics, "mediator_lag is required for mediation under time-series policy.", strict)
        elif spec.lag is not None and spec.mediator_lag > spec.lag:
            _append_issue(diagnostics, "mediator_lag must be <= lag for mediation.", strict)

    if dot_text:
        edges = parse_dot_edges(dot_text)
        violations = validate_temporal_edges(edges)
        if violations:
            diagnostics.temporal_edge_violations.extend(violations)
            _append_issue(diagnostics, "temporal DAG validation failed for one or more edges.", strict)
    else:
        _append_issue(diagnostics, "DOT graph required for temporal DAG validation.", strict)

    if diagnostics.issues:
        diagnostics.valid = False
        if strict:
            raise TimeSeriesValidationError("; ".join(diagnostics.issues))

    return diagnostics

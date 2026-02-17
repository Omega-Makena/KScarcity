"""
Causal Engine Orchestrator.

Entrypoint for the Scarcity production causal inference pipeline.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import logging
import os
import platform
import random
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scarcity.causal.artifacts import ArtifactWriter, compute_data_signature
from scarcity.causal.estimation import EstimatorFactory
from scarcity.causal.feature_layer import FeatureBuilder
from scarcity.causal.graph import load_dot, parse_dot_edges, merge_edges
from scarcity.causal.identification import Identifier
from scarcity.causal.reporting import CausalRunResult, EffectArtifact, RunSummary, RuntimeMetadata, SpecError
from scarcity.causal.specs import EstimandSpec, FailPolicy, ParallelismMode, RuntimeSpec
from scarcity.causal.time_series import TimeSeriesValidationError, validate_time_series
from scarcity.causal.validation import Validator

logger = logging.getLogger(__name__)


@dataclass
class _ChunkResult:
    results: List[EffectArtifact]
    errors: List[SpecError]


def run_causal(
    data: pd.DataFrame,
    spec: EstimandSpec | List[EstimandSpec],
    runtime: Optional[RuntimeSpec] = None,
) -> CausalRunResult:
    """
    Run the causal pipeline for a single spec or a list of specs.

    Returns a CausalRunResult bundle that contains successes, failures, and metadata.
    """
    runtime = runtime or RuntimeSpec()
    runtime.normalize()

    specs = _normalize_specs(spec)
    run_id = runtime.run_id or str(uuid.uuid4())
    runtime.run_id = run_id

    start_time = datetime.utcnow()
    logger.info(f"Starting run_causal with {len(specs)} specs (run_id={run_id})")

    data_signature = compute_data_signature(data, time_column=_resolve_time_column(specs))
    metadata = RuntimeMetadata(
        run_id=run_id,
        runtime=runtime.as_dict(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        dowhy_version=_safe_version("dowhy"),
        econml_version=_safe_version("econml"),
        data_signature=data_signature,
    )

    input_dot = _resolve_input_dot(runtime, specs)

    results: List[EffectArtifact] = []
    errors: List[SpecError] = []

    parallelism = _resolve_parallelism(runtime, len(specs))

    if parallelism == ParallelismMode.NONE or len(specs) == 1:
        _apply_blas_policy(runtime.normalized_blas_policy())
        for index, spec_item in enumerate(specs):
            artifact, error = _run_single_spec(data, spec_item, runtime, run_id, index, data_signature)
            if artifact:
                results.append(artifact)
            if error:
                errors.append(error)
                if runtime.fail_policy == FailPolicy.FAIL_FAST:
                    break
    else:
        results, errors = _run_parallel(data, specs, runtime, run_id, parallelism, data_signature)

    results.sort(key=lambda r: r.index)
    errors.sort(key=lambda e: e.index)

    end_time = datetime.utcnow()
    summary = _build_summary(run_id, specs, results, errors, start_time, end_time, runtime, parallelism)

    learned_edges = _collect_edge_pairs(results)
    artifact_writer = ArtifactWriter(runtime.artifact_root, run_id)
    artifact_writer.write_run(
        CausalRunResult(results=results, errors=errors, summary=summary, metadata=metadata),
        input_dot=input_dot,
        learned_edges=learned_edges,
    )

    return CausalRunResult(results=results, errors=errors, summary=summary, metadata=metadata)


def _resolve_time_column(specs: Sequence[EstimandSpec]) -> Optional[str]:
    for spec in specs:
        if spec.time_column:
            return spec.time_column
    return None


def _normalize_specs(spec: EstimandSpec | List[EstimandSpec]) -> List[EstimandSpec]:
    if isinstance(spec, list):
        return spec
    return [spec]


def _resolve_parallelism(runtime: RuntimeSpec, spec_count: int) -> ParallelismMode:
    if spec_count <= 1:
        return ParallelismMode.NONE
    if runtime.parallelism == ParallelismMode.NONE:
        return ParallelismMode.NONE
    return runtime.parallelism


def _resolve_input_dot(runtime: RuntimeSpec, specs: Sequence[EstimandSpec]) -> Optional[str]:
    if runtime.dot_path:
        return _safe_load_dot(runtime.dot_path)
    dot_paths = [s.dot_path for s in specs if s.dot_path]
    if not dot_paths:
        return None
    if len(set(dot_paths)) == 1:
        return _safe_load_dot(dot_paths[0])
    dot_texts = [_safe_load_dot(path) or "" for path in dot_paths]
    edges = merge_edges([parse_dot_edges(text) for text in dot_texts])
    return "\n".join(["digraph Input {", "  rankdir=LR;"] + [f"  \"{a}\" -> \"{b}\";" for a, b in edges] + ["}"])


def _apply_blas_policy(policy: dict) -> None:
    for key, value in policy.items():
        os.environ[str(key)] = str(value)


def _run_parallel(
    data: pd.DataFrame,
    specs: Sequence[EstimandSpec],
    runtime: RuntimeSpec,
    run_id: str,
    parallelism: ParallelismMode,
    data_signature: dict,
) -> Tuple[List[EffectArtifact], List[SpecError]]:
    max_workers = _resolve_workers(runtime)
    chunk_size = max(1, runtime.chunk_size or 1)
    chunks = list(_chunk_specs(specs, chunk_size))

    results: List[EffectArtifact] = []
    errors: List[SpecError] = []

    executor_cls = concurrent.futures.ProcessPoolExecutor
    init_args = (runtime.normalized_blas_policy(), runtime.resolved_seed())
    executor_kwargs = {
        "max_workers": max_workers,
        "initializer": _worker_init,
        "initargs": init_args,
        "mp_context": multiprocessing.get_context("spawn"),
    }
    if parallelism == ParallelismMode.THREAD:
        executor_cls = concurrent.futures.ThreadPoolExecutor
        executor_kwargs = {"max_workers": max_workers}
        _apply_blas_policy(runtime.normalized_blas_policy())

    with executor_cls(**executor_kwargs) as executor:
        futures = {}
        for chunk_index, chunk in enumerate(chunks):
            future = executor.submit(_run_chunk, data, chunk, runtime, run_id, data_signature)
            futures[future] = chunk_index

        for future in concurrent.futures.as_completed(futures):
            chunk_index = futures[future]
            try:
                chunk_result = future.result()
                results.extend(chunk_result.results)
                errors.extend(chunk_result.errors)
            except Exception as exc:
                errors.append(
                    SpecError(
                        spec_id=f"__chunk__{chunk_index}",
                        index=chunk_index,
                        stage="worker_execution",
                        message=str(exc),
                        exception_type=exc.__class__.__name__,
                        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                        fatal=True,
                    )
                )

            if errors and runtime.fail_policy == FailPolicy.FAIL_FAST:
                for pending in futures:
                    if not pending.done():
                        pending.cancel()
                break

    return results, errors


def _resolve_workers(runtime: RuntimeSpec) -> int:
    if runtime.n_jobs is None or runtime.n_jobs == 0:
        return os.cpu_count() or 1
    if runtime.n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, int(runtime.n_jobs))


def _chunk_specs(specs: Sequence[EstimandSpec], chunk_size: int) -> Iterable[List[Tuple[int, EstimandSpec]]]:
    chunk: List[Tuple[int, EstimandSpec]] = []
    for index, spec in enumerate(specs):
        chunk.append((index, spec))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _worker_init(blas_policy: dict, seed: int) -> None:
    _apply_blas_policy(blas_policy)
    seed_everything(seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _run_chunk(
    data: pd.DataFrame,
    chunk: Sequence[Tuple[int, EstimandSpec]],
    runtime: RuntimeSpec,
    run_id: str,
    data_signature: dict,
) -> _ChunkResult:
    results: List[EffectArtifact] = []
    errors: List[SpecError] = []

    for index, spec in chunk:
        artifact, error = _run_single_spec(data, spec, runtime, run_id, index, data_signature)
        if artifact:
            results.append(artifact)
        if error:
            errors.append(error)
            if runtime.fail_policy == FailPolicy.FAIL_FAST:
                break

    return _ChunkResult(results=results, errors=errors)


def _run_single_spec(
    data: pd.DataFrame,
    spec: EstimandSpec,
    runtime: RuntimeSpec,
    run_id: str,
    index: int,
    data_signature: dict,
) -> Tuple[Optional[EffectArtifact], Optional[SpecError]]:
    spec_id = spec.signature()
    stage = "spec_validation"

    try:
        spec.validate()
        seed_everything(runtime.resolved_seed() + index)

        stage = "dot_load"
        dot_text = _load_dot_or_error(spec.dot_path or runtime.dot_path)
        stage = "time_series_validation"
        temporal_diagnostics = validate_time_series(
            data,
            spec,
            dot_text,
            runtime.time_series_policy,
        ).to_dict()

        stage = "feature_layer"
        clean_data = FeatureBuilder.validate_and_clean(data, spec)

        stage = "identification"
        identifier = Identifier(spec, graph=dot_text)
        model, identified_estimand = identifier.identify(clean_data)

        stage = "estimation"
        estimate_result = EstimatorFactory.estimate(model, identified_estimand, spec, runtime)
        estimate = estimate_result.estimate

        stage = "confidence_intervals"
        confidence_intervals = None
        try:
            cis = estimate.get_confidence_intervals(confidence_level=runtime.confidence_level)
            if cis is not None and len(cis) == 2:
                confidence_intervals = _normalize_ci(cis)
        except Exception as exc:
            logger.warning(f"Could not compute confidence intervals: {exc}")

        stage = "validation"
        refutations = Validator.validate(model, estimate, runtime)

        graph_edges = _extract_graph_edges(model, spec, dot_text)

        diagnostics = {
            "rows": int(len(clean_data)),
            "identified_estimand": str(identified_estimand),
            "method_name": estimate_result.method_name,
            "seed": runtime.resolved_seed() + index,
        }

        provenance = {
            "run_id": run_id,
            "spec_id": spec_id,
            "dot_path": spec.dot_path or runtime.dot_path,
            "data_hash": data_signature.get("hash"),
        }

        artifact = EffectArtifact(
            spec=spec,
            spec_id=spec_id,
            index=index,
            estimand_type=spec.type.value,
            estimate=_normalize_estimate_value(estimate.value),
            confidence_intervals=confidence_intervals,
            diagnostics=diagnostics,
            refuter_results=refutations,
            temporal_diagnostics=temporal_diagnostics,
            provenance=provenance,
            graph_edge_payload=graph_edges,
            backend={
                "name": estimate_result.backend,
                "method_name": estimate_result.method_name,
                "method_params": estimate_result.method_params,
            },
        )

        return artifact, None
    except TimeSeriesValidationError as exc:
        return None, _spec_error(spec_id, index, stage, exc)
    except Exception as exc:
        return None, _spec_error(spec_id, index, stage, exc)


def _extract_graph_edges(model, spec: EstimandSpec, dot_text: Optional[str]) -> List[dict]:
    edges: List[dict] = []
    graph_edges: List[Tuple[str, str]] = []

    if dot_text:
        graph_edges = parse_dot_edges(dot_text)
        for src, tgt in graph_edges:
            edges.append({"source": src, "target": tgt, "origin": "dot_input"})

    try:
        graph = getattr(model, "_graph", None)
        if graph is not None:
            nx_graph = graph.get_graph() if hasattr(graph, "get_graph") else None
            if nx_graph is not None and hasattr(nx_graph, "edges"):
                for src, tgt in nx_graph.edges():
                    edges.append({"source": str(src), "target": str(tgt), "origin": "dowhy_graph"})
    except Exception:
        pass

    if not edges:
        edges = _spec_edges(spec)

    return edges


def _spec_edges(spec: EstimandSpec) -> List[dict]:
    edges: List[dict] = []
    for confounder in spec.confounders:
        edges.append({"source": confounder, "target": spec.treatment, "origin": "spec"})
        edges.append({"source": confounder, "target": spec.outcome, "origin": "spec"})
    edges.append({"source": spec.treatment, "target": spec.outcome, "origin": "spec"})
    if spec.instrument:
        edges.append({"source": spec.instrument, "target": spec.treatment, "origin": "spec"})
    if spec.mediator:
        edges.append({"source": spec.treatment, "target": spec.mediator, "origin": "spec"})
        edges.append({"source": spec.mediator, "target": spec.outcome, "origin": "spec"})
    return edges


def _spec_error(spec_id: str, index: int, stage: str, exc: Exception) -> SpecError:
    return SpecError(
        spec_id=spec_id,
        index=index,
        stage=stage,
        message=str(exc),
        exception_type=exc.__class__.__name__,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        fatal=False,
    )


def _normalize_estimate_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) if isinstance(v, (int, float, np.floating)) else v for v in value]
    try:
        return float(value)
    except Exception:
        return value


def _normalize_ci(cis):
    try:
        lower, upper = cis
    except Exception:
        return cis
    return (_normalize_estimate_value(lower), _normalize_estimate_value(upper))


def _collect_edge_pairs(results: Sequence[EffectArtifact]) -> List[Tuple[str, str]]:
    edge_lists: List[List[Tuple[str, str]]] = []
    for artifact in results:
        edges: List[Tuple[str, str]] = []
        learned_only: List[Tuple[str, str]] = []
        for entry in artifact.graph_edge_payload:
            src = entry.get("source")
            tgt = entry.get("target")
            if src and tgt:
                edges.append((str(src), str(tgt)))
                if entry.get("origin") != "dot_input":
                    learned_only.append((str(src), str(tgt)))
        edge_lists.append(learned_only or edges)
    return merge_edges(edge_lists)


def _build_summary(
    run_id: str,
    specs: Sequence[EstimandSpec],
    results: Sequence[EffectArtifact],
    errors: Sequence[SpecError],
    started_at: datetime,
    finished_at: datetime,
    runtime: RuntimeSpec,
    parallelism: ParallelismMode,
) -> RunSummary:
    total = len(specs)
    succeeded = len(results)
    failed = len(errors)

    if failed == 0:
        status = "success"
    elif succeeded == 0:
        status = "failed"
    else:
        status = "partial"

    return RunSummary(
        run_id=run_id,
        total_specs=total,
        succeeded=succeeded,
        failed=failed,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_sec=(finished_at - started_at).total_seconds(),
        status=status,
        fail_policy=runtime.fail_policy.value,
        parallelism=parallelism.value,
    )


def _safe_version(package: str) -> Optional[str]:
    try:
        module = __import__(package)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _safe_load_dot(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return load_dot(path)
    except Exception as exc:
        logger.warning(f"Failed to load DOT template '{path}': {exc}")
        return None


def _load_dot_or_error(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return load_dot(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load DOT template '{path}': {exc}") from exc

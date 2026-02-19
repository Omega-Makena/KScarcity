"""
Integrative Operators — Tier 4 reasoning orchestration.

Combines structural, relational, causal, and semantic latents into unified
representations and actionable insights while staying online and DRG-aware.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scarcity.engine.operators.attention_ops import attn_linear, pooling_avg
from scarcity.engine.operators.sketch_ops import _deterministic_hash
from scarcity.engine.operators.stability_ops import page_hinkley

logger = logging.getLogger(__name__)


@dataclass
class IntegrativeOutput:
    """Container for Tier-4 operator outputs."""

    latent: np.ndarray
    stats: Dict[str, Any]
    cost_hint_ms: float


def _normalize_energy(latents: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Scale latents to comparable energy."""
    normalized = []
    for latent in latents:
        if latent is None:
            normalized.append(np.zeros(1, dtype=np.float32))
            continue

        arr = np.asarray(latent, dtype=np.float32).flatten()
        if arr.size == 0:
            normalized.append(np.zeros(1, dtype=np.float32))
            continue

        energy = np.sqrt(np.mean(arr ** 2)) + 1e-6
        normalized.append(arr / energy)
    return normalized


def _variance_gate(latents: Sequence[np.ndarray]) -> np.ndarray:
    """Compute adaptive gates from variance."""
    variances = np.array([np.var(latent) for latent in latents], dtype=np.float32)
    variances = np.maximum(variances, 1e-6)
    gates = variances / np.sum(variances)
    return gates.astype(np.float32)


def multi_modal_fusion(
    Z_struct: np.ndarray,
    R_rel: np.ndarray,
    C_causal: np.ndarray,
    S_semantic: np.ndarray,
    use_attention: bool = True,
    fusion_entropy_clip: float = 0.05,
    drg_profile: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> IntegrativeOutput:
    """
    Fuse multi-modal latents into unified representation with adaptive gates.
    """
    start = time.time()
    fallbacks = 0
    drg_profile = drg_profile or {}
    vram_high = bool(drg_profile.get('vram_high', False))

    latents = _normalize_energy([Z_struct, R_rel, C_causal, S_semantic])

    max_dim = max(latent.size for latent in latents)
    padded = []
    for idx, latent in enumerate(latents):
        if latent.size < max_dim:
            latent = np.pad(latent, (0, max_dim - latent.size))
        padded.append(latent)

    stacked = np.stack(padded, axis=0).astype(np.float32)  # [M, D]
    gates = _variance_gate(padded)  # [M]

    if use_attention and not vram_high:
        try:
            query = stacked.mean(axis=0, keepdims=True)
            attn = attn_linear(query, stacked, stacked)  # [1, D]
            attn = np.squeeze(attn, axis=0)
            fusion = gates @ stacked + 0.25 * attn
        except Exception as exc:  # pragma: no cover
            logger.warning(f"MMF: attention failed, fallback to weighted sum: {exc}")
            fusion = gates @ stacked
            fallbacks |= 1
    else:
        fusion = gates @ stacked
        if use_attention and vram_high:
            fallbacks |= 2

    entropy = -np.sum(gates * np.log(gates + 1e-6))
    entropy = min(entropy, fusion_entropy_clip)

    latent = fusion.astype(np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'mmf_entropy': float(entropy),
        'gates': gates.tolist(),
        'fusion_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def hierarchical_context_integrator(
    U_mmf: np.ndarray,
    context: Dict[str, Any],
    normalize_context: bool = True,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Project fused latent into context-aware space using domain/regime metadata.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    latency_high = bool(drg_profile.get('latency_high', False))
    fallbacks = 0

    latent = np.asarray(U_mmf, dtype=np.float32)
    if latent.ndim == 1:
        latent = latent[np.newaxis, :]

    domain_id = float(context.get('domain_id', 0))
    regime_id = float(context.get('regime_id', 0))
    region_id = float(context.get('region_id', 0))
    ctx_vec = np.array([domain_id, regime_id, region_id], dtype=np.float32)

    if normalize_context and not latency_high:
        ctx_norm = np.linalg.norm(ctx_vec) + 1e-6
        ctx_vec = ctx_vec / ctx_norm
    else:
        fallbacks |= 1

    proj = latent * (1.0 + ctx_vec.mean())
    context_stability = float(np.std(ctx_vec))

    pooled = pooling_avg(proj, axis=0)
    latent_out = pooled.astype(np.float16)

    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'context_vector': ctx_vec.tolist(),
        'context_stability': context_stability,
        'hci_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent_out, stats=stats, cost_hint_ms=cost_ms)


def cross_tier_reconciliation(
    latents: Sequence[np.ndarray],
    max_iterations: int = 3,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Align latents from different tiers to minimize disagreement.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    util = float(drg_profile.get('util', 0.0))
    fallbacks = 0

    processed = _normalize_energy(latents)
    max_dim = max(latent.size for latent in processed)
    stack = []
    for latent in processed:
        if latent.size < max_dim:
            latent = np.pad(latent, (0, max_dim - latent.size))
        stack.append(latent)
    stack = np.stack(stack, axis=0)  # [N, D]

    disagreement = []
    for i in range(len(stack)):
        for j in range(i + 1, len(stack)):
            delta = np.mean(np.abs(stack[i] - stack[j]))
            disagreement.append(delta)
    avg_disagreement = float(np.mean(disagreement)) if disagreement else 0.0

    weights = np.ones(len(stack), dtype=np.float32)
    if util > 0.85:
        fallbacks |= 1
        reconciled = np.average(stack, axis=0, weights=weights)
    else:
        for _ in range(max_iterations):
            residual = stack - np.average(stack, axis=0, weights=weights)
            energy = np.linalg.norm(residual, axis=1) + 1e-6
            weights = 1.0 / energy
            weights /= np.sum(weights)
        reconciled = np.average(stack, axis=0, weights=weights)

    consistency_ratio = float(np.exp(-avg_disagreement))
    latent = reconciled.astype(np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'consistency_ratio': consistency_ratio,
        'recon_weights': weights.tolist(),
        'recon_consistency': float(avg_disagreement),
        'ctr_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def adaptive_meta_aggregator(
    latents: Sequence[np.ndarray],
    state: Optional[Dict[str, Any]] = None,
    meta_update_rate: float = 0.2,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Combine latents into a meta-representation with online adaptation.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    latency_high = bool(drg_profile.get('latency_high', False))
    fallbacks = 0

    state = state if state is not None else {}
    meta_vector = np.asarray(state.get('meta_vector', None), dtype=np.float32) if state.get('meta_vector') is not None else None

    flattened = _normalize_energy(latents)
    max_dim = max(vec.size for vec in flattened)
    aligned = []
    for vec in flattened:
        if vec.size < max_dim:
            vec = np.pad(vec, (0, max_dim - vec.size))
        aligned.append(vec)
    matrix = np.stack(aligned, axis=0)

    mean_vector = matrix.mean(axis=0)

    rate = meta_update_rate
    if latency_high:
        rate = meta_update_rate * 0.5
        fallbacks |= 1

    if meta_vector is None:
        meta_vector = mean_vector
    else:
        meta_vector = (1 - rate) * meta_vector + rate * mean_vector

    state['meta_vector'] = meta_vector
    state['meta_update_rate'] = rate

    latent = meta_vector.astype(np.float16)
    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'meta_update_rate': rate,
        'meta_loss_proxy': float(np.mean(np.abs(matrix - mean_vector))),
        'ama_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
        'state': state,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def policy_decision_synthesizer(
    causal_pairs: Sequence[Dict[str, Any]],
    semantic_alignment: Sequence[Dict[str, Any]],
    relational_density: float,
    top_k: int = 5,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Produce interpretable policy decisions from multi-tier insights.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    vram_high = bool(drg_profile.get('vram_high', False))
    fallbacks = 0

    top_k = min(top_k, 5 if vram_high else top_k)

    alpha_lookup = {item['concept']: item.get('score', 0.0) for item in semantic_alignment if 'concept' in item}

    decisions = []
    for pair in causal_pairs:
        concept = max(alpha_lookup.items(), key=lambda kv: kv[1])[0] if alpha_lookup else 'unknown'
        score = pair.get('p_cause', 0.0)
        strength = pair.get('direction_strength', 0.0)
        confidence = 0.5 * score + 0.3 * alpha_lookup.get(concept, 0.0) + 0.2 * relational_density

        decisions.append({
            'src': pair.get('src'),
            'dst': pair.get('dst'),
            'concept': concept,
            'policy': f"Adjust policy weighting towards {concept}",
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'direction_strength': strength,
        })

    decisions.sort(key=lambda item: item['confidence'], reverse=True)
    top_decisions = decisions[:max(1, top_k)]

    latent = np.array([d['confidence'] for d in top_decisions], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'decision_events': top_decisions,
        'policy_confidence': float(np.mean(latent.astype(np.float32))),
        'pds_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def forecast_integrator(
    unified_latents: Iterable[np.ndarray],
    window_size: int = 8,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Generate forward projection using online autoregressive approximation.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    vram_high = bool(drg_profile.get('vram_high', False))
    fallbacks = 0

    window_size = max(3, window_size // 2) if vram_high else window_size

    history = []
    for latent in unified_latents:
        arr = np.asarray(latent, dtype=np.float32).flatten()
        if arr.size == 0:
            continue
        history.append(arr)
        if len(history) >= window_size:
            break

    if not history:
        fallbacks |= 1
        forecast = np.zeros(1, dtype=np.float16)
        variance = 0.0
    else:
        matrix = np.stack(history, axis=0)
        mean_vec = matrix.mean(axis=0)
        if matrix.shape[0] < 2:
            forecast = mean_vec
            variance = 0.0
        else:
            diffs = matrix[1:] - matrix[:-1]
            coef = np.mean(diffs, axis=0)
            forecast = mean_vec + coef
            variance = float(np.mean(np.var(matrix, axis=0)))

    latent = forecast.astype(np.float16)
    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'forecast_var': variance,
        'window_size_used': len(history),
        'fci_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def reasoning_loop_controller(
    insight_scores: Sequence[float],
    explore_lambda: float = 0.3,
    drg_profile: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Adjust exploration rate based on entropy of insight scores.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    state = state if state is not None else {}

    scores = np.asarray(insight_scores, dtype=np.float32)
    if scores.size == 0:
        scores = np.zeros(1, dtype=np.float32)

    entropy = -np.sum(scores * np.log(np.clip(scores, 1e-6, 1.0)))
    lambda_update = explore_lambda * np.exp(-entropy)
    lambda_update = float(np.clip(lambda_update, 0.05, 1.0))

    state['lambda'] = lambda_update
    state['entropy'] = entropy

    latent = np.array([lambda_update, entropy], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'explore_lambda': lambda_update,
        'entropy_score': float(entropy),
        'rlc_latency_ms': cost_ms,
        'state': state,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def integrative_feedback_emitter(
    insights: Sequence[Dict[str, Any]],
    resource_shift: float,
    performance_gain: float,
    telemetry_rate: float = 0.5,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Emit structured feedback events summarizing resource adjustments.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    latency_high = bool(drg_profile.get('latency_high', False))
    fallbacks = 0

    if latency_high:
        telemetry_rate *= 0.5
        fallbacks |= 1

    events = []
    for insight in insights:
        insight_id = insight.get('id', _deterministic_hash(0, str(insight)).integers(0, 1 << 30))
        events.append({
            'insight_id': int(insight_id),
            'resource_shift': resource_shift,
            'performance_gain': performance_gain,
            'telemetry_rate': telemetry_rate,
        })

    if not events:
        events.append({
            'insight_id': -1,
            'resource_shift': resource_shift,
            'performance_gain': performance_gain,
            'telemetry_rate': telemetry_rate,
        })

    latent = np.array([event['performance_gain'] for event in events], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'feedback_events': events,
        'feedback_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def global_insight_assembler(
    insight_sets: Sequence[Sequence[Dict[str, Any]]],
    top_k: int = 10,
    drg_profile: Optional[Dict[str, Any]] = None,
) -> IntegrativeOutput:
    """
    Merge insights across integrative paths into ranked global set.
    """
    start = time.time()
    drg_profile = drg_profile or {}
    fallbacks = 0

    merged: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for insights in insight_sets:
        for insight in insights:
            key = (insight.get('src'), insight.get('dst'))
            score = float(insight.get('confidence', 0.0))
            if key not in merged or merged[key]['confidence'] < score:
                merged[key] = dict(insight)

    ranked = sorted(merged.values(), key=lambda item: item.get('confidence', 0.0), reverse=True)
    top = ranked[:max(1, top_k)]

    confidences = np.array([item.get('confidence', 0.0) for item in top], dtype=np.float32)
    if confidences.size == 0:
        confidences = np.zeros(1, dtype=np.float32)
        fallbacks |= 1

    entropy = -np.sum(confidences * np.log(np.clip(confidences, 1e-6, 1.0)))
    latent = confidences.astype(np.float16)

    cost_ms = (time.time() - start) * 1000.0
    stats = {
        'insight_global': top,
        'insight_count': len(top),
        'confidence_mean': float(np.mean(confidences)),
        'entropy_global': float(entropy),
        'gia_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return IntegrativeOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


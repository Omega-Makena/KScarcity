"""
Causal & Semantic Operators — Tier 3 reasoning layer.

Implements online causal estimation, counterfactual probing, semantic
alignment, and meaning aggregation under tight resource budgets.

All operators are:
- Online / streaming (no offline batches or heavy retraining)
- Deterministic given seeds (uses deterministic RNG helper)
- Resource aware through the Dynamic Resource Governor (DRG) hooks
- Numerically safe with FP16 outputs and FP32 accumulation

See OPERATOR_TAXONOMY.md for tier overview.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scarcity.engine.operators.sketch_ops import _deterministic_hash
from scarcity.engine.operators.stability_ops import page_hinkley
from scarcity.engine.store import EdgeRec, HypergraphStore

logger = logging.getLogger(__name__)


@dataclass
class CausalSemanticOutput:
    """Generic output container for Tier-3 operators."""

    latent: np.ndarray
    stats: Dict[str, Any]
    cost_hint_ms: float


def _sanitize_series(series: np.ndarray) -> np.ndarray:
    """Ensure series is 1D float array without NaN/Inf."""
    if series is None or len(series) == 0:
        return np.zeros(1, dtype=np.float32)

    arr = np.asarray(series, dtype=np.float32).flatten()
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        logger.warning("Tier3: NaN/Inf detected in series, zeroing invalid values.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _directional_gain(
    src_series: np.ndarray,
    dst_series: np.ndarray,
    lag: int,
    ridge: float = 1e-3,
) -> Tuple[float, float]:
    """
    Compute Wiener-Granger style predictive gain and sign.

    Returns:
        gain: Fractional reduction in MSE when including src past.
        sign: Sign of contribution from src.
    """
    x = _sanitize_series(src_series)
    y = _sanitize_series(dst_series)

    if len(x) <= lag + 1 or len(y) <= lag + 1:
        return 0.0, 0.0

    # Align series for lagged regression
    y_target = y[lag:]
    y_past = np.stack([y[lag - i - 1 : -i - 1] for i in range(lag)], axis=1)
    x_past = np.stack([x[lag - i - 1 : -i - 1] for i in range(lag)], axis=1)

    # Baseline: predict y from its own lags
    XtX = y_past.T @ y_past + ridge * np.eye(lag, dtype=np.float32)
    Xty = y_past.T @ y_target
    coeff_base = np.linalg.solve(XtX, Xty)
    residual_base = y_target - y_past @ coeff_base
    mse_base = float(np.mean(residual_base ** 2))

    # Augmented: include src lags
    XY = np.concatenate([y_past, x_past], axis=1)
    XtX_aug = XY.T @ XY + ridge * np.eye(2 * lag, dtype=np.float32)
    Xty_aug = XY.T @ y_target
    coeff_aug = np.linalg.solve(XtX_aug, Xty_aug)
    residual_aug = y_target - XY @ coeff_aug
    mse_aug = float(np.mean(residual_aug ** 2))

    if mse_base <= 1e-9:
        return 0.0, 0.0

    gain = max(0.0, (mse_base - mse_aug) / mse_base)
    contribution = coeff_aug[-lag:]  # coefficients for src lags
    if contribution.size == 0:
        sign = 0.0
    else:
        sign = float(np.tanh(np.sum(contribution)))

    return gain, sign


def _stability_weight(edge: Optional[EdgeRec], default: float = 0.45) -> float:
    """Return signed stability weight given an edge record."""
    if edge is None:
        return default
    signed_weight = edge.stability * np.sign(edge.weight if edge.weight != 0 else 1.0)
    return float(np.clip(signed_weight, -1.0, 1.0))


def _logistic_score(x: float, threshold: float) -> float:
    """Smooth logistic score mapping gain to probability."""
    # Center around threshold, slope tuned for [0,1]
    return float(1.0 / (1.0 + np.exp(-10.0 * (x - threshold))))


def directional_causality(
    series: Dict[int, np.ndarray],
    store: HypergraphStore,
    window_id: int,
    candidate_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_pairs: int = 64,
    min_gain: float = 0.02,
    lag: int = 1,
    min_windows: int = 3,
    history: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
    drg_util: float = 0.0,
) -> CausalSemanticOutput:
    """
    Estimate directional causal strength between variable pairs.

    Args:
        series: Mapping node_id -> 1D time series (latest window).
        store: Hypergraph store for stability metadata.
        window_id: Current window identifier.
        candidate_pairs: Optional explicit candidate edges (src, dst).
        max_pairs: DRG budget for evaluated pairs.
        min_gain: Minimum predictive gain required.
        lag: Number of past lags to consider.
        min_windows: Required stability window count for confirmation.
        history: Rolling history dictionary updated in-place (optional).
        drg_util: DRG utilization ratio (0-1).

    Returns:
        CausalSemanticOutput with directional latent and telemetry.
    """
    start = time.time()
    fallbacks = 0
    history = history if history is not None else {}

    if drg_util > 0.85:
        max_pairs = max(16, max_pairs // 2)
        lag = max(1, min(lag, 1))
        fallbacks |= 1
    elif drg_util > 0.75:
        max_pairs = max(24, int(max_pairs * 0.75))

    keys = list(series.keys())
    if candidate_pairs is None:
        # Generate candidate pairs sorted by store stability if available
        candidate_pairs = []
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                stability = 0.0
                edge = store.get_edge(src, dst) if store else None
                if edge is not None:
                    stability = edge.stability
                candidate_pairs.append(((src, dst), stability))
        candidate_pairs.sort(key=lambda item: item[1], reverse=True)
        candidate_pairs = [pair for pair, _ in candidate_pairs]

    evaluated_pairs = list(candidate_pairs[:max_pairs])

    results: List[Dict[str, Any]] = []
    strengths: List[float] = []
    p_cause_values: List[float] = []
    stable_pairs = 0

    for src, dst in evaluated_pairs:
        gain_forward, sign_forward = _directional_gain(series.get(src), series.get(dst), lag)
        gain_backward, sign_backward = _directional_gain(series.get(dst), series.get(src), lag)

        direction = 0
        gain_delta = gain_forward - gain_backward

        if gain_delta > min_gain:
            direction = 1
            gain = gain_forward
            sign = sign_forward
        elif (-gain_delta) > min_gain:
            direction = -1
            gain = gain_backward
            sign = sign_backward
            src, dst = dst, src  # flip to reflect detected direction
        else:
            continue

        edge_rec = store.get_edge(src, dst) if store else None
        stability_weight = _stability_weight(edge_rec)

        hist = history.setdefault((src, dst), {'count': 0, 'last_window': -1, 'avg_gain': 0.0})
        hist['count'] = min(hist['count'] + 1, 255)
        hist['last_window'] = window_id
        hist['avg_gain'] = 0.8 * hist['avg_gain'] + 0.2 * gain if hist['count'] > 1 else gain

        confirmed = hist['count'] >= min_windows
        if confirmed:
            stable_pairs += 1

        p_cause = np.clip(_logistic_score(gain, min_gain), 0.0, 1.0)
        p_cause *= np.clip(abs(stability_weight), 0.0, 1.0)
        directional_strength = float(np.clip(gain * stability_weight * (1.0 if sign >= 0 else -1.0), -1.0, 1.0))

        strengths.append(directional_strength)
        p_cause_values.append(p_cause)

        results.append({
            'src': src,
            'dst': dst,
            'gain': gain,
            'direction_strength': directional_strength,
            'p_cause': p_cause,
            'sign': sign,
            'stability_weight': stability_weight,
            'confirmed': confirmed,
            'history_count': hist['count'],
        })

    if not strengths:
        fallbacks |= 2
        strengths = [0.0]
        p_cause_values = [0.0]

    latent = np.asarray(strengths, dtype=np.float16)
    p_cause_values = np.asarray(p_cause_values, dtype=np.float32)

    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'window_id': window_id,
        'pairs_evaluated': len(evaluated_pairs),
        'pairs_confirmed': stable_pairs,
        'p_cause_mean': float(np.mean(p_cause_values)),
        'p_cause_var': float(np.var(p_cause_values)),
        'causal_pairs': results,
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def counterfactual_lite(
    baseline: np.ndarray,
    jacobian: np.ndarray,
    perturb_indices: Sequence[int],
    delta_sigma: float = 1.0,
    samples: int = 4,
    drg_util: float = 0.0,
    seed: int = 0,
) -> CausalSemanticOutput:
    """
    Lightweight counterfactual perturbation using stored Jacobian.

    Args:
        baseline: Baseline latent vector [d_in].
        jacobian: Linearized Jacobian [d_out, d_in].
        perturb_indices: Indices to perturb (key drivers).
        delta_sigma: Standard deviation scale for perturbations.
        samples: Number of delta samples (≤ 4 recommended).
        drg_util: DRG utilization ratio.
        seed: Deterministic seed.
    """
    start = time.time()
    fallbacks = 0

    baseline = _sanitize_series(baseline)
    jacobian = np.asarray(jacobian, dtype=np.float32)

    if jacobian.ndim != 2 or jacobian.shape[1] != baseline.shape[0]:
        logger.warning("CF Lite: Jacobian shape mismatch, using identity fallback.")
        d = baseline.shape[0]
        jacobian = np.eye(d, dtype=np.float32)
        fallbacks |= 1

    if drg_util > 0.85:
        samples = max(1, min(samples, 2))
        fallbacks |= 2
    else:
        samples = max(1, min(samples, 4))

    perturb_indices = list(dict.fromkeys(int(idx) for idx in perturb_indices if 0 <= idx < baseline.size))
    if not perturb_indices:
        fallbacks |= 4
        perturb_indices = [0]

    rng = _deterministic_hash(seed, "cf_lite")

    impacts = []
    magnitudes = []
    for i in range(samples):
        delta = np.zeros_like(baseline, dtype=np.float32)
        sampled_noise = rng.normal(loc=0.0, scale=delta_sigma, size=len(perturb_indices))
        delta[perturb_indices] = sampled_noise

        impact = jacobian @ delta
        impacts.append(impact)
        magnitudes.append(np.linalg.norm(impact, ord=2))

    impact_matrix = np.stack(impacts, axis=0) if impacts else np.zeros((1, baseline.size), dtype=np.float32)
    mean_impact = np.mean(impact_matrix, axis=0)
    eta_cf = float(np.mean(magnitudes)) if magnitudes else 0.0

    latent = mean_impact.astype(np.float16, copy=False)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'eta_cf': eta_cf,
        'impact_norm_mean': float(np.mean(magnitudes) if magnitudes else 0.0),
        'impact_norm_max': float(np.max(magnitudes) if magnitudes else 0.0),
        'samples': samples,
        'perturb_indices': perturb_indices,
        'cf_latency_ms': cost_ms,
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def causal_graph_propagation(
    store: HypergraphStore,
    causal_pairs: Sequence[Dict[str, Any]],
    hop_cap: int = 2,
    min_stability: float = 0.2,
    drg_util: float = 0.0,
) -> CausalSemanticOutput:
    """
    Propagate causal signals through the hypergraph to find multi-hop chains.
    """
    start = time.time()
    fallbacks = 0

    if store is None:
        raise ValueError("HypergraphStore is required for causal graph propagation.")

    if drg_util > 0.85:
        hop_cap = 1
        fallbacks |= 1
    else:
        hop_cap = max(1, min(hop_cap, 2))

    queue: List[Tuple[int, float, List[int]]] = []
    for item in causal_pairs:
        p_cause = item.get('p_cause', 0.0)
        if p_cause < 0.1:
            continue
        src = int(item['src'])
        dst = int(item['dst'])
        strength = float(item.get('direction_strength', 0.0))
        queue.append((dst, strength * p_cause, [src, dst]))

    results = []
    chain_strengths = []

    while queue:
        node, strength, path = queue.pop(0)
        chain_strengths.append(strength)
        results.append({'path': path.copy(), 'strength': strength})

        if len(path) - 1 >= hop_cap:
            continue

        # Explore outgoing edges
        neighbors = store.top_k_neighbors(node, k=store.topk_per_node, direction="out")
        for neighbor_id, _ in neighbors:
            edge_rec = store.get_edge(node, neighbor_id)
            if edge_rec is None or edge_rec.stability < min_stability:
                continue

            next_strength = strength * edge_rec.weight * edge_rec.stability
            next_node = neighbor_id

            if next_node in path:
                continue  # avoid cycles

            queue.append((next_node, next_strength, path + [next_node]))

            if len(queue) > 128:
                queue = queue[:128]
                fallbacks |= 4
                break

    if not chain_strengths:
        fallbacks |= 2
        chain_strengths = [0.0]

    latent = np.asarray(chain_strengths, dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'chains_discovered': len(results),
        'chains': results,
        'causal_flow_strength': float(np.mean(np.abs(latent.astype(np.float32)))),
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def policy_semantic_align(
    causal_latent: np.ndarray,
    policy_embeddings: Dict[str, np.ndarray],
    embedding_dim: int = 512,
    smooth_state: Optional[Dict[str, np.ndarray]] = None,
    smooth_alpha: float = 0.3,
    drg_util: float = 0.0,
) -> CausalSemanticOutput:
    """
    Align causal latents with policy or textual embeddings.
    """
    start = time.time()
    fallbacks = 0

    if drg_util > 0.85:
        embedding_dim = max(128, embedding_dim // 2)
        smooth_alpha = min(smooth_alpha, 0.25)
        fallbacks |= 1

    causal_vec = _sanitize_series(causal_latent)
    if causal_vec.size > embedding_dim:
        causal_vec = causal_vec[:embedding_dim]
    elif causal_vec.size < embedding_dim:
        causal_vec = np.pad(causal_vec, (0, embedding_dim - causal_vec.size))

    causal_norm = np.linalg.norm(causal_vec) + 1e-9

    smooth_state = smooth_state if smooth_state is not None else {}
    scores: List[Tuple[str, float]] = []

    for concept, embedding in policy_embeddings.items():
        emb = _sanitize_series(embedding)
        if emb.size > embedding_dim:
            emb = emb[:embedding_dim]
        elif emb.size < embedding_dim:
            emb = np.pad(emb, (0, embedding_dim - emb.size))

        if np.all(emb == 0):
            continue

        if concept in smooth_state:
            emb = smooth_alpha * emb + (1 - smooth_alpha) * smooth_state[concept]

        smooth_state[concept] = emb
        score = float(np.dot(causal_vec, emb) / (causal_norm * (np.linalg.norm(emb) + 1e-9)))
        scores.append((concept, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    top_scores = scores[: min(8, len(scores))]

    if not top_scores:
        fallbacks |= 2
        top_scores = [("none", 0.0)]

    latent = np.asarray([score for _, score in top_scores], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'concept_alignment': [{'concept': c, 'score': s} for c, s in top_scores],
        'alpha_psa_mean': float(np.mean(latent.astype(np.float32))),
        'semantic_entropy': float(-np.sum([s * np.log(max(abs(s), 1e-6)) for _, s in top_scores])),
        'smoothed_embeddings': smooth_state,
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def concept_graph_reason(
    alignment: Sequence[Dict[str, Any]],
    concept_edges: Dict[Tuple[str, str], float],
    max_concepts: int = 20,
    drg_util: float = 0.0,
) -> CausalSemanticOutput:
    """
    Build small semantic graph linking high-scoring concepts.
    """
    start = time.time()
    fallbacks = 0

    if drg_util > 0.85:
        max_concepts = max(10, max_concepts // 2)
        fallbacks |= 1

    concept_scores = {item['concept']: item.get('score', 0.0) for item in alignment}
    top_concepts = sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)[:max_concepts]

    nodes = [concept for concept, _ in top_concepts]
    if not nodes:
        fallbacks |= 2
        nodes = ['none']

    edges = []
    weights = []

    for (c1, c2), weight in concept_edges.items():
        if c1 in concept_scores and c2 in concept_scores:
            combined_score = 0.5 * (concept_scores[c1] + concept_scores[c2])
            strength = combined_score * weight
            edges.append({'src': c1, 'dst': c2, 'weight': strength})
            weights.append(strength)

    if not weights:
        weights = [0.0]

    latent = np.asarray(weights, dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'concept_nodes': nodes,
        'concept_edges': edges,
        'n_concepts_linked': len(edges),
        'concept_entropy': float(np.std(latent.astype(np.float32))),
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def temporal_causal_fusion(
    causal_events: Sequence[Dict[str, Any]],
    semantic_events: Sequence[Dict[str, Any]],
    state: Optional[Dict[str, Any]] = None,
    ema_alpha: float = 0.4,
    drift_threshold: float = 2.0,
) -> CausalSemanticOutput:
    """
    Track causal & semantic alignment over time to detect regime shifts.
    """
    start = time.time()
    fallbacks = 0

    state = state if state is not None else {}
    cause_ema = state.get('p_cause_ema', 0.0)
    semantic_ema = state.get('alpha_psa_ema', 0.0)
    history = state.get('history', [])

    latest_p = np.mean([item.get('p_cause', 0.0) for item in causal_events]) if causal_events else 0.0
    latest_alpha = np.mean([item.get('score', 0.0) for item in semantic_events]) if semantic_events else 0.0

    cause_ema = ema_alpha * latest_p + (1 - ema_alpha) * cause_ema
    semantic_ema = ema_alpha * latest_alpha + (1 - ema_alpha) * semantic_ema

    combined_signal = 0.5 * (cause_ema + semantic_ema)
    history.append(combined_signal)
    if len(history) > 64:
        history = history[-64:]

    drift_flag = page_hinkley(np.asarray(history, dtype=np.float32), delta=0.05, lambda_threshold=drift_threshold)
    regime_change = bool(drift_flag > 0.5)

    state.update({
        'p_cause_ema': cause_ema,
        'alpha_psa_ema': semantic_ema,
        'history': history,
    })

    latent = np.asarray([cause_ema, semantic_ema, combined_signal], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'p_cause_ema': float(cause_ema),
        'alpha_psa_ema': float(semantic_ema),
        'combined_signal': float(combined_signal),
        'regime_shift_flag': regime_change,
        'history_length': len(history),
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)


def meaning_aggregator(
    causal_pairs: Sequence[Dict[str, Any]],
    alpha_psa: Sequence[Dict[str, Any]],
    eta_cf: float,
    top_k: int = 5,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    drg_util: float = 0.0,
) -> CausalSemanticOutput:
    """
    Fuse causal and semantic signals into interpretable insights.
    """
    start = time.time()
    fallbacks = 0

    if drg_util > 0.85:
        top_k = max(3, min(top_k, 5))
        fallbacks |= 1

    w1, w2, w3 = weights
    alpha_dict = {item['concept']: item['score'] for item in alpha_psa if 'concept' in item}

    insights = []
    insight_scores = []
    for pair in causal_pairs:
        concept = None
        best_score = -1.0
        for name, score in alpha_dict.items():
            if score > best_score:
                best_score = score
                concept = name

        p_cause = pair.get('p_cause', 0.0)
        direction_strength = pair.get('direction_strength', 0.0)
        combined_score = (
            w1 * p_cause +
            w2 * max(best_score, 0.0) +
            w3 * float(np.clip(eta_cf, 0.0, 1.0))
        )

        insight_scores.append(combined_score)
        insights.append({
            'src': pair.get('src'),
            'dst': pair.get('dst'),
            'concept': concept,
            'score': combined_score,
            'confidence': float(np.clip(combined_score, 0.0, 1.0)),
            'summary': f"Change in {pair.get('src')} influences {pair.get('dst')} "
                       f"via {concept or 'unknown concept'} with confidence {combined_score:.2f}",
            'direction_strength': direction_strength,
        })

    insights.sort(key=lambda item: item['score'], reverse=True)
    top_insights = insights[:top_k]

    if not top_insights:
        fallbacks |= 2
        top_insights = [{
            'src': None,
            'dst': None,
            'concept': 'none',
            'score': 0.0,
            'confidence': 0.0,
            'summary': "No actionable insight generated in current window.",
            'direction_strength': 0.0,
        }]

    latent = np.asarray([insight['score'] for insight in top_insights], dtype=np.float16)
    cost_ms = (time.time() - start) * 1000.0

    stats = {
        'insight_event': top_insights,
        'insight_count': len(top_insights),
        'avg_confidence': float(np.mean(latent.astype(np.float32))),
        'fallbacks_taken': fallbacks,
    }

    return CausalSemanticOutput(latent=latent, stats=stats, cost_hint_ms=cost_ms)



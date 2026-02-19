"""
Structural Operators — Tier 1 composite feature builders.

Builds interpretable structural features from Tier-0 primitives:
temporal fusion, cross-variable alignment, low-rank compression,
sparse interactions, and schema-aware pooling.

All operators are online, bounded, deterministic, and DRG-aware.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from scarcity.engine.operators.attention_ops import attn_linear, layernorm, rmsnorm, pooling_avg
from scarcity.engine.operators.sketch_ops import countsketch, latent_clip, _deterministic_hash

logger = logging.getLogger(__name__)


@dataclass
class StructuralOutput:
    """Output from a structural operator."""
    latent: np.ndarray
    stats: Dict[str, Any]
    cost_hint_ms: float


def temporal_fusion(
    X_sel: np.ndarray,
    H: Optional[np.ndarray] = None,
    k_short: int = 5,
    k_medium: int = 17,
    use_residual_attention: bool = True,
    drg_util: float = 0.0,
    return_sequence: bool = True,
    seed: int = 0,
    path_id: str = "default"
) -> StructuralOutput:
    """
    Temporal fusion: fuse short- and medium-horizon temporal patterns.
    
    Fuses short- and medium-horizon temporal patterns for a path without full
    recurrent state or multi-head attention blow-up.
    
    Args:
        X_sel: Selected variables [W, d_v]
        H: Optional Tier-0 latent [W, D0]
        k_short: Short kernel size
        k_medium: Medium kernel size
        use_residual_attention: Whether to apply residual attention
        drg_util: Current DRG utilization (0-1)
        return_sequence: Return full sequence vs pooled
        seed: Deterministic seed
        path_id: Path identifier
        
    Returns:
        StructuralOutput with fused latent
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate inputs
    if np.any(np.isnan(X_sel)) or np.any(np.isinf(X_sel)):
        logger.warning(f"TFU: NaN/Inf detected in input, zeroing")
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    W, d_v = X_sel.shape
    
    # DRG adaptation: reduce complexity under pressure
    if drg_util > 0.8:
        k_medium = k_short
        use_residual_attention = False
        fallbacks |= 2
    elif drg_util > 0.7:
        # Reduce medium kernel slightly
        k_medium = max(k_short + 2, int(k_medium * 0.8))
    
    # Step 1: Temporal bases via causal convolution
    # Depthwise-separable convolutions with FP32 accumulation
    conv_short = _depthwise_causal_conv(X_sel, k_short)
    conv_medium = _depthwise_causal_conv(X_sel, k_medium)
    
    # Step 2: Gated fusion with lightweight linear map
    # Compute gate from combined features [conv_s, conv_m]
    combined = np.concatenate([conv_short, conv_medium], axis=1)  # [W, 2*d_v]
    
    # Lightweight linear map: fixed scaling from statistics
    # Use per-channel statistics to compute gate
    stats_s = np.mean(np.abs(conv_short), axis=0, keepdims=True)
    stats_m = np.mean(np.abs(conv_medium), axis=0, keepdims=True)
    
    # Gate: sigmoid-like squashing of relative variance
    # g_t favors the mode with higher activity
    scale = np.maximum(np.maximum(stats_s, stats_m), 1e-6)
    g_raw = (stats_s - stats_m) / scale
    g = 1.0 / (1.0 + np.exp(-10.0 * g_raw))  # Fast sigmoid approximation
    
    # Broadcast g to [W, d_v]
    if g.shape[0] == 1:
        g = np.broadcast_to(g, (W, d_v))
    
    # Fused output per timestep
    Y = g * conv_short + (1 - g) * conv_medium
    
    # Validate fusion output
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        logger.warning(f"TFU: NaN/Inf after fusion, using fallback")
        Y = 0.5 * (conv_short + conv_medium)
        fallbacks |= 4
    
    # Step 3: Residual attention (if enabled and within budget)
    kernels_used = 2
    if use_residual_attention and drg_util < 0.8:
        try:
            Y_attn = attn_linear(Y, Y, Y)
            if not (np.any(np.isnan(Y_attn)) or np.any(np.isinf(Y_attn))):
                Y = 0.5 * Y + 0.5 * Y_attn
                kernels_used = 3
            else:
                fallbacks |= 8
        except Exception as e:
            logger.warning(f"TFU: Attention failed: {e}")
            fallbacks |= 8
    
    # Step 4: Normalization & clipping
    Y_norm = rmsnorm(Y)
    
    # Percentile clamping (p99.9)
    Y_flat = Y_norm.flatten()
    Y_clipped = latent_clip(Y_flat, p=99.9)
    Y = Y_clipped.reshape(Y_norm.shape)
    
    # Energy accounting
    input_energy = np.sum(X_sel ** 2)
    output_energy = np.sum(Y ** 2)
    energy_ratio = output_energy / (input_energy + 1e-12)
    
    if energy_ratio > 1.05:
        logger.warning(f"TFU: Energy amplification {energy_ratio:.3f}, rescaling")
        Y = Y / np.sqrt(energy_ratio)
        fallbacks |= 16
    
    # Pool if requested
    if not return_sequence:
        Y = pooling_avg(Y, axis=0)
    
    cost_ms = (time.time() - start_time) * 1000

    drg_level = float(np.clip(drg_util, 0.0, 1.0))
    complexity_score = float((k_short + k_medium) * max(1, d_v))
    attention_factor = 1.3 if kernels_used == 3 else 0.9
    budget_factor = 1.0 - 0.35 * drg_level
    synthetic_cost = (0.015 * complexity_score + 0.002 * W) * attention_factor * budget_factor
    cost_hint = max(0.25, 0.25 * cost_ms + synthetic_cost)
    
    stats = {
        'tfu_latency_ms': cost_ms,
        'tfu_kernels_used': kernels_used,
        'gate_mean': float(np.mean(g)),
        'fusion_std': float(np.std(Y)),
        'energy_ratio': float(energy_ratio),
        'fallbacks_taken': fallbacks
    }
    
    return StructuralOutput(
        latent=Y.astype(np.float16),
        stats=stats,
        cost_hint_ms=cost_hint
    )


def _depthwise_causal_conv(X: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Depthwise separable causal convolution.
    
    Efficient causal convolution with FP32 accumulation.
    
    Args:
        X: Input tensor [W, d_v] in FP16
        kernel_size: Kernel size
        
    Returns:
        Convolved output [W, d_v] in FP32
    """
    W, _ = X.shape
    k = max(1, int(kernel_size))

    # Convert to FP32 for accumulation
    X_fp32 = X.astype(np.float32)

    # Prefix sum for O(W) causal window means
    prefix = np.vstack([np.zeros((1, X_fp32.shape[1]), dtype=np.float32),
                        np.cumsum(X_fp32, axis=0)])
    idx = np.arange(W)
    start = np.maximum(0, idx - k + 1)
    end = idx + 1
    window_len = (end - start).astype(np.float32).reshape(-1, 1)
    summed = prefix[end] - prefix[start]
    return summed / window_len


def cross_align(
    X_pair: np.ndarray,
    lag_offsets: Optional[List[int]] = None,
    max_lag: int = 6,
    use_fft: bool = True,
    drg_util: float = 0.0,
    return_sequence: bool = True,
    seed: int = 0,
    path_id: str = "default"
) -> StructuralOutput:
    """
    Cross-align variables by latent covariance with lead-lag detection.
    
    Aligns variables by their latent covariance to emphasize co-movement and
    lead-lag structure across selected variables.
    
    Args:
        X_pair: Pair of series [W, 2] or group [W, d_v]
        lag_offsets: Optional lag offsets from Controller/Encoder
        max_lag: Maximum lag to test
        use_fft: Use FFT for efficiency
        drg_util: Current DRG utilization
        return_sequence: Return full sequence vs pooled
        seed: Deterministic seed
        path_id: Path identifier
        
    Returns:
        StructuralOutput with aligned latent and alignment score
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG adaptation
    if drg_util > 0.8:
        max_lag = min(2, max_lag)
        use_fft = False
        fallbacks |= 1
    elif drg_util > 0.7:
        max_lag = min(4, max_lag)
    
    W, d_v = X_pair.shape
    
    # Skip if single variable or W too small
    if d_v == 1 or W < 32:
        output = X_pair.astype(np.float16)
        if not return_sequence:
            output = pooling_avg(output, axis=0)
        return StructuralOutput(
            latent=output,
            stats={'xal_latency_ms': (time.time() - start_time) * 1000,
                   'xal_rho_hat': 0.0, 'xal_best_lag': 0},
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Step 1: Covariance sketch using CountSketch for efficiency
    # For pairs, we can use direct correlation; for groups, use sketched covariance
    best_rho = 0.0
    best_lag = 0
    
    if d_v >= 2:
        # Pair-wise alignment: test lead-lag structure
        x1, x2 = X_pair[:, 0].astype(np.float32), X_pair[:, 1].astype(np.float32)
        
        # Normalize for correlation
        x1_norm = (x1 - np.mean(x1)) / (np.std(x1) + 1e-12)
        x2_norm = (x2 - np.mean(x2)) / (np.std(x2) + 1e-12)
        
        # Test lags: bounded symmetric set
        lag_set = list(range(-max_lag, max_lag + 1))
        if lag_offsets:
            lag_set = sorted(set(lag_set + lag_offsets))
            lag_set = [l for l in lag_set if abs(l) <= max_lag]
        
        for lag in lag_set:
            if abs(lag) >= W // 4:
                continue
            
            # Shift x2 by lag
            if lag == 0:
                x2_shifted = x2_norm
            elif lag > 0:
                x2_shifted = np.concatenate([np.zeros(lag, dtype=np.float32), x2_norm[:-lag]])
            else:
                x2_shifted = np.concatenate([x2_norm[-lag:], np.zeros(-lag, dtype=np.float32)])
            
            # Correlation: use FFT for efficiency if enabled and window large enough
            if use_fft and W >= 256:
                rho = _fft_cross_corr(x1_norm, x2_shifted)
            else:
                # Time-domain covariance
                rho = np.mean(x1_norm * x2_shifted)
                rho = np.clip(rho, -1.0, 1.0)
            
            if not np.isnan(rho) and not np.isinf(rho) and abs(rho) > abs(best_rho):
                best_rho = rho
                best_lag = lag
    
    # Step 2: Alignment weights via sigmoid-like squashing
    # Convert best lag's score to weight, guarding against noise
    alpha = 1.0 / (1.0 + np.exp(-10.0 * best_rho))  # Fast sigmoid approximation
    
    # Step 3: Attention reweight: scale/shift by alignment weights
    # Emphasize aligned segments
    aligned_baseline = np.mean(X_pair, axis=1, keepdims=True)
    output = alpha * X_pair + (1 - alpha) * aligned_baseline
    
    # Validate output
    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        logger.warning(f"XAL: NaN/Inf after alignment, using fallback")
        output = X_pair
        fallbacks |= 2
    
    # Normalize
    output = rmsnorm(output)
    
    if not return_sequence:
        output = pooling_avg(output, axis=0)
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=output.astype(np.float16),
        stats={
            'xal_latency_ms': cost_ms,
            'xal_rho_hat': float(best_rho),
            'xal_best_lag': best_lag,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def _fft_cross_corr(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    FFT-based cross-correlation approximation.
    
    Efficiently computes correlation via FFT convolution.
    """
    n = len(x1)
    
    # Zero-pad to next power of 2 for efficiency
    n_padded = 2 ** int(np.ceil(np.log2(max(n, 16))))
    
    x1_pad = np.pad(x1.astype(np.float32), (0, n_padded - n), mode='constant')
    x2_pad = np.pad(x2.astype(np.float32), (0, n_padded - n), mode='constant')
    
    # FFT
    X1_fft = np.fft.rfft(x1_pad)
    X2_fft = np.fft.rfft(x2_pad)
    
    # Cross-correlation: IFFT of conjugate product
    cross = np.fft.irfft(X1_fft * np.conj(X2_fft))
    
    # Normalize by L2 norms
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    if norm1 * norm2 < 1e-12:
        return 0.0
    
    # Take zero-lag correlation
    corr = cross[0] / (norm1 * norm2)
    
    return float(np.clip(corr, -1.0, 1.0))


def lowrank_mix(
    H: np.ndarray,
    rank: Optional[int] = None,
    power_iter: bool = True,
    eta_min: float = 0.6,
    drg_util: float = 0.0,
    drg_sketch_dim: int = 512,
    seed: int = 0,
    path_id: str = "default"
) -> StructuralOutput:
    """
    Low-rank compression via randomized SVD (Halko-Martinsson).
    
    Compresses high-dimensional latents into a compact subspace while keeping
    most energy and decorrelating features.
    
    Args:
        H: Tier-0 latent [W, D0]
        rank: Target rank (if None, inferred from sketch_dim)
        power_iter: Use power iteration (at most one)
        eta_min: Minimum energy retention
        drg_util: Current DRG utilization
        drg_sketch_dim: Sketch dimension from DRG
        seed: Deterministic seed
        path_id: Path identifier
        
    Returns:
        StructuralOutput with compressed latent and energy retention
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        logger.warning(f"LRM: NaN/Inf in input, zeroing")
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    W, D0 = H.shape
    
    # Determine rank from sketch_dim
    if rank is None:
        rank = min(256, drg_sketch_dim // 4)
    
    rank = min(rank, D0, W)  # Bounded by dimensions
    rank = max(1, rank)  # At least rank 1
    
    # DRG adaptation
    if drg_util > 0.55:
        power_iter = False
        fallbacks |= 2
    elif drg_util > 0.45:
        # Reduce rank slightly
        rank = max(1, int(rank * 0.9))
    
    # Step 1: Randomized range finder (Halko-Martinsson)
    # Generate Omega ∈ ℝ^{D0×(r+p)} with p small (oversampling)
    rng = _deterministic_hash(seed, path_id)
    p = min(8, rank // 2, max(1, D0 - rank))  # Oversampling parameter
    target_cols = rank + p
    target_cols = min(target_cols, W)  # Can't have more columns than rows
    
    Omega = rng.standard_normal(size=(D0, target_cols)).astype(np.float32)
    
    # Compute Y = HΩ
    Y = H.astype(np.float32) @ Omega  # [W, target_cols], FP32 accumulation
    
    # Step 2: Power iteration (at most one) to sharpen spectrum
    if power_iter and drg_util < 0.55:
        try:
            Y = H @ (H.T @ Y)  # [W, target_cols]
        except Exception as e:
            logger.warning(f"LRM: Power iteration failed: {e}")
            fallbacks |= 4
    
    # Step 3: Orthogonalization via online Gram-Schmidt or QR
    try:
        Q, R = np.linalg.qr(Y, mode='reduced')  # Q: [W, min(W, target_cols)]
    except Exception as e:
        logger.warning(f"LRM: QR failed: {e}, using fallback")
        # Fallback: simple normalization
        Q = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-12)
        fallbacks |= 8
    
    # Q spans the found subspace
    # Step 4: Projection Z = HQ (but we want Z ∈ ℝ^{W×r}, so use Q directly)
    # Actually, we want the compressed representation: columns of Q
    Z = Q
    
    # Truncate to requested rank
    final_rank = min(rank, Z.shape[1])
    if Z.shape[1] > final_rank:
        Z = Z[:, :final_rank]
    
    actual_rank = Z.shape[1]
    
    # Step 5: Rescale and normalize
    # Energy retention check
    input_energy = np.sum(H.astype(np.float32) ** 2)
    output_energy = np.sum(Z.astype(np.float32) ** 2)
    eta = output_energy / (input_energy + 1e-12)
    
    # Rescale to maintain energy if needed
    if eta < eta_min and actual_rank >= rank:
        logger.warning(f"LRM: Low energy retention {eta:.3f} < {eta_min}")
        # Attempt to increase rank if headroom available
        if rank < min(D0 // 2, W // 2):
            # Recursive call with increased rank
            return lowrank_mix(H, rank=min(rank + 32, W, D0), power_iter=False,
                             eta_min=eta_min, drg_util=min(1.0, drg_util + 0.2),
                             drg_sketch_dim=drg_sketch_dim, seed=seed, path_id=path_id)
        else:
            # Scale up to meet minimum
            scale = np.sqrt(eta_min / (eta + 1e-12))
            Z = Z * scale
            eta = eta_min
            fallbacks |= 16
    
    # Clamp energy if too high (guard against amplification)
    if eta > 1.05:
        logger.warning(f"LRM: Energy amplification {eta:.3f}, rescaling")
        Z = Z / np.sqrt(eta)
        eta = 1.0
        fallbacks |= 32
    
    # Normalize by RMS
    Z = rmsnorm(Z)
    
    # Validate output
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.warning(f"LRM: NaN/Inf in output, using fallback")
        Z = H[:, :actual_rank] if actual_rank <= H.shape[1] else H
        Z = rmsnorm(Z)
        fallbacks |= 64
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=Z.astype(np.float16),
        stats={
            'lrm_latency_ms': cost_ms,
            'lrm_rank': actual_rank,
            'lrm_energy_eta': float(eta),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def sparse_interact(
    X_or_Z: np.ndarray,
    d_hash: Optional[int] = None,
    max_pairs_factor: float = 3.0,
    target_sparsity: float = 0.15,
    drg_util: float = 0.0,
    drg_sketch_dim: int = 512,
    seed: int = 0,
    path_id: str = "default"
) -> StructuralOutput:
    """
    Sparse multiplicative interactions via feature hashing.
    
    Introduces approximate multiplicative interactions between selected variables
    while keeping memory bounded via feature hashing.
    
    Args:
        X_or_Z: Input [W, d_v] or [W, r]
        d_hash: Hash dimension (if None, from sketch_dim)
        max_pairs_factor: Max pairs multiplier
        target_sparsity: Target sparsity rate (fraction of non-zero activations)
        drg_util: Current DRG utilization
        drg_sketch_dim: Sketch dimension from DRG
        seed: Deterministic seed
        path_id: Path identifier
        
    Returns:
        StructuralOutput with hashed interactions and sparsity
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(X_or_Z)) or np.any(np.isinf(X_or_Z)):
        logger.warning(f"SIN: NaN/Inf in input, zeroing")
        X_or_Z = np.nan_to_num(X_or_Z, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    W, d_v = X_or_Z.shape
    
    # Determine hash dimension from sketch_dim
    if d_hash is None:
        d_hash = drg_sketch_dim
    
    d_hash = min(d_hash, 2048)  # Cap for 4 GB profile
    
    # DRG adaptation
    if drg_util > 0.7:
        target_sparsity = target_sparsity / 2.0
        max_pairs_factor = max_pairs_factor * 0.5
        d_hash = min(d_hash, drg_sketch_dim // 2)
        fallbacks |= 2
    elif drg_util > 0.55:
        target_sparsity = target_sparsity * 0.8
        max_pairs_factor = max_pairs_factor * 0.8
    
    # Limit pairs: B bounded by DRG
    max_pairs = min(max(int(d_v * max_pairs_factor), d_v), int(3.0 * d_v))
    max_pairs = min(max_pairs, 50)  # Hard cap
    
    # Step 1: Feature hashing for bounded subset of pairs
    rng = _deterministic_hash(seed, path_id)
    
    # Select pairs: prioritize adjacent and balanced pairs
    pairs = []
    if d_v >= 2:
        # Adjacent pairs
        for i in range(min(d_v - 1, max_pairs // 2)):
            pairs.append((i, i + 1))
        
        # Additional pairs if budget allows
        remaining = max_pairs - len(pairs)
        for i in range(min(remaining, d_v - 1)):
            idx1 = i % d_v
            idx2 = (i + 2) % d_v
            if idx1 != idx2 and (idx1, idx2) not in pairs:
                pairs.append((idx1, idx2))
    
    if len(pairs) == 0:
        # Fallback: no pairs, return input
        logger.warning(f"SIN: No pairs selected, returning input")
        Z_gated = X_or_Z[:, :min(d_hash, d_v)]
        if Z_gated.shape[1] < d_hash:
            # Pad with zeros
            Z_padded = np.zeros((W, d_hash), dtype=np.float32)
            Z_padded[:, :Z_gated.shape[1]] = Z_gated
            Z_gated = Z_padded
        actual_sparsity = 1.0
    else:
        # Feature hashing: CountSketch-style
        Z_hash = np.zeros((W, d_hash), dtype=np.float32)
        
        # Generate hash functions once (deterministic per (seed, path_id))
        # For efficiency, use same hash function across pairs but different sign
        for pair_idx, (idx1, idx2) in enumerate(pairs):
            # Product feature
            prod = X_or_Z[:, idx1].astype(np.float32) * X_or_Z[:, idx2].astype(np.float32)
            
            # CountSketch hash: signed hash
            # Use deterministic hash per pair
            pair_rng = _deterministic_hash(seed + pair_idx, path_id + str(pair_idx))
            h = pair_rng.integers(0, d_hash, size=W)
            s = pair_rng.choice([-1, 1], size=W)
            
            # Accumulate into hash table
            for t in range(W):
                Z_hash[t, h[t]] += s[t] * prod[t]
        
        # Normalize by expected collisions to keep variance stable
        collision_scale = np.sqrt(d_hash / max(len(pairs), 1))
        Z_hash = Z_hash * collision_scale
        
        # Step 2: Gating with moving threshold (EMA of previous absolute medians)
        # For first pass, use percentile
        tau = np.percentile(np.abs(Z_hash), 100 * (1 - target_sparsity))
        tau = max(tau, 1e-6)  # Guard against zero threshold
        
        # ReLU-like gate with threshold
        Z_gated = np.maximum(Z_hash - tau, 0.0)
        
        # Sparsity check
        actual_sparsity = np.mean(Z_gated > 0.0)
        
        # Adjust threshold if sparsity far from target
        if actual_sparsity > target_sparsity * 2.0:
            # Too many non-zeros: increase threshold
            tau_new = np.percentile(np.abs(Z_hash), 100 * (1 - target_sparsity))
            Z_gated = np.maximum(Z_hash - tau_new, 0.0)
            actual_sparsity = np.mean(Z_gated > 0.0)
        elif actual_sparsity < target_sparsity * 0.5 and target_sparsity > 0:
            # Too few non-zeros: decrease threshold
            tau_new = np.percentile(np.abs(Z_hash), 100 * (1 - target_sparsity * 2.0))
            Z_gated = np.maximum(Z_hash - tau_new, 0.0)
            actual_sparsity = np.mean(Z_gated > 0.0)
    
    # Step 3: Normalization: divide by √(expected collisions) and RMSNorm
    Z_norm = rmsnorm(Z_gated)
    
    # Validate output
    if np.any(np.isnan(Z_norm)) or np.any(np.isinf(Z_norm)):
        logger.warning(f"SIN: NaN/Inf after gating, using fallback")
        Z_norm = np.zeros((W, d_hash), dtype=np.float32)
        fallbacks |= 4
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=Z_norm.astype(np.float16),
        stats={
            'sin_latency_ms': cost_ms,
            'sin_d_hash': d_hash,
            'sin_sparsity': float(actual_sparsity),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def structural_pool(
    X_grouped: np.ndarray,
    group_indices: Optional[List[List[int]]] = None,
    mode: str = 'trimmed_mean',
    trim_pct: float = 0.1,
    drg_util: float = 0.0,
    return_sequence: bool = True
) -> StructuralOutput:
    """
    Aggregate features by schema groups.
    
    Aggregates features by schema groups (e.g., "prices", "employment", "policy")
    to create interpretable group signals.
    
    Args:
        X_grouped: Input [W, P] or any latent [W, d]
        group_indices: List of group variable indices (from SchemaManager)
        mode: Pooling mode (mean, max, trimmed_mean)
        trim_pct: Trim percentage for trimmed_mean (q=80-90% → trim_pct=0.1-0.2)
        drg_util: Current DRG utilization
        return_sequence: Return full sequence vs pooled
        
    Returns:
        StructuralOutput with aggregated latent
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(X_grouped)) or np.any(np.isinf(X_grouped)):
        logger.warning(f"SPool: NaN/Inf in input, zeroing")
        X_grouped = np.nan_to_num(X_grouped, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    W, P = X_grouped.shape
    
    # If no groups specified, treat as single group
    if group_indices is None or len(group_indices) == 0:
        group_indices = [list(range(P))]
    
    # DRG adaptation: if latency_high, produce only pooled vector
    if drg_util > 0.8:
        return_sequence = False
        fallbacks |= 2
    
    # Aggregate per group
    Z_groups = []
    
    for group in group_indices:
        if len(group) == 0:
            continue
        
        # Filter valid indices
        valid_indices = [idx for idx in group if 0 <= idx < P]
        if len(valid_indices) == 0:
            continue
        
        group_data = X_grouped[:, valid_indices].astype(np.float32)
        
        if mode == 'mean':
            # GroupMean: per-group mean
            z_group = np.mean(group_data, axis=1, keepdims=True)
        elif mode == 'max':
            # GroupMax: robust to outliers
            z_group = np.max(group_data, axis=1, keepdims=True)
        elif mode == 'trimmed_mean':
            # GroupTrimmedMean: mean of middle q% (q=80-90)
            if len(valid_indices) < 3:
                # Too few elements for trimming
                z_group = np.mean(group_data, axis=1, keepdims=True)
            else:
                k = max(1, int(len(valid_indices) * trim_pct))
                sorted_data = np.sort(group_data, axis=1)
                if k * 2 < len(valid_indices):
                    trimmed = sorted_data[:, k:-k]
                else:
                    trimmed = sorted_data
                z_group = np.mean(trimmed, axis=1, keepdims=True)
        else:
            # Default: mean
            z_group = np.mean(group_data, axis=1, keepdims=True)
        
        Z_groups.append(z_group)
    
    # Concatenate groups
    if Z_groups:
        Z = np.concatenate(Z_groups, axis=1)
    else:
        # Fallback: single group mean
        Z = np.mean(X_grouped, axis=1, keepdims=True)
        fallbacks |= 4
    
    # Validate output
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.warning(f"SPool: NaN/Inf after pooling, using fallback")
        Z = np.mean(X_grouped, axis=1, keepdims=True)
        fallbacks |= 8
    
    # Pool to vector if requested
    if not return_sequence:
        Z = pooling_avg(Z, axis=0)
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=Z.astype(np.float16),
        stats={
            'spool_groups': len(group_indices),
            'spool_mode': mode,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def regime_gate(
    Z: np.ndarray,
    regime_indicator: float = 0.0,
    regime_stat: float = 0.0,
    drg_util: float = 0.0
) -> StructuralOutput:
    """
    Condition structural outputs on detected regimes.
    
    Conditions structural outputs on detected regimes (e.g., monetary tightening),
    gently modulating amplitudes.
    
    Args:
        Z: Structural latent [W, D]
        regime_indicator: Regime ID r ∈ {0,1,…} (normalized to [0,1])
        regime_stat: Regime statistic κ (e.g., inflation surprise)
        drg_util: Current DRG utilization
        
    Returns:
        StructuralOutput with regime-gated latent and mixing coefficient
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.warning(f"RGate: NaN/Inf in input, zeroing")
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    # DRG hooks: if meta signals absent or util > 0.8, set π_r ≡ 1 (no gating)
    if regime_stat == 0.0 or drg_util > 0.8:
        pi_r = 1.0
        if drg_util > 0.8:
            fallbacks |= 2
    else:
        # Compute mixing coefficient: π_r = σ(a_r·κ + b_r)
        # where σ is sigmoid approximation, (a_r, b_r) are fixed scalings
        a_r, b_r = 2.0, 0.0  # Fixed scalings
        
        # Fast sigmoid approximation: 1 / (1 + exp(-x))
        pi_r = 1.0 / (1.0 + np.exp(-(a_r * regime_stat + b_r)))
        pi_r = float(np.clip(pi_r, 0.0, 1.0))
    
    # Baseline: conservative baseline (e.g., GroupMean)
    z_base = np.mean(Z, axis=0, keepdims=True)
    
    # Output: Z_RG = π_r·Z_* + (1−π_r)·Z_base
    Z_gated = pi_r * Z + (1 - pi_r) * z_base
    
    # Validate output
    if np.any(np.isnan(Z_gated)) or np.any(np.isinf(Z_gated)):
        logger.warning(f"RGate: NaN/Inf after gating, using fallback")
        Z_gated = Z
        fallbacks |= 4
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=Z_gated.astype(np.float16),
        stats={
            'rgate_pi': pi_r,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def denoise_lite(
    Z: np.ndarray,
    alpha_fast: float = 0.3,
    alpha_slow: float = 0.05,
    blend_alpha: float = 0.6,
    drg_util: float = 0.0
) -> StructuralOutput:
    """
    Stabilize noisy structural signals without Kalman state explosion.
    
    Uses two-time-scale EMA to stabilize noisy structural signals without
    requiring expensive Kalman filtering state.
    
    Args:
        Z: Structural latent [W, D]
        alpha_fast: Fast EMA coefficient (higher = more responsive)
        alpha_slow: Slow EMA coefficient (lower = more smoothing)
        blend_alpha: Blending weight α (Z_DNL = α·m_f + (1−α)·m_s)
        drg_util: Current DRG utilization
        
    Returns:
        StructuralOutput with denoised latent and noise score
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.warning(f"DNL: NaN/Inf in input, zeroing")
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    W, D = Z.shape
    
    # DRG hooks: increase smoothing (reduce α) when latency_high or unstable
    if drg_util > 0.7:
        blend_alpha = blend_alpha * 0.8  # Favor slow EMA
        fallbacks |= 2
    
    # Two-time-scale EMA: fast m_f and slow m_s
    # Initialize EMAs (use first value)
    m_fast = Z[0].copy().astype(np.float32)
    m_slow = Z[0].copy().astype(np.float32)
    
    output = np.zeros_like(Z, dtype=np.float32)
    
    # Online EMA updates
    for t in range(W):
        # Update EMAs: m_t = α·z_t + (1−α)·m_{t-1}
        m_fast = alpha_fast * Z[t].astype(np.float32) + (1 - alpha_fast) * m_fast
        m_slow = alpha_slow * Z[t].astype(np.float32) + (1 - alpha_slow) * m_slow
        
        # Blend: Z_DNL = α·m_f + (1−α)·m_s
        output[t] = blend_alpha * m_fast + (1 - blend_alpha) * m_slow
    
    # Noise score: ν = |m_f − m_s| / (|m_s|+ε)
    # Telemetered for drift detection
    noise_diff = np.abs(m_fast - m_slow)
    noise_score = np.mean(noise_diff) / (np.mean(np.abs(m_slow)) + 1e-12)
    noise_score = float(noise_score)
    
    # Validate output
    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        logger.warning(f"DNL: NaN/Inf after denoising, using fallback")
        output = Z.astype(np.float32)
        fallbacks |= 4
    
    cost_ms = (time.time() - start_time) * 1000
    
    return StructuralOutput(
        latent=output.astype(np.float16),
        stats={
            'dnl_noise_score': noise_score,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


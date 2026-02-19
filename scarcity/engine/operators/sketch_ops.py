"""
Sketch Operators — Polynomial and tensor sketching primitives.

Efficient projections for higher-order interactions without explicit enumeration.
Uses deterministic hashing for reproducibility.
"""

import numpy as np
import hashlib
from typing import Tuple


def _deterministic_hash(seed: int, path_id: str) -> np.random.Generator:
    """
    Generate deterministic RNG from seed and path_id.
    
    Args:
        seed: Base seed
        path_id: Path identifier
        
    Returns:
        NumPy RNG with deterministic state
    """
    combined = f"{seed}_{path_id}".encode('utf-8')
    hash_val = int(hashlib.md5(combined).hexdigest()[:16], 16)
    return np.random.default_rng(hash_val)


def poly_sketch(x, degree: int = 2, dim: int = 512, seed: int = 0, path_id: str = "default"):
    """
    Polynomial sketch projection using CountSketch + FFT convolution.
    
    Approximates polynomial feature maps efficiently.
    
    Args:
        x: Input tensor [seq, d] or [d]
        degree: Polynomial degree
        dim: Output dimension
        seed: Base seed for determinism
        path_id: Path identifier for determinism
        
    Returns:
        Sketched tensor [dim] in FP16
    """
    # Flatten if needed
    x_flat = x.flatten()
    d = len(x_flat)
    
    if d == 0:
        return np.zeros(dim, dtype=np.float16)
    
    # Generate deterministic CountSketch hash functions
    rng = _deterministic_hash(seed, path_id)
    
    # CountSketch: random sign and hash
    h = rng.integers(0, dim, size=d)  # Hash function
    s = rng.choice([-1, 1], size=d)   # Sign function
    
    # Compute sketch via convolution approximation
    # For degree > 1, we approximate polynomial features
    result = np.zeros(dim, dtype=np.float32)  # Accumulate in FP32
    
    if degree == 1:
        # Linear: just CountSketch
        for i in range(d):
            result[h[i]] += s[i] * x_flat[i]
    
    elif degree == 2:
        # Quadratic: approximate by squaring then sketching
        x_sq = x_flat ** 2
        for i in range(d):
            result[h[i]] += s[i] * x_flat[i]
        for i in range(d):
            # Second-order term
            j = h[i]
            result[j] += 0.5 * s[i] * x_sq[i]
    
    else:
        # Higher degree: recursive approximation
        for deg in range(1, degree + 1):
            x_deg = x_flat ** deg
            weight = 1.0 / (deg if deg > 0 else 1.0)
            for i in range(d):
                result[h[i]] += weight * s[i] * x_deg[i]
    
    # Normalize and convert to FP16
    result = result / np.linalg.norm(result + 1e-12)
    return result.astype(np.float16)


def tensor_sketch(x1, x2, dim: int = 512, seed: int = 0, path_id: str = "default"):
    """
    Tensor sketch for pairwise interactions.
    
    Efficiently approximates x1 ⊗ x2 (Kronecker product sketch).
    
    Args:
        x1: First input tensor [d1]
        x2: Second input tensor [d2]
        dim: Output dimension
        seed: Base seed
        path_id: Path identifier
        
    Returns:
        Sketched tensor [dim] in FP16
    """
    d1, d2 = len(x1), len(x2)
    
    if d1 == 0 or d2 == 0:
        return np.zeros(dim, dtype=np.float16)
    
    # Generate deterministic hash functions
    rng = _deterministic_hash(seed, path_id)
    
    h1 = rng.integers(0, dim, size=d1)
    s1 = rng.choice([-1, 1], size=d1)
    h2 = rng.integers(0, dim, size=d2)
    s2 = rng.choice([-1, 1], size=d2)
    
    # Tensor sketch via FFT convolution: O(dim log dim) + O(d1+d2)
    cs1 = np.zeros(dim, dtype=np.float32)
    cs2 = np.zeros(dim, dtype=np.float32)
    for i in range(d1):
        cs1[h1[i]] += s1[i] * x1[i]
    for j in range(d2):
        cs2[h2[j]] += s2[j] * x2[j]

    fft1 = np.fft.fft(cs1)
    fft2 = np.fft.fft(cs2)
    result = np.fft.ifft(fft1 * fft2).real.astype(np.float32)
    
    # Normalize
    result = result / np.linalg.norm(result + 1e-12)
    return result.astype(np.float16)


def countsketch(x, dim: int = 512, seed: int = 0, path_id: str = "default"):
    """
    CountSketch: linear-time dimensionality reduction.
    
    Args:
        x: Input tensor [d]
        dim: Output dimension
        seed: Base seed
        path_id: Path identifier
        
    Returns:
        Sketched tensor [dim]
    """
    if len(x) == 0:
        return np.zeros(dim, dtype=np.float16)
    
    rng = _deterministic_hash(seed, path_id)
    h = rng.integers(0, dim, size=len(x))
    s = rng.choice([-1, 1], size=len(x))
    
    result = np.zeros(dim, dtype=np.float32)
    for i, val in enumerate(x):
        result[h[i]] += s[i] * val
    
    return result.astype(np.float16)


def latent_clip(latent, p: float = 99.9):
    """
    Percentile-based clipping to prevent extreme activations.
    
    Args:
        latent: Input latent tensor
        p: Percentile threshold
        
    Returns:
        Clipped latent
    """
    if len(latent) == 0:
        return latent
    
    threshold = np.percentile(np.abs(latent), p)
    return np.clip(latent, -threshold, threshold)

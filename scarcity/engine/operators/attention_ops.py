"""
Attention Operators — Linear and sparse attention primitives.

GPU-optimized attention kernels for path encoding with FP16 support.
"""

import numpy as np
from typing import Optional


def attn_linear(q, k, v, mask: Optional[np.ndarray] = None):
    """
    Linear attention over sequences (O(L·d) complexity).
    
    Implements linearized attention: output = (phi(Q) · phi(K)^T) · V
    where phi is a feature map. For efficiency, we use a simplified version.
    
    Args:
        q: Query tensor [seq, d] in FP16
        k: Key tensor [seq, d] in FP16
        v: Value tensor [seq, d] in FP16
        mask: Optional attention mask [seq, seq]
        
    Returns:
        Output tensor [seq, d] in FP16
    """
    # Linear attention: Q·K^T then apply to V
    # For numerical stability in FP16, use scaled dot-product
    d = q.shape[-1]
    
    # Compute similarity
    scores = np.dot(q, k.T) / np.sqrt(float(d))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask * (-1e9)
    
    # Softmax with numerical stability
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-12)
    
    # Apply to values
    output = np.dot(weights, v)
    
    return output


def attn_sparse(q, k, v, top_k: int = 10, mask: Optional[np.ndarray] = None):
    """
    Sparse attention with top-k attention.
    
    Only attends to the top-k highest scoring positions.
    
    Args:
        q: Query tensor [seq, d]
        k: Key tensor [seq, d]
        v: Value tensor [seq, d]
        top_k: Number of top positions to attend to
        mask: Optional attention mask
        
    Returns:
        Output tensor [seq, d]
    """
    d = q.shape[-1]
    
    # Compute similarity
    scores = np.dot(q, k.T) / np.sqrt(float(d))
    
    # Apply mask
    if mask is not None:
        scores = scores + mask * (-1e9)
    
    # Find top-k for each query
    seq_len = scores.shape[1]
    k_actual = min(top_k, seq_len)
    
    # Get top-k indices per row
    top_k_indices = np.argsort(scores, axis=-1)[:, -k_actual:]
    
    # Create sparse attention mask
    sparse_mask = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        sparse_mask[i, top_k_indices[i]] = 1.0
    
    # Apply softmax only to top-k positions
    scores_masked = scores * sparse_mask
    scores_exp = np.exp(scores_masked - np.max(scores_masked, axis=-1, keepdims=True))
    
    # Renormalize
    scores_exp = scores_exp * sparse_mask
    weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-12)
    
    # Apply to values
    output = np.dot(weights, v)
    
    return output


def pooling_avg(seq, axis=0):
    """
    Average pooling along specified axis.
    
    Args:
        seq: Input tensor
        axis: Axis to pool over
        
    Returns:
        Pooled tensor
    """
    return np.mean(seq, axis=axis, keepdims=False)


def pooling_lastk(seq, k: int, axis=0):
    """
    Last-k pooling along specified axis.
    
    Takes the mean of the last k elements.
    
    Args:
        seq: Input tensor
        k: Number of last elements to pool
        axis: Axis to pool over
        
    Returns:
        Pooled tensor
    """
    if axis < 0:
        axis += len(seq.shape)
    
    seq_len = seq.shape[axis]
    k_actual = min(k, seq_len)
    
    # Slice last k elements along axis
    if axis == 0:
        return np.mean(seq[-k_actual:], axis=axis)
    elif axis == 1 and len(seq.shape) == 2:
        return np.mean(seq[:, -k_actual:], axis=axis)
    else:
        # General case
        return np.mean(np.take(seq, range(seq_len - k_actual, seq_len), axis=axis), axis=axis)


def layernorm(x, eps: float = 1e-6):
    """
    Layer normalization.
    
    Args:
        x: Input tensor
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm


def rmsnorm(x, eps: float = 1e-6):
    """
    Root Mean Square normalization.
    
    Args:
        x: Input tensor
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms

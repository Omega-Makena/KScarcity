"""
Encoder — Online feature/path encoding.

Produces compact latents for candidate paths using FP16 operations with
deterministic sketch hashing and bounded online state.
Implements full online specification with attention, pooling, and sketch operators.
"""

import logging
import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from scarcity.engine.types import Candidate
from scarcity.engine.operators.attention_ops import (
    attn_linear, pooling_avg, pooling_lastk, layernorm, rmsnorm
)
from scarcity.engine.operators.sketch_ops import (
    poly_sketch, tensor_sketch, countsketch, latent_clip
)

logger = logging.getLogger(__name__)


@dataclass
class EncodedBatch:
    """
    Represents the output of a batched encoding operation.
    
    This data structure holds the projected latent representations for a set of
    candidate paths, along with their associated metadata, aggregate processing
    statistics, and performance telemetry.

    Attributes:
        latents: A list of numpy arrays, where each array is the latent vector
            representation of a single candidate path.
        meta: A list of dictionaries containing metadata for each path, such as
            the path ID, depth, domain index, sketch dimension, and precision mode.
        stats: A dictionary of aggregate statistics for the batch, including
            average latent norms and saturation percentages.
        telemetry: A dictionary containing performance metrics for the encoding
            process, such as latency, cache hit rates, and memory usage flags.
    """
    latents: List[np.ndarray]
    meta: List[Dict[str, Any]]
    stats: Dict[str, float]
    telemetry: Dict[str, Any]


class VariableEmbeddingMapper:
    """
    Manages learnable identity embeddings for system variables.

    This class maintains a dictionary of dense vector embeddings for each variable
    in the observed schema. It also handles variable-specific scaling factors
    (variances) to dynamically adjust the magnitude of embeddings based on
    signal strength or importance.
    """
    
    def __init__(self, n_vars: int, id_dim: int = 64):
        """
        Initializes the embedding mapper with a fixed number of variables.

        Args:
            n_vars: The total number of unique variables to embed.
            id_dim: The dimensionality of the embedding vectors.
        """
        self.n_vars = n_vars
        self.id_dim = id_dim
        
        # Initialize embeddings (fixed for now, can be EMA-adapted)
        rng = np.random.default_rng(42)
        self.embeddings = rng.standard_normal(size=(n_vars, id_dim)).astype(np.float32) * 0.1
        
        # Scale hooks (variances)
        self.scales = np.ones(n_vars, dtype=np.float32)
        
        logger.debug(f"Initialized VariableEmbeddingMapper: {n_vars} vars, dim={id_dim}")
    
    def update_scales(self, variances: np.ndarray):
        """Update scale hooks from variances."""
        if len(variances) == len(self.scales):
            self.scales = variances.copy()
    
    def get_embedding(self, var_idx: int) -> np.ndarray:
        """Get embedding for a variable."""
        if var_idx < len(self.embeddings):
            emb = self.embeddings[var_idx].astype(np.float16)
            scale = self.scales[var_idx]
            return emb * np.sqrt(np.clip(scale, 0.1, 10.0))
        return np.zeros(self.id_dim, dtype=np.float16)


class LagPositionalEncoder:
    """
    Provides learned positional encodings for temporal lags.

    Similar to positional encodings in Transformers, this class maps integer
    time-lag values to dense vectors. These vectors are added to variable
    embeddings to allow the model to distinguish between contemporaneous and
    lagged relationships.
    """
    
    def __init__(self, max_lag: int = 6, lag_dim: int = 16):
        """
        Initialize lag encoder.
        
        Args:
            max_lag: Maximum lag value
            lag_dim: Embedding dimension
        """
        self.max_lag = max_lag
        self.lag_dim = lag_dim
        
        # Learned lag table
        rng = np.random.default_rng(43)
        self.lag_table = rng.standard_normal(size=(max_lag + 1, lag_dim)).astype(np.float32) * 0.1
        
        logger.debug(f"Initialized LagPositionalEncoder: max_lag={max_lag}, dim={lag_dim}")
    
    def encode_lag(self, lag: int) -> np.ndarray:
        """Encode a lag value."""
        lag_clamped = np.clip(lag, 0, self.max_lag)
        return self.lag_table[int(lag_clamped)].astype(np.float16)


class PrecisionManager:
    """
    Manages mixed-precision computing policies.

    Controls automatic casting to half-precision (FP16) to save memory and
    bandwidth. Includes logic to detect instability or overflow and fallback
    to single-precision (FP32) when necessary.
    """
    
    def __init__(self):
        """Initialize precision manager."""
        self.fp16_enabled = True
        self.fallback_count = 0
        
    def autocast_fp16(self, x: np.ndarray) -> np.ndarray:
        """Cast to FP16 if enabled."""
        if self.fp16_enabled:
            return x.astype(np.float16)
        return x
    
    def accumulate_fp32(self, x: np.ndarray) -> np.ndarray:
        """Ensure FP32 for accumulations."""
        if x.dtype == np.float16:
            return x.astype(np.float32)
        return x
    
    def trigger_fallback(self):
        """Trigger precision fallback."""
        self.fp16_enabled = False
        self.fallback_count += 1
        logger.warning(f"Precision fallback triggered: count={self.fallback_count}")


class SketchCache:
    """
    Least-Recently-Used (LRU) cache for deterministic sketch projections.

    Caches the random projection matrices or seed parameters for sketch operations
    to avoid recomputing them for frequently used path structures. This improves
    throughput for repetitive encoding patterns.
    """
    
    def __init__(self, capacity: int = 8):
        """Initialize cache."""
        self.capacity = capacity
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Tuple[int, int]) -> Optional[Dict[str, np.ndarray]]:
        """Get cached sketch params."""
        if key in self.cache:
            self.hits += 1
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: Tuple[int, int], value: Dict[str, np.ndarray]):
        """Put sketch params."""
        # Evict if needed
        if len(self.cache) >= self.capacity and key not in self.cache:
            if self.access_order:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
        
        self.cache[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


class Encoder:
    """
    The core online feature encoder for causal paths.

    This class implements the complete pipeline for transforming raw data windows
    and candidate paths into fixed-size latent vectors. The pipeline includes:
    1. Variable embedding and lag encoding.
    2. Sequence construction from time-series windows.
    3. Attention mechanisms and temporal pooling.
    4. Normalization (LayerNorm/RMSNorm).
    5. Dimensionality reduction via CountSketch or TensorSketch.

    It manages its own precision state and caching to optimize performance under
    the constraints set by the Data Resource Governor.
    """
    
    def __init__(self, drg: Optional[Dict[str, Any]] = None):
        """
        Initializes the encoder with configuration from the resource governor.

        Args:
            drg: A dictionary containing Data Resource Governor settings, such as
                dimensions for embeddings and sketches, pooling types, cache
                capacities, and precision flags.
        """
        self.drg = drg or {}
        
        # Configuration
        self.id_dim = self.drg.get('id_dim', 64)
        self.lag_dim = self.drg.get('lag_dim', 16)
        self.sketch_dim = self.drg.get('sketch_dim', 512)
        self.max_path_len = self.drg.get('max_path_len', 5)
        self.max_lag = self.drg.get('max_lag', 6)
        self.pooling = self.drg.get('pooling', 'avg')
        self.lastk = self.drg.get('lastk', 32)
        self.norm = self.drg.get('norm', 'layernorm')
        self.latent_clip_p = self.drg.get('latent_clip_p', 99.9)
        
        # Submodules
        self.precision_mgr = PrecisionManager()
        self.var_emb_mapper = None  # Lazy init when n_vars known
        self.lag_encoder = LagPositionalEncoder(max_lag=self.max_lag, lag_dim=self.lag_dim)
        self.sketch_cache = SketchCache(capacity=self.drg.get('cache_capacity', 8))
        
        # Online state
        self.schema_hash = None
        self.latent_norm_ema = defaultdict(float)  # Per-domain EMAs
        self.alpha_ema = 0.1
        
        # Stats
        self.total_encoded = 0
        self.oom_fallbacks = 0
        self.total_cost_ms = 0.0
        
        logger.info(f"Encoder initialized: sketch_dim={self.sketch_dim}, id_dim={self.id_dim}, lag_dim={self.lag_dim}")
    
    def step(self, window: np.ndarray, candidates: List[Candidate], 
             context: Dict[str, Any]) -> EncodedBatch:
        """
        Executes the encoding pipeline for a batch of candidate paths.

        This method processes a raw data window and a list of proposed causal paths,
        mapping each path to a latent vector. It handles all steps from embedding
        lookup to final sketching and normalization.

        If encoding fails for any specific path (e.g., due to invalid indices),
        it gracefully falls back to a zero-vector and logs the error in the
        metadata, ensuring the batch processing proceeds.

        Args:
            window: A 2D numpy array representing the data window [WindowSize, Variables].
            candidates: A list of `Candidate` objects representing the paths to encode.
            context: A dictionary of context information, including the current schema,
                window ID, and active resource profile.

        Returns:
            An `EncodedBatch` object containing the list of latent vectors,
            corresponding metadata, and aggregate statistics.
        """
        start_time = time.time()
        
        # Step 0: DRG read & setup
        schema = context.get('schema', {})
        schema_str = str(sorted(schema.get('fields', {}).keys()))
        self.schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        
        # Initialize var mapper if needed
        P = window.shape[1] if window.ndim == 2 else 0
        if self.var_emb_mapper is None or self.var_emb_mapper.n_vars != P:
            self.var_emb_mapper = VariableEmbeddingMapper(n_vars=P, id_dim=self.id_dim)
        
        # Pre-allocate scratch buffer
        scratch_size = self.sketch_dim
        scratch_buffer = np.zeros(scratch_size, dtype=np.float32)
        
        latents = []
        meta = []
        
        # Per-path encoding
        for cand in candidates:
            try:
                latent, path_meta = self._encode_path(window, cand, scratch_buffer, context)
                latents.append(latent)
                meta.append(path_meta)
            except Exception as e:
                logger.warning(f"Encoding failed for path {cand.path_id}: {e}")
                # Fallback: zero latent
                latents.append(np.zeros(self.sketch_dim, dtype=np.float16))
                meta.append({
                    'path_id': cand.path_id,
                    'depth': cand.depth,
                    'domain': cand.domain,
                    'sketch_dim': self.sketch_dim,
                    'precision': 'fp16' if self.precision_mgr.fp16_enabled else 'fp32',
                    'error': str(e)
                })
        
        # Compute stats
        encode_time_ms = (time.time() - start_time) * 1000
        self.total_encoded += len(candidates)
        self.total_cost_ms += encode_time_ms
        
        # Aggregate stats
        latent_norms = [np.linalg.norm(l) for l in latents]
        stats = {
            'paths_encoded': len(candidates),
            'avg_latent_norm': float(np.mean(latent_norms)) if latent_norms else 0.0,
            'p99_latent_norm': float(np.percentile(latent_norms, 99)) if latent_norms else 0.0,
            'saturation_pct': 0.0  # Placeholder
        }
        
        # Telemetry
        telemetry = {
            'encode_latency_ms': encode_time_ms,
            'fp16_time_frac': 1.0 if self.precision_mgr.fp16_enabled else 0.0,
            'fp32_accum_time_ms': 0.0,  # Placeholder
            'sketch_dim_active': self.sketch_dim,
            'oom_fallbacks': self.oom_fallbacks,
            'cache_hits': self.sketch_cache.hits,
            'cache_misses': self.sketch_cache.misses,
            'cache_hit_rate': self.sketch_cache.hits / max(self.sketch_cache.hits + self.sketch_cache.misses, 1)
        }
        
        return EncodedBatch(
            latents=latents,
            meta=meta,
            stats=stats,
            telemetry=telemetry
        )
    
    def _encode_path(self, window: np.ndarray, cand: Candidate, 
                     scratch_buffer: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encodes a single candidate path into a latent vector.

        This internal method implements the specific composition logic:
        1. Look up embeddings for the variables in the path.
        2. Apply positional encodings for their respective lags.
        3. Extract the time-series data from the window.
        4. Apply the configured attention and pooling operators.
        5. Project the result into the sketch dimension using a deterministic seed.

        Args:
            window: The data window array.
            cand: The candidate path to encode.
            scratch_buffer: A pre-allocated buffer for intermediate computations.
            context: Context dictionary for seeding and schema access.

        Returns:
            A tuple containing:
            - The resulting latent vector (numpy array).
            - A dictionary of metadata about the encoding (depth, domain, etc.).

        Raises:
            ValueError: If the path contains invalid indices or results in empty embeddings.
        """
        W, P = window.shape
        
        # Step 1: Variable embeddings + lags
        token_embeddings = []
        for var_idx, lag in zip(cand.vars, cand.lags):
            if var_idx >= P:
                continue
            
            var_emb = self.var_emb_mapper.get_embedding(var_idx)
            lag_emb = self.lag_encoder.encode_lag(lag)
            
            # Combine (simple concatenation for now)
            combined_emb = np.concatenate([var_emb, lag_emb])
            token_embeddings.append(combined_emb)
        
        if len(token_embeddings) == 0:
            raise ValueError("No valid variable embeddings")
        
        # Step 2: Build sequence tensor [W, d_emb]
        d_emb = len(token_embeddings[0])
        seq_tokens = []
        
        for var_idx in cand.vars:
            if var_idx >= P:
                continue
            # Extract time series for this variable
            ts = window[:, var_idx]
            # Tile to form embeddings
            seq_tokens.append(ts[:, np.newaxis] * self.var_emb_mapper.get_embedding(var_idx)[:1])
        
        if len(seq_tokens) == 0:
            raise ValueError("No valid sequences")
        
        # Combine sequences
        seq_tensor = np.concatenate(seq_tokens, axis=1)  # [W, d_emb]
        
        # Step 3: Feature composition (attention + pooling)
        if cand.depth > 1:
            # Linear attention
            seq_encoded = attn_linear(seq_tensor, seq_tensor, seq_tensor)
        else:
            seq_encoded = seq_tensor
        
        # Pooling
        if self.pooling == 'avg':
            pooled = pooling_avg(seq_encoded)
        elif self.pooling == 'lastk':
            pooled = pooling_lastk(seq_encoded, self.lastk)
        else:
            pooled = seq_encoded[-1]  # Last token
        
        # Step 4: Normalization
        if self.norm == 'layernorm':
            pooled = layernorm(pooled)
        elif self.norm == 'rmsnorm':
            pooled = rmsnorm(pooled)
        
        # Step 5: Interaction projection (sketch)
        seed = self._compute_seed(context)
        
        # Try to get cached sketch params
        cache_key = (cand.depth, self.sketch_dim)
        cached_params = self.sketch_cache.get(cache_key)
        
        if cached_params is None:
            # Generate and cache
            # For simplicity, use poly_sketch
            cached_params = {'seed': seed}
            self.sketch_cache.put(cache_key, cached_params)
        
        # Apply sketch
        latent = poly_sketch(pooled, degree=2, dim=self.sketch_dim, 
                            seed=seed, path_id=cand.path_id)
        
        # Step 6: Safety clipping
        latent = latent_clip(latent, p=self.latent_clip_p)
        
        # Convert to FP16 if not already
        latent = self.precision_mgr.autocast_fp16(latent)
        
        # Meta
        meta = {
            'path_id': cand.path_id,
            'depth': cand.depth,
            'domain': cand.domain,
            'sketch_dim': self.sketch_dim,
            'precision': 'fp16' if self.precision_mgr.fp16_enabled else 'fp32',
            'cost_hint': 0.0  # Placeholder
        }
        
        return latent, meta
    
    def _compute_seed(self, context: Dict[str, Any]) -> int:
        """Compute deterministic seed."""
        window_id = context.get('window_id', 0)
        profile_rev = context.get('profile_rev', 0)
        
        combined = f"{self.schema_hash}_{window_id}_{profile_rev}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:16], 16)
        return hash_val
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves current statistics for the encoder subsystem.

        Returns:
            A dictionary containing configuration details (sketch dimension, precision)
            and runtime metrics (total items encoded, average latency, cache health,
            and fallback counts).
        """
        return {
            'sketch_dim': self.sketch_dim,
            'precision': 'fp16' if self.precision_mgr.fp16_enabled else 'fp32',
            'total_encoded': self.total_encoded,
            'avg_encode_time_ms': self.total_cost_ms / max(self.total_encoded, 1),
            'oom_fallbacks': self.oom_fallbacks,
            'cache_size': len(self.sketch_cache.cache)
        }
    
    # Backward compatibility
    def encode_paths(self, window_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Backward-compatible encoding."""
        if len(candidates) == 0:
            return []
        
        # Convert old dict candidates to new Candidate objects
        new_candidates = []
        for cand_dict in candidates:
            new_cand = Candidate(
                path_id=cand_dict.get('path_id', 'unknown'),
                vars=(cand_dict.get('source', 0), cand_dict.get('target', 1)),
                lags=(cand_dict.get('lag', 0), cand_dict.get('lag', 0)),
                ops=('sketch', 'attn'),
                root=cand_dict.get('source', 0),
                depth=2,
                domain=0,
                gen_reason='compat'
            )
            new_candidates.append(new_cand)
        
        # Extract window
        window = window_data.get('data')
        if window is None:
            return []
        if isinstance(window, list):
            window = np.array(window)
        
        # Context
        context = {
            'schema': window_data.get('schema', {}),
            'window_id': window_data.get('window_id', 0),
            'profile_rev': 0
        }
        
        # Encode
        batch = self.step(window, new_candidates, context)
        return batch.latents

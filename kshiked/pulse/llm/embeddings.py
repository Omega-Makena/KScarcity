"""
Ollama Embedding Provider for KShield Pulse

Provides semantic embeddings via Ollama's /api/embed endpoint.
Replaces/augments the rule-based TF-IDF embeddings in nlp.py with
actual semantic understanding from local LLMs.

Features:
- Semantic similarity between texts
- Cluster detection for narrative grouping
- Anomaly detection for novel threats
- Batch embedding with caching
- Compatible with numpy/sklearn downstream

Usage:
    from kshiked.pulse.llm.embeddings import OllamaEmbeddings
    
    embedder = OllamaEmbeddings()
    async with embedder:
        vec = await embedder.embed("Serikali wezi!")
        sims = await embedder.similarity_matrix(texts)
        clusters = await embedder.cluster_texts(texts, n_clusters=5)
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from .config import OllamaConfig, AnalysisTask, InferenceMetrics, SessionMetrics

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Semantic embeddings via Ollama local models.
    
    Uses Ollama's /api/embed endpoint for dense vector embeddings.
    Supports nomic-embed-text (768d) and mxbai-embed-large (1024d).
    
    Architecture:
        Text → Ollama /api/embed → numpy array (768d or 1024d)
             → Cosine similarity / K-means clustering / Anomaly detection
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        self.config = config or OllamaConfig(
            base_url=base_url, embedding_model=model
        )
        self.base_url = self.config.base_url
        self.model = self.config.embedding_model
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, np.ndarray] = {}  # text_hash → embedding
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self.metrics = SessionMetrics()

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.config.connect_timeout, total=60.0
            )
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Core Embedding
    # =========================================================================

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a single text.
        
        Args:
            text: Input text
            
        Returns:
            numpy array of shape (dim,) or None on failure
        """
        # Check cache
        key = self._cache_key(text)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1

        await self._ensure_session()
        
        start = time.monotonic()
        metric = InferenceMetrics(
            task=AnalysisTask.EMBEDDING, model=self.model
        )

        try:
            payload = {"model": self.model, "input": text}
            async with self._session.post(
                f"{self.base_url}/api/embed", json=payload
            ) as resp:
                elapsed = (time.monotonic() - start) * 1000
                metric.latency_ms = elapsed

                if resp.status != 200:
                    logger.error(f"Embed failed: HTTP {resp.status}")
                    metric.success = False
                    self.metrics.record(metric)
                    return None

                data = await resp.json()
                embeddings = data.get("embeddings", [])
                
                if not embeddings:
                    logger.warning("No embeddings returned")
                    metric.success = False
                    self.metrics.record(metric)
                    return None

                vec = np.array(embeddings[0], dtype=np.float32)
                metric.success = True
                metric.total_tokens = data.get("prompt_eval_count", 0)
                self.metrics.record(metric)

                # Cache it
                self._cache[key] = vec
                return vec

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            metric.success = False
            metric.error = str(e)
            self.metrics.record(metric)
            return None

    async def embed_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            show_progress: Log progress every 100 texts
            
        Returns:
            List of numpy arrays (same length as texts, None for failures)
        """
        results: List[Optional[np.ndarray]] = []
        
        # Check cache first
        uncached_indices = []
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                self._cache_hits += 1
                results.append(self._cache[key])
            else:
                self._cache_misses += 1
                results.append(None)
                uncached_indices.append(i)

        if not uncached_indices:
            return results

        # Process uncached in batches
        batch_size = 32  # Ollama embed supports batching
        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]

            if show_progress and batch_start % 100 == 0:
                logger.info(
                    f"Embedding batch {batch_start}/{len(uncached_indices)}"
                )

            await self._ensure_session()
            try:
                payload = {"model": self.model, "input": batch_texts}
                async with self._session.post(
                    f"{self.base_url}/api/embed", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        embeddings = data.get("embeddings", [])
                        for j, idx in enumerate(batch_indices):
                            if j < len(embeddings):
                                vec = np.array(embeddings[j], dtype=np.float32)
                                results[idx] = vec
                                self._cache[self._cache_key(texts[idx])] = vec
                    else:
                        logger.warning(f"Batch embed failed: HTTP {resp.status}")

            except Exception as e:
                logger.error(f"Batch embed error: {e}")

            # Small delay between batches
            await asyncio.sleep(0.05)

        return results

    # =========================================================================
    # Similarity Operations
    # =========================================================================

    async def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Similarity score 0.0 to 1.0 (or -1.0 on failure)
        """
        vec_a, vec_b = await asyncio.gather(
            self.embed(text_a), self.embed(text_b)
        )
        if vec_a is None or vec_b is None:
            return -1.0
        return float(self._cosine_similarity(vec_a, vec_b))

    async def similarity_matrix(
        self, texts: List[str]
    ) -> Optional[np.ndarray]:
        """
        Compute pairwise cosine similarity matrix.
        
        Returns:
            NxN numpy array of similarities
        """
        embeddings = await self.embed_batch(texts)
        valid = [e for e in embeddings if e is not None]
        
        if len(valid) < 2:
            return None

        matrix = np.stack(valid)
        # Normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        
        return normalized @ normalized.T

    async def find_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Find most similar texts in a corpus.
        
        Returns:
            List of (index, similarity, text) tuples sorted by similarity
        """
        query_vec = await self.embed(query)
        if query_vec is None:
            return []

        corpus_vecs = await self.embed_batch(corpus)
        
        results = []
        for i, vec in enumerate(corpus_vecs):
            if vec is not None:
                sim = self._cosine_similarity(query_vec, vec)
                results.append((i, float(sim), corpus[i]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # =========================================================================
    # Clustering
    # =========================================================================

    async def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[str, any]:
        """
        Cluster texts by semantic similarity.
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            method: "kmeans" or "agglomerative"
            
        Returns:
            {
                "labels": [0, 1, 2, ...],  # cluster assignment per text
                "centroids": [[...], ...],  # cluster centers
                "sizes": {0: 10, 1: 15, ...},  # cluster sizes
                "representatives": {0: "text...", ...}  # nearest to centroid
            }
        """
        embeddings = await self.embed_batch(texts, show_progress=True)
        
        valid_mask = [e is not None for e in embeddings]
        valid_vecs = [e for e in embeddings if e is not None]
        
        if len(valid_vecs) < n_clusters:
            logger.warning(
                f"Too few valid embeddings ({len(valid_vecs)}) for {n_clusters} clusters"
            )
            return {"labels": [-1] * len(texts), "centroids": [], "sizes": {}, "representatives": {}}

        matrix = np.stack(valid_vecs)
        
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
            centroids = model.cluster_centers_
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(matrix)
            # Compute centroids manually
            centroids = np.zeros((n_clusters, matrix.shape[1]))
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = matrix[mask].mean(axis=0)

        # Map back to full text list
        full_labels = [-1] * len(texts)
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                full_labels[i] = int(labels[valid_idx])
                valid_idx += 1

        # Find representative texts (nearest to centroid)
        representatives = {}
        sizes = {}
        for k in range(n_clusters):
            cluster_mask = labels == k
            sizes[k] = int(cluster_mask.sum())
            if cluster_mask.any():
                dists = np.linalg.norm(matrix[cluster_mask] - centroids[k], axis=1)
                best_idx = np.argmin(dists)
                # Map back to original index
                cluster_indices = np.where(cluster_mask)[0]
                valid_indices = [i for i, v in enumerate(valid_mask) if v]
                orig_idx = valid_indices[cluster_indices[best_idx]]
                representatives[k] = texts[orig_idx]

        return {
            "labels": full_labels,
            "centroids": centroids.tolist(),
            "sizes": sizes,
            "representatives": representatives,
        }

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    async def detect_anomalies(
        self,
        texts: List[str],
        threshold: float = 2.0,
    ) -> List[Tuple[int, float, str]]:
        """
        Detect semantically anomalous texts (potential novel threats).
        
        Uses distance from mean embedding to find outliers.
        
        Args:
            texts: Corpus of texts
            threshold: Z-score threshold for anomaly (default 2.0)
            
        Returns:
            List of (index, z_score, text) for anomalous texts
        """
        embeddings = await self.embed_batch(texts)
        valid_vecs = [(i, e) for i, e in enumerate(embeddings) if e is not None]
        
        if len(valid_vecs) < 10:
            return []

        indices, vecs = zip(*valid_vecs)
        matrix = np.stack(vecs)
        
        # Compute mean and std of distances
        mean_vec = matrix.mean(axis=0)
        distances = np.linalg.norm(matrix - mean_vec, axis=1)
        
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        if std_dist == 0:
            return []

        z_scores = (distances - mean_dist) / std_dist
        
        anomalies = []
        for j, (orig_idx, z) in enumerate(zip(indices, z_scores)):
            if z > threshold:
                anomalies.append((orig_idx, float(z), texts[orig_idx]))
        
        anomalies.sort(key=lambda x: x[1], reverse=True)
        return anomalies

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics."""
        return {
            "cache_size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(
                self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3
            ),
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

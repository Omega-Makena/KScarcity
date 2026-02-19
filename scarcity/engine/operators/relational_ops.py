"""
Relational Operators — Tier 2 graph-based feature builders.

Leverages the current hypergraph (edges, hyperedges, regimes, domains) to
propagate signals, score neighborhoods, and extract multi-hop relational
structure for the MPIE — all online.

All operators are online, bounded, deterministic, and DRG-aware.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass
from collections import defaultdict

from scarcity.engine.operators.attention_ops import attn_linear, layernorm, rmsnorm, pooling_avg
from scarcity.engine.operators.sketch_ops import countsketch, tensor_sketch, _deterministic_hash
from scarcity.engine.store import HypergraphStore, EdgeRec

logger = logging.getLogger(__name__)


@dataclass
class RelationalOutput:
    """Output from a relational operator."""
    latent: np.ndarray
    stats: Dict[str, Any]
    cost_hint_ms: float


@dataclass
class SampledSubgraph:
    """Sampled subgraph from store."""
    nodes: List[int]  # Node IDs in subgraph
    edges: List[Tuple[int, int]]  # (src, dst) pairs
    edge_weights: Dict[Tuple[int, int], float]  # Edge weights
    edge_stability: Dict[Tuple[int, int], float]  # Edge stability
    coverage: Dict[int, float]  # Domain/group coverage vector


def neighborhood_sampling(
    S: List[int],
    store: HypergraphStore,
    deg_cap: int = 16,
    hop_cap: int = 2,
    drg_util: float = 0.0,
    domain: Optional[int] = None,
    seed: int = 0,
    window_id: int = 0
) -> Tuple[SampledSubgraph, RelationalOutput]:
    """
    Select a compact, diverse local subgraph around variables in a candidate path.
    
    Samples a compact neighborhood around seed nodes S with diversity thinning
    to favor under-covered schema groups.
    
    Args:
        S: Seed node IDs (variables in the path)
        store: HypergraphStore instance
        deg_cap: Maximum neighbors per node
        hop_cap: Maximum number of hops (1 or 2)
        drg_util: Current DRG utilization
        domain: Optional domain filter
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        Tuple of (SampledSubgraph, RelationalOutput with coverage vector)
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG adaptation
    if drg_util > 0.8:
        deg_cap = max(8, deg_cap // 2)
        hop_cap = 1
        fallbacks |= 1
    elif drg_util > 0.7:
        deg_cap = max(8, int(deg_cap * 0.75))
        hop_cap = min(hop_cap, 2)
    
    hop_cap = max(1, min(hop_cap, 2))  # Bounded to 1-2 hops
    deg_cap = max(4, min(deg_cap, 32))  # Bounded to 4-32
    
    # Track nodes and edges in subgraph
    subgraph_nodes: Set[int] = set(S)
    subgraph_edges: List[Tuple[int, int]] = []
    edge_weights: Dict[Tuple[int, int], float] = {}
    edge_stability: Dict[Tuple[int, int], float] = {}
    
    # Track domain/group coverage for diversity thinning
    coverage: Dict[int, int] = defaultdict(int)  # domain -> count
    rng = _deterministic_hash(seed, f"nbs_{window_id}")
    
    # Step 1: For each seed v∈S, fetch top-k neighbors
    current_nodes = list(S)
    current_deg_cap = deg_cap
    
    for hop in range(hop_cap):
        next_nodes: Set[int] = set()
        
        for v in current_nodes:
            # Get top-k neighbors from store
            neighbors = store.top_k_neighbors(
                v, k=current_deg_cap, direction="out", domain=domain
            )
            
            # Apply diversity thinning: probabilistic down-sampling
            # favoring under-covered schema groups
            for neighbor_id, score in neighbors:
                # Get neighbor's domain
                neighbor_domain = 0
                if neighbor_id in store.nodes:
                    neighbor_domain = store.nodes[neighbor_id].get('domain', 0)
                
                # Compute inverse coverage weight (higher for under-covered)
                coverage_count = coverage[neighbor_domain]
                inverse_coverage = 1.0 / (1.0 + coverage_count)
                
                # Softmax-like probability (favor under-covered)
                prob = inverse_coverage / (1.0 + coverage_count)
                
                # Sample with probability
                if rng.random() < prob or len(next_nodes) < current_deg_cap // 2:
                    # Add neighbor and edge
                    edge_key = (v, neighbor_id)
                    edge = store.get_edge(v, neighbor_id)
                    
                    if edge is not None:
                        # Prune edges with stability < ε_s or age > T_stale
                        age = window_id - edge.last_seen
                        epsilon_stability = 0.4
                        T_stale = 600
                        
                        if edge.stability >= epsilon_stability and age <= T_stale:
                            subgraph_nodes.add(neighbor_id)
                            if edge_key not in subgraph_edges:
                                subgraph_edges.append(edge_key)
                            edge_weights[edge_key] = edge.weight
                            edge_stability[edge_key] = edge.stability
                            next_nodes.add(neighbor_id)
                            coverage[neighbor_domain] += 1
        
        # Update for next hop (reduce deg_cap)
        if hop < hop_cap - 1:
            current_deg_cap = max(4, current_deg_cap // 2)
        
        current_nodes = list(next_nodes)
    
    # Normalize coverage vector to [0,1]
    total_coverage = sum(coverage.values()) if coverage else 1
    coverage_normalized = {
        domain: count / total_coverage
        for domain, count in coverage.items()
    }
    
    cost_ms = (time.time() - start_time) * 1000
    
    subgraph = SampledSubgraph(
        nodes=list(subgraph_nodes),
        edges=subgraph_edges,
        edge_weights=edge_weights,
        edge_stability=edge_stability,
        coverage=coverage_normalized
    )
    
    output = RelationalOutput(
        latent=np.array([list(coverage_normalized.values())], dtype=np.float16)
            if coverage_normalized else np.zeros((1, 1), dtype=np.float16),
        stats={
            'nbs_nodes': len(subgraph_nodes),
            'nbs_edges': len(subgraph_edges),
            'hop_cap_used': hop_cap,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )
    
    return subgraph, output


def diffusion_sketch(
    Z: np.ndarray,
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    T: int = 2,
    alpha: float = 0.6,
    drg_util: float = 0.0,
    drg_sketch_dim: int = 512,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Propagate a Tier-1 latent over the sampled subgraph using fast, sketched diffusion.
    
    Uses power-series approximation: R = (1-α) Σ_{t=0..T} α^t Â^t Z_node
    
    Args:
        Z: Tier-1 latent [W, D] or [D] (pooled)
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        T: Diffusion steps
        alpha: Damping factor
        drg_util: Current DRG utilization
        drg_sketch_dim: Sketch dimension from DRG
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with diffused latent and energy ratio
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        logger.warning(f"DFS: NaN/Inf in input, zeroing")
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    # DRG adaptation
    if drg_util > 0.8:
        T = 1
        alpha = 0.5
        drg_sketch_dim = max(256, drg_sketch_dim // 2)
        fallbacks |= 2
    elif drg_util > 0.7:
        T = min(T, 2)
        alpha = max(0.3, alpha * 0.9)
    
    T = max(1, min(T, 3))  # Bounded to 1-3 steps
    alpha = max(0.3, min(alpha, 0.8))  # Bounded to [0.3, 0.8]
    
    # Handle sequence vs pooled input
    is_sequence = len(Z.shape) == 2
    if is_sequence:
        W, D = Z.shape
        Z_node = Z  # [W, D]
    else:
        D = len(Z)
        Z_node = Z.reshape(1, D)  # [1, D]
        W = 1
    
    # Map node IDs to indices in subgraph
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    n_nodes = len(subgraph.nodes)
    
    if n_nodes == 0:
        # Fallback: return input
        return RelationalOutput(
            latent=Z.astype(np.float16),
            stats={
                'dfs_latency_ms': (time.time() - start_time) * 1000,
                'dfs_steps': 0,
                'dfs_energy_ratio': 1.0,
                'fallbacks_taken': 4
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Build local adjacency matrix A (sparse representation)
    # Normalize to row-stochastic: Â = D^{-1/2} A D^{-1/2}
    # For efficiency, we'll use sparse matvec operations
    
    # Initialize node features: map Z to nodes (if Z_node is per-node, use as-is)
    # For simplicity, assign Z_node to seed nodes (first |S| nodes)
    # In practice, Z_node would be mapped via variable IDs
    H = np.zeros((n_nodes, D), dtype=np.float32)
    
    # Assign Z_node to nodes (simple assignment for now)
    seed_count = min(W, n_nodes)
    H[:seed_count] = Z_node[:seed_count].astype(np.float32)
    
    # Build sparse adjacency with normalized weights
    # Compute degree matrix (for normalization)
    out_degree = np.zeros(n_nodes, dtype=np.float32)
    edge_list = []  # List of (i, j, weight) tuples
    
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            weight = subgraph.edge_weights.get((src, dst), 0.0)
            if abs(weight) > 1e-12:
                edge_list.append((i, j, weight))
                out_degree[i] += abs(weight)
    
    # Normalize degrees (avoid division by zero)
    out_degree = np.maximum(out_degree, 1e-12)
    sqrt_degree = np.sqrt(out_degree)
    
    # Normalize adjacency: Â_ij = A_ij / (sqrt(deg_i) * sqrt(deg_j))
    A_normalized = {}
    in_degree = np.zeros(n_nodes, dtype=np.float32)
    for i, j, weight in edge_list:
        normalized_weight = weight / (sqrt_degree[i] * sqrt_degree[j] + 1e-12)
        A_normalized[(i, j)] = normalized_weight
        in_degree[j] += abs(normalized_weight)
    
    # Power-series approximation: R = (1-α) Σ_{t=0..T} α^t Â^t H
    R = np.zeros_like(H, dtype=np.float32)
    H_t = H.copy()
    
    input_energy = np.sum(H ** 2)
    
    for t in range(T + 1):
        # Add contribution: (1-α) * α^t * Â^t * H
        if t == 0:
            contribution = H_t
        else:
            # Compute Â * H_t via sparse matvec
            H_next = np.zeros_like(H_t, dtype=np.float32)
            for (i, j), weight in A_normalized.items():
                H_next[j] += weight * H_t[i]
            
            H_t = H_next
            
            # Clip to prevent explosion
            H_t = np.clip(H_t, -10.0, 10.0)
            contribution = H_t
        
        # Weight by (1-α) * α^t
        weight_coeff = (1.0 - alpha) * (alpha ** t)
        R += weight_coeff * contribution
        
        # Early stopping if incremental energy < ε
        if t > 0:
            incremental = np.sum(contribution ** 2)
            if incremental < 1e-8 * input_energy:
                break
    
    # Normalize
    R = rmsnorm(R)
    
    # Map back to output shape
    if is_sequence:
        # Output [W, D] - take first W nodes or pad
        if R.shape[0] >= W:
            R_output = R[:W]
        else:
            R_output = np.zeros((W, D), dtype=np.float32)
            R_output[:R.shape[0]] = R
    else:
        # Pooled output
        R_output = pooling_avg(R, axis=0)
    
    # Energy ratio
    output_energy = np.sum(R_output ** 2)
    energy_ratio = output_energy / (input_energy + 1e-12)
    
    # Validate output
    if np.any(np.isnan(R_output)) or np.any(np.isinf(R_output)):
        logger.warning(f"DFS: NaN/Inf after diffusion, using fallback")
        R_output = Z.astype(np.float32)
        fallbacks |= 4
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=R_output.astype(np.float16),
        stats={
            'dfs_latency_ms': cost_ms,
            'dfs_steps': t + 1,
            'dfs_energy_ratio': float(energy_ratio),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def random_walk_with_restart(
    seeds: List[int],
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    gamma: float = 0.15,
    max_steps: int = 8,
    drg_util: float = 0.0,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Produce a proximity vector over sampled nodes via random walk with restart.
    
    Highlights influential neighbors of candidate variables.
    
    Args:
        seeds: Seed node IDs
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        gamma: Restart probability
        max_steps: Maximum iteration steps
        drg_util: Current DRG utilization
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with stationary distribution π
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG adaptation
    if drg_util > 0.7:
        max_steps = max(4, max_steps // 2)
        gamma = max(0.15, min(gamma, 0.25))
        fallbacks |= 1
    
    max_steps = max(4, min(max_steps, 8))  # Bounded to 4-8
    gamma = max(0.1, min(gamma, 0.3))  # Bounded to [0.1, 0.3]
    
    n_nodes = len(subgraph.nodes)
    if n_nodes == 0:
        return RelationalOutput(
            latent=np.zeros(1, dtype=np.float16),
            stats={
                'rwr_latency_ms': (time.time() - start_time) * 1000,
                'rwr_convergence': 0.0,
                'rwr_mass_on_seeds': 0.0,
                'fallbacks_taken': 2
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Map node IDs to indices
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    
    # Build normalized adjacency Â (row-stochastic)
    out_degree = np.zeros(n_nodes, dtype=np.float32)
    A_normalized = {}
    
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            weight = abs(subgraph.edge_weights.get((src, dst), 0.0))
            if weight > 1e-12:
                out_degree[i] += weight
    
    out_degree = np.maximum(out_degree, 1e-12)
    
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            weight = abs(subgraph.edge_weights.get((src, dst), 0.0))
            if weight > 1e-12:
                # Row-stochastic: Â_ij = A_ij / deg_i
                A_normalized[(i, j)] = weight / out_degree[i]
    
    # Initialize: π_0 = e_S (one-hot on seeds)
    pi = np.zeros(n_nodes, dtype=np.float32)
    seed_indices = [node_to_idx[s] for s in seeds if s in node_to_idx]
    
    if len(seed_indices) == 0:
        # Fallback: uniform
        pi[:] = 1.0 / n_nodes
    else:
        # Uniform over seeds
        pi[seed_indices] = 1.0 / len(seed_indices)
    
    # Iterate: π_{t+1} = (1-γ)Â π_t + γ e_S
    e_S = pi.copy()
    tau = 1e-6  # Convergence threshold
    
    for step in range(max_steps):
        pi_prev = pi.copy()
        
        # Compute (1-γ)Â π_t via sparse matvec
        pi_new = np.zeros(n_nodes, dtype=np.float32)
        for (i, j), weight in A_normalized.items():
            pi_new[j] += (1.0 - gamma) * weight * pi_prev[i]
        
        # Add restart: γ e_S
        pi = pi_new + gamma * e_S
        
        # Normalize (ensure sum = 1)
        pi_sum = np.sum(pi)
        if pi_sum > 1e-12:
            pi = pi / pi_sum
        else:
            pi = e_S
            break
        
        # Check convergence: ||π_{t+1} - π_t||_1 < τ
        diff = np.sum(np.abs(pi - pi_prev))
        if diff < tau:
            break
    
    # Compute metrics
    convergence = 1.0 - float(diff) if step < max_steps - 1 else 0.0
    mass_on_seeds = float(np.sum(pi[seed_indices])) if seed_indices else 0.0
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=pi.astype(np.float16),
        stats={
            'rwr_latency_ms': cost_ms,
            'rwr_convergence': convergence,
            'rwr_mass_on_seeds': mass_on_seeds,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def relational_attention(
    H_nodes: np.ndarray,
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    bias_beta: float = 0.8,
    drg_util: float = 0.0,
    return_sequence: bool = True,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Attend across neighbors using edge weights and stability as learn-free priors.
    
    Args:
        H_nodes: Node features [n_nodes, D] or [D] (pooled)
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        bias_beta: Edge bias coefficient
        drg_util: Current DRG utilization
        return_sequence: Return full sequence vs pooled
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with attended features and entropy
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(H_nodes)) or np.any(np.isinf(H_nodes)):
        logger.warning(f"RAT: NaN/Inf in input, zeroing")
        H_nodes = np.nan_to_num(H_nodes, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    # DRG adaptation: fall back to mean aggregation if latency_high
    if drg_util > 0.8:
        # Use mean pooling
        H_mean = np.mean(H_nodes.reshape(-1, H_nodes.shape[-1]), axis=0, keepdims=True)
        if return_sequence:
            H_output = np.broadcast_to(H_mean, H_nodes.shape)
        else:
            H_output = H_mean
        return RelationalOutput(
            latent=H_output.astype(np.float16),
            stats={
                'rat_latency_ms': (time.time() - start_time) * 1000,
                'rat_entropy': 0.0,
                'fallbacks_taken': 2
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Handle input shape
    is_sequence = len(H_nodes.shape) == 2
    if is_sequence:
        n_nodes, D = H_nodes.shape
        H = H_nodes.astype(np.float32)
    else:
        D = len(H_nodes)
        n_nodes = len(subgraph.nodes)
        if n_nodes == 0:
            n_nodes = 1
        # Broadcast to nodes
        H = np.broadcast_to(H_nodes.reshape(1, D), (n_nodes, D)).astype(np.float32)
    
    # Map node IDs to indices
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    
    # Build edge bias matrix: b_ij = β·W_e(i→j)
    # where W_e = f(weight, stability)
    bias_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            
            # Edge prior: W_e = weight * stability
            weight = subgraph.edge_weights.get((src, dst), 0.0)
            stability = subgraph.edge_stability.get((src, dst), 0.0)
            W_e = abs(weight) * stability
            
            # Additive bias
            bias_matrix[i, j] = bias_beta * W_e
    
    # Fixed linear maps for Q, K, V (stateless projections)
    # For simplicity, use identity maps scaled by small factors
    # Simple fixed projections (could be learned but kept stateless for now)
    Q = H  # [n_nodes, D]
    K = H  # [n_nodes, D]
    V = H  # [n_nodes, D]
    
    # Compute attention with bias: scores = Q·K^T + bias
    scores = np.dot(Q, K.T) / np.sqrt(float(D))  # [n_nodes, n_nodes]
    scores = scores + bias_matrix  # Add edge bias
    
    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-12)
    
    # Apply to values
    R = np.dot(attn_weights, V)  # [n_nodes, D]
    
    # Normalize with RMSNorm
    R = rmsnorm(R)
    
    # Compute attention entropy: H = -Σ p log p
    entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-12), axis=-1)
    avg_entropy = float(np.mean(entropy))
    
    # Aggregate to seeds (if needed)
    if not return_sequence:
        R = pooling_avg(R, axis=0)
    
    # Validate output
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        logger.warning(f"RAT: NaN/Inf after attention, using fallback")
        R = np.mean(H, axis=0, keepdims=True) if return_sequence else np.mean(H, axis=0)
        fallbacks |= 4
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=R.astype(np.float16),
        stats={
            'rat_latency_ms': cost_ms,
            'rat_entropy': avg_entropy,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def signed_message_passing(
    x: np.ndarray,
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    rounds: int = 1,
    drg_util: float = 0.0,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Propagate signed effects (positive/negative relations) without cancellation.
    
    Two-channel message passing: m⁺ = A⁺ x, m⁻ = A⁻ x
    Combine via x' = σ(m⁺ − m⁻)
    
    Args:
        x: Per-node scalar or low-D latent [n_nodes, d] or [d]
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        rounds: Number of propagation rounds (1 or 2)
        drg_util: Current DRG utilization
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with propagated features and consistency score
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        logger.warning(f"SMP: NaN/Inf in input, zeroing")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    # DRG adaptation
    if drg_util > 0.7:
        rounds = 1
        fallbacks |= 2
    
    rounds = max(1, min(rounds, 2))  # Bounded to 1-2 rounds
    
    # Handle input shape
    is_sequence = len(x.shape) == 2
    if is_sequence:
        n_nodes, d = x.shape
        x_nodes = x.astype(np.float32)
    else:
        d = len(x)
        n_nodes = len(subgraph.nodes)
        if n_nodes == 0:
            n_nodes = 1
        x_nodes = np.broadcast_to(x.reshape(1, d), (n_nodes, d)).astype(np.float32)
    
    # Map node IDs to indices
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    
    # Build signed adjacency: A⁺ and A⁻
    A_plus = {}  # (i, j) -> weight for positive edges
    A_minus = {}  # (i, j) -> weight for negative edges
    
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            weight = subgraph.edge_weights.get((src, dst), 0.0)
            
            if weight > 0:
                A_plus[(i, j)] = abs(weight)
            elif weight < 0:
                A_minus[(i, j)] = abs(weight)
    
    # Initialize
    x_t = x_nodes.copy()
    
    for round_idx in range(rounds):
        # Two-channel message passing
        m_plus = np.zeros_like(x_t, dtype=np.float32)
        m_minus = np.zeros_like(x_t, dtype=np.float32)
        
        # Positive channel: m⁺ = A⁺ x
        for (i, j), weight in A_plus.items():
            if i < n_nodes and j < n_nodes:
                m_plus[j] += weight * x_t[i]
        
        # Negative channel: m⁻ = A⁻ x
        for (i, j), weight in A_minus.items():
            if i < n_nodes and j < n_nodes:
                m_minus[j] += weight * x_t[i]
        
        # Combine: x' = σ(m⁺ − m⁻) where σ is bounded activation
        # Use tanh as bounded activation
        m_diff = m_plus - m_minus
        x_t = np.tanh(m_diff)  # Bounded to [-1, 1]
        
        # Clip to prevent extreme values
        x_t = np.clip(x_t, -3.0, 3.0)
    
    # Compute consistency score: agreement of sign with seed
    if len(subgraph.nodes) > 0:
        # Compare final signs with original
        sign_original = np.sign(x_nodes)
        sign_final = np.sign(x_t)
        consistency = np.mean(sign_original == sign_final)
    else:
        consistency = 0.0
    
    # Validate output
    if np.any(np.isnan(x_t)) or np.any(np.isinf(x_t)):
        logger.warning(f"SMP: NaN/Inf after propagation, using fallback")
        x_t = x_nodes
        fallbacks |= 4
    
    # Map back to output shape
    if not is_sequence:
        x_t = pooling_avg(x_t, axis=0)
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=x_t.astype(np.float16),
        stats={
            'smp_latency_ms': cost_ms,
            'smp_consistency': float(consistency),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def relational_contrast(
    R: np.ndarray,
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    null_samples: int = 2,
    drg_util: float = 0.0,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Quantify how distinct the local neighborhood signal is versus shuffled baseline.
    
    Provides robustness score to fight spurious correlations.
    
    Args:
        R: Relational latent (any from previous operators)
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        null_samples: Number of null samplings (shuffled graphs)
        drg_util: Current DRG utilization
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with contrast score
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG adaptation
    if drg_util > 0.8:
        null_samples = 1
        fallbacks |= 1
    
    null_samples = max(1, min(null_samples, 4))  # Bounded to 1-4
    
    # Compute scalar summary of true relational latent
    # Use energy as summary statistic
    mu_true = np.sum(R ** 2)
    
    # Generate null samplings (edge-shuffled subgraphs)
    rng = _deterministic_hash(seed, f"rcl_{window_id}")
    mu_nulls = []
    
    for null_idx in range(null_samples):
        # Shuffle edges by randomizing connections
        # Simple approach: randomly reassign destination nodes
        shuffled_edges = []
        if len(subgraph.edges) > 0:
            dst_nodes = [dst for _, dst in subgraph.edges]
            shuffled_dst = rng.choice(dst_nodes, size=len(dst_nodes), replace=False)
            
            for idx, (src, _) in enumerate(subgraph.edges):
                if idx < len(shuffled_dst):
                    shuffled_edges.append((src, shuffled_dst[idx]))
        
        # Compute summary on shuffled graph (same topology, different connections)
        # For simplicity, use same R but permuted
        if len(shuffled_edges) > 0:
            # Permute R to simulate shuffled connections
            R_perm = rng.permutation(R.flatten()).reshape(R.shape)
            mu_null = np.sum(R_perm ** 2)
        else:
            mu_null = mu_true  # No change if no edges
        
        mu_nulls.append(mu_null)
    
    # Compute contrast: c = (μ_true − μ_null) / σ_null
    mu_nulls_arr = np.array(mu_nulls)
    mu_null_mean = np.mean(mu_nulls_arr)
    mu_null_std = np.std(mu_nulls_arr)
    
    if mu_null_std < 1e-12:
        contrast = 0.0
    else:
        contrast = (mu_true - mu_null_mean) / mu_null_std
    
    # Output contrast as scalar latent
    contrast_vec = np.array([float(contrast)], dtype=np.float16)
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=contrast_vec,
        stats={
            'rcl_latency_ms': cost_ms,
            'rcl_contrast': float(contrast),
            'nulls_used': null_samples,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def community_aware_pooling(
    H_nodes: np.ndarray,
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    c_cap: int = 4,
    drg_util: float = 0.0,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Pool node features by local communities to produce interpretable group signals.
    
    Uses lightweight online label propagation or one-pass modularity heuristic
    on the sampled subgraph to derive ≤ c_cap groups.
    
    Args:
        H_nodes: Node features [n_nodes, D]
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        c_cap: Maximum number of communities (2-6)
        drg_util: Current DRG utilization
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with pooled features per community and entropy
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # Validate input
    if np.any(np.isnan(H_nodes)) or np.any(np.isinf(H_nodes)):
        logger.warning(f"CAP: NaN/Inf in input, zeroing")
        H_nodes = np.nan_to_num(H_nodes, nan=0.0, posinf=0.0, neginf=0.0)
        fallbacks |= 1
    
    # DRG adaptation
    if drg_util > 0.8:
        c_cap = max(2, c_cap // 2)
        fallbacks |= 2
    
    c_cap = max(2, min(c_cap, 6))  # Bounded to 2-6
    
    # Handle input shape
    if len(H_nodes.shape) == 2:
        n_nodes, D = H_nodes.shape
        H = H_nodes.astype(np.float32)
    else:
        D = len(H_nodes)
        n_nodes = len(subgraph.nodes)
        if n_nodes == 0:
            return RelationalOutput(
                latent=np.zeros((1, D), dtype=np.float16),
                stats={
                    'cap_latency_ms': (time.time() - start_time) * 1000,
                    'cap_groups': 0,
                    'cap_entropy': 0.0,
                    'fallbacks_taken': 4
                },
                cost_hint_ms=(time.time() - start_time) * 1000
            )
        H = np.broadcast_to(H_nodes.reshape(1, D), (n_nodes, D)).astype(np.float32)
    
    if n_nodes == 0:
        return RelationalOutput(
            latent=np.zeros((1, D), dtype=np.float16),
            stats={
                'cap_latency_ms': (time.time() - start_time) * 1000,
                'cap_groups': 0,
                'cap_entropy': 0.0,
                'fallbacks_taken': 4
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Map node IDs to indices
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    
    # Lightweight one-pass modularity heuristic
    # Assign nodes to communities based on edge weights
    communities = np.zeros(n_nodes, dtype=np.int32)
    
    # Initialize: each node in its own community
    for i in range(n_nodes):
        communities[i] = i
    
    # Merge communities greedily based on edge weights
    # Build adjacency list
    adj_list = defaultdict(list)
    for src, dst in subgraph.edges:
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            weight = abs(subgraph.edge_weights.get((src, dst), 0.0))
            if weight > 1e-12:
                adj_list[i].append((j, weight))
                adj_list[j].append((i, weight))
    
    # Greedy merging: merge nodes with strongest connections
    # until we have ≤ c_cap communities
    current_communities = n_nodes
    
    while current_communities > c_cap and len(adj_list) > 0:
        # Find strongest edge connecting different communities
        best_weight = 0.0
        best_pair = None
        
        for i, neighbors in adj_list.items():
            for j, weight in neighbors:
                if communities[i] != communities[j] and weight > best_weight:
                    best_weight = weight
                    best_pair = (communities[i], communities[j])
        
        if best_pair is None:
            break
        
        # Merge communities
        c1, c2 = best_pair
        if c1 > c2:
            c1, c2 = c2, c1
        
        # Reassign all nodes from c2 to c1
        for i in range(n_nodes):
            if communities[i] == c2:
                communities[i] = c1
        
        current_communities -= 1
    
    # Get final community assignments
    unique_communities = np.unique(communities)
    n_communities = len(unique_communities)
    
    # Pool features per community (trimmed mean)
    R_pooled = []
    for c in unique_communities:
        mask = communities == c
        community_features = H[mask]
        
        if len(community_features) > 0:
            # Trimmed mean (remove outliers)
            if len(community_features) >= 3:
                sorted_features = np.sort(community_features, axis=0)
                k = max(1, len(sorted_features) // 10)  # Trim 10%
                trimmed = sorted_features[k:-k] if k > 0 else sorted_features
                z_group = np.mean(trimmed, axis=0)
            else:
                z_group = np.mean(community_features, axis=0)
            
            R_pooled.append(z_group)
    
    if len(R_pooled) == 0:
        R_pooled = [np.mean(H, axis=0)]
        n_communities = 1
    
    R = np.array(R_pooled, dtype=np.float32)  # [n_communities, D]
    
    # Compute membership entropy
    community_sizes = [np.sum(communities == c) for c in unique_communities]
    total_size = sum(community_sizes)
    if total_size > 0:
        probs = [size / total_size for size in community_sizes]
        entropy = -sum(p * np.log(p + 1e-12) for p in probs)
    else:
        entropy = 0.0
    
    # Sort by energy and drop lowest if over budget
    if len(R) > c_cap:
        energies = np.sum(R ** 2, axis=1)
        top_indices = np.argsort(energies)[-c_cap:]
        R = R[top_indices]
        n_communities = c_cap
    
    # Validate output
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        logger.warning(f"CAP: NaN/Inf after pooling, using fallback")
        R = np.mean(H, axis=0, keepdims=True)
        fallbacks |= 8
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=R.astype(np.float16),
        stats={
            'cap_latency_ms': cost_ms,
            'cap_groups': n_communities,
            'cap_entropy': float(entropy),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def hyperedge_reducer(
    hyperedge_sources: List[List[int]],
    source_latents: List[np.ndarray],
    store: HypergraphStore,
    max_sources: int = 3,
    drg_util: float = 0.0,
    drg_sketch_dim: int = 512,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Compress a hyperedge [A,B,…]→Y from Store into a single latent.
    
    Captures joint contribution without explicit tensor blow-up via TensorSketch.
    
    Args:
        hyperedge_sources: List of source node ID lists (one per hyperedge)
        source_latents: List of Tier-1 latents for each source [D]
        store: HypergraphStore instance
        max_sources: Maximum number of sources to process (≤3)
        drg_util: Current DRG utilization
        drg_sketch_dim: Sketch dimension from DRG
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with compressed latent and joint-gain estimate
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG adaptation
    if drg_util > 0.7:
        max_sources = max(2, max_sources - 1)
        drg_sketch_dim = max(256, drg_sketch_dim // 2)
        fallbacks |= 1
    
    max_sources = max(2, min(max_sources, 3))  # Bounded to 2-3
    
    if len(hyperedge_sources) == 0 or len(source_latents) == 0:
        return RelationalOutput(
            latent=np.zeros(drg_sketch_dim, dtype=np.float16),
            stats={
                'her_latency_ms': (time.time() - start_time) * 1000,
                'her_sources': 0,
                'fallbacks_taken': 2
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Limit sources
    n_sources = min(len(hyperedge_sources), len(source_latents), max_sources)
    sources = hyperedge_sources[:n_sources]
    latents = source_latents[:n_sources]
    
    # Validate and normalize latents
    D = len(latents[0]) if len(latents) > 0 else drg_sketch_dim
    latents_valid = []
    
    for i, z in enumerate(latents):
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            logger.warning(f"HER: NaN/Inf in source {i}, zeroing")
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            fallbacks |= 4
        
        if len(z) != D:
            # Pad or truncate
            if len(z) < D:
                z_padded = np.zeros(D, dtype=np.float32)
                z_padded[:len(z)] = z
                z = z_padded
            else:
                z = z[:D]
        
        latents_valid.append(z.astype(np.float32))
    
    if len(latents_valid) == 0:
        return RelationalOutput(
            latent=np.zeros(drg_sketch_dim, dtype=np.float16),
            stats={
                'her_latency_ms': (time.time() - start_time) * 1000,
                'her_sources': 0,
                'fallbacks_taken': 8
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Apply TensorSketch across sources with consistent hashing
    # For hyperedge id, use hash of source node IDs
    rng = _deterministic_hash(seed, f"her_{window_id}")
    
    # Sequential tensor sketch: sketch each source, then combine
    # For efficiency, use CountSketch on each source and combine
    sketched = []
    
    for i, z in enumerate(latents_valid):
        # Use deterministic hash per source
        source_rng = _deterministic_hash(seed + i, f"her_{window_id}_{i}")
        sketch = countsketch(z, dim=drg_sketch_dim, seed=seed + i, path_id=f"her_{window_id}_{i}")
        sketched.append(sketch.astype(np.float32))
    
    # Combine sketches: element-wise product (approximates tensor product)
    R = np.ones(drg_sketch_dim, dtype=np.float32)
    for sketch in sketched:
        R = R * sketch
    
    # Normalize
    R = R / (np.linalg.norm(R) + 1e-12)
    
    # Compute leave-one-out approximations to estimate marginal contributions
    # R_i = R - R_{-i} (approximate)
    marginals = []
    for i in range(len(sketched)):
        # Approximate R_{-i} by removing i-th source's contribution
        R_minus_i = np.ones(drg_sketch_dim, dtype=np.float32)
        for j, sketch in enumerate(sketched):
            if j != i:
                R_minus_i = R_minus_i * sketch
        
        R_minus_i = R_minus_i / (np.linalg.norm(R_minus_i) + 1e-12)
        marginal = R - R_minus_i
        marginals.append(marginal)
    
    # Combine with stability weights (if available from store)
    # For now, use uniform weights
    if len(marginals) > 0:
        joint_gain = np.mean([np.sum(m ** 2) for m in marginals])
    else:
        joint_gain = np.sum(R ** 2)
    
    # Validate output
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        logger.warning(f"HER: NaN/Inf after reduction, using fallback")
        R = np.mean([z for z in latents_valid], axis=0)
        R = R[:drg_sketch_dim] if len(R) > drg_sketch_dim else np.pad(R, (0, drg_sketch_dim - len(R)))
        fallbacks |= 16
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=R.astype(np.float16),
        stats={
            'her_latency_ms': cost_ms,
            'her_sources': n_sources,
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )


def causal_hint_prop(
    seed_hints: Dict[Tuple[int, int], float],
    subgraph: SampledSubgraph,
    store: HypergraphStore,
    min_accept_rate: float = 0.05,
    drg_util: float = 0.0,
    seed: int = 0,
    window_id: int = 0
) -> RelationalOutput:
    """
    Diffuse weak causal hints (from past Granger-like signals) over local subgraph.
    
    Suggests promising directions for exploration (lightweight hints, not identification).
    
    Args:
        seed_hints: Dictionary mapping (src, dst) edge keys to hint values
        subgraph: Sampled subgraph
        store: HypergraphStore instance
        min_accept_rate: Minimum acceptance rate to enable (avoid biasing when weak)
        drg_util: Current DRG utilization
        seed: Deterministic seed
        window_id: Current window ID
        
    Returns:
        RelationalOutput with directional preferences and hint-consistency
    """
    import time
    start_time = time.time()
    
    fallbacks = 0
    
    # DRG hooks: disabled if acceptance rate too low (avoid biasing when evidence is weak)
    # For now, always enabled (acceptance rate tracking would come from Evaluator)
    if drg_util > 0.9:
        # Return zero hints under extreme pressure
        n_nodes = len(subgraph.nodes)
        return RelationalOutput(
            latent=np.zeros(n_nodes, dtype=np.float16) if n_nodes > 0 else np.zeros(1, dtype=np.float16),
            stats={
                'chp_latency_ms': (time.time() - start_time) * 1000,
                'chp_hint_consistency': 0.0,
                'fallbacks_taken': 1
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    n_nodes = len(subgraph.nodes)
    if n_nodes == 0 or len(seed_hints) == 0:
        return RelationalOutput(
            latent=np.zeros(n_nodes, dtype=np.float16) if n_nodes > 0 else np.zeros(1, dtype=np.float16),
            stats={
                'chp_latency_ms': (time.time() - start_time) * 1000,
                'chp_hint_consistency': 0.0,
                'fallbacks_taken': 2
            },
            cost_hint_ms=(time.time() - start_time) * 1000
        )
    
    # Map node IDs to indices
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph.nodes)}
    
    # Initialize hint scores per node
    hint_scores = np.zeros(n_nodes, dtype=np.float32)
    
    # Single-step propagation: score of node j is sum over incoming hinted edges
    # weighted by Store stability and by π from RWR (for now, uniform)
    for (src, dst), hint_value in seed_hints.items():
        if src in node_to_idx and dst in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[dst]
            
            # Get edge stability from store
            edge = store.get_edge(src, dst)
            stability = edge.stability if edge else 0.5
            
            # Weight hint by stability
            hint_scores[j] += hint_value * stability
    
    # Normalize and clamp to [-1, 1]
    max_abs = np.max(np.abs(hint_scores))
    if max_abs > 1e-12:
        hint_scores = hint_scores / max_abs
    
    hint_scores = np.clip(hint_scores, -1.0, 1.0)
    
    # Compute hint-consistency: variance of hint signs (lower = more consistent)
    if len(hint_scores) > 1:
        consistency = 1.0 - float(np.var(np.sign(hint_scores)))
    else:
        consistency = 1.0
    
    cost_ms = (time.time() - start_time) * 1000
    
    return RelationalOutput(
        latent=hint_scores.astype(np.float16),
        stats={
            'chp_latency_ms': cost_ms,
            'chp_hint_consistency': float(consistency),
            'fallbacks_taken': fallbacks
        },
        cost_hint_ms=cost_ms
    )

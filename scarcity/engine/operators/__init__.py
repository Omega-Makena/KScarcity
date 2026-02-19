"""
GPU operators for online inference — Tiered Architecture.

Tier 0 (Basic Operators): ✅ Implemented
- Mathematical primitives, tensor utilities, sketching, attention, stability
- Mandatory for every window; bounded online complexity

Tier 1 (Structural): ✅ Implemented
- Composite features: temporal fusion, cross-variable alignment, low-rank projections

Tier 2 (Relational): ✅ Implemented
- Graph reasoning: neighborhood sampling, diffusion, random walks, attention, message passing

Tier 3 (Causal/Semantic): 🌐 Future
- Counterfactual analysis, policy impact, language embeddings

Tier 4 (Integrative): 🚀 Future
- Meta-learning orchestration, adaptive fusion

See OPERATOR_TAXONOMY.md for full architecture.
"""

# Tier 0: Attention & Normalization
from .attention_ops import (
    attn_linear,
    attn_sparse,
    pooling_avg,
    pooling_lastk,
    layernorm,
    rmsnorm
)

# Tier 0: Sketching
from .sketch_ops import (
    poly_sketch,
    tensor_sketch,
    countsketch,
    latent_clip,
    _deterministic_hash
)

# Tier 0: Evaluation
from .evaluation_ops import (
    r2_gain,
    nll_gain,
    granger_step
)

# Tier 0: Stability & Drift
from .stability_ops import (
    spearman_concordance,
    sign_agreement,
    page_hinkley,
    stability_score
)

# Tier 1: Structural Operators
try:
    from .structural_ops import (
        temporal_fusion,
        cross_align,
        lowrank_mix,
        sparse_interact,
        structural_pool,
        regime_gate,
        denoise_lite,
        StructuralOutput
    )
    _tier1_available = True
except ImportError:
    _tier1_available = False

# Tier 2: Relational Operators
try:
    from .relational_ops import (
        neighborhood_sampling,
        diffusion_sketch,
        random_walk_with_restart,
        relational_attention,
        signed_message_passing,
        relational_contrast,
        community_aware_pooling,
        hyperedge_reducer,
        causal_hint_prop,
        RelationalOutput,
        SampledSubgraph
    )
    _tier2_available = True
except ImportError:
    _tier2_available = False

# Tier 3: Causal & Semantic Operators
try:
    from .causal_semantic_ops import (
        directional_causality,
        counterfactual_lite,
        causal_graph_propagation,
        policy_semantic_align,
        concept_graph_reason,
        temporal_causal_fusion,
        meaning_aggregator,
        CausalSemanticOutput,
    )
    _tier3_available = True
except ImportError:
    _tier3_available = False

# Tier 4: Integrative Operators
try:
    from .integrative_ops import (
        multi_modal_fusion,
        hierarchical_context_integrator,
        cross_tier_reconciliation,
        adaptive_meta_aggregator,
        policy_decision_synthesizer,
        forecast_integrator,
        reasoning_loop_controller,
        integrative_feedback_emitter,
        global_insight_assembler,
        IntegrativeOutput,
    )
    _tier4_available = True
except ImportError:
    _tier4_available = False

__all__ = [
    # Tier 0: Attention & Normalization
    'attn_linear',
    'attn_sparse',
    'pooling_avg',
    'pooling_lastk',
    'layernorm',
    'rmsnorm',
    
    # Tier 0: Sketching
    'poly_sketch',
    'tensor_sketch',
    'countsketch',
    'latent_clip',
    '_deterministic_hash',
    
    # Tier 0: Evaluation
    'r2_gain',
    'nll_gain',
    'granger_step',
    
    # Tier 0: Stability & Drift
    'spearman_concordance',
    'sign_agreement',
    'page_hinkley',
    'stability_score',
]

# Add Tier 1 exports if available
if _tier1_available:
    __all__.extend([
        # Tier 1: Structural
        'temporal_fusion',
        'cross_align',
        'lowrank_mix',
        'sparse_interact',
        'structural_pool',
        'regime_gate',
        'denoise_lite',
        'StructuralOutput',
    ])

# Add Tier 2 exports if available
if _tier2_available:
    __all__.extend([
        # Tier 2: Relational
        'neighborhood_sampling',
        'diffusion_sketch',
        'random_walk_with_restart',
        'relational_attention',
        'signed_message_passing',
        'relational_contrast',
        'community_aware_pooling',
        'hyperedge_reducer',
        'causal_hint_prop',
        'RelationalOutput',
        'SampledSubgraph',
    ])

# Add Tier 3 exports if available
if _tier3_available:
    __all__.extend([
        'directional_causality',
        'counterfactual_lite',
        'causal_graph_propagation',
        'policy_semantic_align',
        'concept_graph_reason',
        'temporal_causal_fusion',
        'meaning_aggregator',
        'CausalSemanticOutput',
    ])

# Add Tier 4 exports if available
if _tier4_available:
    __all__.extend([
        'multi_modal_fusion',
        'hierarchical_context_integrator',
        'cross_tier_reconciliation',
        'adaptive_meta_aggregator',
        'policy_decision_synthesizer',
        'forecast_integrator',
        'reasoning_loop_controller',
        'integrative_feedback_emitter',
        'global_insight_assembler',
        'IntegrativeOutput',
    ])

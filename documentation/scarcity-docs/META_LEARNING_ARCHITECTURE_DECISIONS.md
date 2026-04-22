# SCARCITY Meta-Learning: Architecture Decisions & Known Weaknesses

**Date:** 2026-04-20  
**Status:** In-progress design — pre-implementation

---

## 1. What We Discovered: Current Meta Module Is Not Meta-Learning

The existing `scarcity/meta/` module was labelled "meta-learning" but operates as **adaptive learning**:

- `DomainMetaLearner` — EMA-based confidence scoring + adaptive learning rate. No learning-to-learn.
- `OnlineReptileOptimizer` — REPTILE interpolates toward task-specific parameters. Closer to online fine-tuning than meta-optimization.
- `CrossDomainMetaAggregator` — trimmed mean over domain deltas. Pure aggregation, no learned update rule.

**Consequence:** the system cannot warm-start an unseen domain from prior experience. Every domain begins cold from defaults. This is an adaptive system, not a meta-learning system.

**Decision:** rename the existing module's role to **adaptive fallback layer**. It remains active and becomes the cold-start and failure-recovery path when the true meta-learner is unavailable or has rolled back.

---

## 2. Target Architecture: Federated Meta-Learning with Hierarchical Memory

### Three-Layer Learning Separation

```
Layer 3 — Cross-Domain Meta-Learner (global, abstract)
               learns transferable invariants across all domains

Layer 2 — Domain Server / Basket (per-domain, domain-specific)
               holds pretrained base model for that domain
               coordinates clients within the basket

Layer 1 — Client Node (local, fast adaptation)
               adapts from domain base model using local memory
               sends adaptation signals (Δθ, embeddings, metrics) upward
```

### Key Principle

No raw data leaves a client. What travels upward:
- `Δθ` — parameter update delta (low-rank approximation)
- Embedding summary — clustered context representations
- Performance metrics — loss before/after adaptation, convergence speed

---

## 3. Domain Server Topology Decision

### Problem Identified

The original architecture assumed "one server per domain" but this does not map to reality:
- Multiple independent clients exist within the same domain (e.g., three hospitals all in "healthcare")
- You cannot assign them all to the same server — they are separate entities
- Spinning up a dedicated server process per domain is operationally expensive and unnecessary

### Solution

**Domain servers are logical agents, not network processes.**

The existing `BasketModel` in `scarcity/federation/layers.py:66` already groups clients by domain. It needs to be **elevated** into a full `DomainServer` object that owns:

1. A domain-specific pretrained base model (initialization for all clients in that basket)
2. The episodic memory buffer for that domain
3. Coordination of client adaptation within the basket
4. Upward communication of compressed signals to the global meta-learner

This means no new infrastructure is needed for the topology. `HierarchicalFederation` already provides the grouping. The change is in what each basket *holds and does*, not how baskets are created or networked.

**Benefit:** domain-specific pretraining becomes tractable. Each domain server accumulates episodes and builds a specialized base model. Clients warm-start from that base rather than a cold global prior.

---

## 4. Optimizer Decision: Keep REPTILE, Add Episodic Memory

### Why Not Replace REPTILE with Full Model-Based Meta-Learning

Full model-based meta-learning (NTM, MANN, SNAIL) was considered and rejected for two reasons:

**Bug reproducibility:** model-based meta-learning makes debugging significantly harder.
- Memory state depends on the full history of past episodes
- Retrieval is approximate — the same query returns different entries after any memory update
- A bad adaptation could trace back to a memory entry from hundreds of steps ago
- "Which episode caused this behavior?" is exactly the category of bug that is hard to reproduce and pin to a specific point

**Complexity vs. benefit:** the optimizer (REPTILE) is not the limiting factor. The limiting factor is that REPTILE has no episodic memory — every new domain starts cold.

### Decision: Deterministic Episodic Memory Wrapper

Keep REPTILE as the update rule. Add a fixed-capacity episodic memory with deterministic retrieval on top:

```
Context → encoder → embedding
                        ↓
              top-k cosine similarity against episode buffer
                        ↓
              hit:  apply stored delta  (fast path — model-based behavior)
              miss: fall back to REPTILE (cold path — adaptive fallback)
                        ↓
              store new episode after adaptation
```

**Why this preserves debuggability:**
- Episode buffer is a fixed-size inspectable list
- Retrieval is deterministic: top-k cosine similarity, ties broken by recency
- Any bug can be reproduced by logging (query embedding, buffer state at time of query)
- No neural memory controller — no hidden state that changes on every step

**Why this achieves true meta-learning behavior:**
- The system reuses past adaptation patterns instead of always starting cold
- A new domain that resembles a past domain gets a warm start from stored deltas
- The episode buffer accumulates across rounds, giving the system a real history to learn from

---

## 5. Component Maturity Assessment

### Federation Layer

| Component | File | Maturity | Notes |
|---|---|---|---|
| BasketManager | `federation/basket.py` | 90% | Domain grouping, DP fingerprints, sub-basket k-means — solid |
| GossipProtocol | `federation/gossip.py` | 85% | Intra-basket communication with local DP |
| UpdateBuffer | `federation/buffer.py` | 85% | Staleness-aware storage with triggers |
| Layer1Aggregator | `federation/layers.py` | 80% | Intra-basket trimmed mean aggregation |
| Layer2Aggregator | `federation/layers.py` | 80% | Cross-basket Bulyan + DP + secure agg |
| SecureAggregation | `federation/secure_aggregation.py` | 75% | Masking-based, not cryptographic |
| PrivacyAccountant | `federation/buffer.py` | 80% | DP budget tracking |
| FederationClientAgent | `federation/client_agent.py` | 75% (transport) / 0% (meta role) | Transport/DP/validation mature; no Base Model, Memory, Encoder, Adaptation Engine |
| GlobalMetaModel | `federation/layers.py:80` | 5% | Stub — just `hypothesis_params: Dict[str, float]` |
| Transport / WsTransport | `federation/transport.py` | 80% | Functional |

### Adaptive Fallback Layer (current "meta" module)

| Component | File | Maturity | Notes |
|---|---|---|---|
| DomainMetaLearner | `meta/domain_meta.py` | 90% | EMA + confidence — will be fallback |
| CrossDomainMetaAggregator | `meta/cross_meta.py` | 90% | Trimmed mean — will be fallback |
| OnlineReptileOptimizer | `meta/optimizer.py` | 85% | Will remain as cold-start path |
| MetaLearningAgent | `meta/meta_learning.py` | 85% | EventBus orchestration — will be fallback orchestrator |
| MetaSupervisor | `meta/integrative_meta.py` | 85% | Rollback suppression — reusable |
| MetaStorageManager | `meta/storage.py` | 90% | Filesystem persistence — reusable |

### Missing Components (not yet built)

| Component | Purpose | Effort |
|---|---|---|
| Local Memory Module | Per-domain episode store: `(key, value, context, Δ, policy)` | ~1 week |
| Context Encoder | Structured context dict → fixed embedding vector | ~3-4 days |
| Adaptation Engine | Cosine similarity retrieval + delta application | ~1 week |
| DomainServer | Elevated BasketModel with base model + memory + client coordination | ~1 week |
| Global Meta-Memory | Compressed abstractions: adaptation distributions, invariant patterns | ~1 week |
| Protocol Bridge | Extend `packets.py` / `codec.py` to carry `(Δθ, embedding_summary, metrics)` | ~4-5 days |

**Estimated total to stable federated memory layer: 4–5 weeks**

The domain-based and cross-domain meta-learners (real meta-learning at the global level) come after this — an additional 3–4 weeks.

---

## 6. Known Weaknesses

### W1 — No Episodic Memory Anywhere

Neither the federation layer nor the meta module stores past adaptation episodes. Every round begins without knowledge of what worked before. This is the primary scalability bottleneck.

### W2 — GlobalMetaModel Is a Stub

`GlobalMetaModel` in `layers.py` only stores a flat `Dict[str, float]`. The compressed abstraction layer (adaptation distributions, invariant patterns, domain relationships) described in the architecture does not exist.

### W3 — Client Agent Has No Meta-Learning Role

`FederationClientAgent` performs transport, DP, and reconciliation. It has no Base Model, no Memory Module, no Encoder, and no Adaptation Engine. It cannot generate adaptation signals (`Δθ`, embeddings) — it generates knowledge graph packets (`EdgeDelta`, `PathPack`). This is a different communication model from what the new architecture requires.

### W4 — Protocol Mismatch

The federation protocol currently exchanges structural knowledge (graph edges, causal packs). The new architecture requires exchanging adaptation signals. These are different payloads. `packets.py` and `codec.py` need extension — not replacement, but addition of new packet types alongside existing ones.

### W5 — Bug Reproducibility Risk in Memory Systems

Any approximate retrieval mechanism (cosine similarity) creates non-determinism if the memory buffer is not snapshotted at query time. A bug manifesting as a bad adaptation may trace back to an episode that was later evicted. **Mitigation already decided:** fixed-capacity buffer, deterministic top-k retrieval, ties broken by recency, query + buffer state logged together.

### W6 — Domain Server Topology Was Underspecified

The original design implied one server per domain without specifying how multiple independent clients in the same domain would be coordinated. This has been resolved: domain servers are logical agents (elevated `BasketModel`), not network processes.

### W7 — Secure Aggregation Is Non-Cryptographic

The current `SecureAggregation` uses masking-based summation (in-process). This is adequate for development but not production-grade under adversarial conditions. Noted as a future hardening item, not a blocker for the meta-learning work.

### W8 — Scalability of Episode Retrieval

Cosine similarity over a fixed-capacity buffer is O(n) per query. At scale (many clients, large buffers) this becomes a bottleneck. Mitigation path: approximate nearest-neighbor index (e.g., FAISS or a simple LSH). Not needed for initial implementation but must be in scope before production scale.

---

## 7. Build Order

Given the above, the recommended build sequence is:

```
Phase 1 — Local memory foundation (independent of federation changes)
  1a. Context Encoder              scarcity/meta/encoder.py
  1b. Local Memory Module          scarcity/meta/memory.py
  1c. Adaptation Engine            scarcity/meta/adaptation.py

Phase 2 — Domain Server elevation (extends BasketModel)
  2a. DomainServer                 scarcity/federation/domain_server.py
  2b. Wire into HierarchicalFederation

Phase 3 — Global Meta-Memory (replaces GlobalMetaModel stub)
  3a. Global Meta-Memory           scarcity/federation/global_meta_memory.py

Phase 4 — Protocol bridge (extends, does not break existing)
  4a. New packet types             scarcity/federation/packets.py (additive)
  4b. Codec extension              scarcity/federation/codec.py (additive)

Phase 5 — True meta-learner at global level
  5a. Domain-Based Meta-Learner    scarcity/meta/domain_server_meta.py
  5b. Cross-Domain Meta-Learner    extend scarcity/meta/cross_meta.py
```

Phases 1–4 deliver the **stable federated memory layer**.  
Phase 5 delivers **true meta-learning** (learning to learn).  
The existing adaptive fallback (`meta/`) remains active throughout and handles cold starts, rollbacks, and failure recovery.

---

## 8. What Does Not Need to Change

- `BasketManager` — mature, no changes needed
- `GossipProtocol` — mature, no changes needed
- `UpdateBuffer` + `TriggerEngine` — mature, no changes needed
- `PrivacyAccountant` — mature, no changes needed
- `DomainMetaLearner`, `CrossDomainMetaAggregator`, `OnlineReptileOptimizer` — promoted to fallback role, no deletion
- `MetaStorageManager` — reusable for episode persistence
- `MetaSupervisor` — rollback suppression logic reusable in new architecture

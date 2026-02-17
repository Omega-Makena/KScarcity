# KShield Federation (Aegis Protocol) — Overview

> `kshiked.federation` — Defense-sector federated collaboration layer.

---

## Purpose

The KShield Federation module (codenamed **Aegis Protocol**) provides defense-sector-specific federation capabilities that sit atop the core `scarcity.federation` primitives. While `scarcity.federation` provides generic federated learning (aggregation, privacy, transport), this module adds:

- **Node identity and governance** for defense agencies
- **Gossip-based peer discovery** for decentralized topologies
- **Security hardening** for classified environments
- **Schema exchange** for inter-agency data models

---

## Distinction from `scarcity.federation`

| Aspect | `scarcity.federation` | `kshiked.federation` |
|--------|----------------------|---------------------|
| Level | Core FL primitives | Defense-sector integration |
| Key Class | `FederationClientAgent` | `FederationNode` |
| Aggregation | `SecureAggregator`, `FederatedAggregator` | Uses scarcity aggregators internally |
| Transport | Generic (loopback, gRPC) | Gossip-based peer discovery |
| Security | DP, secure aggregation | Agency-level access control, classified handling |

---

## File Guide

| File | Size | Purpose |
|------|------|---------|
| `coordinator.py` | 2.6 KB | Federation round coordinator — manages participation, rounds, and results |
| `gossip.py` | 5.4 KB | Gossip protocol for peer discovery and topology management |
| `governance.py` | 6.3 KB | Federation governance policies — trust scoring, access control, contribution rules |
| `integration.py` | 6.0 KB | Integration layer connecting KShield intelligence to federation rounds |
| `node.py` | 6.7 KB | `FederationNode` — represents a single agency node with local model and state |
| `schemas.py` | 3.4 KB | Data schemas for federation messages, packets, and metadata exchange |
| `security.py` | 5.1 KB | Security layer — encryption, authentication, audit logging for classified contexts |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              Aegis Protocol                   │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Node A  │  │  Node B  │  │  Node C  │   │
│  │ (Agency) │  │ (Agency) │  │ (Agency) │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │         │
│       └──────gosip.py───────────────┘         │
│                    │                          │
│          ┌─────────▼─────────┐                │
│          │  coordinator.py   │                │
│          │  (Round Manager)  │                │
│          └─────────┬─────────┘                │
│                    │                          │
│          ┌─────────▼─────────┐                │
│          │  governance.py    │                │
│          │  (Trust & Policy) │                │
│          └─────────┬─────────┘                │
│                    │                          │
│          ┌─────────▼─────────┐                │
│          │  security.py      │                │
│          │  (Crypto & Audit) │                │
│          └───────────────────┘                │
│                                               │
│  ┌────────────────────────────────────────┐   │
│  │      scarcity.federation (core FL)     │   │
│  │  SecureAggregator · PrivacyGuard       │   │
│  └────────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

---

*Source: `kshiked/federation/` · Last updated: 2026-02-11*

# SCARCITY Core Library - Complete Documentation

## Overview

This documentation covers the complete `scarcity/` core library - the heart of the SCARCITY framework. The core library implements all machine learning algorithms, data structures, and mathematical operations.

**Location**: `scarcity/`

**Purpose**: Reusable, production-grade implementations of:
- Multi-Path Inference Engine (MPIE) for causal discovery
- Dynamic Resource Governor (DRG) for adaptive resource management
- Federation layer for decentralized learning
- Meta-learning for cross-domain optimization
- Simulation engine for 3D hypergraph visualization
- Streaming data processing
- Runtime event bus

## Documentation Structure

### Core Modules
1. [Runtime Bus](./01-runtime.md) - Event-driven communication fabric
2. [Engine (MPIE)](./02-engine.md) - Multi-Path Inference Engine
3. [Governor (DRG)](./03-governor.md) - Dynamic Resource Governor
4. [Federation](./04-federation.md) - Federated learning layer
5. [Meta-Learning](./05-meta.md) - Cross-domain optimization
6. [Simulation](./06-simulation.md) - 3D hypergraph visualization
7. [Stream Processing](./07-stream.md) - Data streaming utilities
8. [FMI](./08-fmi.md) - Federated Model Interface
9. [Dashboard](./09-dashboard.md) - Dashboard integration

### Technical Details
- [Data Structures](./10-data-structures.md) - Core data structures
- [Algorithms](./11-algorithms.md) - Algorithm implementations
- [Mathematical Operations](./12-math-operations.md) - Mathematical primitives
- [Operators](./13-operators.md) - Causal operators
- [Data Flow](./14-data-flow.md) - How data flows through components

## Module Dependency Graph

```
runtime (EventBus)
    |
    +-- engine (MPIE)
    |     |
    |     +-- controller (BanditRouter)
    |     +-- encoder (Encoder)
    |     +-- evaluator (Evaluator)
    |     +-- store (HypergraphStore)
    |     +-- operators (Causal operators)
    |
    +-- governor (DRG)
    |     |
    |     +-- monitor (ResourceMonitor)
    |     +-- policies (ControlPolicies)
    |     +-- actuators (ResourceActuators)
    |
    +-- federation
    |     |
    |     +-- coordinator (FederationCoordinator)
    |     +-- client_agent (FederationClientAgent)
    |     +-- aggregator (ModelAggregator)
    |     +-- privacy_guard (DifferentialPrivacy)
    |
    +-- meta
    |     |
    |     +-- meta_learning (MetaLearningAgent)
    |     +-- integrative_meta (MetaSupervisor)
    |     +-- optimizer (MetaOptimizer)
    |
    +-- simulation
    |     |
    |     +-- engine (SimulationEngine)
    |     +-- agents (AgentRegistry)
    |     +-- visualization3d (3D renderer)
    |
    +-- stream
          |
          +-- source (StreamSource)
          +-- window (WindowBuilder)
          +-- schema (SchemaManager)
```

## Quick Reference

### File Count by Module
- runtime: 2 files
- engine: 10 files + 9 operator files
- governor: 9 files
- federation: 12 files
- meta: 10 files
- simulation: 10 files
- stream: 8 files
- fmi: 9 files
- dashboard: 6 files + API modules

### Total Lines of Code
Approximately 15,000+ lines of production Python code

### Key Algorithms Implemented
- PC Algorithm (causal discovery)
- Multi-armed bandit (UCB, Thompson Sampling)
- Bootstrap resampling
- Federated averaging (FedAvg)
- Differential privacy (Gaussian mechanism)
- PID control
- Force-directed graph layout
- Sketch-based dimensionality reduction

### Mathematical Foundations
- Structural Causal Models (SCM)
- Conditional independence testing
- Information theory (mutual information, entropy)
- Bayesian inference
- Online learning theory
- Control theory
- Graph theory

## Navigation Guide

### For Algorithm Developers
Start with [Engine (MPIE)](./02-engine.md) to understand causal discovery implementation

### For System Engineers
Start with [Runtime Bus](./01-runtime.md) to understand event-driven architecture

### For ML Researchers
Start with [Mathematical Operations](./12-math-operations.md) and [Algorithms](./11-algorithms.md)

### For Integration Developers
Start with [Data Flow](./14-data-flow.md) to understand component interactions

## Conventions

### Code Organization
- Each module has `__init__.py` for exports
- Core classes are in module root files
- Utilities and helpers in separate files
- Operators in dedicated `operators/` subfolder

### Naming Conventions
- Classes: PascalCase (e.g., `MPIEOrchestrator`)
- Functions: snake_case (e.g., `compute_gain`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_CANDIDATES`)
- Private methods: _leading_underscore (e.g., `_internal_method`)

### Type Annotations
All public APIs use Python type hints for clarity and IDE support

### Documentation
All classes and public methods have docstrings following Google style

## Version Information

**Core Library Version**: 1.0.0
**Python Requirement**: 3.11+
**Key Dependencies**: NumPy, asyncio

## Related Documentation

- [Product Overview](../01-product-overview.md)
- [Architecture](../02-architecture.md)
- [Mathematical Foundations](../03-mathematical-foundations.md)
- [Backend Implementation](../05-backend-implementation.md)

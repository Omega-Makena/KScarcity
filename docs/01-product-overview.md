# Product Overview

## What is SCARCITY?

**SCARCITY** (Scarcity-aware Causal Adaptive Resource-efficient Intelligence Training sYstem) is an advanced machine learning framework designed for online, resource-constrained environments where data arrives in streams and computational resources are limited.

### Core Value Proposition

SCARCITY enables organizations to:
1. **Learn from streaming data** in real-time without batch processing
2. **Discover causal relationships** automatically from observational data
3. **Adapt to resource constraints** dynamically (CPU, memory, GPU)
4. **Collaborate across domains** through federated learning
5. **Optimize across tasks** using meta-learning

## Problem Statement

Traditional machine learning systems face critical challenges:

- **Data Scarcity**: Limited labeled data in specialized domains
- **Resource Constraints**: Edge devices with limited compute/memory
- **Distribution Shift**: Data distributions change over time
- **Privacy Requirements**: Cannot centralize sensitive data
- **Real-time Demands**: Decisions needed immediately, not after batch training

SCARCITY addresses all these challenges through an integrated framework.

## Key Features

### 1. Multi-Path Inference Engine (MPIE)
**Purpose**: Discover causal relationships from streaming data

**Capabilities**:
- Automatic causal graph discovery
- Multiple hypothesis testing
- Bootstrap-based statistical validation
- Hypergraph representation of causal structures

### 2. Dynamic Resource Governor (DRG)
**Purpose**: Adapt system behavior to resource availability

**Capabilities**:
- Real-time CPU/memory/GPU monitoring
- Predictive resource forecasting
- Adaptive policy enforcement
- Graceful degradation under constraints

### 3. Federation Layer
**Purpose**: Enable decentralized learning across organizations

**Capabilities**:
- Peer-to-peer model sharing
- Multiple aggregation strategies (FedAvg, Weighted, Adaptive)
- Differential privacy protection
- Flexible topology (mesh, ring, star)

### 4. Meta-Learning Agent
**Purpose**: Transfer knowledge across domains and tasks

**Capabilities**:
- Cross-domain optimization
- Prior knowledge extraction
- Performance prediction
- Adaptive hyperparameter tuning

### 5. 3D Simulation Engine
**Purpose**: Visualize and explore causal hypergraphs

**Capabilities**:
- Interactive 3D visualization
- Force-directed graph layout
- Real-time updates
- What-if scenario analysis


## Use Cases

### Healthcare
- **Problem**: Limited patient data per hospital, privacy concerns
- **Solution**: Federated learning across hospitals, causal discovery for treatment effects
- **Benefit**: Better models without sharing patient data

### Finance
- **Problem**: Market conditions change rapidly, fraud patterns evolve
- **Solution**: Online learning from transaction streams, adaptive resource allocation
- **Benefit**: Real-time fraud detection with minimal latency

### Manufacturing
- **Problem**: Equipment failures are rare, sensor data is high-volume
- **Solution**: Causal discovery for failure prediction, resource-efficient edge deployment
- **Benefit**: Predictive maintenance without cloud dependency

### Retail
- **Problem**: Customer behavior varies by region, seasonal patterns
- **Solution**: Multi-domain learning, meta-learning for new stores
- **Benefit**: Faster adaptation to new markets

## System Components

### Backend (Python/FastAPI)
- **Location**: `backend/`
- **Purpose**: Core ML engine, API server, data processing
- **Key Technologies**: FastAPI, NumPy, asyncio

### Frontend (React/TypeScript)
- **Location**: `scarcity-deep-dive/`
- **Purpose**: Interactive dashboard, visualization, monitoring
- **Key Technologies**: React, TypeScript, Vite, shadcn/ui

### Core Library (Python)
- **Location**: `scarcity/`
- **Purpose**: Reusable ML algorithms and components
- **Key Technologies**: NumPy, custom implementations

## Current Status

### ✅ Implemented
- Runtime Bus (event-driven communication)
- MPIE Orchestrator (causal discovery)
- Dynamic Resource Governor (resource monitoring)
- Multi-domain data generation
- Federation coordinator with multiple strategies
- Domain management system
- Real-time data visualization
- Demo mode for presentations

### 🚧 In Progress
- Advanced privacy mechanisms
- Distributed simulation engine
- Enhanced meta-learning algorithms

### 📋 Planned
- GPU acceleration
- Kubernetes deployment
- Advanced visualization features
- Model export/import

## Performance Characteristics

### Throughput
- **Data Ingestion**: 100-500 windows/second
- **Causal Discovery**: 50-200 candidate paths/second
- **Resource Monitoring**: 2 Hz (every 0.5 seconds)

### Latency
- **API Response**: < 100ms (p95)
- **Data Window Processing**: < 50ms (p95)
- **Federation Round**: 1-5 seconds

### Resource Usage
- **Memory**: 500MB - 2GB (depends on hypergraph size)
- **CPU**: 2-4 cores recommended
- **GPU**: Optional, not yet utilized

## Architecture Principles

1. **Event-Driven**: All components communicate via async message bus
2. **Modular**: Each component can be enabled/disabled independently
3. **Observable**: Comprehensive telemetry and logging
4. **Scalable**: Designed for horizontal scaling (future)
5. **Testable**: Clear interfaces, dependency injection

# Core Algorithms

## 1. Multi-Path Inference Engine (MPIE)

### 1.1 Overview

MPIE discovers causal relationships from streaming data using a multi-stage pipeline:

```
Data Window → Encoder → Controller → Evaluator → Hypergraph Store
```

### 1.2 Encoder Algorithm

**Purpose**: Transform raw features into causal representations

**Location**: `scarcity/engine/encoder.py`

**Algorithm**:
```python
def encode(window: np.ndarray) -> EncodedFeatures:
    """
    Encode raw features for causal discovery.
    
    Steps:
    1. Normalize features (z-score)
    2. Detect non-linear relationships
    3. Compute pairwise statistics
    4. Extract temporal patterns
    
    Returns:
        EncodedFeatures with correlation matrix, MI matrix, etc.
    """
    # Normalize
    normalized = (window - mean) / std
    
    # Compute correlations
    corr_matrix = np.corrcoef(normalized.T)
    
    # Compute mutual information
    mi_matrix = compute_mutual_information(normalized)
    
    # Detect non-linearity
    nonlinear_scores = detect_nonlinearity(normalized)
    
    return EncodedFeatures(
        corr=corr_matrix,
        mi=mi_matrix,
        nonlinear=nonlinear_scores
    )
```

**Complexity**: O(d²·n) where d=features, n=samples

### 1.3 Controller Algorithm

**Purpose**: Generate and rank candidate causal paths

**Location**: `scarcity/engine/controller.py`

**Algorithm**:
```python
def generate_candidates(
    encoded: EncodedFeatures,
    max_candidates: int = 200
) -> List[CandidatePath]:
    """
    Generate candidate causal paths.
    
    Steps:
    1. Identify strong pairwise relationships
    2. Extend to multi-hop paths
    3. Filter by statistical significance
    4. Rank by combined score
    
    Returns:
        Top-k candidate paths
    """
    candidates = []
    
    # Find strong edges (correlation > threshold)
    strong_edges = find_strong_edges(encoded.corr, threshold=0.3)
    
    # Generate paths up to length 3
    for length in [1, 2, 3]:
        paths = generate_paths(strong_edges, length)
        candidates.extend(paths)
    
    # Score each path
    for path in candidates:
        path.score = compute_path_score(path, encoded)
    
    # Sort and return top-k
    candidates.sort(key=lambda p: p.score, reverse=True)
    return candidates[:max_candidates]
```

**Path Scoring**:
```python
def compute_path_score(path: Path, encoded: EncodedFeatures) -> float:
    """
    Score = α·strength + β·stability + γ·novelty
    """
    strength = compute_strength(path, encoded)
    stability = compute_stability(path, encoded)
    novelty = compute_novelty(path, history)
    
    return 0.4*strength + 0.4*stability + 0.2*novelty
```

**Complexity**: O(d³) for path generation

### 1.4 Evaluator Algorithm

**Purpose**: Validate candidate paths using statistical tests

**Location**: `scarcity/engine/evaluator.py`

**Algorithm**:
```python
def evaluate_candidates(
    candidates: List[CandidatePath],
    data: np.ndarray,
    n_resamples: int = 1000
) -> List[ValidatedPath]:
    """
    Validate paths using bootstrap resampling.
    
    Steps:
    1. For each candidate path
    2. Perform bootstrap resampling
    3. Compute confidence intervals
    4. Test statistical significance
    5. Return validated paths
    """
    validated = []
    
    for candidate in candidates:
        # Bootstrap resampling
        bootstrap_scores = []
        for _ in range(n_resamples):
            # Resample with replacement
            indices = np.random.choice(len(data), len(data))
            resampled = data[indices]
            
            # Compute score on resample
            score = compute_path_score(candidate, resampled)
            bootstrap_scores.append(score)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        # Test significance
        if ci_lower > 0:  # Significant
            validated.append(ValidatedPath(
                path=candidate,
                score=np.mean(bootstrap_scores),
                ci=(ci_lower, ci_upper),
                p_value=compute_p_value(bootstrap_scores)
            ))
    
    return validated
```

**Complexity**: O(k·B·n) where k=candidates, B=resamples, n=samples

### 1.5 Hypergraph Store Algorithm

**Purpose**: Maintain causal graph with incremental updates

**Location**: `scarcity/engine/store.py`

**Data Structure**:
```python
class HypergraphStore:
    nodes: Dict[int, Node]      # Node ID → Node
    edges: Dict[Tuple, Edge]    # (src, dst) → Edge
    regimes: Dict[int, Regime]  # Regime ID → Regime
```

**Update Algorithm**:
```python
def update(self, validated_paths: List[ValidatedPath]):
    """
    Incrementally update hypergraph.
    
    Steps:
    1. Add new nodes if needed
    2. Update edge weights (exponential moving average)
    3. Update edge stability
    4. Prune weak edges
    5. Detect regime changes
    """
    for path in validated_paths:
        for i in range(len(path) - 1):
            src, dst = path[i], path[i+1]
            
            # Get or create edge
            edge = self.get_edge(src, dst)
            
            # Update weight (EMA)
            alpha = 0.1  # Learning rate
            edge.weight = (1-alpha)*edge.weight + alpha*path.score
            
            # Update stability
            edge.stability = compute_stability(edge.history)
            
            # Update timestamp
            edge.last_updated = now()
    
    # Prune weak edges
    self.prune_edges(threshold=0.1)
    
    # Detect regime changes
    self.detect_regime_change()
```

**Complexity**: O(k·l) where k=paths, l=path length


## 2. Dynamic Resource Governor (DRG)

### 2.1 Overview

DRG monitors system resources and adapts behavior to prevent overload.

**Location**: `scarcity/governor/drg_core.py`

### 2.2 Resource Monitoring Algorithm

**Purpose**: Track CPU, memory, GPU usage in real-time

**Location**: `scarcity/governor/monitor.py`

**Algorithm**:
```python
async def monitor_loop(self, interval: float = 0.5):
    """
    Continuously monitor system resources.
    
    Steps:
    1. Read CPU usage
    2. Read memory usage
    3. Read GPU usage (if available)
    4. Publish to event bus
    5. Sleep for interval
    """
    while self.running:
        # Read CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Read memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Read GPU (if available)
        gpu_percent = 0.0
        vram_percent = 0.0
        if torch.cuda.is_available():
            gpu_percent = get_gpu_utilization()
            vram_percent = get_vram_utilization()
        
        # Create profile
        profile = ResourceProfile(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            vram_percent=vram_percent,
            timestamp=time.time()
        )
        
        # Publish to bus
        await self.bus.publish("resource_profile", profile)
        
        # Sleep
        await asyncio.sleep(interval)
```

**Complexity**: O(1) per iteration

### 2.3 Control Policy Algorithm

**Purpose**: Decide actions based on resource state

**Location**: `scarcity/governor/policies.py`

**Algorithm**:
```python
def compute_action(self, profile: ResourceProfile) -> Action:
    """
    Compute control action using PID controller.
    
    Steps:
    1. Compute error (threshold - current)
    2. Update integral and derivative terms
    3. Compute PID output
    4. Map to discrete action
    """
    # Compute error for each resource
    cpu_error = self.cpu_threshold - profile.cpu_percent
    mem_error = self.mem_threshold - profile.memory_percent
    
    # Update PID terms
    self.cpu_integral += cpu_error * self.dt
    self.cpu_derivative = (cpu_error - self.cpu_error_prev) / self.dt
    
    # Compute PID output
    cpu_output = (
        self.Kp * cpu_error +
        self.Ki * self.cpu_integral +
        self.Kd * self.cpu_derivative
    )
    
    # Similar for memory...
    
    # Map to action
    if cpu_output < -10 or mem_output < -10:
        return Action.THROTTLE_HEAVY
    elif cpu_output < -5 or mem_output < -5:
        return Action.THROTTLE_LIGHT
    elif cpu_output > 10 and mem_output > 10:
        return Action.INCREASE_RATE
    else:
        return Action.MAINTAIN
```

**PID Tuning**:
- Kp = 1.0 (proportional gain)
- Ki = 0.1 (integral gain)
- Kd = 0.05 (derivative gain)

### 2.4 Actuator Algorithm

**Purpose**: Execute control actions

**Location**: `scarcity/governor/actuators.py`

**Algorithm**:
```python
async def execute_action(self, action: Action):
    """
    Execute control action on subsystems.
    
    Actions:
    - THROTTLE_HEAVY: Reduce rate by 50%
    - THROTTLE_LIGHT: Reduce rate by 25%
    - MAINTAIN: No change
    - INCREASE_RATE: Increase rate by 25%
    """
    if action == Action.THROTTLE_HEAVY:
        # Reduce MPIE processing rate
        await self.mpie.set_rate(self.mpie.rate * 0.5)
        
        # Reduce data ingestion rate
        await self.data_source.set_rate(self.data_source.rate * 0.5)
        
    elif action == Action.THROTTLE_LIGHT:
        await self.mpie.set_rate(self.mpie.rate * 0.75)
        await self.data_source.set_rate(self.data_source.rate * 0.75)
        
    elif action == Action.INCREASE_RATE:
        await self.mpie.set_rate(self.mpie.rate * 1.25)
        await self.data_source.set_rate(self.data_source.rate * 1.25)
```

### 2.5 Forecasting Algorithm

**Purpose**: Predict future resource usage

**Location**: `scarcity/governor/profiler.py`

**Algorithm**:
```python
def forecast(self, history: List[float], horizon: int = 10) -> List[float]:
    """
    Forecast future resource usage using exponential smoothing.
    
    Steps:
    1. Fit exponential smoothing model
    2. Generate forecasts
    3. Return predictions
    """
    # Exponential smoothing
    alpha = 0.3  # Smoothing parameter
    
    # Initialize
    forecast = [history[0]]
    
    # Smooth historical data
    for i in range(1, len(history)):
        smoothed = alpha * history[i] + (1 - alpha) * forecast[-1]
        forecast.append(smoothed)
    
    # Generate future forecasts
    last_value = forecast[-1]
    for _ in range(horizon):
        forecast.append(last_value)
    
    return forecast[-horizon:]
```

**Alternative**: ARIMA model for more sophisticated forecasting

## 3. Federation Coordinator

### 3.1 Overview

Coordinates model sharing across multiple domains.

**Location**: `backend/app/core/federation_coordinator.py`

### 3.2 FedAvg Algorithm

**Purpose**: Aggregate models using federated averaging

**Algorithm**:
```python
def federated_average(self, updates: List[ModelUpdate]) -> np.ndarray:
    """
    Aggregate models using FedAvg.
    
    Formula:
        w_global = Σₖ (nₖ/n) · wₖ
    
    Steps:
    1. Compute total samples
    2. Weight each model by sample count
    3. Sum weighted models
    """
    total_samples = sum(u.num_samples for u in updates)
    
    if total_samples == 0:
        # Unweighted average
        return np.mean([u.weights for u in updates], axis=0)
    
    # Weighted average
    aggregated = np.zeros_like(updates[0].weights)
    for update in updates:
        weight = update.num_samples / total_samples
        aggregated += weight * update.weights
    
    return aggregated
```

**Complexity**: O(k·m) where k=domains, m=model size

### 3.3 Adaptive Aggregation Algorithm

**Purpose**: Weight models by performance

**Algorithm**:
```python
def adaptive_average(self, updates: List[ModelUpdate]) -> np.ndarray:
    """
    Aggregate models weighted by inverse loss.
    
    Better models get higher weight.
    
    Formula:
        αₖ = (1/lossₖ) / Σⱼ(1/lossⱼ)
        w_global = Σₖ αₖ · wₖ
    """
    # Compute inverse losses
    inv_losses = [1.0 / (u.loss + 1e-6) for u in updates]
    total_inv_loss = sum(inv_losses)
    
    # Compute weights
    weights = [inv_loss / total_inv_loss for inv_loss in inv_losses]
    
    # Aggregate
    aggregated = np.zeros_like(updates[0].weights)
    for update, weight in zip(updates, weights):
        aggregated += weight * update.weights
    
    return aggregated
```

### 3.4 Differential Privacy Algorithm

**Purpose**: Add noise for privacy

**Algorithm**:
```python
def add_privacy_noise(
    self,
    weights: np.ndarray,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0
) -> np.ndarray:
    """
    Add Gaussian noise for (ε, δ)-differential privacy.
    
    Formula:
        σ = √(2·ln(1.25/δ)) · Δf / ε
        w̃ = w + N(0, σ²I)
    """
    # Compute noise scale
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    # Generate noise
    noise = np.random.normal(0, sigma, weights.shape)
    
    # Add noise
    noised_weights = weights + noise
    
    return noised_weights
```

**Privacy Guarantee**: (ε, δ)-differential privacy
- ε = 1.0: Moderate privacy
- δ = 1e-5: Failure probability

## 4. Multi-Domain Data Generation

### 4.1 Overview

Generates synthetic data with domain-specific distributions.

**Location**: `backend/app/core/multi_domain_generator.py`

### 4.2 Distribution Generators

**Normal Distribution**:
```python
def generate_normal(
    window_size: int,
    features: int,
    mean: float = 0.0,
    std: float = 1.0
) -> np.ndarray:
    """Generate normally distributed data."""
    return np.random.normal(mean, std, (window_size, features))
```

**Skewed Distribution**:
```python
def generate_skewed(
    window_size: int,
    features: int,
    shape: float = 2.0,
    scale: float = 1.0
) -> np.ndarray:
    """Generate log-normal (skewed) data."""
    return np.random.lognormal(shape, scale, (window_size, features))
```

**Bimodal Distribution**:
```python
def generate_bimodal(
    window_size: int,
    features: int,
    mean1: float = -1.0,
    std1: float = 0.5,
    mean2: float = 1.0,
    std2: float = 0.5,
    weight: float = 0.5
) -> np.ndarray:
    """Generate mixture of two Gaussians."""
    n1 = int(window_size * weight)
    n2 = window_size - n1
    
    data1 = np.random.normal(mean1, std1, (n1, features))
    data2 = np.random.normal(mean2, std2, (n2, features))
    
    data = np.vstack([data1, data2])
    np.random.shuffle(data)
    return data
```

### 4.3 Staggered Generation Algorithm

**Purpose**: Generate data for multiple domains with time offsets

**Algorithm**:
```python
async def staggered_generation(
    domains: List[Domain],
    base_interval: float = 5.0
):
    """
    Generate data for domains with staggered timing.
    
    Steps:
    1. Calculate offset for each domain
    2. Start generation tasks with offsets
    3. Each task generates data periodically
    """
    num_domains = len(domains)
    
    for i, domain in enumerate(domains):
        # Calculate offset
        offset = (i * base_interval) / num_domains
        
        # Start task with offset
        asyncio.create_task(
            generate_with_offset(domain, base_interval, offset)
        )

async def generate_with_offset(
    domain: Domain,
    interval: float,
    offset: float
):
    """Generate data for domain with initial offset."""
    # Wait for offset
    await asyncio.sleep(offset)
    
    # Generate periodically
    while True:
        data = generate_for_domain(domain)
        await publish_data(domain, data)
        await asyncio.sleep(interval)
```

**Benefit**: Spreads load evenly over time

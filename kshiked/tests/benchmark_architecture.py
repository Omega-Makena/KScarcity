"""
Benchmark: OOP vs Vectorized Architecture.
Determines if we need GPU or just Vectorized CPU for 1M hypotheses.

Usage: python -m kshiked.tests.benchmark_architecture
(requires: pip install -e .)
"""
import time
import numpy as np

# Mock RLS State
N_HYPOTHESES = 1000
N_FEATURES = 2

def benchmark_oop():
    """Current Architecture: List of Objects."""
    class MockRLS:
        def __init__(self):
            self.w = np.zeros(N_FEATURES)
            self.P = np.eye(N_FEATURES)
        def update(self, x, y):
            # Simplified update
            err = y - np.dot(self.w, x)
            self.w += 0.01 * err * x

    population = [MockRLS() for _ in range(N_HYPOTHESES)]
    x = np.random.randn(2)
    y = 1.0
    
    start = time.time()
    for _ in range(10): # Process 10 rows
        for model in population:
            model.update(x, y)
            
    dt = time.time() - start
    print(f"[OOP] 10 Rows x {N_HYPOTHESES} Models: {dt:.4f}s")
    return dt

def benchmark_vectorized():
    """Tensor Architecture: Real VectorizedRLS."""
    from scarcity.engine.vectorized_core import VectorizedRLS
    
    rls = VectorizedRLS(N_HYPOTHESES, N_FEATURES)
    
    # Input data per model
    # X: (M, F)
    # let's assume all models see same features for this micro-benchmark
    # In reality, they see different features.
    # To fully test broadcasting, let's create random X per model
    X = np.random.randn(N_HYPOTHESES, N_FEATURES).astype(np.float32)
    Y = np.random.randn(N_HYPOTHESES).astype(np.float32)
    
    start = time.time()
    for _ in range(10): 
        # Full Batch Update
        rls.update(X, Y)
        
    dt = time.time() - start
    print(f"[Vectorized Core] 10 Rows x {N_HYPOTHESES} Models: {dt:.4f}s")
    return dt

if __name__ == "__main__":
    t_oop = benchmark_oop()
    t_vec = benchmark_vectorized()
    
    speedup = t_oop / t_vec
    print(f"\nSpeedup Factor: {speedup:.1f}x")
    
    if speedup > 50:
        print("RECOMMENDATION: Refactor to Vectorized CPU (NumPy). GPU likely overkill for N=1 streaming.")
    else:
        print("RECOMMENDATION: OOP might be acceptable.")

"""
robust statistics module.

provides industrial-grade online estimators robust to outliers (non-gaussian noise).
includes:
- onlinewinsorizer: clips inputs to [p05, p95] to prevent spikes.
- onlinemad: median absolute deviation (robust variance).
- huber loss: gradient clipping for rls.
"""

import numpy as np
import math
from typing import List, Optional

# style: lowercase constant for huber
HUBER_DELTA = 1.345

class OnlineWinsorizer:
    """
    tracks percentiles (p01, p99) using a sliding window to clip extreme outliers.
    """
    def __init__(self, window_size: int = 1000, lower_p: float = 1.0, upper_p: float = 99.0):
        self.window: List[float] = []
        self.window_size = window_size
        self.lower_p = lower_p
        self.upper_p = upper_p
        self.lower_bound = -float('inf')
        self.upper_bound = float('inf')
        self._step = 0

    def update(self, x: float) -> float:
        if not math.isfinite(x): return x
        
        # update buffer
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0) # simple fifo
            
        # re-calculate bounds periodically (every 10 steps to save cpu)
        self._step += 1
        if self._step % 10 == 0 and len(self.window) > 100:
            # simple sort-based quantile (robust and fast enough for n=1000)
            sorted_w = sorted(self.window)
            n = len(sorted_w)
            li = int(n * (self.lower_p / 100.0))
            ui = int(n * (self.upper_p / 100.0))
            self.lower_bound = sorted_w[min(li, n-1)]
            self.upper_bound = sorted_w[min(ui, n-1)]
            
        # clip
        if len(self.window) < 20: 
            return x # too early to clip
            
        return max(self.lower_bound, min(x, self.upper_bound))

class OnlineMAD:
    """
    tracks median and mad (median absolute deviation) online.
    more robust than mean/std (welford).
    """
    def __init__(self, window_size: int = 1000):
        self.window: List[float] = []
        self.window_size = window_size
        self.median = 0.0
        self.mad = 0.0

    def update(self, x: float) -> None:
        if not math.isfinite(x): return
        
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0)
            
        # exact median on window
        if len(self.window) > 0:
            sorted_w = sorted(self.window)
            n = len(sorted_w)
            mid = n // 2
            self.median = sorted_w[mid]
            
            # mad
            abs_devs = sorted([abs(v - self.median) for v in sorted_w])
            self.mad = abs_devs[mid] * 1.4826 # scale factor for normal consistency

    @property
    def std_proxy(self) -> float:
        return self.mad if self.mad > 1e-9 else 1e-9

def huber_gradient(error: float, delta: float = HUBER_DELTA) -> float:
    """
    calculates the effective error term for gradient descent.
    if error is small (quadratic region), returns error.
    if error is large (linear region), clips it to delta * sign(error).
    """
    if abs(error) <= delta:
        return error
    else:
        # linear gradient: delta * sign(error)
        return delta * (1.0 if error > 0 else -1.0)

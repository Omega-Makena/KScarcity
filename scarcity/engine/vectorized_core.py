"""
vectorized core engine (numpy).

replaces object-oriented 'hypothesis per object' architecture with 
batch matrix operations (tensor state).

shape conventions:
- M: number of hypotheses (e.g. 10,000)
- F: number of features (e.g. 2 for linear: [1, x])
- weights (W): (M, F)
- covariance (P): (M, F, F)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class VectorizedRLS:
    """
    massively parallel rls engine.
    updates M models in O(1) python overhead (O(M) C overhead).
    """
    def __init__(self, n_models: int, n_features: int = 2, lambda_forget: float = 0.99):
        self.M = n_models
        self.F = n_features
        self.lam = lambda_forget
        
        # state tensors
        self.W = np.zeros((self.M, self.F), dtype=np.float32)
        
        # covariance matrices p initialized to 10*I
        # shape: (M, F, F)
        self.P = np.zeros((self.M, self.F, self.F), dtype=np.float32)
        idx = np.arange(self.F)
        self.P[:, idx, idx] = 10.0 # batched identity scaling

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        batch predict.
        X: (M, F) - input feature vector for each model.
        returns: (M,) - predictions.
        """
        # dot product per row: (M, F) * (M, F) -> sum over F
        # einsum 'ij,ij->i'
        return np.einsum('ij,ij->i', self.W, X)

    def update(self, X: np.ndarray, Y: np.ndarray, active_mask: Optional[np.ndarray] = None) -> None:
        """
        batch update.
        X: (M, F)
        Y: (M,)
        active_mask: (M,) boolean - if provided, only update these models.
        """
        # if mask provided, we could do boolean indexing, but that might trigger copies.
        # for full speed on mostly-active, we might just update all and mask gradients?
        # let's use boolean indexing for safety if M is huge.
        
        if active_mask is not None:
             # sub-select indices
             idxs = np.where(active_mask)[0]
             if len(idxs) == 0: return
             
             X_sub = X[idxs] # (m, F)
             Y_sub = Y[idxs] # (m,)
             P_sub = self.P[idxs] # (m, F, F)
             W_sub = self.W[idxs] # (m, F)
             
             # call internal update on subsets
             W_new, P_new = self._batch_rls_step(X_sub, Y_sub, P_sub, W_sub)
             
             # scatter back
             self.W[idxs] = W_new
             self.P[idxs] = P_new
        else:
             self.W, self.P = self._batch_rls_step(X, Y, self.P, self.W)

    def update_subset(self, indices: np.ndarray, X_sub: np.ndarray, Y_sub: np.ndarray) -> None:
        """
        efficiently update only the specified models.
        indices: (N_active,)
        X_sub: (N_active, F)
        Y_sub: (N_active,)
        """
        if len(indices) == 0: return
        
        P_sub = self.P[indices]
        W_sub = self.W[indices]
        
        W_new, P_new = self._batch_rls_step(X_sub, Y_sub, P_sub, W_sub)
        
        self.W[indices] = W_new
        self.P[indices] = P_new

    def _batch_rls_step(self, X, Y, P, W):
        """
        the math core.
        P: (M, F, F)
        X: (M, F)
        """
        # 1. Px = P @ x
        # (M, F, F) @ (M, F, 1) -> (M, F, 1)
        # squeeze last dim
        X_expanded = X[:, :, np.newaxis] # (M, F, 1)
        Px = np.matmul(P, X_expanded).squeeze(2) # (M, F)
        
        # 2. Denom = lambda + x.T @ Px
        # x dot Px: (M, F) * (M, F) -> sum -> (M,)
        xPx = np.einsum('ij,ij->i', X, Px)
        denom = self.lam + xPx
        
        # stability check
        valid = np.abs(denom) > 1e-9
        # avoid div by zero by clamping denom (soft landing)
        denom[~valid] = 1.0 
        
        # 3. Gain g = Px / denom
        # (M, F) / (M, 1)
        denom_view = denom[:, np.newaxis]
        g = Px / denom_view
        
        # 4. Error = y - w @ x
        preds = np.einsum('ij,ij->i', W, X)
        err = Y - preds
        
        # zero out invalid updates
        err[~valid] = 0.0
        
        # 5. Update W: w = w + g * err
        # (M, F) + (M, F) * (M, 1)
        W_new = W + g * err[:, np.newaxis]
        
        
        # 6. Update P: P = (1/lam) * (P - g @ Px.T)
        # g @ Px.T is outer product of g and Px: (M, F, 1) @ (M, 1, F) -> (M, F, F)
        
        g_exp = g[:, :, np.newaxis] # (M, F, 1)
        Px_exp = Px[:, np.newaxis, :] # (M, 1, F)
        numerator = np.matmul(g_exp, Px_exp) # (M, F, F)
        
        P_new = (P - numerator) / self.lam
        
        return W_new, P_new

class VectorizedHypothesisPool:
    """
    manages the mapping from (variable_a, variable_b) -> tensor index.
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        # we pre-allocate the rls engine for max capacity
        self.engine = VectorizedRLS(capacity, n_features=2)
        
        # mapping
        self.key_to_idx: Dict[Tuple[str, str], int] = {}
        self.idx_to_key: Dict[int, Tuple[str, str]] = {}
        self.free_indices = list(range(capacity - 1, -1, -1)) # stack of free slots
        
    def get_or_create(self, var_a: str, var_b: str) -> int:
        key = (var_a, var_b)
        if key in self.key_to_idx:
            return self.key_to_idx[key]
        
        if not self.free_indices:
            raise MemoryError("vectorized_pool full")
            
        idx = self.free_indices.pop()
        self.key_to_idx[key] = idx
        self.idx_to_key[idx] = key
        # reset state for this slot
        self._reset_slot(idx)
        return idx
        
    def _reset_slot(self, idx: int):
        self.engine.W[idx] = 0
        self.engine.P[idx] = 0
        # reset diagonal to 10
        for i in range(2):
            self.engine.P[idx, i, i] = 10.0

"""
Evaluator — Predictive, causal, uncertainty, and stability scoring.

Scores candidate paths online with predictive gain, uncertainty bounds, stability
metrics, and reward shaping per Controller ⇆ Evaluator contract.
"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable

from scarcity.engine.types import Candidate, EvalResult, Reward
from scarcity.engine.utils import (
    clip, safe_div, robust_zscore, robust_quantiles, 
    compute_median_mad, softplus
)
from scarcity.engine.operators.evaluation_ops import r2_gain

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Online path evaluator.
    
    Scores paths for acceptance based on predictive gain, uncertainty,
    stability, and produces shaped rewards for bandit learning.
    """
    
    def __init__(self, drg: Optional[Dict[str, Any]] = None, operators: Optional[Dict[str, Any]] = None, rng: Optional[np.random.Generator] = None):
        """
        Initializes the path evaluator with resource constraints and operators.

        Sets up the scoring parameters (gain thresholds, stability requirements),
        reward shaping weights, and internal tracking history.

        Args:
            drg: A dictionary defining the resource governor profile, which controls
                thresholds like `gain_min`, `stability_min`, and resampling counts.
            operators: A registry of available data operators (not currently used
                directly in this class but preserved for interface compatibility).
            rng: A numpy random number generator for bootstrap resampling. If None,
                a default generator is instantiated.
        """
        self.drg = drg or {}
        self.operators = operators or {}
        self.rng = rng if rng else np.random.default_rng()
        
        # Acceptance thresholds
        self.gain_min = self.drg.get('gain_min', 0.01)
        self.stability_min = self.drg.get('stability_min', 0.7)
        self.ci_width_lambda = self.drg.get('lambda', 0.5)
        self.resamples = self.drg.get('resamples', 8)
        
        # Reward shaping weights
        self.w_g = self.drg.get('w_g', 0.55)  # gain weight
        self.w_s = self.drg.get('w_s', 0.30)  # stability weight
        self.w_c = self.drg.get('w_c', 0.15)  # CI weight
        self.alpha_L = self.drg.get('alpha_L', 0.20)  # latency penalty
        self.beta_D = self.drg.get('beta_D', 0.10)  # diversity bonus
        self.L_target = self.drg.get('L_target', 150.0)  # latency target ms
        
        # Baselines (EMA of simple predictors)
        self.baselines = {}
        self.holdout_size = 16  # Will be adjusted per window
        self.last_window_stability = None
        
        # Tracking
        self.total_evaluated = 0
        self.acceptance_count = 0
        self.gain_history = []
        self.ci_width_history = []
        self.stability_history = []
        self._relax_steps = 0
        
        logger.info(f"Evaluator initialized: gain_min={self.gain_min}, stability_min={self.stability_min}, resamples={self.resamples}")
    
    def score(self, window_tensor: np.ndarray, candidates: List[Candidate]) -> List[EvalResult]:
        """
        Evaluates a batch of candidate paths against the provided data window.

        This method orchestrates the scoring process for each candidate:
        1. Builds a design matrix (features vs target) given the path's lags.
        2. Performs bootstrap resampling to estimate predictive gain (R²) and its
           confidence intervals.
        3. Computes a stability score based on sign agreement across resamples.
        4. Determines acceptance based on the configured thresholds (gain,
           stability, and CI width).
        5. Updates internal baselines and statistics.

        It adapts the holdout size for validation based on the window length to
        ensure robust statistics even with limited data.

        Args:
            window_tensor: A 2D numpy array containing the time-series data.
            candidates: A list of `Candidate` objects to evaluate.

        Returns:
            A list of `EvalResult` objects, one for each valid candidate. Candidates
            that fail technical validation (e.g., insufficient data) are returned
            with error flags in their result.
        """
        if len(candidates) == 0:
            return []
        
        # Adjust hold-out size
        self.holdout_size = max(16, window_tensor.shape[0] // 8)
        
        results = []
        
        for cand in candidates:
            result = self._score_single(window_tensor, cand)
            if result:
                results.append(result)
        
        # Update tracking
        self.total_evaluated += len(results)
        self.acceptance_count += sum(1 for r in results if r.accepted)
        
        # Store gains for robust normalization in make_rewards
        self.gain_history.extend([r.gain for r in results])
        self.ci_width_history.extend([r.ci_hi - r.ci_lo for r in results])
        self.stability_history.extend([r.stability for r in results])
        
        # Truncate history
        if len(self.gain_history) > 1000:
            self.gain_history = self.gain_history[-1000:]
        if len(self.ci_width_history) > 1000:
            self.ci_width_history = self.ci_width_history[-1000:]
        if len(self.stability_history) > 1000:
            self.stability_history = self.stability_history[-1000:]

        self._maybe_relax_thresholds()
    
        return results

    def apply_meta_update(self, g_min: Optional[float] = None, lambda_ci: Optional[float] = None) -> None:
        """
        Updates internal thresholds based on meta-learning policy signals.

        Allows the meta-learner to dynamically adjust the stringency of the
        evaluator. For example, it can lower the minimum gain threshold in
        low-signal environments or tighten the confidence interval requirements
        when false positives are high.

        Args:
            g_min: The new minimum gain threshold (clipped to [0.0, 0.1]).
            lambda_ci: The new confidence interval width multiplier (clipped to [0.1, 1.0]).
        """
        if g_min is not None:
            self.gain_min = float(np.clip(g_min, 0.0, 0.1))
        if lambda_ci is not None:
            self.ci_width_lambda = float(np.clip(lambda_ci, 0.1, 1.0))
    
    def _score_single(self, window_tensor: np.ndarray, candidate: Candidate) -> Optional[EvalResult]:
        """
        Computes the score for a single candidate path.

        Performs the heavy lifting of:
        - Constructing the regression verification problem.
        - Running bootstrap resamples.
        - Calculating robust statistics (median gain, quantiles).
        - Making the final acceptance decision.

        Args:
            window_tensor: The full data window.
            candidate: The specific path to evaluate.

        Returns:
            An `EvalResult` object containing valid scores, or a result with `accepted=False`
            and error details if evaluation could not proceed (e.g., collinearity,
            insufficient data).
        """
        start_time = time.time()
        
        try:
            # Check if candidate variables are valid
            if len(candidate.vars) == 0 or any(v >= window_tensor.shape[1] for v in candidate.vars):
                return EvalResult(
                    path_id=candidate.path_id,
                    gain=0.0,
                    ci_lo=0.0,
                    ci_hi=0.0,
                    stability=0.5,
                    cost_ms=0.0,
                    accepted=False,
                    extras={'error': 'invalid_vars'}
                )
            
            design = self._build_design_matrix(window_tensor, candidate)
            if design is None:
                cost_ms = (time.time() - start_time) * 1000
                return EvalResult(
                    path_id=candidate.path_id,
                    gain=0.0,
                    ci_lo=0.0,
                    ci_hi=0.0,
                    stability=0.5,
                    cost_ms=cost_ms,
                    accepted=False,
                    extras={'error': 'insufficient_support'}
                )
            X, y = design
            if X.shape[0] < max(8, self.holdout_size):
                cost_ms = (time.time() - start_time) * 1000
                return EvalResult(
                    path_id=candidate.path_id,
                    gain=0.0,
                    ci_lo=0.0,
                    ci_hi=0.0,
                    stability=0.5,
                    cost_ms=cost_ms,
                    accepted=False,
                    extras={'error': 'insufficient_rows'}
                )

            holdout_rows = max(4, min(self.holdout_size, max(4, X.shape[0] // 3)))
            if holdout_rows >= X.shape[0]:
                holdout_rows = max(4, X.shape[0] // 2)
            gains: List[float] = []
            for _ in range(self.resamples):
                gain = self._bootstrap_gain(X, y, holdout_rows)
                if gain is not None and np.isfinite(gain):
                    gains.append(float(gain))
            
            if len(gains) == 0:
                cost_ms = (time.time() - start_time) * 1000
                return EvalResult(
                    path_id=candidate.path_id,
                    gain=0.0,
                    ci_lo=0.0,
                    ci_hi=0.0,
                    stability=0.5,
                    cost_ms=cost_ms,
                    accepted=False,
                    extras={'error': 'no_valid_gains'}
                )
            
            # Robust quantiles for CI
            quantiles = robust_quantiles(gains, [0.16, 0.84])
            ci_lo, ci_hi = quantiles[0], quantiles[1]
            ci_width = ci_hi - ci_lo
            gain_median = float(np.median(gains))
            
            # Compute stability
            stability = self._compute_stability(gain_median, gains)
            
            # Acceptance decision
            accepted = (
                gain_median >= self.gain_min and
                stability >= self.stability_min and
                ci_width <= self.ci_width_lambda * max(abs(gain_median), 1e-6)
            )
            
            # Cost
            cost_ms = (time.time() - start_time) * 1000

            target_idx = candidate.vars[-1]
            baseline_key = f"baseline_{target_idx}"
            target_mean = float(np.mean(y))
            prev_baseline = self.baselines.get(baseline_key, target_mean)
            self.baselines[baseline_key] = 0.95 * prev_baseline + 0.05 * target_mean
            
            return EvalResult(
                path_id=candidate.path_id,
                gain=gain_median,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                stability=stability,
                cost_ms=cost_ms,
                accepted=accepted,
                extras={'resamples': len(gains), 'holdout_rows': holdout_rows}
            )
            
        except Exception as e:
            logger.warning(f"Error scoring candidate {candidate.path_id}: {e}")
            cost_ms = (time.time() - start_time) * 1000
            return EvalResult(
                path_id=candidate.path_id,
                gain=0.0,
                ci_lo=0.0,
                ci_hi=0.0,
                stability=0.5,
                cost_ms=cost_ms,
                accepted=False,
                extras={'error': str(e)}
            )
    
    def _build_design_matrix(
        self,
        window_tensor: np.ndarray,
        candidate: Candidate
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Constructs the feature matrix (X) and target vector (y) for evaluation.

        Processes the raw time-series window to apply the specified lags for each
        variable in the candidate path. It aligns the series by trimming invalid
        start/end periods caused by lags.

        Args:
            window_tensor: The raw data window.
            candidate: The path definition containing variable indices and lags.

        Returns:
            A tuple (X, y) where X is the design matrix and y is the target vector,
            or None if the construction is impossible (e.g., lag exceeds window size).
        """
        if window_tensor.ndim != 2:
            return None
        if len(candidate.vars) < 2:
            return None

        max_lag = int(max(candidate.lags)) if candidate.lags else 0
        rows = window_tensor.shape[0] - max_lag
        if rows <= 4:
            return None

        series: List[np.ndarray] = []
        for var_idx, lag in zip(candidate.vars, candidate.lags):
            if var_idx >= window_tensor.shape[1]:
                return None
            start = max_lag - int(lag)
            end = start + rows
            if start < 0 or end > window_tensor.shape[0]:
                return None
            series.append(window_tensor[start:end, var_idx].astype(np.float32))

        if len(series) < 2:
            return None

        target = series[-1]
        features = np.column_stack(series[:-1]).astype(np.float32)
        if np.any(np.isnan(features)) or np.any(np.isnan(target)):
            return None

        return features, target

    def _bootstrap_gain(self, X: np.ndarray, y: np.ndarray, holdout_rows: int) -> Optional[float]:
        """
        Estimates predictive gain using a single bootstrap resample.

        Randomly splits the data into training and holdout sets. Fits a linear
        model on the training set and evaluates R² (improvement over mean baseline)
        on the holdout set.

        Args:
            X: Design matrix.
            y: Target vector.
            holdout_rows: Number of rows to hold out for validation.

        Returns:
            The calculated R² gain, or None if the linear solve fails (e.g., singular matrix).
        """
        if X.shape[0] <= holdout_rows:
            return None
        idx = self.rng.permutation(X.shape[0])
        hold = idx[:holdout_rows]
        train = idx[holdout_rows:]
        if len(train) <= X.shape[1]:
            return None
        X_train, y_train = X[train], y[train]
        X_hold, y_hold = X[hold], y[hold]
        try:
            weights, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        except np.linalg.LinAlgError:
            return None
        preds = X_hold @ weights
        baseline = np.full_like(preds, np.mean(y_train))
        return float(r2_gain(y_hold, preds, baseline))

    def _maybe_relax_thresholds(self) -> None:
        """
        Dynamically lowers acceptance standards if the system is starving.

        If the acceptance rate is critically low over a number of evaluations,
        this method gradually reduces `gain_min` and `stability_min` and widens
        the allowed confidence intervals. This ensures the system produces *some*
        hypotheses rather than stalling completely, though they may be lower quality.
        """
        if self.total_evaluated < 64:
            return
        accept_rate = self.acceptance_count / max(1, self.total_evaluated)
        # target floor rises slightly with each relaxation to avoid runaway lowering
        required_rate = 0.01 + 0.01 * self._relax_steps
        if accept_rate >= required_rate or self._relax_steps >= 4:
            return

        old_gain = self.gain_min
        old_stability = self.stability_min
        old_lambda = self.ci_width_lambda

        self.gain_min = max(0.001, self.gain_min * 0.7)
        self.stability_min = max(0.2, self.stability_min * 0.85)
        self.ci_width_lambda = min(6.0, self.ci_width_lambda * 1.3)
        self._relax_steps += 1

        logger.info(
            "Evaluator auto-relaxed thresholds (step %d): gain_min %.4f->%.4f, stability_min %.3f->%.3f, lambda %.3f->%.3f (accept_rate=%.4f)",
            self._relax_steps,
            old_gain,
            self.gain_min,
            old_stability,
            self.stability_min,
            old_lambda,
            self.ci_width_lambda,
            accept_rate,
        )
    
    def _compute_stability(self, current_gain: float, gains: List[float]) -> float:
        """
        Calculates a stability score for the candidate.

        Stability is measured by the consistency of the sign of the gain across
        bootstrap resamples, and agreement with the previous window's stability
        (if available). 1.0 implies perfect stability, 0.0 implies chaos/noise.

        Args:
            current_gain: The median gain of the current evaluation.
            gains: The list of gains from all bootstrap resamples.

        Returns:
            A float stability score in [0.0, 1.0].
        """
        if self.last_window_stability is None:
            self.last_window_stability = gains
            return 0.5
        
        # Sign agreement
        if len(gains) == 0 or len(self.last_window_stability) == 0:
            return 0.5
        
        signs_current = [1 if g >= 0 else -1 for g in gains]
        signs_last = [1 if g >= 0 else -1 for g in self.last_window_stability]
        
        agreement = sum(1 for i in range(min(len(signs_current), len(signs_last))) 
                        if signs_current[i] == signs_last[i]) / max(len(signs_current), len(signs_last))
        
        # Spearman-like (simplified)
        # Full implementation would use scipy.stats.spearmanr
        stability = 0.5 * agreement + 0.5  # Normalize to [0,1]
        
        self.last_window_stability = gains
        return float(clip(stability, 0.0, 1.0))
    
    def make_rewards(self, results: List[EvalResult], D_lookup: Callable[[str], float], 
                     candidates: Optional[List[Candidate]] = None) -> List[Reward]:
        """
        Computes the final rewards for the Multi-Armed Bandit controller.

        Applies "Reward Shaping" to the raw evaluation results. This involves:
        1. Normalizing gain and stability scores.
        2. Applying penalties for high latency.
        3. Adding bonuses for exploring diverse/uncovered causal paths.
        4. Clipping the final reward to the [-1, 1] range expected by the bandit.

        Args:
            results: The list of evaluation results from the scoring phase.
            D_lookup: A callable to retrieve diversity scores for a path ID.
            candidates: Optional list of original candidates (used to extract arm keys).

        Returns:
            A list of `Reward` objects to be fed back to the Controller.
        """
        if len(results) == 0:
            return []
        
        # Build lookup from path_id to candidate
        cand_lookup = {}
        if candidates:
            for cand in candidates:
                cand_lookup[cand.path_id] = cand
        
        # Compute window-local normalization
        gains = [r.gain for r in results if 'error' not in r.extras]
        ci_widths = [r.ci_hi - r.ci_lo for r in results if 'error' not in r.extras]
        
        if len(gains) == 0:
            median_gain, mad_gain = 0.0, 1.0
            kappa = 1.0
        else:
            median_gain, mad_gain = compute_median_mad(gains)
            kappa = 0.5 * abs(median_gain) + 1e-6
        
        rewards = []
        
        for result in results:
            # Skip error cases
            if 'error' in result.extras:
                continue
            
            # Normalize gain (ĝ)
            g_hat = robust_zscore(result.gain, median_gain, mad_gain)
            g_hat = clip(g_hat, -3.0, 3.0)
            
            # Normalize stability (ŝ)
            s_hat = 2.0 * result.stability - 1.0  # [0,1] -> [-1,1]
            
            # Normalize CI (ĉ)
            ci_width = result.ci_hi - result.ci_lo
            c_hat = clip(1.0 - ci_width / kappa, 0.0, 1.0)
            
            # Core reward
            r_core = self.w_g * np.tanh(g_hat) + self.w_s * s_hat + self.w_c * (c_hat - 0.5)
            
            # Latency penalty
            latency_penalty = self.alpha_L * softplus((result.cost_ms - self.L_target) / self.L_target)
            
            # Diversity bonus
            D_path = D_lookup(result.path_id)
            diversity_bonus = self.beta_D * D_path
            
            # Final reward
            r_total = clip(r_core - latency_penalty + diversity_bonus, -1.0, 1.0)
            
            # Clamp non-accepted paths with high latency
            if not result.accepted and latency_penalty > 0.5:
                r_total = max(0.0, r_total)
            
            # Partial credit for close-to-acceptance
            if not result.accepted and result.gain >= self.gain_min:
                r_total = max(r_total, 0.05)
            
            # Extract arm_key from candidate
            arm_key = (0, 0)  # Default
            if result.path_id in cand_lookup:
                cand = cand_lookup[result.path_id]
                # Use (root, depth) as arm_key
                arm_key = (cand.root, cand.depth)
            
            reward = Reward(
                path_id=result.path_id,
                arm_key=arm_key,
                value=r_total,
                latency_penalty=latency_penalty,
                diversity_bonus=diversity_bonus,
                accepted=result.accepted
            )
            rewards.append(reward)
        
        return rewards
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Returns aggregate statistics of the evaluator.

        Returns:
            A dictionary containing metrics like acceptance rate, current thresholds
            (p50/p90 gains), average CI widths, and total candidates evaluated.
        """
        accept_rate = safe_div(self.acceptance_count, self.total_evaluated, 0.0)
        
        gains = self.gain_history[-100:] if len(self.gain_history) > 0 else []
        ci_widths = self.ci_width_history[-100:] if len(self.ci_width_history) > 0 else []
        stability_vals = self.stability_history[-100:] if len(self.stability_history) > 0 else []
        
        gain_p50 = np.median(gains) if len(gains) > 0 else 0.0
        gain_p90 = np.percentile(gains, 90) if len(gains) > 0 else 0.0
        ci_width_avg = np.mean(ci_widths) if len(ci_widths) > 0 else 0.0
        stability_avg = np.mean(stability_vals) if len(stability_vals) > 0 else 0.0
        
        return {
            'accept_rate': accept_rate,
            'gain_p50': gain_p50,
            'gain_p90': gain_p90,
            'ci_width_avg': ci_width_avg,
            'stability_avg': stability_avg,
            'total_evaluated': self.total_evaluated,
            'acceptance_count': self.acceptance_count
        }

"""
Bayesian Parameter Estimation for SFC Models.

Extends scarcity's KalmanFilter1D (engine/algorithms_online.py) and
meta-learning prior concepts (meta/optimizer.py) to provide:

1. Metropolis-Hastings MCMC for joint posterior estimation of SFCConfig params
2. Bayesian credible intervals on simulation trajectories
3. Marginal likelihood / model comparison (Bayesian Information Criterion)
4. Online Bayesian updating via extended Kalman filter on parameters

All priors are derived from kenya_calibration.py's CalibratedParam ranges
and the meta-learning global_prior concept.

Dependencies: numpy only (no PyMC/Stan — pure Python for portability).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from scarcity.engine.algorithms_online import KalmanFilter1D, KalmanConfig
from scarcity.simulation.sfc import SFCConfig, SFCEconomy

logger = logging.getLogger("scarcity.simulation.bayesian")


# =========================================================================
# Prior Specification
# =========================================================================

@dataclass
class ParameterPrior:
    """
    Prior distribution for a single SFCConfig parameter.

    Supports Normal, Uniform, LogNormal, and Beta priors.
    Prior bounds from CalibratedParam ranges or economic theory.
    """
    name: str
    distribution: str = "normal"  # "normal", "uniform", "lognormal", "beta"

    # Normal / LogNormal
    mean: float = 0.0
    std: float = 1.0

    # Uniform / bounds
    lower: float = -np.inf
    upper: float = np.inf

    # Beta (for parameters on [0,1])
    alpha: float = 2.0
    beta_param: float = 2.0

    def log_prior(self, value: float) -> float:
        """Compute log-prior density at value."""
        # Hard bounds check
        if value < self.lower or value > self.upper:
            return -np.inf

        if self.distribution == "normal":
            z = (value - self.mean) / max(self.std, 1e-10)
            return -0.5 * z ** 2 - np.log(self.std) - 0.5 * np.log(2 * np.pi)

        elif self.distribution == "uniform":
            width = self.upper - self.lower
            if width <= 0:
                return -np.inf
            return -np.log(width)

        elif self.distribution == "lognormal":
            if value <= 0:
                return -np.inf
            log_val = np.log(value)
            z = (log_val - self.mean) / max(self.std, 1e-10)
            return -0.5 * z ** 2 - np.log(self.std) - log_val - 0.5 * np.log(2 * np.pi)

        elif self.distribution == "beta":
            if value <= 0 or value >= 1:
                return -np.inf
            from math import lgamma
            a, b = self.alpha, self.beta_param
            log_B = lgamma(a) + lgamma(b) - lgamma(a + b)
            return (a - 1) * np.log(value) + (b - 1) * np.log(1 - value) - log_B

        return 0.0  # Improper uniform

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the prior."""
        if self.distribution == "normal":
            val = rng.normal(self.mean, self.std)
        elif self.distribution == "uniform":
            val = rng.uniform(self.lower, self.upper)
        elif self.distribution == "lognormal":
            val = rng.lognormal(self.mean, self.std)
        elif self.distribution == "beta":
            val = rng.beta(self.alpha, self.beta_param)
        else:
            val = rng.normal(self.mean, self.std)
        return float(np.clip(val, self.lower, self.upper))


# =========================================================================
# Default Kenya-Calibrated Priors
# =========================================================================

def default_sfc_priors() -> Dict[str, ParameterPrior]:
    """
    Default priors for key SFCConfig parameters.

    Informed by Kenya calibration ranges (World Bank data) and
    standard macroeconomic theory.
    """
    return {
        "consumption_propensity": ParameterPrior(
            name="consumption_propensity",
            distribution="beta", alpha=8.0, beta_param=2.0,
            lower=0.5, upper=0.99,
        ),
        "investment_sensitivity": ParameterPrior(
            name="investment_sensitivity",
            distribution="normal", mean=0.5, std=0.2,
            lower=0.0, upper=2.0,
        ),
        "tax_rate": ParameterPrior(
            name="tax_rate",
            distribution="normal", mean=0.16, std=0.05,
            lower=0.05, upper=0.40,
        ),
        "target_inflation": ParameterPrior(
            name="target_inflation",
            distribution="normal", mean=0.05, std=0.02,
            lower=0.0, upper=0.15,
        ),
        "taylor_rule_phi": ParameterPrior(
            name="taylor_rule_phi",
            distribution="normal", mean=1.5, std=0.3,
            lower=1.0, upper=3.0,
        ),
        "taylor_rule_psi": ParameterPrior(
            name="taylor_rule_psi",
            distribution="normal", mean=0.5, std=0.2,
            lower=0.0, upper=1.5,
        ),
        "phillips_coefficient": ParameterPrior(
            name="phillips_coefficient",
            distribution="normal", mean=0.15, std=0.08,
            lower=0.01, upper=0.5,
        ),
        "inflation_anchor_weight": ParameterPrior(
            name="inflation_anchor_weight",
            distribution="beta", alpha=5.0, beta_param=3.0,
            lower=0.0, upper=1.0,
        ),
        "okun_coefficient": ParameterPrior(
            name="okun_coefficient",
            distribution="normal", mean=0.02, std=0.01,
            lower=0.005, upper=0.10,
        ),
        "neutral_rate": ParameterPrior(
            name="neutral_rate",
            distribution="normal", mean=0.04, std=0.02,
            lower=0.0, upper=0.10,
        ),
        "spending_ratio": ParameterPrior(
            name="spending_ratio",
            distribution="normal", mean=0.20, std=0.05,
            lower=0.10, upper=0.40,
        ),
        "gdp_adjustment_speed": ParameterPrior(
            name="gdp_adjustment_speed",
            distribution="normal", mean=0.1, std=0.05,
            lower=0.01, upper=0.5,
        ),
    }


# =========================================================================
# Likelihood Function
# =========================================================================

@dataclass
class ObservedData:
    """Historical observations for likelihood computation."""
    gdp_growth: Optional[List[float]] = None
    inflation: Optional[List[float]] = None
    unemployment: Optional[List[float]] = None
    interest_rate: Optional[List[float]] = None

    @property
    def n_obs(self) -> int:
        lengths = []
        for series in [self.gdp_growth, self.inflation, self.unemployment, self.interest_rate]:
            if series is not None:
                lengths.append(len(series))
        return max(lengths) if lengths else 0


def log_likelihood(
    params: Dict[str, float],
    observed: ObservedData,
    base_config: Optional[SFCConfig] = None,
    sigma_gdp: float = 1.0,
    sigma_infl: float = 2.0,
    sigma_unemp: float = 1.0,
) -> float:
    """
    Gaussian log-likelihood of observed data given SFC parameters.

    Runs the SFC model with the proposed parameters and computes the
    log-likelihood of the observed time series under Gaussian errors.

    Args:
        params: Dict of parameter_name → value to override on SFCConfig.
        observed: Historical observations.
        base_config: Base SFCConfig to modify.  Uses defaults if None.
        sigma_*: Observation noise standard deviations.

    Returns:
        Log-likelihood (float).  Returns -inf for invalid configurations.
    """
    config = base_config or SFCConfig()

    # Apply parameter overrides
    for key, val in params.items():
        if hasattr(config, key):
            setattr(config, key, val)

    n_steps = observed.n_obs
    if n_steps < 2:
        return -np.inf

    config.steps = n_steps

    try:
        trajectory = SFCEconomy.run_scenario(config)
    except Exception:
        return -np.inf

    ll = 0.0

    for t in range(min(n_steps, len(trajectory))):
        outcomes = trajectory[t].get("outcomes", {})

        if observed.gdp_growth is not None and t < len(observed.gdp_growth):
            sim_val = outcomes.get("gdp_growth", 0.0)
            obs_val = observed.gdp_growth[t]
            ll += -0.5 * ((sim_val - obs_val) / sigma_gdp) ** 2

        if observed.inflation is not None and t < len(observed.inflation):
            sim_val = outcomes.get("inflation", 0.0)
            obs_val = observed.inflation[t]
            ll += -0.5 * ((sim_val - obs_val) / sigma_infl) ** 2

        if observed.unemployment is not None and t < len(observed.unemployment):
            sim_val = outcomes.get("unemployment", 0.0)
            obs_val = observed.unemployment[t]
            ll += -0.5 * ((sim_val - obs_val) / sigma_unemp) ** 2

    return ll


# =========================================================================
# MCMC Sampler (Metropolis-Hastings)
# =========================================================================

@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling."""
    n_iterations: int = 5000
    burn_in: int = 1000
    proposal_scale: float = 0.01     # Relative proposal width
    adapt_interval: int = 200        # Adapt proposal every N steps
    target_acceptance: float = 0.234  # Optimal for high-dim MH
    seed: int = 42
    thin: int = 1                    # Keep every N-th sample


@dataclass
class MCMCResult:
    """MCMC sampling results with posterior diagnostics."""
    chains: Dict[str, np.ndarray]       # parameter_name → samples array
    log_posteriors: np.ndarray           # log-posterior at each sample
    acceptance_rate: float = 0.0
    n_effective_samples: int = 0

    def posterior_mean(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.chains.items()}

    def posterior_std(self) -> Dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.chains.items()}

    def credible_interval(
        self, param: str, level: float = 0.95
    ) -> Tuple[float, float]:
        """HPD credible interval for a parameter."""
        if param not in self.chains:
            return (0.0, 0.0)
        samples = np.sort(self.chains[param])
        alpha = 1.0 - level
        n = len(samples)
        interval_size = int(np.ceil(level * n))

        best_width = np.inf
        best_lo = 0.0
        best_hi = 0.0
        for i in range(n - interval_size):
            width = samples[i + interval_size] - samples[i]
            if width < best_width:
                best_width = width
                best_lo = samples[i]
                best_hi = samples[i + interval_size]

        return (float(best_lo), float(best_hi))

    def effective_sample_size(self, param: str) -> float:
        """Estimate effective sample size using autocorrelation."""
        if param not in self.chains:
            return 0.0
        x = self.chains[param]
        n = len(x)
        if n < 10:
            return float(n)

        x_centered = x - np.mean(x)
        var = np.var(x)
        if var < 1e-15:
            return float(n)

        # Compute autocorrelation up to lag n//2
        max_lag = min(n // 2, 500)
        autocorr_sum = 0.0
        for lag in range(1, max_lag):
            rho = np.mean(x_centered[:-lag] * x_centered[lag:]) / var
            if rho < 0.05:
                break
            autocorr_sum += rho

        ess = n / (1 + 2 * autocorr_sum)
        return max(1.0, ess)

    def gelman_rubin(self, other_chain: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Gelman-Rubin R-hat diagnostic for convergence (requires 2 chains).

        R-hat ≈ 1.0 indicates convergence.  R-hat > 1.1 suggests non-convergence.
        """
        r_hats = {}
        for param in self.chains:
            if param not in other_chain:
                continue
            chain1 = self.chains[param]
            chain2 = other_chain[param]
            n = min(len(chain1), len(chain2))
            if n < 10:
                continue

            c1, c2 = chain1[:n], chain2[:n]
            m1, m2 = np.mean(c1), np.mean(c2)
            v1, v2 = np.var(c1, ddof=1), np.var(c2, ddof=1)

            grand_mean = (m1 + m2) / 2.0
            B = n * ((m1 - grand_mean) ** 2 + (m2 - grand_mean) ** 2)
            W = (v1 + v2) / 2.0

            if W < 1e-15:
                r_hats[param] = 1.0
            else:
                var_hat = ((n - 1) / n) * W + (1.0 / n) * B
                r_hats[param] = float(np.sqrt(var_hat / W))

        return r_hats


class BayesianEstimator:
    """
    Metropolis-Hastings MCMC for joint posterior estimation of SFC parameters.

    Builds on:
    - KalmanFilter1D (algorithms_online.py) for online parameter tracking
    - OnlineReptileOptimizer (meta/optimizer.py) for prior aggregation concept
    - CalibratedParam (kenya_calibration.py) for parameter ranges

    Usage:
        estimator = BayesianEstimator()
        result = estimator.estimate(observed_data)
        posterior_config = estimator.posterior_config(result)
    """

    def __init__(
        self,
        priors: Optional[Dict[str, ParameterPrior]] = None,
        mcmc_config: Optional[MCMCConfig] = None,
        base_config: Optional[SFCConfig] = None,
    ):
        self.priors = priors or default_sfc_priors()
        self.mcmc_cfg = mcmc_config or MCMCConfig()
        self.base_config = base_config or SFCConfig()

        # Online Kalman trackers for each parameter (real-time updating)
        self._kalman_trackers: Dict[str, KalmanFilter1D] = {
            name: KalmanFilter1D(
                config=KalmanConfig(process_noise=prior.std ** 2 * 0.01,
                                    observation_noise=prior.std ** 2)
            )
            for name, prior in self.priors.items()
        }
        # Initialize Kalman states to prior means
        for name, prior in self.priors.items():
            self._kalman_trackers[name].x = prior.mean
            self._kalman_trackers[name].p = prior.std ** 2

    def estimate(
        self,
        observed: ObservedData,
        param_names: Optional[List[str]] = None,
    ) -> MCMCResult:
        """
        Run MCMC to estimate posterior distributions of SFC parameters.

        Args:
            observed: Historical data to condition on.
            param_names: Subset of parameters to estimate.  Defaults to all priors.

        Returns:
            MCMCResult with posterior chains and diagnostics.
        """
        rng = np.random.default_rng(self.mcmc_cfg.seed)
        names = param_names or list(self.priors.keys())
        n_params = len(names)

        # Initialize at prior means (or Kalman-filtered estimates)
        current = {}
        for name in names:
            if name in self._kalman_trackers:
                current[name] = self._kalman_trackers[name].x
            elif name in self.priors:
                current[name] = self.priors[name].mean
            else:
                current[name] = getattr(self.base_config, name, 0.0)

        # Proposal scale (adaptive)
        scales = {name: self.priors[name].std * self.mcmc_cfg.proposal_scale
                  for name in names if name in self.priors}

        current_ll = log_likelihood(current, observed, self.base_config)
        current_lp = current_ll + sum(
            self.priors[n].log_prior(current[n]) for n in names if n in self.priors
        )

        # Storage
        total = self.mcmc_cfg.n_iterations
        n_stored = (total - self.mcmc_cfg.burn_in) // self.mcmc_cfg.thin
        chains = {name: np.zeros(max(1, n_stored)) for name in names}
        log_posts = np.zeros(max(1, n_stored))
        accepted = 0
        stored = 0

        for i in range(total):
            # Propose
            proposal = {}
            for name in names:
                s = scales.get(name, 0.01)
                proposal[name] = current[name] + rng.normal(0, s)

            # Evaluate
            prop_ll = log_likelihood(proposal, observed, self.base_config)
            prop_lp = prop_ll + sum(
                self.priors[n].log_prior(proposal[n])
                for n in names if n in self.priors
            )

            # Accept/reject
            log_alpha = prop_lp - current_lp
            if np.log(rng.uniform()) < log_alpha:
                current = proposal
                current_lp = prop_lp
                accepted += 1

            # Store post-burnin
            if i >= self.mcmc_cfg.burn_in and (i - self.mcmc_cfg.burn_in) % self.mcmc_cfg.thin == 0:
                if stored < n_stored:
                    for name in names:
                        chains[name][stored] = current[name]
                    log_posts[stored] = current_lp
                    stored += 1

            # Adaptive proposal scaling
            if (i + 1) % self.mcmc_cfg.adapt_interval == 0 and i < self.mcmc_cfg.burn_in:
                rate = accepted / (i + 1)
                for name in names:
                    if rate < self.mcmc_cfg.target_acceptance - 0.05:
                        scales[name] *= 0.8  # shrink
                    elif rate > self.mcmc_cfg.target_acceptance + 0.05:
                        scales[name] *= 1.2  # widen

        # Trim chains if fewer stored than expected
        for name in names:
            chains[name] = chains[name][:stored]
        log_posts = log_posts[:stored]

        result = MCMCResult(
            chains=chains,
            log_posteriors=log_posts,
            acceptance_rate=accepted / total,
        )
        result.n_effective_samples = int(
            np.mean([result.effective_sample_size(n) for n in names])
        )

        logger.info(
            f"MCMC complete: {total} iterations, acceptance={result.acceptance_rate:.1%}, "
            f"ESS≈{result.n_effective_samples}"
        )

        return result

    def posterior_config(self, result: MCMCResult) -> SFCConfig:
        """
        Build an SFCConfig using posterior means from MCMC result.
        """
        config = SFCConfig()
        # Copy base config first
        for attr in vars(self.base_config):
            if not attr.startswith('_'):
                try:
                    setattr(config, attr, getattr(self.base_config, attr))
                except (AttributeError, TypeError):
                    pass

        # Override with posterior means
        means = result.posterior_mean()
        for name, val in means.items():
            if hasattr(config, name):
                setattr(config, name, val)

        return config

    def posterior_predictive(
        self,
        result: MCMCResult,
        n_draws: int = 100,
        steps: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate posterior predictive distribution by running the SFC model
        at sampled parameter values.

        Returns:
            Dict of outcome_name → (n_draws × steps) array.
        """
        rng = np.random.default_rng(self.mcmc_cfg.seed + 999)
        n_stored = len(next(iter(result.chains.values())))
        indices = rng.choice(n_stored, size=min(n_draws, n_stored), replace=False)

        sim_steps = steps or self.base_config.steps

        predictions: Dict[str, List[List[float]]] = {}

        for idx in indices:
            params = {name: float(chain[idx]) for name, chain in result.chains.items()}
            config = SFCConfig()
            for attr in vars(self.base_config):
                if not attr.startswith('_'):
                    try:
                        setattr(config, attr, getattr(self.base_config, attr))
                    except (AttributeError, TypeError):
                        pass
            for name, val in params.items():
                if hasattr(config, name):
                    setattr(config, name, val)
            config.steps = sim_steps

            try:
                traj = SFCEconomy.run_scenario(config)
                for frame in traj:
                    outcomes = frame.get("outcomes", {})
                    for key, val in outcomes.items():
                        if key not in predictions:
                            predictions[key] = []
                        # Append per trajectory per step
            except Exception:
                continue

            # Collect full trajectory for this draw
            for key in list(predictions.keys()):
                pass  # predictions already accumulated above

        # Restructure: collect all draws properly
        all_predictions: Dict[str, List[np.ndarray]] = {}
        for idx in indices:
            params = {name: float(chain[idx]) for name, chain in result.chains.items()}
            config = SFCConfig()
            for attr in vars(self.base_config):
                if not attr.startswith('_'):
                    try:
                        setattr(config, attr, getattr(self.base_config, attr))
                    except (AttributeError, TypeError):
                        pass
            for name, val in params.items():
                if hasattr(config, name):
                    setattr(config, name, val)
            config.steps = sim_steps

            try:
                traj = SFCEconomy.run_scenario(config)
                for key in ["gdp_growth", "inflation", "unemployment"]:
                    series = [f.get("outcomes", {}).get(key, 0.0) for f in traj]
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(series)
            except Exception:
                continue

        result_arrays = {}
        for key, draws in all_predictions.items():
            if draws:
                max_len = max(len(d) for d in draws)
                arr = np.full((len(draws), max_len), np.nan)
                for i, d in enumerate(draws):
                    arr[i, :len(d)] = d
                result_arrays[key] = arr

        return result_arrays

    def bayesian_information_criterion(
        self, result: MCMCResult, observed: ObservedData
    ) -> float:
        """
        Compute BIC for model comparison.

        BIC = k·ln(n) - 2·ln(L_max)
        where k = number of parameters, n = number of observations,
        L_max = maximum likelihood achieved.
        """
        k = len(result.chains)
        n = observed.n_obs
        if n < 1 or k < 1:
            return np.inf

        # Max log-likelihood from chain
        max_ll = float(np.max(result.log_posteriors))

        bic = k * np.log(n) - 2.0 * max_ll
        return bic

    def online_update(self, param_name: str, observed_value: float) -> float:
        """
        Online Bayesian update of a single parameter using Kalman filter.

        This provides real-time parameter tracking between full MCMC runs.
        Extends KalmanFilter1D from algorithms_online.py.

        Args:
            param_name: Name of the SFCConfig parameter.
            observed_value: New observation informing this parameter.

        Returns:
            Updated parameter estimate.
        """
        if param_name not in self._kalman_trackers:
            prior = self.priors.get(param_name)
            if prior is None:
                return observed_value
            self._kalman_trackers[param_name] = KalmanFilter1D(
                config=KalmanConfig(
                    process_noise=prior.std ** 2 * 0.01,
                    observation_noise=prior.std ** 2,
                )
            )
            self._kalman_trackers[param_name].x = prior.mean

        return self._kalman_trackers[param_name].update(observed_value)

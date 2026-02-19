"""
Tier-5 Meta-Integrative Layer — global reasoning and governance.

Provides online, rule-based meta-governance over Scarcity's lower tiers.
Aggregates telemetry, evaluates health, recommends controller/evaluator/DRG
adjustments, enforces safety, and persists decisions for later analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from scarcity.runtime import EventBus, get_bus
from scarcity.engine.resource_profile import clone_default_profile
from .integrative_config import (
    IntegrativeMetaConfig,
    MetaScoreConfig,
    ControllerPolicyConfig,
    EvaluatorPolicyConfig,
    DRGPolicyConfig,
    SafetyConfig,
    DEFAULT_INTEGRATIVE_CONFIG,
)

logger = logging.getLogger(__name__)

# Legacy DEFAULT_CONFIG kept for backward compatibility
# New code should use IntegrativeMetaConfig dataclass
DEFAULT_CONFIG: Dict[str, Any] = DEFAULT_INTEGRATIVE_CONFIG.to_dict()


def _clamp(value: float, bounds: List[float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def _select_nearest(value: int, choices: List[int]) -> int:
    arr = np.asarray(choices, dtype=np.int64)
    idx = np.argmin(np.abs(arr - value))
    return int(arr[idx])


@dataclass
class MetaState:
    """State vector for the meta-integrative layer."""
    tau: float = 0.9
    gamma_diversity: float = 0.3
    g_min: float = 0.01
    lambda_ci: float = 0.5
    tier2_enabled: bool = True
    tier3_topk: int = 5
    ema_reward: float = 0.0
    last_reward: float = 0.0
    cooldowns: Dict[str, int] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    decision_count: int = 0
    rollback_count: int = 0


class MetaIntegrativeLayer:
    """
    Rule-based meta-governance layer coordinating Tier-0..Tier-4 outputs.
    
    Responsible for:
    - Computing meta-reward based on telemetry
    - Adjusting high-level hyperparameters (tau, gamma, g_min, etc.)
    - Managing safety rollbacks and cooldowns
    - Logging meta-decisions
    
    Args:
        config: Configuration as IntegrativeMetaConfig dataclass or legacy Dict.
        log_path: Path to write the policy log.
    """

    def __init__(
        self,
        config: Optional[Union[IntegrativeMetaConfig, Dict[str, Any]]] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the meta-integrative layer.
        
        Args:
            config: Configuration (dataclass or dict). Defaults to default config.
            log_path: Path to write the policy log.
        """
        # Handle both dataclass and dict configs for backward compatibility
        if config is None:
            self._config = DEFAULT_INTEGRATIVE_CONFIG
            self.config = DEFAULT_CONFIG
        elif isinstance(config, IntegrativeMetaConfig):
            self._config = config
            self.config = config.to_dict()
        else:
            # Legacy dict config
            self._config = None
            self.config = self._merge_config(config)
        
        self.state = MetaState()
        self.log_path = log_path or Path("artifacts/meta/policy_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
        for section, values in overrides.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
        return cfg

    def update(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the latest telemetry snapshot and return meta outputs.
        
        Args:
            telemetry: Dictionary of current system telemetry.
            
        Returns:
            Dictionary containing policy updates, resource hints, scoring metrics, and logs.
        """
        prev_snapshot = self._policy_snapshot()
        reward = self._compute_reward(telemetry)
        ema_reward = self._update_ema(reward)

        policy_update, changed_knobs = self._apply_policies(telemetry, reward, ema_reward)
        resource_hint = self._resource_policy(telemetry)
        rollback_triggered = self._safety_checks(reward, ema_reward, prev_snapshot, changed_knobs)
        post_snapshot = self._policy_snapshot()

        event_record = {
            'timestamp': time.time(),
            'reward': reward,
            'reward_avg': ema_reward,
            'policy_update': policy_update,
            'knobs_before': prev_snapshot,
            'knobs_after': post_snapshot,
            'resource_hint': resource_hint,
            'rollback': rollback_triggered,
        }
        self._record_event(event_record)

        output = {
            'meta_policy_update': policy_update,
            'resource_profile_hint': resource_hint,
            'meta_score': reward,
            'meta_score_avg': ema_reward,
            'meta_event_log': list(self.state.history),
            'meta_telemetry': {
                'meta_reward': reward,
                'meta_reward_avg': ema_reward,
                'meta_decision_count': self.state.decision_count,
                'rollback_count': self.state.rollback_count,
                'tau': self.state.tau,
                'gamma_diversity': self.state.gamma_diversity,
                'g_min': self.state.g_min,
                'lambda_ci': self.state.lambda_ci,
                'vram_util': telemetry.get('vram_util', 0.0),
                'latency_ms': telemetry.get('latency_ms', 0.0),
                'policy_changes_pending': sum(1 for v in self.state.cooldowns.values() if v > 0),
                'meta_success_rate': self._success_rate(),
            },
        }

        return output

    # --------------------------------------------------------------------- #
    # Reward computation
    # --------------------------------------------------------------------- #

    def _compute_reward(self, telemetry: Dict[str, Any]) -> float:
        weights = self.config['meta_score']['weights']
        penalties = self.config['meta_score']['penalties']

        accept = telemetry.get('accept_rate', 0.0)
        stability = telemetry.get('stability_avg', 0.0)
        contrast = telemetry.get('rcl_contrast', 0.0)
        latency = telemetry.get('latency_ms', 0.0) / 120.0  # normalize
        latency = float(np.clip(latency, 0.0, 2.0))
        vram = telemetry.get('vram_util', 0.0)
        oom_flag = 1.0 if telemetry.get('oom_flag', False) else 0.0

        reward = (
            weights.get('accept', 0.0) * accept +
            weights.get('stability', 0.0) * stability +
            weights.get('contrast', 0.0) * contrast -
            penalties.get('latency', 0.0) * latency -
            penalties.get('vram', 0.0) * vram -
            penalties.get('oom', 0.0) * oom_flag
        )

        reward = float(np.clip(reward, -1.0, 1.0))
        self.state.last_reward = reward
        return reward

    def _update_ema(self, reward: float) -> float:
        alpha = self.config['meta_score'].get('ema_alpha', 0.3)
        ema = (1 - alpha) * self.state.ema_reward + alpha * reward
        self.state.ema_reward = float(ema)
        return self.state.ema_reward

    # --------------------------------------------------------------------- #
    # Policy logic
    # --------------------------------------------------------------------- #

    def _apply_policies(
        self,
        telemetry: Dict[str, Any],
        reward: float,
        ema_reward: float,
    ) -> Tuple[Dict[str, Any], List[str]]:
        change_map: Dict[str, Any] = {'rev': self.state.decision_count + 1}
        changed_knobs: List[str] = []
        controller_changes = {}
        evaluator_changes = {}
        operator_changes = {}

        self._cooldown_tick()

        # Controller exploration adjustments
        accept = telemetry.get('accept_rate', 0.0)
        stability = telemetry.get('stability_avg', 0.0)
        gain_p50 = telemetry.get('gain_p50', 0.0)

        tau_bounds = self.config['controller_policy']['tau_bounds']
        gamma_bounds = self.config['controller_policy']['gamma_bounds']

        if accept < 0.06 and ema_reward < self.state.ema_reward + 1e-9:
            tau_new = _clamp(self.state.tau + 0.1, tau_bounds)
            gamma_new = _clamp(self.state.gamma_diversity + 0.05, gamma_bounds)
            if self._eligible('tau') and tau_new != self.state.tau:
                self.state.tau = tau_new
                controller_changes['tau'] = tau_new
                changed_knobs.append('tau')
            if self._eligible('gamma_diversity') and gamma_new != self.state.gamma_diversity:
                self.state.gamma_diversity = gamma_new
                controller_changes['gamma_diversity'] = gamma_new
                changed_knobs.append('gamma_diversity')

        if stability > 0.8 and gain_p50 > telemetry.get('gain_prev', gain_p50):
            tau_new = _clamp(self.state.tau - 0.1, tau_bounds)
            if self._eligible('tau') and tau_new != self.state.tau:
                self.state.tau = tau_new
                controller_changes['tau'] = tau_new
                changed_knobs.append('tau')

        if controller_changes:
            change_map['controller'] = controller_changes

        # Evaluator thresholds
        if telemetry.get('accept_low_windows', 0) >= 5 and accept < 0.03:
            g_min_bounds = self.config['evaluator_policy']['g_min_bounds']
            g_new = _clamp(self.state.g_min - 0.002, g_min_bounds)
            if self._eligible('g_min') and g_new != self.state.g_min:
                self.state.g_min = g_new
                evaluator_changes['g_min'] = g_new
                changed_knobs.append('g_min')

        if telemetry.get('ci_width', 0.0) > telemetry.get('ci_width_target', 0.1):
            lambda_bounds = self.config['evaluator_policy']['lambda_ci_bounds']
            lambda_new = _clamp(self.state.lambda_ci - 0.05, lambda_bounds)
            if self._eligible('lambda_ci') and lambda_new != self.state.lambda_ci:
                self.state.lambda_ci = lambda_new
                evaluator_changes['lambda_ci'] = lambda_new
                changed_knobs.append('lambda_ci')

        if evaluator_changes:
            change_map['evaluator'] = evaluator_changes

        # Operator tier toggling
        latency_ms = telemetry.get('latency_ms', 0.0)
        vram = telemetry.get('vram_util', 0.0)
        under_load = latency_ms > 120.0 or vram > 0.85

        if under_load:
            if self.state.tier3_topk > 0 and self._eligible('tier3_topk'):
                self.state.tier3_topk = 0
                operator_changes['tier3_topk'] = 0
                changed_knobs.append('tier3_topk')
            elif self.state.tier2_enabled and self._eligible('tier2_enabled'):
                self.state.tier2_enabled = False
                operator_changes['tier2'] = 'off'
                changed_knobs.append('tier2_enabled')
        else:
            if not self.state.tier2_enabled and self._eligible('tier2_enabled'):
                self.state.tier2_enabled = True
                operator_changes['tier2'] = 'on'
                changed_knobs.append('tier2_enabled')
            if self.state.tier3_topk == 0 and self._eligible('tier3_topk'):
                self.state.tier3_topk = 5
                operator_changes['tier3_topk'] = 5
                changed_knobs.append('tier3_topk')

        if operator_changes:
            change_map['operators'] = operator_changes

        # Reason tag
        reasons = []
        if accept < 0.06:
            reasons.append('low_accept')
        if under_load:
            reasons.append('latency_high')
        if telemetry.get('oom_flag', False):
            reasons.append('oom')
        if not reasons:
            reasons.append('steady')
        change_map['reason'] = ' + '.join(reasons)

        self.state.decision_count += 1
        return change_map, changed_knobs

    def _resource_policy(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        vram = telemetry.get('vram_util', 0.0)
        latency = telemetry.get('latency_ms', 0.0)
        drg_cfg = self.config['drg_policy']

        hint: Dict[str, Any] = {}

        if vram > 0.85 or telemetry.get('oom_flag', False):
            hint['n_paths_delta'] = -0.15
            hint['sketch_dim_target'] = drg_cfg['sketch_dim_set'][0]
        elif vram < 0.55 and latency < 100.0:
            hint['n_paths_delta'] = 0.10
            hint['resamples_target'] = telemetry.get('resamples_default', 10) + 2

        if 'sketch_dim_target' in hint:
            hint['sketch_dim_target'] = _select_nearest(int(hint['sketch_dim_target']), drg_cfg['sketch_dim_set'])

        return hint

    # --------------------------------------------------------------------- #
    # Safety & logging
    # --------------------------------------------------------------------- #

    def _safety_checks(
        self,
        reward: float,
        ema_reward: float,
        previous_snapshot: Dict[str, Any],
        changed_knobs: List[str],
    ) -> bool:
        rollback_delta = self.config['safety']['rollback_delta']
        cooldown = self.config['safety']['cooldown_cycles']

        if len(self.state.history) >= 1:
            prev_entry = self.state.history[-1]
            delta = prev_entry.get('reward_avg', 0.0) - ema_reward
            if delta > rollback_delta:
                self._rollback_previous(prev_entry.get('knobs_before', {}))
                self.state.rollback_count += 1
                logger.warning("MetaIntegrativeLayer: rollback triggered due to reward drop %.3f", delta)
                return True

        # apply cooldown for newly changed knobs
        for knob in changed_knobs:
            self.state.cooldowns[knob] = cooldown
        return False

    def _rollback_previous(self, previous_snapshot: Dict[str, Any]) -> None:
        if not previous_snapshot:
            return
        self.state.tau = previous_snapshot.get('tau', self.state.tau)
        self.state.gamma_diversity = previous_snapshot.get('gamma_diversity', self.state.gamma_diversity)
        self.state.g_min = previous_snapshot.get('g_min', self.state.g_min)
        self.state.lambda_ci = previous_snapshot.get('lambda_ci', self.state.lambda_ci)
        self.state.tier2_enabled = previous_snapshot.get('tier2_enabled', self.state.tier2_enabled)
        self.state.tier3_topk = previous_snapshot.get('tier3_topk', self.state.tier3_topk)

    def _record_event(self, event: Dict[str, Any]) -> None:
        self.state.history.append(event)
        if len(self.state.history) > 10:
            self.state.history = self.state.history[-10:]
        try:
            with self.log_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError as exc:  # pragma: no cover
            logger.error("Failed to write meta policy log: %s", exc)

    def _cooldown_tick(self) -> None:
        for key in list(self.state.cooldowns.keys()):
            self.state.cooldowns[key] = max(0, self.state.cooldowns[key] - 1)

    def _eligible(self, knob: str) -> bool:
        return self.state.cooldowns.get(knob, 0) <= 0

    def _set_cooldown(self, knob: str) -> None:
        self.state.cooldowns[knob] = self.config['safety']['cooldown_cycles']

    def _success_rate(self) -> float:
        if not self.state.history:
            return 1.0
        successes = sum(1 for event in self.state.history if not event.get('rollback', False))
        return float(successes / len(self.state.history))

    def _policy_snapshot(self) -> Dict[str, Any]:
        return {
            'tau': self.state.tau,
            'gamma_diversity': self.state.gamma_diversity,
            'g_min': self.state.g_min,
            'lambda_ci': self.state.lambda_ci,
            'tier2_enabled': self.state.tier2_enabled,
            'tier3_topk': self.state.tier3_topk,
        }


class MetaSupervisor:
    """
    Runtime integration layer for Tier-5 meta governance.
    
    Bridge between the runtime EventBus and the MetaIntegrativeLayer.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        bus: Optional[EventBus] = None,
        log_path: Optional[Path] = None,
        initial_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the meta supervisor.
        
        Args:
            config: Configuration dictionary.
            bus: EventBus instance.
            log_path: Path for logging.
            initial_profile: Initial resource profile.
        """
        self.bus = bus if bus else get_bus()
        self.layer = MetaIntegrativeLayer(config=config, log_path=log_path)
        self.config = self.layer.config
        self.current_profile = initial_profile.copy() if initial_profile else clone_default_profile()

        self.last_processing: Dict[str, Any] = {}
        self.last_telemetry: Dict[str, Any] = {}
        self.prev_gain_p50: Optional[float] = None
        self.low_accept_windows = 0
        self.running = False
        self._lock = asyncio.Lock()

        self._processing_handler = self._handle_processing_metrics
        self._telemetry_handler = self._handle_telemetry

    async def start(self) -> None:
        """Start the supervisor."""
        if self.running:
            return
        self.running = True
        self.bus.subscribe("processing_metrics", self._processing_handler)
        self.bus.subscribe("telemetry", self._telemetry_handler)

    async def stop(self) -> None:
        """Stop the supervisor."""
        if not self.running:
            return
        self.running = False
        self.bus.unsubscribe("processing_metrics", self._processing_handler)
        self.bus.unsubscribe("telemetry", self._telemetry_handler)

    async def _handle_processing_metrics(self, topic: str, data: Dict[str, Any]) -> None:
        self.last_processing = data or {}
        accept = self.last_processing.get('accept_rate', 0.0)
        if accept < 0.03:
            self.low_accept_windows += 1
        else:
            self.low_accept_windows = 0

        await self._maybe_update()

    async def _handle_telemetry(self, topic: str, data: Dict[str, Any]) -> None:
        self.last_telemetry = data or {}

    async def _maybe_update(self) -> None:
        if not self.running or not self.last_processing:
            return

        async with self._lock:
            meta_input = self._build_meta_input()
            outputs = self.layer.update(meta_input)

            await self.bus.publish("meta_policy_update", outputs.get('meta_policy_update', {}))

            resource_hint = outputs.get('resource_profile_hint', {})
            profile = self._apply_resource_hint(resource_hint)
            if profile:
                await self.bus.publish("resource_profile", profile)

            meta_metrics = outputs.get('meta_telemetry', {})
            if meta_metrics:
                await self.bus.publish("meta_metrics", meta_metrics)

            self.prev_gain_p50 = self.last_processing.get('gain_p50', self.prev_gain_p50)

    def _build_meta_input(self) -> Dict[str, Any]:
        processing = dict(self.last_processing)
        telemetry = self.last_telemetry

        combined: Dict[str, Any] = {}
        combined.update(processing)

        combined['latency_ms'] = processing.get('engine_latency_ms', telemetry.get('bus_latency_ms', 0.0))
        combined['vram_util'] = self._extract_vram_util(telemetry)
        combined['accept_low_windows'] = self.low_accept_windows
        combined['gain_prev'] = self.prev_gain_p50 if self.prev_gain_p50 is not None else processing.get('gain_p50', 0.0)
        combined.setdefault('stability_avg', processing.get('stability_avg', 0.0))
        combined.setdefault('ci_width', processing.get('ci_width_avg', 0.0))
        combined.setdefault('ci_width_target', 0.1)
        combined.setdefault('diversity_index', processing.get('diversity_index', 0.0))
        combined.setdefault('rcl_contrast', processing.get('rcl_contrast', 0.0))
        combined.setdefault('oom_flag', processing.get('oom_flag', False))

        return combined

    def _apply_resource_hint(self, hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not hint:
            return None

        profile = self.current_profile.copy()
        drg_cfg = self.config.get('drg_policy', {})
        max_paths = drg_cfg.get('n_paths_max', 128)

        if 'n_paths_delta' in hint:
            delta = hint['n_paths_delta']
            current = profile.get('n_paths', 200)
            profile['n_paths'] = int(max(1, min(max_paths, current * (1 + delta))))

        if 'sketch_dim_target' in hint:
            profile['sketch_dim'] = int(hint['sketch_dim_target'])

        if 'resamples_target' in hint:
            profile['resamples'] = int(max(1, hint['resamples_target']))

        changed = profile != self.current_profile
        if changed:
            self.current_profile = profile
            return profile
        return None

    @staticmethod
    def _extract_vram_util(telemetry: Dict[str, Any]) -> float:
        if not telemetry:
            return 0.0
        if 'gpu_memory_util' in telemetry:
            return float(telemetry['gpu_memory_util'])
        used = telemetry.get('vram_used_gb')
        total = telemetry.get('vram_total_gb', 0.0)
        if used is not None and total:
            return float(np.clip(used / total, 0.0, 1.0))
        return float(telemetry.get('vram_util', 0.0))


__all__ = [
    'MetaIntegrativeLayer',
    'MetaState',
    'DEFAULT_CONFIG',
    'MetaSupervisor',
    'IntegrativeMetaConfig',
]

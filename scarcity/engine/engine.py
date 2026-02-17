"""
MPIE Orchestrator — Event-driven engine coordinator.

Coordinates the online inference pipeline: Controller → Encoder → Evaluator → Store → Exporter.
Implements full Controller ⇆ Evaluator online interaction contract.
"""

import asyncio
import logging
import numpy as np  # type: ignore
from typing import Dict, Any, Optional, List
from collections import deque
import time
from dataclasses import asdict

from scarcity.runtime import EventBus, get_bus
from scarcity.engine.bandit_router import BanditRouter
from scarcity.engine.encoder import Encoder
from scarcity.engine.evaluator import Evaluator
from scarcity.engine.store import HypergraphStore
from scarcity.engine.exporter import Exporter
from scarcity.engine.types import Candidate, EvalResult, Reward
from scarcity.engine.resource_profile import clone_default_profile

logger = logging.getLogger(__name__)


class MPIEOrchestrator:
    """
    Multi-Path Inference Engine orchestrator.
    
    Coordinates online inference pipeline under resource constraints.
    Never blocks; maintains bounded state only.
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        """
        Initializes the MPIE orchestrator and its constituent subsystems.

        This constructor sets up the entire online inference pipeline including the
        BanditRouter for path proposals, Encoder for feature extraction, Evaluator
        for statistical validation, HypergraphStore for graph persistence, and
        Exporter for insight broadcasting.

        It also initializes rolling statistics trackers for latency and acceptance
        rates, and sets the initial resource profile using the default system
        configuration.

        Args:
            bus: An optional EventBus instance. If not provided, the global
                singleton event bus is retrieved via `get_bus()`. This bus is
                used for all asynchronous communication between components.
        """
        self.bus = bus if bus else get_bus()
        
        # Get default resource profile
        self.last_resource_profile = self._get_default_profile()
        
        # Initialize subsystems
        rng = np.random.default_rng()
        self.controller = BanditRouter(drg=self.last_resource_profile, rng=rng)
        self.encoder = Encoder(drg=self.last_resource_profile)
        self.evaluator = Evaluator(drg=self.last_resource_profile, rng=rng)
        self.store = HypergraphStore()
        self.exporter = Exporter()
        
        # Bounded state
        self.profile_history = deque(maxlen=3)
        
        # Rolling statistics
        self.latency_ema = 0.0
        self.acceptance_counts = deque(maxlen=100)
        
        # Flags
        self.oom_backoff = False
        self.running = False
        
        # Stats
        self._stats = {
            'windows_processed': 0,
            'oom_incidents': 0,
            'avg_latency_ms': 0.0
        }
        
        logger.info("MPIE Orchestrator initialized with full Controller⇆Evaluator contract")
    
    async def start(self) -> None:
        """
        Starts the orchestrator and activates event subscriptions.

        This method marks the orchestrator as running and subscribes to the critical
        event topics:
        - `data_window`: Triggers the main inference pipeline when new data arrives.
        - `resource_profile`: Updates internal resource constraints from the Governor.
        - `meta_policy_update`: Applies high-level policy changes from Meta-Learning.

        This operation is idempotent; calling it on an already running orchestrator
        logs a warning but has no side effects.
        """
        if self.running:
            logger.warning("MPIE already running")
            return
        
        self.running = True
        
        # Subscribe to input events
        self.bus.subscribe("data_window", self._handle_data_window)
        self.bus.subscribe("resource_profile", self._handle_resource_profile)
        self.bus.subscribe("meta_policy_update", self._handle_meta_policy_update)
        self.bus.subscribe("meta_prior_update", self._handle_meta_policy_update)
        # FMI topics (bridge outputs from federation-meta interface)
        self.bus.subscribe("fmi.meta_prior_update", self._handle_meta_policy_update)
        self.bus.subscribe("fmi.meta_policy_hint", self._handle_fmi_policy_hint)
        self.bus.subscribe("fmi.warm_start_profile", self._handle_fmi_warm_start)
        self.bus.subscribe("fmi.telemetry", self._handle_fmi_telemetry)
        
        logger.info("MPIE Orchestrator started with full subsystems")
    
    async def stop(self) -> None:
        """
        Stops the orchestrator and releases event subscriptions.

        This method marks the orchestrator as stopped and unsubscribes from all
        active event topics (`data_window`, `resource_profile`, `meta_policy_update`).
        It gracefully halts processing of new events, though currently executing
        tasks may complete depending on the event bus behavior.

        This operation is idempotent; calling it on a stopped orchestrator does
        nothing.
        """
        if not self.running:
            return
        
        self.running = False
        
        # Unsubscribe
        self.bus.unsubscribe("data_window", self._handle_data_window)
        self.bus.unsubscribe("resource_profile", self._handle_resource_profile)
        self.bus.unsubscribe("meta_policy_update", self._handle_meta_policy_update)
        self.bus.unsubscribe("meta_prior_update", self._handle_meta_policy_update)
        self.bus.unsubscribe("fmi.meta_prior_update", self._handle_meta_policy_update)
        self.bus.unsubscribe("fmi.meta_policy_hint", self._handle_fmi_policy_hint)
        self.bus.unsubscribe("fmi.warm_start_profile", self._handle_fmi_warm_start)
        self.bus.unsubscribe("fmi.telemetry", self._handle_fmi_telemetry)
        
        logger.info("MPIE Orchestrator stopped")
    
    async def _handle_data_window(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Process an incoming data window through the full inference pipeline.

        This is the core event handler that implements the Controller-Evaluator
        interaction contract. It executes the following sequence:
        1. Retrieves the current resource profile.
        2. Requests path proposals from the Controller (BanditRouter).
        3. Encodes the window data for feature extraction.
        4. Scores the proposed paths using the Evaluator.
        5. Computes diversity metrics and shapes rewards.
        6. Updates the Controller with performance rewards (bandit learning).
        7. Persists accepted causal edges to the HypergraphStore.
        8. Emits insights via the Exporter.
        9. Publishes comprehensive processing metrics to the event bus.

        Args:
            topic: The topic string on which the event was received.
            data: The event payload containing the data window, schema information,
                and optional metadata like window ID.
        """
        if not self.running:
            return
        
        start_time = time.time()
        
        try:
            # Step 1: Get current resource profile
            resource_profile = self.last_resource_profile or self._get_default_profile()
            
            # Step 2: Propose paths (Controller.propose returns List[Candidate])
            candidates = self.controller.propose(
                n_proposals=resource_profile.get('n_paths', 200),
                context={
                    'schema': data.get('schema', {}),
                    'window_meta': {'length': len(data.get('data', [])), 'timestamp': time.time()},
                },
            )
            if candidates and not isinstance(candidates[0], Candidate):
                logger.error("BanditRouter returned non-Candidate proposals; skipping window.")
                return
            candidate_lookup = {cand.path_id: cand for cand in candidates}
            
            # Step 3: Extract window tensor
            window_tensor = data.get('data')
            if window_tensor is None:
                logger.warning("No data in window")
                return
            
            if isinstance(window_tensor, list):
                window_tensor = np.array(window_tensor)

            schema_obj = data.get('schema', {}) or {}
            var_names = self._resolve_var_names(schema_obj, window_tensor.shape[1])
            
            # Step 4: Score candidates (Evaluator.score returns List[EvalResult])
            results = self.evaluator.score(window_tensor, candidates)
            
            # Step 5: Build diversity lookup from Controller
            diversity_dict = {cand.path_id: self.controller.diversity_score(cand) for cand in candidates}
            D_lookup = lambda pid: diversity_dict.get(pid, 0.0)
            
            # Step 6: Produce rewards (Evaluator.make_rewards returns List[Reward])
            rewards = self.evaluator.make_rewards(results, D_lookup, candidates=candidates)
            
            # Step 7: Update Controller with rewards (bandit learning)
            self.controller.update(rewards)
            accepted_candidates: List[Candidate] = []
            store_payloads: List[Dict[str, Any]] = []
            for result in results:
                if not result.accepted:
                    continue
                candidate = candidate_lookup.get(result.path_id)
                if not candidate:
                    continue
                accepted_candidates.append(candidate)
                resolved_vars = [self._var_name(var_names, idx) for idx in candidate.vars]
                store_payloads.append({
                    'path_id': result.path_id,
                    'gain': result.gain,
                    'ci_lo': result.ci_lo,
                    'ci_hi': result.ci_hi,
                    'stability': result.stability,
                    'cost_ms': result.cost_ms,
                    'domain': candidate.domain,
                    'window_id': data.get('window_id', self._stats['windows_processed']),
                    'schema_version': schema_obj.get('version', 0),
                    'source': resolved_vars[0],
                    'target': resolved_vars[-1],
                    'vars': resolved_vars,
                    'var_indices': list(candidate.vars),
                    'lags': list(candidate.lags),
                    'ops': list(candidate.ops),
                })
            self.controller.register_acceptances(accepted_candidates)
            
            # Step 8: Update store with accepted results
            accepted = [r for r in results if r.accepted]
            accepted_payloads = [asdict(r) for r in accepted]
            if self.store and store_payloads:
                self.store.update_edges(store_payloads)
                await self.bus.publish(
                    "engine.insight",
                    {
                        "edges": store_payloads,
                        "window_id": data.get('window_id', self._stats['windows_processed']),
                        "timestamp": time.time(),
                    },
                )
            
            # Step 9: Export insights
            if self.exporter:
                self.exporter.emit_insights(accepted_payloads, resource_profile)
            
            # Step 10: Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            self.acceptance_counts.append(len(accepted))
            self._stats['windows_processed'] += 1
            
            # Step 11: Publish comprehensive metrics
            await self._publish_metrics(latency_ms, len(candidates), len(accepted))
            
        except Exception as e:
            logger.error(f"Error processing data window: {e}", exc_info=True)
    
    async def _handle_resource_profile(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Updates the internal resource profile based on Governor feedback.

        This handler receives resource updates (e.g., from the Dynamic Resource
        Governor) and propagates them to all subsystems. This allows the engine
        to adaptively throttle or expand its operations (like batch size or
        resampling counts) in real-time based on system load.

        Args:
            topic: The event topic.
            data: A dictionary containing resource parameters and constraints.
        """
        self.last_resource_profile = data
        self.profile_history.append(data)
        
        logger.debug(f"Resource profile updated: {data}")
        
        # Propagate to subsystems
        if self.controller:
            self.controller.update_resource_profile(data)
        if self.evaluator:
            self.evaluator.drg = data
            if 'resamples' in data:
                self.evaluator.resamples = int(max(1, data['resamples']))
            if 'gain_min' in data:
                self.evaluator.gain_min = float(data['gain_min'])
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """
        Retrieves the default resource profile configuration.

        Returns:
            A dictionary containing default values for resource-related settings,
            such as the number of paths to propose, resampling counts, and
            allocation limits.
        """
        return clone_default_profile()
    
    def _update_latency(self, latency_ms: float) -> None:
        """
        Updates the exponential moving average (EMA) of processing latency.

        Args:
            latency_ms: The processing time of the most recent window in milliseconds.
                This value is blended with the historical average using a fixed alpha.
        """
        alpha = 0.3
        if self.latency_ema == 0:
            self.latency_ema = latency_ms
        else:
            self.latency_ema = alpha * latency_ms + (1 - alpha) * self.latency_ema
        
        self._stats['avg_latency_ms'] = self.latency_ema
    
    async def _publish_metrics(self, latency_ms: float, n_candidates: int, n_accepted: int) -> None:
        """
        Publishes comprehensive system telemetry to the event bus.

        Collects statistics from the Controller, Evaluator, Store, and the
        Orchestrator itself, aggregates them into a single metrics payload, and
        broadcasts it on the `processing_metrics` topic. This data is consumed
        by the Meta-Learning system and the Dashboard.

        Args:
            latency_ms: The latency of the current processing cycle.
            n_candidates: The total number of paths proposed by the Controller.
            n_accepted: The number of paths that passed Evaluator validation.
        """
        # Get Controller stats
        ctrl_stats = self.controller.get_stats() if self.controller else {}
        
        # Get Evaluator stats
        eval_stats = self.evaluator.get_stats() if self.evaluator else {}
        
        # Compute diversity index from candidates (if available)
        # This is a placeholder - full implementation would track proposed diversity
        diversity_index = 0.0
        
        metrics = {
            # Orchestrator metrics
            'engine_latency_ms': latency_ms,
            'n_candidates': n_candidates,
            'accepted_count': n_accepted,
            'accept_rate': n_accepted / max(n_candidates, 1),
            'edges_active': self.store.get_edge_count() if self.store else 0,
            'oom_flag': self.oom_backoff,
            
            # Controller metrics (from §9)
            'proposal_entropy': ctrl_stats.get('proposal_entropy', 0.0),
            'diversity_index': diversity_index,
            'arm_mean_r_topk': ctrl_stats.get('arm_mean_r_topk', []),
            'drift_detections': ctrl_stats.get('drift_detections', 0),
            'thompson_mode': ctrl_stats.get('thompson_mode', False),
            
            # Evaluator metrics (from §9)
            'eval_accept_rate': eval_stats.get('accept_rate', 0.0),
            'gain_p50': eval_stats.get('gain_p50', 0.0),
            'gain_p90': eval_stats.get('gain_p90', 0.0),
            'ci_width_avg': eval_stats.get('ci_width_avg', 0.0),
            'stability_avg': eval_stats.get('stability_avg', 0.0),
            'total_evaluated': eval_stats.get('total_evaluated', 0)
        }
        
        await self.bus.publish("processing_metrics", metrics)

    async def _handle_meta_policy_update(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Applies meta-learning policy updates to the subsystems.

        This handler receives high-level parameter updates (meta-policies) from
        the Meta-Learning layer and routes them to the appropriate components:
        - Controller: Updates exploration (tau) and diversity (gamma) parameters.
        - Evaluator: Updates acceptance thresholds (gain_min, CI lambda).
        - Operators: Updates configuration for tiered operators.

        Args:
            topic: The event topic.
            data: The policy update payload containing configuration dictionaries
                for each subsystem.
        """
        if not data:
            return
        if "prior" in data and isinstance(data["prior"], dict):
            data = data["prior"]

        controller_cfg = data.get('controller', {})
        if controller_cfg and self.controller:
            self.controller.apply_meta_update(
                tau=controller_cfg.get('tau'),
                gamma_diversity=controller_cfg.get('gamma_diversity')
            )

        evaluator_cfg = data.get('evaluator', {})
        if evaluator_cfg and self.evaluator:
            self.evaluator.apply_meta_update(
                g_min=evaluator_cfg.get('g_min'),
                lambda_ci=evaluator_cfg.get('lambda_ci')
            )

        operator_cfg = data.get('operators', {})
        if operator_cfg:
            if 'tier3_topk' in operator_cfg:
                self.last_resource_profile['tier3_topk'] = operator_cfg['tier3_topk']
            if 'tier2' in operator_cfg:
                self.last_resource_profile['tier2_enabled'] = (operator_cfg['tier2'] != 'off')

    async def _handle_fmi_policy_hint(self, topic: str, payload: Dict[str, Any]) -> None:
        """Apply FMI policy hints by forwarding bundle to meta policy updater."""
        bundle = payload.get("bundle", {}) if isinstance(payload, dict) else {}
        if not isinstance(bundle, dict) or not bundle:
            return
        await self._handle_meta_policy_update(topic, bundle)

    async def _handle_fmi_warm_start(self, topic: str, payload: Dict[str, Any]) -> None:
        """Apply FMI warm-start init payload to controller/evaluator/ops."""
        init_cfg = payload.get("init", {}) if isinstance(payload, dict) else {}
        if not isinstance(init_cfg, dict) or not init_cfg:
            return
        await self._handle_meta_policy_update(topic, init_cfg)

    async def _handle_fmi_telemetry(self, topic: str, payload: Dict[str, Any]) -> None:
        """Currently a no-op hook for FMI telemetry."""
        _ = topic, payload

    def _resolve_var_names(self, schema: Dict[str, Any], width: int) -> List[str]:
        """
        Resolves variable names for the data window from the schema.

        Attempts to extract human-readable variable names from the provided schema.
        If the schema is missing or malformed, it falls back to generating
        positional names (e.g., 'var_0', 'var_1').

        Args:
            schema: The schema dictionary describing the data format.
            width: The number of columns/variables in the data window.

        Returns:
            A list of string names corresponding to the columns in the data.
        """
        fields = schema.get('fields', {}) if isinstance(schema, dict) else {}
        if isinstance(fields, dict) and fields:
            return list(fields.keys())
        if isinstance(fields, list) and fields:
            names = []
            for idx, entry in enumerate(fields):
                if isinstance(entry, dict) and entry.get('name'):
                    names.append(str(entry['name']))
                else:
                    names.append(f"var_{idx}")
            return names
        return [f"var_{idx}" for idx in range(width)]

    @staticmethod
    def _var_name(var_names: List[str], index: int) -> str:
        if 0 <= index < len(var_names):
            return var_names[index]
        return f"var_{index}"

    
    def set_oom_flag(self) -> None:
        """
        Sets the Out-Of-Memory (OOM) backoff flag.

        This signals the orchestrator that the system is under memory pressure and
        should reduce its workload (e.g., by proposing fewer paths) in subsequent
        cycles. Increments the OOM incident counter.
        """
        self.oom_backoff = True
        self._stats['oom_incidents'] += 1
        logger.warning("OOM flag set - reducing work next cycle")
    
    def clear_oom_flag(self) -> None:
        """
        Clears the Out-Of-Memory (OOM) backoff flag.
        
        This indicates that memory pressure has subsided and the orchestrator can
        gradually resume normal workload levels.
        """
        self.oom_backoff = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregates and returns current orchestrator statistics.

        Returns:
            A dictionary containing internal counters (windows processed, OOM incidents),
            current running state, backoff status, and moving averages of performance
            metrics like the acceptance rate.
        """
        return {
            **self._stats,
            'running': self.running,
            'oom_backoff': self.oom_backoff,
            'avg_accept_rate': sum(self.acceptance_counts) / max(len(self.acceptance_counts), 1)
        }


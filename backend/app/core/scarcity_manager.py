"""
ScarcityCoreManager - Coordinates all scarcity core components.

This module manages the lifecycle of all scarcity framework components:
- Runtime Bus (event-driven communication)
- MPIE Orchestrator (causal discovery)
- Dynamic Resource Governor (resource monitoring)
- Federation Coordinator (decentralized learning)
- Meta Learning Agent (cross-domain optimization)
- Simulation Engine (hypergraph visualization)
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from collections import deque

from scarcity.runtime import EventBus, get_bus
from scarcity.engine.engine import MPIEOrchestrator
from scarcity.governor.drg_core import DynamicResourceGovernor, DRGConfig
from scarcity.meta.meta_learning import MetaLearningAgent, MetaLearningConfig
from scarcity.federation import FederationClientAgent, FederationCoordinator, StoreReconciler, build_reconciler
from scarcity.federation.client_agent import ClientAgentConfig
from scarcity.federation.coordinator import CoordinatorConfig
from scarcity.federation.aggregator import AggregationConfig, AggregationMethod
from scarcity.federation.privacy_guard import PrivacyConfig
from scarcity.federation.validator import ValidatorConfig
from scarcity.federation.transport import TransportConfig
from scarcity.simulation.engine import SimulationEngine, SimulationConfig
from scarcity.simulation.agents import AgentRegistry
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ScarcityCoreManager:
    """
    Manages lifecycle of all scarcity core components.
    
    Coordinates initialization, startup, and shutdown of:
    - Runtime Bus
    - MPIE Orchestrator
    - Dynamic Resource Governor
    - Federation Coordinator (TODO)
    - Meta Learning Agent
    - Simulation Engine (TODO)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the manager with None references for all components."""
        self.settings = settings or get_settings()
        self.bus: Optional[EventBus] = None
        self.mpie: Optional[MPIEOrchestrator] = None
        self.drg: Optional[DynamicResourceGovernor] = None
        self.federation_coordinator: Optional[FederationCoordinator] = None
        self.federation_client: Optional[FederationClientAgent] = None
        self.meta: Optional[MetaLearningAgent] = None
        self.simulation: Optional[SimulationEngine] = None
        
        # Multi-domain support
        self.domain_manager: Optional[Any] = None  # DomainManager
        self.multi_domain_generator: Optional[Any] = None  # MultiDomainDataGenerator
        self.federation_coordinator_v2: Optional[Any] = None  # FederationCoordinator (v2)
        self.domain_data_store: Optional[Any] = None  # DomainDataStore
        
        self._initialized = False
        self._started = False
        
        # Telemetry collection
        self._telemetry_history: deque = deque(maxlen=1000)  # Keep last 1000 events
        self._subscribed_topics: List[str] = []
        
        logger.info("ScarcityCoreManager created")

    def _parse_aggregation_method(self, value: str) -> AggregationMethod:
        try:
            return AggregationMethod(value.lower())
        except Exception:
            logger.warning(f"Unknown aggregation method '{value}', defaulting to trimmed_mean")
            return AggregationMethod.TRIMMED_MEAN

    def _build_federation_client_config(self) -> ClientAgentConfig:
        settings = self.settings
        aggregation = AggregationConfig(
            method=self._parse_aggregation_method(settings.scarcity_federation_aggregation_method),
            trim_alpha=settings.scarcity_federation_aggregation_trim_alpha,
            multi_krum_m=settings.scarcity_federation_aggregation_multi_krum_m,
            adaptive_metric_is_loss=settings.scarcity_federation_aggregation_adaptive_metric_is_loss,
        )
        privacy = PrivacyConfig(
            secure_aggregation=settings.scarcity_federation_privacy_secure_aggregation,
            dp_noise_sigma=settings.scarcity_federation_privacy_noise_sigma,
            dp_epsilon=settings.scarcity_federation_privacy_epsilon,
            dp_delta=settings.scarcity_federation_privacy_delta,
            dp_sensitivity=settings.scarcity_federation_privacy_sensitivity,
            dp_noise_type=settings.scarcity_federation_privacy_noise_type,
            seed_length=settings.scarcity_federation_privacy_seed_length,
        )
        validator = ValidatorConfig(
            trust_min=settings.scarcity_federation_validator_trust_min,
            max_edges=settings.scarcity_federation_validator_max_edges,
            max_concepts=settings.scarcity_federation_validator_max_concepts,
        )
        transport = TransportConfig(
            protocol=settings.scarcity_federation_transport_protocol,
            endpoint=settings.scarcity_federation_transport_endpoint,
            reconnect_backoff=settings.scarcity_federation_transport_reconnect_backoff,
        )
        return ClientAgentConfig(
            aggregation=aggregation,
            privacy=privacy,
            validator=validator,
            transport=transport,
        )

    def _build_coordinator_config(self) -> CoordinatorConfig:
        settings = self.settings
        return CoordinatorConfig(
            heartbeat_timeout=settings.scarcity_federation_coordinator_heartbeat_timeout,
            fairness_quota_kb_min=settings.scarcity_federation_coordinator_fairness_quota_kb_min,
            mode=settings.scarcity_federation_coordinator_mode,
        )
    
    async def initialize(self) -> None:
        """
        Initialize all components in dependency order.
        
        Order matters:
        1. Runtime Bus (foundation for all communication)
        2. MPIE Orchestrator (depends on bus)
        3. DRG (depends on bus, monitors MPIE)
        4. Federation (depends on bus, MPIE)
        5. Meta Learning (depends on bus, federation)
        6. Simulation (depends on MPIE hypergraph)
        """
        if self._initialized:
            logger.warning("ScarcityCoreManager already initialized")
            return
        
        initialized_components = []
        
        try:
            # 1. Initialize Runtime Bus
            logger.info("Initializing Runtime Bus...")
            self.bus = get_bus()
            initialized_components.append("bus")
            
            # Subscribe to core topics for telemetry collection
            self._subscribe_to_core_topics()
            
            # 1.5. Initialize Domain Manager and Multi-Domain Generator
            logger.info("Initializing Domain Manager...")
            try:
                from app.core.domain_manager import DomainManager, DistributionType
                from app.core.multi_domain_generator import MultiDomainDataGenerator
                from app.core.domain_data_store import DomainDataStore
                
                self.domain_manager = DomainManager()
                self.multi_domain_generator = MultiDomainDataGenerator(
                    domain_manager=self.domain_manager,
                    bus=self.bus
                )
                self.domain_data_store = DomainDataStore(
                    max_windows_per_domain=1000,
                    bus=self.bus
                )
                
                # Create 4 default domains with distinct distributions
                logger.info("Creating default domains...")
                self.domain_manager.create_domain(
                    name="Healthcare",
                    distribution_type=DistributionType.NORMAL,
                    distribution_params={"mean": 0.5, "std": 0.8}
                )
                self.domain_manager.create_domain(
                    name="Finance",
                    distribution_type=DistributionType.SKEWED,
                    distribution_params={"shape": 1.5, "scale": 1.2}
                )
                self.domain_manager.create_domain(
                    name="Retail",
                    distribution_type=DistributionType.BIMODAL,
                    distribution_params={"mean1": -0.8, "std1": 0.4, "mean2": 0.8, "std2": 0.4, "weight": 0.6}
                )
                self.domain_manager.create_domain(
                    name="Economics",
                    distribution_type=DistributionType.NORMAL,
                    distribution_params={"mean": -0.3, "std": 1.2}
                )
                logger.info("Created 4 default domains")
                
                if self.settings.scarcity_federation_enabled:
                    # Initialize Federation Coordinator v2
                    from app.core.federation_coordinator import FederationCoordinator as FedCoordV2
                    from app.core.federation_coordinator import AggregationStrategy

                    strategy_value = self.settings.scarcity_federation_v2_strategy.lower()
                    try:
                        strategy = AggregationStrategy(strategy_value)
                    except Exception:
                        logger.warning(
                            f"Unknown federation v2 strategy '{strategy_value}', defaulting to fedavg"
                        )
                        strategy = AggregationStrategy.FEDAVG

                    self.federation_coordinator_v2 = FedCoordV2(
                        strategy=strategy,
                        enable_privacy=self.settings.scarcity_federation_v2_enable_privacy,
                    )
                    logger.info("Federation Coordinator v2 initialized")

                    # Enable federation by default
                    self.federation_coordinator_v2.enable_federation()

                    # Create full mesh topology between all domains
                    domain_ids = [d.id for d in self.domain_manager.list_domains()]
                    self.federation_coordinator_v2.create_full_mesh(domain_ids)
                    logger.info(f"Federation enabled with full mesh topology for {len(domain_ids)} domains")

                    initialized_components.append("federation_coordinator_v2")
                else:
                    self.federation_coordinator_v2 = None

                initialized_components.append("domain_manager")
                initialized_components.append("multi_domain_generator")
                initialized_components.append("domain_data_store")
            except Exception as e:
                logger.error(f"Failed to initialize Domain Manager: {e}")
                self.domain_manager = None
                self.multi_domain_generator = None
                self.domain_data_store = None
                self.federation_coordinator_v2 = None
            
            # 2. Initialize MPIE Orchestrator
            if self.settings.scarcity_mpie_enabled:
                logger.info("Initializing MPIE Orchestrator...")
                try:
                    self.mpie = MPIEOrchestrator(bus=self.bus)
                    initialized_components.append("mpie")
                except Exception as e:
                    logger.error(f"Failed to initialize MPIE: {e}")
                    self.mpie = None
            else:
                logger.info("MPIE disabled by configuration")
                self.mpie = None
            
            # 3. Initialize Dynamic Resource Governor
            if self.settings.scarcity_drg_enabled:
                logger.info("Initializing Dynamic Resource Governor...")
                try:
                    drg_config = DRGConfig()
                    self.drg = DynamicResourceGovernor(config=drg_config, bus=self.bus)
                    initialized_components.append("drg")

                    # Register MPIE with DRG for resource monitoring
                    if self.mpie and self.drg:
                        self.drg.register_subsystem("mpie", self.mpie)
                except Exception as e:
                    logger.error(f"Failed to initialize DRG: {e}")
                    self.drg = None
            else:
                logger.info("DRG disabled by configuration")
                self.drg = None
            
            # 4. Initialize Federation
            if self.settings.scarcity_federation_enabled:
                logger.info("Initializing Federation Layer...")
                try:
                    # Initialize coordinator
                    coordinator_config = self._build_coordinator_config()
                    self.federation_coordinator = FederationCoordinator(config=coordinator_config)
                    initialized_components.append("federation_coordinator")

                    # Initialize client agent
                    if self.mpie and self.mpie.store:
                        reconciler = build_reconciler(self.mpie.store)
                        client_config = self._build_federation_client_config()
                        self.federation_client = FederationClientAgent(
                            node_id="local_node",
                            reconciler=reconciler,
                            bus=self.bus,
                            config=client_config,
                        )
                        initialized_components.append("federation_client")
                    else:
                        logger.warning("Cannot initialize federation client: MPIE store not available")
                except Exception as e:
                    logger.error(f"Failed to initialize Federation: {e}")
                    self.federation_coordinator = None
                    self.federation_client = None
            else:
                logger.info("Federation disabled by configuration")
                self.federation_coordinator = None
                self.federation_client = None
            
            # 5. Initialize Meta Learning Agent
            if self.settings.scarcity_meta_enabled:
                logger.info("Initializing Meta Learning Agent...")
                try:
                    meta_config = MetaLearningConfig()
                    self.meta = MetaLearningAgent(bus=self.bus, config=meta_config)
                    initialized_components.append("meta")
                except Exception as e:
                    logger.error(f"Failed to initialize Meta Learning: {e}")
                    self.meta = None
            else:
                logger.info("Meta learning disabled by configuration")
                self.meta = None
            
            # 6. Initialize Simulation Engine
            if self.settings.scarcity_simulation_enabled:
                logger.info("Initializing Simulation Engine...")
                try:
                    # Create agent registry from MPIE hypergraph
                    if self.mpie and self.mpie.store:
                        agent_registry = AgentRegistry()
                        
                        # Populate registry from hypergraph store
                        snapshot = self.mpie.store.snapshot()
                        nodes = snapshot.get("nodes", {})
                        edges = snapshot.get("edges", {})
                        
                        # Convert edges to list format for registry
                        edge_list = []
                        for edge_key, edge_data in edges.items():
                            try:
                                # Parse edge key "(src, dst)"
                                inner = edge_key.strip()[1:-1]
                                src_s, dst_s = inner.split(",")
                                src_id = int(src_s.strip())
                                dst_id = int(dst_s.strip())
                                
                                edge_list.append({
                                    'source': nodes.get(src_id, {}).get('name', f'node_{src_id}'),
                                    'target': nodes.get(dst_id, {}).get('name', f'node_{dst_id}'),
                                    'weight': float(edge_data.get('weight', 0.0)),
                                    'stability': float(edge_data.get('stability', 0.0)),
                                    'domain': int(edge_data.get('domain', 0))
                                })
                            except (ValueError, TypeError, KeyError):
                                continue
                        
                        if edge_list:
                            agent_registry.update_edges(edge_list)

                        sim_config = SimulationConfig()
                        self.simulation = SimulationEngine(
                            registry=agent_registry,
                            config=sim_config,
                            bus=self.bus,
                        )
                        initialized_components.append("simulation")
                    else:
                        logger.warning("Cannot initialize simulation: MPIE store not available")
                except Exception as e:
                    logger.error(f"Failed to initialize Simulation: {e}")
                    self.simulation = None
            else:
                logger.info("Simulation disabled by configuration")
                self.simulation = None
            
            self._initialized = True
            logger.info(f"ScarcityCoreManager initialization complete. Components: {initialized_components}")
            
        except Exception as e:
            logger.error(f"Critical failure during ScarcityCoreManager initialization: {e}", exc_info=True)
            # Attempt cleanup of partially initialized components
            await self._cleanup_partial_init()
            raise
    
    async def start(self) -> None:
        """
        Start all components.
        
        Must be called after initialize().
        """
        if not self._initialized:
            raise RuntimeError("Cannot start ScarcityCoreManager before initialization")
        
        if self._started:
            logger.warning("ScarcityCoreManager already started")
            return
        
        try:
            # Start components in order
            logger.info("Starting scarcity core components...")
            
            if self.mpie:
                await self.mpie.start()
                logger.info("MPIE Orchestrator started")
            
            if self.drg:
                await self.drg.start()
                logger.info("Dynamic Resource Governor started")
            
            if self.federation_client:
                await self.federation_client.start()
                logger.info("Federation Client Agent started")
            
            if self.meta:
                await self.meta.start()
                logger.info("Meta Learning Agent started")
            
            if self.simulation:
                await self.simulation.start()
                logger.info("Simulation Engine started")
            
            # Start domain data store
            if self.domain_data_store:
                await self.domain_data_store.start()
                logger.info("Domain Data Store started")
            
            # Start multi-domain data generation
            if self.multi_domain_generator:
                await self.multi_domain_generator.start()
                logger.info("Multi-Domain Data Generator started")
            
            self._started = True
            logger.info("All scarcity core components started")
            
        except Exception as e:
            logger.error(f"Failed to start ScarcityCoreManager: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """
        Gracefully shutdown all components.
        
        Shutdown in reverse order of startup.
        """
        if not self._started:
            logger.warning("ScarcityCoreManager not started, nothing to stop")
            return
        
        try:
            logger.info("Stopping scarcity core components...")
            
            # Stop in reverse order
            if self.multi_domain_generator:
                await self.multi_domain_generator.stop()
                logger.info("Multi-Domain Data Generator stopped")
            
            if self.domain_data_store:
                await self.domain_data_store.stop()
                logger.info("Domain Data Store stopped")
            
            if self.simulation:
                await self.simulation.stop()
                logger.info("Simulation Engine stopped")
            
            if self.meta:
                await self.meta.stop()
                logger.info("Meta Learning Agent stopped")
            
            if self.federation_client:
                await self.federation_client.stop()
                logger.info("Federation Client Agent stopped")
            
            if self.drg:
                await self.drg.stop()
                logger.info("Dynamic Resource Governor stopped")
            
            if self.mpie:
                await self.mpie.stop()
                logger.info("MPIE Orchestrator stopped")
            
            if self.bus:
                await self.bus.shutdown()
                logger.info("Runtime Bus shut down")
            
            self._started = False
            logger.info("All scarcity core components stopped")
            
        except Exception as e:
            logger.error(f"Error during ScarcityCoreManager shutdown: {e}", exc_info=True)
            raise
    
    async def _cleanup_partial_init(self) -> None:
        """Clean up partially initialized components."""
        logger.info("Cleaning up partially initialized components...")
        
        try:
            if self.simulation and self.simulation._running:
                await self.simulation.stop()
        except Exception as e:
            logger.error(f"Error stopping simulation during cleanup: {e}")
        
        try:
            if self.meta and self.meta._running:
                await self.meta.stop()
        except Exception as e:
            logger.error(f"Error stopping meta during cleanup: {e}")
        
        try:
            if self.federation_client and self.federation_client._running:
                await self.federation_client.stop()
        except Exception as e:
            logger.error(f"Error stopping federation during cleanup: {e}")
        
        try:
            if self.drg and self.drg._running:
                await self.drg.stop()
        except Exception as e:
            logger.error(f"Error stopping DRG during cleanup: {e}")
        
        try:
            if self.mpie and self.mpie.running:
                await self.mpie.stop()
        except Exception as e:
            logger.error(f"Error stopping MPIE during cleanup: {e}")
        
        try:
            if self.bus:
                await self.bus.shutdown(timeout=2.0)
        except Exception as e:
            logger.error(f"Error shutting down bus during cleanup: {e}")
    
    def get_status(self) -> dict:
        """
        Get status of all components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            "initialized": self._initialized,
            "started": self._started,
            "components": {
                "runtime_bus": "online" if self.bus else "offline",
                "domain_manager": "online" if self.domain_manager else "offline",
                "multi_domain_generator": "online" if self.multi_domain_generator and self.multi_domain_generator.running else "offline",
                "domain_data_store": "online" if self.domain_data_store and self.domain_data_store._running else "offline",
                "mpie": "online" if self.mpie and self.mpie.running else "offline",
                "drg": "online" if self.drg and self.drg._running else "offline",
                "federation_coordinator": "online" if self.federation_coordinator else "offline",
                "federation_coordinator_v2": "online" if self.federation_coordinator_v2 and self.federation_coordinator_v2.active else "offline",
                "federation_client": "online" if self.federation_client and self.federation_client._running else "offline",
                "meta": "online" if self.meta and self.meta._running else "offline",
                "simulation": "online" if self.simulation and self.simulation._running else "offline",
            },
            "domains": len(self.domain_manager.list_domains()) if self.domain_manager else 0
        }
    
    def _subscribe_to_core_topics(self) -> None:
        """
        Subscribe to core topics for telemetry collection.
        
        Subscribes to:
        - data_window: Data ingestion events
        - processing_metrics: MPIE processing metrics
        - resource_profile: DRG resource monitoring
        - federation_update: Federation learning updates
        - meta_update: Meta learning updates
        - simulation_state: Simulation state changes
        """
        if not self.bus:
            logger.warning("Cannot subscribe to topics: bus not initialized")
            return
        
        core_topics = [
            "data_window",
            "processing_metrics",
            "resource_profile",
            "federation_update",
            "meta_update",
            "simulation_state",
        ]
        
        for topic in core_topics:
            self.bus.subscribe(topic, self._telemetry_callback)
            self._subscribed_topics.append(topic)
        
        logger.info(f"Subscribed to {len(core_topics)} core topics for telemetry")
    
    async def _telemetry_callback(self, topic: str, data: Any) -> None:
        """
        Callback for telemetry events from the bus.
        
        Args:
            topic: Event topic
            data: Event payload
        """
        telemetry_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "topic": topic,
            "data": data
        }
        self._telemetry_history.append(telemetry_event)
        logger.debug(f"Captured telemetry event on topic '{topic}'")
    
    def get_bus_statistics(self) -> Dict[str, Any]:
        """
        Get Runtime Bus statistics.
        
        Returns:
            Dictionary with bus stats including message counts and active topics
        """
        if not self.bus:
            return {"error": "Bus not initialized"}
        
        return self.bus.get_stats()
    
    def get_active_topics(self) -> List[str]:
        """
        Get list of active topics on the bus.
        
        Returns:
            List of topic names with active subscribers
        """
        if not self.bus:
            return []
        
        return self.bus.topics()
    
    def get_telemetry_history(self, limit: Optional[int] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get telemetry event history.
        
        Args:
            limit: Maximum number of events to return (most recent first)
            topic: Filter by specific topic (optional)
        
        Returns:
            List of telemetry events
        """
        events = list(self._telemetry_history)
        
        # Filter by topic if specified
        if topic:
            events = [e for e in events if e["topic"] == topic]
        
        # Reverse to get most recent first
        events.reverse()
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events

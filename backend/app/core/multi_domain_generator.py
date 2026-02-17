"""Multi-domain synthetic data generation."""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging

from app.core.domain_manager import DomainManager, Domain, DistributionType, DomainStatus

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates synthetic data for a specific distribution type."""
    
    def __init__(self, distribution_type: DistributionType, params: Dict[str, float], features: int = 32):
        self.distribution_type = distribution_type
        self.params = params
        self.features = features
    
    def generate(self, window_size: int) -> np.ndarray:
        """Generate one window of data."""
        if self.distribution_type == DistributionType.NORMAL:
            return self._generate_normal(window_size)
        elif self.distribution_type == DistributionType.SKEWED:
            return self._generate_skewed(window_size)
        elif self.distribution_type == DistributionType.BIMODAL:
            return self._generate_bimodal(window_size)
        else:
            return self._generate_normal(window_size)
    
    def _generate_normal(self, window_size: int) -> np.ndarray:
        """Generate normally distributed data."""
        mean = self.params.get("mean", 0.0)
        std = self.params.get("std", 1.0)
        return np.random.normal(mean, std, (window_size, self.features))
    
    def _generate_skewed(self, window_size: int) -> np.ndarray:
        """Generate skewed (log-normal) data."""
        shape = self.params.get("shape", 2.0)
        scale = self.params.get("scale", 1.0)
        return np.random.lognormal(shape, scale, (window_size, self.features))
    
    def _generate_bimodal(self, window_size: int) -> np.ndarray:
        """Generate bimodal (mixture of Gaussians) data."""
        mean1 = self.params.get("mean1", -1.0)
        std1 = self.params.get("std1", 0.5)
        mean2 = self.params.get("mean2", 1.0)
        std2 = self.params.get("std2", 0.5)
        weight = self.params.get("weight", 0.5)
        
        # Generate from two Gaussians
        n1 = int(window_size * weight)
        n2 = window_size - n1
        
        data1 = np.random.normal(mean1, std1, (n1, self.features))
        data2 = np.random.normal(mean2, std2, (n2, self.features))
        
        # Combine and shuffle
        data = np.vstack([data1, data2])
        np.random.shuffle(data)
        return data


class DomainScheduler:
    """Schedules staggered data generation across domains."""
    
    def __init__(self):
        self.tasks: Dict[int, asyncio.Task] = {}
        self.intervals: Dict[int, float] = {}
    
    def calculate_offset(self, domain_id: int, num_domains: int, base_interval: float) -> float:
        """Calculate time offset for staggered generation."""
        if num_domains <= 1:
            return 0.0
        return (domain_id * base_interval) / num_domains
    
    async def schedule_domain(
        self,
        domain_id: int,
        interval: float,
        offset: float,
        callback
    ):
        """Schedule periodic generation for a domain."""
        # Wait for offset
        if offset > 0:
            await asyncio.sleep(offset)
        
        while True:
            try:
                await callback(domain_id)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating data for domain {domain_id}: {e}")
                await asyncio.sleep(interval)
    
    def start_domain(self, domain_id: int, interval: float, offset: float, callback):
        """Start generation task for domain."""
        if domain_id in self.tasks:
            self.stop_domain(domain_id)
        
        task = asyncio.create_task(
            self.schedule_domain(domain_id, interval, offset, callback)
        )
        self.tasks[domain_id] = task
        self.intervals[domain_id] = interval
    
    def stop_domain(self, domain_id: int):
        """Stop generation task for domain."""
        if domain_id in self.tasks:
            self.tasks[domain_id].cancel()
            del self.tasks[domain_id]
            del self.intervals[domain_id]
    
    def stop_all(self):
        """Stop all generation tasks."""
        for task in self.tasks.values():
            task.cancel()
        self.tasks.clear()
        self.intervals.clear()


class MultiDomainDataGenerator:
    """Generates synthetic data for multiple domains with distinct distributions."""
    
    def __init__(self, domain_manager: DomainManager, bus=None):
        self.domain_manager = domain_manager
        self.bus = bus
        self.generators: Dict[int, DataGenerator] = {}
        self.scheduler = DomainScheduler()
        self.running = False
        self.window_size = 100
        self.features = 32
    
    async def start(self):
        """Start generating data for all active domains."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting multi-domain data generation")
        
        # Start generation for all active domains
        domains = self.domain_manager.list_domains()
        num_domains = len(domains)
        
        for domain in domains:
            if domain.status == DomainStatus.ACTIVE and domain.synthetic_enabled:
                await self._start_domain_generation(domain, num_domains)
    
    async def stop(self):
        """Stop all data generation."""
        if not self.running:
            return
        
        self.running = False
        self.scheduler.stop_all()
        logger.info("Stopped multi-domain data generation")
    
    async def _start_domain_generation(self, domain: Domain, num_domains: int):
        """Start generation for a specific domain."""
        # Create generator if not exists
        if domain.id not in self.generators:
            self.generators[domain.id] = DataGenerator(
                domain.distribution_type,
                domain.distribution_params,
                self.features
            )
        
        # Calculate staggered offset
        base_interval = 5.0  # Default 5 seconds
        offset = self.scheduler.calculate_offset(domain.id, num_domains, base_interval)
        
        # Start scheduled generation
        self.scheduler.start_domain(
            domain.id,
            base_interval,
            offset,
            self._generate_and_publish
        )
        
        logger.info(f"Started generation for domain {domain.id} ({domain.name}) with offset {offset:.2f}s")
    
    async def _generate_and_publish(self, domain_id: int):
        """Generate one window of data for domain and publish to bus."""
        domain = self.domain_manager.get_domain(domain_id)
        if domain is None or domain.status != DomainStatus.ACTIVE:
            return
        
        # Generate data
        data = await self.generate_for_domain(domain_id)
        
        # Update domain statistics
        domain.total_windows += 1
        domain.last_data_at = datetime.utcnow()
        
        # Calculate scarcity signal (random for now, could be based on data characteristics)
        scarcity_signal = float(np.random.uniform(0.0, 1.0))
        
        # Publish to bus if available
        if self.bus:
            await self.bus.publish("data_window", {
                "data": data,
                "domain_id": domain_id,
                "domain_name": domain.name,
                "window_id": domain.total_windows,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "synthetic",
                "scarcity_signal": scarcity_signal
            })
        
        logger.debug(f"Generated window {domain.total_windows} for domain {domain.name}")
    
    async def generate_for_domain(self, domain_id: int) -> np.ndarray:
        """
        Generate one window of data for specific domain.
        
        Args:
            domain_id: Domain ID to generate for
        
        Returns:
            Generated data array
        
        Raises:
            ValueError: If domain not found
        """
        domain = self.domain_manager.get_domain(domain_id)
        if domain is None:
            raise ValueError(f"Domain {domain_id} not found")
        
        # Get or create generator
        if domain_id not in self.generators:
            self.generators[domain_id] = DataGenerator(
                domain.distribution_type,
                domain.distribution_params,
                self.features
            )
        
        # Generate data
        generator = self.generators[domain_id]
        data = generator.generate(self.window_size)
        
        return data
    
    def set_interval(self, domain_id: int, interval_seconds: float):
        """
        Configure generation interval for domain.
        
        Args:
            domain_id: Domain ID
            interval_seconds: Interval in seconds
        """
        if domain_id in self.scheduler.tasks:
            # Restart with new interval
            domain = self.domain_manager.get_domain(domain_id)
            if domain:
                self.scheduler.stop_domain(domain_id)
                num_domains = len(self.domain_manager.list_domains())
                offset = self.scheduler.calculate_offset(domain_id, num_domains, interval_seconds)
                self.scheduler.start_domain(
                    domain_id,
                    interval_seconds,
                    offset,
                    self._generate_and_publish
                )
    
    def pause_domain(self, domain_id: int):
        """Pause generation for domain."""
        self.scheduler.stop_domain(domain_id)
        logger.info(f"Paused generation for domain {domain_id}")
    
    def resume_domain(self, domain_id: int):
        """Resume generation for domain."""
        domain = self.domain_manager.get_domain(domain_id)
        if domain and domain.status == DomainStatus.ACTIVE:
            num_domains = len(self.domain_manager.list_domains())
            asyncio.create_task(self._start_domain_generation(domain, num_domains))
            logger.info(f"Resumed generation for domain {domain_id}")

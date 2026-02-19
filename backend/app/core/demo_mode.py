"""
Demo Mode - Accelerates data generation for impressive live demos.

This module temporarily speeds up data generation and processing
to make demos more visually impressive while keeping everything real.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DemoMode:
    """Manages demo mode state and acceleration."""
    
    def __init__(self):
        self.active = False
        self.original_intervals = {}
        self.acceleration_factor = 5  # 5x faster
    
    def activate(self, scarcity_manager):
        """
        Activate demo mode - speeds up data generation.
        
        Args:
            scarcity_manager: The ScarcityCoreManager instance
        """
        if self.active:
            logger.warning("Demo mode already active")
            return
        
        logger.info("🎬 Activating DEMO MODE - Accelerating data generation")
        
        # Speed up multi-domain generator
        if scarcity_manager.multi_domain_generator:
            generator = scarcity_manager.multi_domain_generator
            
            # Store original intervals
            for domain_id, interval in generator.scheduler.intervals.items():
                self.original_intervals[domain_id] = interval
            
            # Accelerate each domain
            for domain_id in list(generator.scheduler.intervals.keys()):
                original_interval = generator.scheduler.intervals[domain_id]
                new_interval = original_interval / self.acceleration_factor
                generator.set_interval(domain_id, new_interval)
                logger.info(f"Domain {domain_id}: {original_interval}s → {new_interval}s")
        
        self.active = True
        logger.info("✅ Demo mode activated - Data flowing 5x faster!")
    
    def deactivate(self, scarcity_manager):
        """
        Deactivate demo mode - restore normal speed.
        
        Args:
            scarcity_manager: The ScarcityCoreManager instance
        """
        if not self.active:
            logger.warning("Demo mode not active")
            return
        
        logger.info("🎬 Deactivating DEMO MODE - Restoring normal speed")
        
        # Restore original intervals
        if scarcity_manager.multi_domain_generator:
            generator = scarcity_manager.multi_domain_generator
            
            for domain_id, original_interval in self.original_intervals.items():
                generator.set_interval(domain_id, original_interval)
                logger.info(f"Domain {domain_id}: Restored to {original_interval}s")
        
        self.original_intervals.clear()
        self.active = False
        logger.info("✅ Demo mode deactivated - Normal operation resumed")
    
    def get_status(self) -> dict:
        """Get demo mode status."""
        return {
            "active": self.active,
            "acceleration_factor": self.acceleration_factor if self.active else 1,
            "accelerated_domains": len(self.original_intervals)
        }


# Global demo mode instance
_demo_mode = DemoMode()


def get_demo_mode() -> DemoMode:
    """Get the global demo mode instance."""
    return _demo_mode

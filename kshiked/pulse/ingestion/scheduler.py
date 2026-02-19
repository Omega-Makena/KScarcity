"""
Scheduler for KShield Pulse Ingestion

APScheduler-based job scheduling for:
- Social media scraping (every 30 minutes)
- E-commerce price collection (every 6 hours)
- LLM batch processing (hourly)
- Report generation (daily)

Usage:
    scheduler = IngestionScheduler(orchestrator)
    await scheduler.start()
    
    # Runs continuously until stopped
    await scheduler.wait()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Any

logger = logging.getLogger("kshield.pulse.ingestion.scheduler")


# =============================================================================
# Job Configuration
# =============================================================================

class JobConfig:
    """Configuration for a scheduled job."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        interval_minutes: int = 30,
        enabled: bool = True,
        max_instances: int = 1,
    ):
        self.name = name
        self.func = func
        self.interval_minutes = interval_minutes
        self.enabled = enabled
        self.max_instances = max_instances


# =============================================================================
# Scheduler
# =============================================================================

class IngestionScheduler:
    """
    Manages scheduled ingestion jobs.
    
    Uses asyncio for scheduling instead of APScheduler
    to avoid extra dependencies. Simple but effective.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize scheduler.
        
        Args:
            orchestrator: IngestionOrchestrator instance.
        """
        self.orchestrator = orchestrator
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._stats: dict[str, Any] = {}
    
    async def start(self) -> None:
        """Start all scheduled jobs."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting ingestion scheduler...")
        
        # Start social media job (every 30 minutes)
        self._tasks["social_media"] = asyncio.create_task(
            self._run_periodic(
                name="social_media",
                func=self._scrape_social_media,
                interval_minutes=30,
            )
        )
        
        # Start e-commerce job (every 6 hours)
        self._tasks["ecommerce"] = asyncio.create_task(
            self._run_periodic(
                name="ecommerce",
                func=self._scrape_ecommerce,
                interval_minutes=360,  # 6 hours
            )
        )
        
        # Start processing job (hourly)
        self._tasks["processing"] = asyncio.create_task(
            self._run_periodic(
                name="processing",
                func=self._process_signals,
                interval_minutes=60,
            )
        )
        
        logger.info(f"Started {len(self._tasks)} scheduled jobs")
    
    async def stop(self) -> None:
        """Stop all scheduled jobs."""
        self._running = False
        
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._tasks.clear()
        logger.info("Scheduler stopped")
    
    async def wait(self) -> None:
        """Wait for scheduler to complete (runs indefinitely)."""
        while self._running:
            await asyncio.sleep(1)
    
    async def _run_periodic(
        self,
        name: str,
        func: Callable,
        interval_minutes: int,
    ) -> None:
        """Run a function periodically."""
        interval_seconds = interval_minutes * 60
        
        while self._running:
            start_time = datetime.utcnow()
            
            try:
                logger.info(f"Running job: {name}")
                await func()
                
                self._stats[name] = {
                    "last_run": start_time.isoformat(),
                    "status": "success",
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                }
                
            except Exception as e:
                logger.error(f"Job {name} failed: {e}")
                self._stats[name] = {
                    "last_run": start_time.isoformat(),
                    "status": "error",
                    "error": str(e),
                }
            
            # Wait for next interval
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            wait_time = max(0, interval_seconds - elapsed)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    async def _scrape_social_media(self) -> None:
        """Job: Scrape social media platforms."""
        await self.orchestrator.scrape_social_media()
    
    async def _scrape_ecommerce(self) -> None:
        """Job: Scrape e-commerce for prices."""
        await self.orchestrator.scrape_ecommerce()
    
    async def _process_signals(self) -> None:
        """Job: Process unprocessed posts through LLM."""
        # Get unprocessed posts
        db = self.orchestrator._db
        if not db:
            return
        
        posts = await db.get_unprocessed_posts(limit=100)
        
        if not posts:
            logger.debug("No unprocessed posts to analyze")
            return
        
        logger.info(f"Processing {len(posts)} posts through LLM")
        
        # TODO: Integrate with GeminiProvider for batch classification
        # For now, just mark as processed
        post_ids = [p.id for p in posts]
        await db.mark_posts_processed(post_ids)
    
    async def run_now(self, job_name: str) -> bool:
        """
        Manually trigger a job immediately.
        
        Args:
            job_name: Name of job to run.
            
        Returns:
            True if job was triggered.
        """
        if job_name == "social_media":
            await self._scrape_social_media()
            return True
        elif job_name == "ecommerce":
            await self._scrape_ecommerce()
            return True
        elif job_name == "processing":
            await self._process_signals()
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get job statistics."""
        return dict(self._stats)
    
    def get_next_run_times(self) -> dict[str, datetime]:
        """Get next scheduled run time for each job."""
        # Simplified - would need to track actual next run times
        return {
            "social_media": datetime.utcnow() + timedelta(minutes=30),
            "ecommerce": datetime.utcnow() + timedelta(hours=6),
            "processing": datetime.utcnow() + timedelta(hours=1),
        }


# =============================================================================
# Standalone Runner
# =============================================================================

async def run_scheduler():
    """Run scheduler as standalone process."""
    from .orchestrator import IngestionOrchestrator, IngestionConfig
    
    config = IngestionConfig.from_env()
    
    async with IngestionOrchestrator(config) as orchestrator:
        scheduler = IngestionScheduler(orchestrator)
        await scheduler.start()
        
        try:
            await scheduler.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down scheduler...")
            await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(run_scheduler())

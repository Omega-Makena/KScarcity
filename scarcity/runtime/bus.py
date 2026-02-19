"""
Event Bus — central pub/sub communication fabric for SCARCITY.

This module implements an asynchronous event-driven message broker that enables
decoupled communication between all system components.

Algorithmic approach: Reactive Message Broker optimized for intra-process communication.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Set
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class EventBus:
    """
    Asynchronous pub/sub event broker for SCARCITY runtime.
    
    Features:
    - Non-blocking concurrent dispatch
    - Topic-based message routing
    - Automatic error isolation (one subscriber failure doesn't affect others)
    - Graceful shutdown support
    """
    
    def __init__(self):
        """Initialize the event bus with empty subscriber registry."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = True
        self._tasks: Set[asyncio.Task] = set()
        self._stats = {
            'messages_published': 0,
            'messages_delivered': 0,
            'delivery_errors': 0,
            'topics_active': 0
        }
        logger.info("EventBus initialized")
    
    async def publish(self, topic: str, data: Any) -> None:
        """
        Publish an event to all subscribers of a topic.
        
        Args:
            topic: Message topic (e.g., "data_window", "telemetry")
            data: Payload to deliver to subscribers
            
        Raises:
            RuntimeError: If bus is shutting down or shut down
        """
        if not self._running:
            raise RuntimeError("EventBus is shutting down or shut down")
        
        # Increment stats
        self._stats['messages_published'] += 1
        
        # Get subscribers for this topic (if any)
        subscribers = self._subscribers.get(topic, [])
        
        if not subscribers:
            logger.debug(f"No subscribers for topic '{topic}'")
            return
        
        # Dispatch to all subscribers concurrently
        for callback in subscribers:
            task = asyncio.create_task(self._dispatch(callback, topic, data))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        logger.debug(f"Published to topic '{topic}' with {len(subscribers)} subscribers")
    
    async def _dispatch(self, callback: Callable, topic: str, data: Any) -> None:
        """
        Dispatch event to a single subscriber with error handling.
        
        Args:
            callback: Subscriber coroutine
            topic: Topic name
            data: Payload
        """
        try:
            # Call the subscriber's callback
            if asyncio.iscoroutinefunction(callback):
                await callback(topic, data)
            else:
                callback(topic, data)
            
            self._stats['messages_delivered'] += 1
        except Exception as e:
            self._stats['delivery_errors'] += 1
            logger.error(
                f"Error delivering message to subscriber for topic '{topic}': {e}",
                exc_info=True
            )
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Register a subscriber callback for a topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Callable that will receive (topic, data) as arguments
            
        Example:
            async def my_handler(topic, data):
                print(f"Received {data} on {topic}")
            
            bus.subscribe("data_window", my_handler)
        """
        self._subscribers[topic].append(callback)
        self._stats['topics_active'] = len(self._subscribers)
        logger.info(f"Subscribed to topic '{topic}'")
    
    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        """
        Remove a subscriber from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            callback: Callback to remove
            
        Returns:
            True if removed, False if not found
        """
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
                
                # Clean up empty topics
                if not self._subscribers[topic]:
                    del self._subscribers[topic]
                    self._stats['topics_active'] = len(self._subscribers)
                
                logger.info(f"Unsubscribed from topic '{topic}'")
                return True
            except ValueError:
                logger.warning(f"Callback not found in subscribers for topic '{topic}'")
                return False
        return False
    
    def topics(self) -> List[str]:
        """
        Get list of all active topics.
        
        Returns:
            List of topic names that have at least one subscriber
        """
        return list(self._subscribers.keys())

    def get_stats(self) -> Dict[str, int]:
        """
        Get current bus statistics.
        
        Returns:
            Dictionary with message counts and active topics
        """
        return {
            'messages_published': self._stats['messages_published'],
            'messages_delivered': self._stats['messages_delivered'],
            'delivery_errors': self._stats['delivery_errors'],
            'topics_active': len(self._subscribers),
            'total_subscribers': sum(len(callbacks) for callbacks in self._subscribers.values())
        }

    async def wait_for_idle(self) -> None:
        """
        Wait until all in-flight subscriber tasks complete.
        """
        pending = [task for task in list(self._tasks) if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    
    async def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the bus.
        
        Waits for all pending tasks to complete or timeout.
        
        Args:
            timeout: Maximum seconds to wait for tasks to complete
        """
        logger.info("Shutting down EventBus...")
        self._running = False
        
        # Wait for all pending tasks
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Some tasks did not complete within {timeout}s timeout")
                # Cancel remaining tasks
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
        
        self._tasks.clear()
        self._subscribers.clear()
        logger.info("EventBus shut down complete")


# Global singleton instance
_global_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """
    Get or create the global EventBus instance.
    
    Returns:
        Global EventBus singleton
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_bus() -> None:
    """Reset the global bus (useful for testing)."""
    global _global_bus
    _global_bus = None

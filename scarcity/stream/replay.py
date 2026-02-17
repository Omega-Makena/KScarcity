"""
ReplayManager — Fault-tolerant replay controller.

Provides recovery from crashes or dropped connections using append-only log
with offset tracking and consistency checksums.
"""

import logging
import json
import os
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logging.warning("aiofiles not available, using synchronous file I/O")

logger = logging.getLogger(__name__)


class ReplayManager:
    """
    Fault-tolerant replay controller.
    
    Features:
    - Append-only log for events
    - Offset tracking
    - Consistency checksums
    - Selective replay (domain, timestamp)
    """
    
    def __init__(self, log_file: str = "logs/stream/replay.log", checkpoint_dir: str = "logs/stream/checkpoints"):
        """
        Initialize replay manager.
        
        Args:
            log_file: Path to append-only log file
            checkpoint_dir: Directory for checkpoints
        """
        self.log_file = log_file
        self.checkpoint_dir = checkpoint_dir
        self.last_offset = 0
        self.heartbeat_timeout = 30.0  # seconds
        self.last_heartbeat = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"ReplayManager initialized: log_file={log_file}")
    
    async def log_event(self, event: Dict[str, Any]) -> int:
        """
        Append event to log.
        
        Args:
            event: Event data
            
        Returns:
            Offset (byte position) in log
        """
        # Add metadata
        log_entry = {
            'offset': self.last_offset,
            'timestamp': datetime.utcnow().isoformat(),
            'checksum': self._compute_checksum(event),
            'event': event
        }
        
        # Write to log
        line = json.dumps(log_entry) + "\n"
        offset = self.last_offset
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(line)
        else:
            with open(self.log_file, 'a') as f:
                f.write(line)
        
        self.last_offset += len(line.encode('utf-8'))
        self.last_heartbeat = datetime.utcnow()
        
        return offset
    
    async def replay_events(self, start_offset: int = 0, end_offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Replay events from log.
        
        Args:
            start_offset: Starting byte offset
            end_offset: Ending byte offset (None = to end)
            
        Returns:
            List of events
        """
        events = []
        current_offset = 0
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.log_file, 'r') as f:
                    async for line in f:
                        if current_offset >= start_offset and (end_offset is None or current_offset < end_offset):
                            try:
                                log_entry = json.loads(line)
                                
                                # Verify checksum
                                if self._verify_checksum(log_entry):
                                    events.append(log_entry['event'])
                                else:
                                    logger.error(f"Checksum mismatch at offset {current_offset}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse log entry: {e}")
                        
                        current_offset += len(line.encode('utf-8'))
            else:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if current_offset >= start_offset and (end_offset is None or current_offset < end_offset):
                            try:
                                log_entry = json.loads(line)
                                
                                # Verify checksum
                                if self._verify_checksum(log_entry):
                                    events.append(log_entry['event'])
                                else:
                                    logger.error(f"Checksum mismatch at offset {current_offset}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse log entry: {e}")
                        
                        current_offset += len(line.encode('utf-8'))
        
        except FileNotFoundError:
            logger.info("No replay log found")
        
        return events
    
    async def save_checkpoint(self, offset: int, metadata: Optional[Dict] = None) -> str:
        """
        Save checkpoint.
        
        Args:
            offset: Current log offset
            metadata: Optional metadata
            
        Returns:
            Checkpoint file path
        """
        checkpoint = {
            'offset': offset,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.json")
        
        checkpoint_str = json.dumps(checkpoint, indent=2)
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(checkpoint_str)
        else:
            with open(checkpoint_file, 'w') as f:
                f.write(checkpoint_str)
        
        logger.info(f"Checkpoint saved: offset={offset}, file={checkpoint_file}")
        return checkpoint_file
    
    async def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load most recent checkpoint.
        
        Returns:
            Checkpoint data or None
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        if not checkpoints:
            return None
        
        latest = sorted(checkpoints)[-1]
        checkpoint_file = os.path.join(self.checkpoint_dir, latest)
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(checkpoint_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            else:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    async def recover(self) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Recover from last checkpoint.
        
        Returns:
            Tuple of (last_offset, events to replay)
        """
        checkpoint = await self.load_latest_checkpoint()
        
        if checkpoint:
            start_offset = checkpoint['offset']
            logger.info(f"Recovering from checkpoint: offset={start_offset}")
        else:
            start_offset = 0
            logger.info("No checkpoint found, starting from beginning")
        
        events = await self.replay_events(start_offset=start_offset)
        self.last_offset = checkpoint['offset'] if checkpoint else 0
        
        logger.info(f"Recovery complete: loaded {len(events)} events")
        return self.last_offset, events
    
    def check_heartbeat(self) -> bool:
        """
        Check if heartbeat is recent.
        
        Returns:
            True if heartbeat is recent, False if timeout
        """
        if self.last_heartbeat is None:
            return False
        
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed < self.heartbeat_timeout
    
    def _compute_checksum(self, data: Dict) -> str:
        """Compute checksum for data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _verify_checksum(self, log_entry: Dict) -> bool:
        """Verify checksum of log entry."""
        expected = log_entry.get('checksum')
        actual = self._compute_checksum(log_entry.get('event', {}))
        return expected == actual
    
    def get_stats(self) -> Dict:
        """Get replay statistics."""
        return {
            'log_file': self.log_file,
            'last_offset': self.last_offset,
            'heartbeat_ok': self.check_heartbeat(),
            'log_size_bytes': os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        }

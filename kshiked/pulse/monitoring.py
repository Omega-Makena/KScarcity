"""
Pulse Engine Monitoring Manager (Sentinel Roadmap v2.0)

Manages persistent monitoring targets (Topics, Actors, Locations).
Handles:
- Target CRUD (Create, Read, Update, Delete)
- Expiry Logic (7-day default)
- JSON Persistence
- HITL Review (Expired Targets)
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import asdict

from .llm.signals import MonitoringTarget

logger = logging.getLogger(__name__)

class MonitoringManager:
    """
    Manages key intelligence targets (KITs) for the Ingestor.
    """
    def __init__(self, storage_path: str = "monitoring_targets.json"):
        self.storage_path = storage_path
        self.targets: Dict[str, MonitoringTarget] = {}
        self._load()

    def add_target(self, identifier: str, target_type: str, reason: str, days: int = 7) -> MonitoringTarget:
        """Add or update a monitoring target."""
        target = MonitoringTarget(
            identifier=identifier,
            target_type=target_type,
            reason=reason,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=days),
            active=True
        )
        self.targets[identifier] = target
        self._save()
        logger.info(f"Added monitoring target: {identifier} (Expires: {target.expires_at})")
        return target

    def get_target(self, identifier: str) -> Optional[MonitoringTarget]:
        """Get a target by identifier."""
        return self.targets.get(identifier)

    def get_active_targets(self) -> List[MonitoringTarget]:
        """Get all currently active, non-expired targets."""
        now = datetime.now()
        active = []
        for t in self.targets.values():
            if t.active and t.expires_at > now:
                active.append(t)
            elif t.active and t.expires_at <= now:
                # Auto-mark as inactive if expired? 
                # Roadmap says they enter "EXPIRED" state for review.
                # For this getter, we only return TRUE active ones.
                pass
        return active

    def get_expired_targets(self) -> List[MonitoringTarget]:
        """Get targets that have expired but are still marked active (pending review)."""
        now = datetime.now()
        return [t for t in self.targets.values() if t.active and t.expires_at <= now]

    def renew_target(self, identifier: str, days: int = 7) -> bool:
        """Renew a target for N days."""
        target = self.targets.get(identifier)
        if not target:
            return False
        
        target.renew(days)
        self._save()
        logger.info(f"Renewed target: {identifier} until {target.expires_at}")
        return True

    def drop_target(self, identifier: str) -> bool:
        """Deactivate a target."""
        target = self.targets.get(identifier)
        if not target:
            return False
        
        target.drop()
        self._save()
        logger.info(f"Dropped target: {identifier}")
        return True

    def _save(self):
        """Persist to JSON."""
        data = {
            k: asdict(v) for k, v in self.targets.items()
        }
        # Handle datetime serialization
        def dt_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, default=dt_serializer, indent=2)
        except Exception as e:
            logger.error(f"Failed to save targets: {e}")

    def _load(self):
        """Load from JSON."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            for k, v in data.items():
                # Deserialize datetimes
                if 'created_at' in v:
                    v['created_at'] = datetime.fromisoformat(v['created_at'])
                if 'expires_at' in v:
                    v['expires_at'] = datetime.fromisoformat(v['expires_at'])
                
                self.targets[k] = MonitoringTarget(**v)
        except Exception as e:
            logger.error(f"Failed to load targets: {e}")

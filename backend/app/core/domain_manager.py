"""Domain management for multi-domain simulation."""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DomainStatus(str, Enum):
    """Domain status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"


class DistributionType(str, Enum):
    """Data distribution types."""
    NORMAL = "normal"
    SKEWED = "skewed"
    BIMODAL = "bimodal"


@dataclass
class Domain:
    """Domain model representing a client/organization."""
    id: int
    name: str
    distribution_type: DistributionType
    distribution_params: Dict[str, float]
    status: DomainStatus = DomainStatus.ACTIVE
    synthetic_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_data_at: Optional[datetime] = None
    
    # Statistics
    total_windows: int = 0
    manual_uploads: int = 0
    federation_rounds: int = 0


# Predefined domain names pool
DOMAIN_NAMES = [
    "Healthcare",
    "Finance",
    "Retail",
    "Economics",
    "Manufacturing",
    "Energy",
    "Technology",
    "Pharmaceuticals",
    "Transportation",
    "Agriculture",
    "Education",
    "Telecommunications",
]


class DomainManager:
    """Manages domain lifecycle, naming, and configuration."""
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.domains: Dict[int, Domain] = {}
        self.name_registry: Dict[str, int] = {}
        self.next_id: int = 0
        self.available_names: List[str] = DOMAIN_NAMES.copy()
        
        # Persistence
        if persistence_path:
            self.persistence_path = Path(persistence_path)
        else:
            self.persistence_path = Path("data/domains.json")
        
        # Load existing domains if available
        self._load_from_disk()
    
    def create_domain(
        self,
        name: Optional[str] = None,
        distribution_type: DistributionType = DistributionType.NORMAL,
        distribution_params: Optional[Dict[str, float]] = None
    ) -> Domain:
        """
        Create a new named domain with specified data distribution.
        
        Args:
            name: Domain name (auto-assigned if None)
            distribution_type: Type of data distribution
            distribution_params: Distribution parameters (mean, std, etc.)
        
        Returns:
            Created Domain instance
        
        Raises:
            ValueError: If name already exists
        """
        # Assign name
        if name is None:
            if not self.available_names:
                name = f"Domain-{self.next_id}"
            else:
                name = self.available_names.pop(0)
        
        # Check for name conflict
        if name in self.name_registry:
            raise ValueError(f"Domain name '{name}' already exists")
        
        # Set default distribution params
        if distribution_params is None:
            distribution_params = self._get_default_params(distribution_type)
        
        # Create domain
        domain = Domain(
            id=self.next_id,
            name=name,
            distribution_type=distribution_type,
            distribution_params=distribution_params
        )
        
        # Register
        self.domains[self.next_id] = domain
        self.name_registry[name] = self.next_id
        self.next_id += 1
        
        # Persist
        self._save_to_disk()
        
        return domain
    
    def get_domain(self, domain_id: int) -> Optional[Domain]:
        """Retrieve domain by ID."""
        return self.domains.get(domain_id)
    
    def get_domain_by_name(self, name: str) -> Optional[Domain]:
        """Retrieve domain by name."""
        domain_id = self.name_registry.get(name)
        if domain_id is not None:
            return self.domains.get(domain_id)
        return None
    
    def pause_domain(self, domain_id: int) -> None:
        """
        Pause synthetic data generation for domain.
        
        Args:
            domain_id: Domain ID to pause
        
        Raises:
            ValueError: If domain not found or already paused
        """
        domain = self.get_domain(domain_id)
        if domain is None:
            raise ValueError(f"Domain {domain_id} not found")
        
        if domain.status == DomainStatus.PAUSED:
            raise ValueError(f"Domain {domain_id} is already paused")
        
        domain.status = DomainStatus.PAUSED
        domain.synthetic_enabled = False
        
        # Persist
        self._save_to_disk()
    
    def resume_domain(self, domain_id: int) -> None:
        """
        Resume synthetic data generation for domain.
        
        Args:
            domain_id: Domain ID to resume
        
        Raises:
            ValueError: If domain not found or not paused
        """
        domain = self.get_domain(domain_id)
        if domain is None:
            raise ValueError(f"Domain {domain_id} not found")
        
        if domain.status != DomainStatus.PAUSED:
            raise ValueError(f"Domain {domain_id} is not paused")
        
        domain.status = DomainStatus.ACTIVE
        domain.synthetic_enabled = True
        
        # Persist
        self._save_to_disk()
    
    def remove_domain(self, domain_id: int) -> None:
        """
        Remove domain and clean up resources.
        
        Args:
            domain_id: Domain ID to remove
        
        Raises:
            ValueError: If domain not found
        """
        domain = self.get_domain(domain_id)
        if domain is None:
            raise ValueError(f"Domain {domain_id} not found")
        
        # Mark as stopped
        domain.status = DomainStatus.STOPPED
        
        # Remove from registries
        del self.name_registry[domain.name]
        del self.domains[domain_id]
        
        # Return name to pool if it was from the pool
        if domain.name in DOMAIN_NAMES:
            self.available_names.append(domain.name)
        
        # Persist
        self._save_to_disk()
    
    def list_domains(self) -> List[Domain]:
        """List all domains with their status."""
        return list(self.domains.values())
    
    def _save_to_disk(self) -> None:
        """Save domain configuration to disk."""
        try:
            # Create directory if it doesn't exist
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize domains
            data = {
                "next_id": self.next_id,
                "available_names": self.available_names,
                "domains": [
                    {
                        **asdict(domain),
                        "distribution_type": domain.distribution_type.value,
                        "status": domain.status.value,
                        "created_at": domain.created_at.isoformat(),
                        "last_data_at": domain.last_data_at.isoformat() if domain.last_data_at else None
                    }
                    for domain in self.domains.values()
                ]
            }
            
            # Write to file
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.domains)} domains to {self.persistence_path}")
        
        except Exception as e:
            logger.error(f"Failed to save domains to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load domain configuration from disk."""
        if not self.persistence_path.exists():
            logger.info(f"No persistence file found at {self.persistence_path}")
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            # Restore state
            self.next_id = data.get("next_id", 0)
            self.available_names = data.get("available_names", DOMAIN_NAMES.copy())
            
            # Restore domains
            for domain_data in data.get("domains", []):
                domain = Domain(
                    id=domain_data["id"],
                    name=domain_data["name"],
                    distribution_type=DistributionType(domain_data["distribution_type"]),
                    distribution_params=domain_data["distribution_params"],
                    status=DomainStatus(domain_data["status"]),
                    synthetic_enabled=domain_data["synthetic_enabled"],
                    created_at=datetime.fromisoformat(domain_data["created_at"]),
                    last_data_at=datetime.fromisoformat(domain_data["last_data_at"]) if domain_data["last_data_at"] else None,
                    total_windows=domain_data["total_windows"],
                    manual_uploads=domain_data["manual_uploads"],
                    federation_rounds=domain_data["federation_rounds"]
                )
                
                self.domains[domain.id] = domain
                self.name_registry[domain.name] = domain.id
            
            logger.info(f"Loaded {len(self.domains)} domains from {self.persistence_path}")
        
        except Exception as e:
            logger.error(f"Failed to load domains from disk: {e}")
    
    def save(self) -> None:
        """Public method to trigger save."""
        self._save_to_disk()
    
    def _get_default_params(self, distribution_type: DistributionType) -> Dict[str, float]:
        """Get default parameters for distribution type."""
        if distribution_type == DistributionType.NORMAL:
            return {"mean": 0.0, "std": 1.0}
        elif distribution_type == DistributionType.SKEWED:
            return {"shape": 2.0, "scale": 1.0}
        elif distribution_type == DistributionType.BIMODAL:
            return {"mean1": -1.0, "std1": 0.5, "mean2": 1.0, "std2": 0.5, "weight": 0.5}
        else:
            return {"mean": 0.0, "std": 1.0}

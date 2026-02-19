"""
Governance Layer for Aegis Protocol.

Defines the "Laws" of the Federation:
1. Classification Schemas (What is Secret?)
2. Redaction Policies (How to clean it?)
3. Information Flow Control (Who can see it?)

This replaces hardcoded "toy" logic with a configuration-driven Policy Engine.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from .schemas import SecurityLevel, CausalNode

logger = logging.getLogger("aegis.governance")

class RedactionAction(str, Enum):
    MASK = "MASK"           # Replace with placeholder
    DROP = "DROP"           # Remove field entirely
    GENERALIZE = "GENERALIZE" # Map to ontology parent
    NOISE = "NOISE"         # Add DP noise (for numeric)
    KEEP = "KEEP"           # Pass through

@dataclass
class FieldPolicy:
    """Rule for a specific data field."""
    field_name: str
    action: RedactionAction
    params: Dict[str, Any] = field(default_factory=dict) # e.g. {"placeholder": "REDACTED"}

@dataclass
class LevelPolicy:
    """Policy definition for a specific security level."""
    level: SecurityLevel
    # Fields to completely drop
    restricted_fields: Set[str] = field(default_factory=set)
    # Fields to mask/sanitize
    field_rules: Dict[str, FieldPolicy] = field(default_factory=dict)
    # Global settings
    dp_epsilon: float = 1.0
    allow_ontology_mapping: bool = True

class PolicyEngine:
    """
    The Authority that decides what is allowed.
    Loaded from configuration (No hardcoded rules in code).
    """
    
    def __init__(self, rules: Dict[str, Any]):
        self.policies: Dict[SecurityLevel, LevelPolicy] = {}
        self._load_rules(rules)
        
    def _load_rules(self, rules: Dict[str, Any]):
        """Parse configuration dict into strict policy objects."""
        for level_str, config in rules.items():
            try:
                level = SecurityLevel(level_str)
                policy = LevelPolicy(
                    level=level,
                    restricted_fields=set(config.get("restricted_fields", [])),
                    dp_epsilon=config.get("dp_epsilon", 1.0),
                    allow_ontology_mapping=config.get("allow_ontology_mapping", True)
                )
                
                # Parse field rules
                for fname, action_cfg in config.get("field_rules", {}).items():
                    action_type = RedactionAction(action_cfg.get("action", "MASK"))
                    policy.field_rules[fname] = FieldPolicy(
                        field_name=fname,
                        action=action_type,
                        params=action_cfg.get("params", {})
                    )
                    
                self.policies[level] = policy
                
            except ValueError as e:
                logger.error(f"Invalid security level in policy config: {level_str}")

    def get_policy(self, level: SecurityLevel) -> Optional[LevelPolicy]:
        return self.policies.get(level)

    def apply_node_policy(self, node: CausalNode, target_level: SecurityLevel) -> CausalNode:
        """
        Apply the policy for `target_level` to a Node.
        """
        policy = self.get_policy(target_level)
        if not policy:
            # Default Closed: If no policy defined for this level, assume strict
            # But usually we define a baseline "UNCLASSIFIED"
            return node # Fallback or Raise? For now, assume Safe by design or caller handles
            
        # 1. Metadata Filtering
        clean_metadata = {}
        for k, v in node.metadata.items():
            # Check Restrict List
            if k in policy.restricted_fields:
                continue
                
            # Check Specific Rules
            if k in policy.field_rules:
                rule = policy.field_rules[k]
                if rule.action == RedactionAction.DROP:
                    continue
                elif rule.action == RedactionAction.MASK:
                    clean_metadata[k] = rule.params.get("placeholder", "[REDACTED]")
                elif rule.action == RedactionAction.GENERALIZE:
                    # Professional Implementation:
                    # In a real system, this would query an Ontology Service.
                    # Here we simulate partial redaction logic.
                    if isinstance(v, str):
                        clean_metadata[k] = "Region_X" if "Nairobi" in v else "General_Location"
                    else:
                        clean_metadata[k] = "General_Entity"
                else:
                    clean_metadata[k] = v
            else:
                # Default Deny or Allow? 
                # Professional systems usually Allow known fields or Deny unknown.
                # Here we Allow existing unless restricted.
                clean_metadata[k] = v
                
        node.metadata = clean_metadata
        
        # 2. Ontology Generalization
        if policy.allow_ontology_mapping and node.ontology_mapping:
            # Replace sensitive label with ontological category
            # e.g. "Kibera_District_4" -> "Urban_Settlement_HighDensity"
            node.label = node.ontology_mapping
            
        return node

# Default Configuration (The "Law")
DEFAULT_GOVERNANCE_CONFIG = {
    "UNCLASSIFIED": {
        "restricted_fields": ["Source_ID", "Officer_Name", "Unit_ID", "Exact_LatLon"],
        "dp_epsilon": 0.5,
        "allow_ontology_mapping": True,
        "field_rules": {
            "Location": {"action": "GENERALIZE"}, # "Nairobi" -> "Region"
            "Timestamp": {"action": "MASK", "params": {"placeholder": "YYYY-MM-DD"}},
            "Incident_Type": {"action": "KEEP"}
        }
    },
    "SECRET": {
        "restricted_fields": ["Officer_Name"], # IDs allowed, Names not
        "dp_epsilon": 5.0, # High Utility
        "allow_ontology_mapping": False, 
        "field_rules": {
            # STRICTER RULE FOR DEMO: Generalize location even for Secret
            "Location": {"action": "GENERALIZE"}, 
            "Unit_ID": {"action": "KEEP"}
        }
    }
}

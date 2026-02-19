"""
Coarse-to-Fine Grouping Logic.

Manages the lifecycle of VariableGroups: 
- Initialization (Coarse grouping)
- Splitting (High residual pressure)
- Merging (High interaction strength)
"""

from __future__ import annotations

import numpy as np
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .robustness import OnlineMAD

logger = logging.getLogger(__name__)

@dataclass
class VariableGroup:
    """
    A set of variables treated as a single unit or cluster.

    Groups allow the system to reason about macro-level relationships (e.g., "Macro Factors" -> "Asset A")
    before drilling down into specific variable-to-variable links.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variables: Set[str] = field(default_factory=set)
    
    # Internal statistics
    variance_stats: OnlineMAD = field(default_factory=OnlineMAD)
    residual_stats: OnlineMAD = field(default_factory=OnlineMAD)
    
    def update_stats(self, row: Dict[str, float]) -> None:
        """
        Updates the internal variance statistics of the group based on member values.
        
        Args:
            row: The current data row.
        """
        if not self.variables:
            return
            
        # Simplified: average value of group members
        vals = [row[v] for v in self.variables if v in row]
        if not vals:
            return
            
        mean_val = float(np.mean(vals))
        self.variance_stats.update(mean_val)
        
    def add_residual(self, error: float) -> None:
        """
        Tracks unexplained variance (prediction error) for this group.
        
        High residuals indicate that the group may naturally be incoherent and
        should satisfy the split criteria.
        
        Args:
            error: The prediction error magnitude.
        """
        self.residual_stats.update(error)

class AdaptiveGrouper:
    """
    Manages the dynamic hierarchical clustering of variables.
    
    The Grouper monitors model performance on variable groups. If a group cannot
    be well-predicted (high residuals), it is "shattered" (split) into smaller
    components or atomic variables to allow for more fine-grained modeling.
    """
    def __init__(self, split_threshold: float = 1.0, minimize_groups: bool = False):
        """
        Args:
            split_threshold: The residual error threshold that triggers a group split.
            minimize_groups: If True, attempts to merge groups more aggressively.
        """
        self.groups: Dict[str, VariableGroup] = {}
        self.var_to_group: Dict[str, str] = {}
        self.split_threshold = split_threshold
        
    def initialize(self, variable_names: List[str]) -> None:
        """
        Initializes the grouping strategy.
        
        By default, starts with atomic groups (one per variable) for maximum
        resolution in 'Online Relationship Discovery' mode.
        """
        # For 'Online Relationship Discovery', start with atomic groups (1 var per group)
        # unless specifically asked to compress.
        for var in variable_names:
            self._create_group({var})
            
    def _create_group(self, variables: Set[str]) -> VariableGroup:
        group = VariableGroup(variables=variables)
        self.groups[group.id] = group
        for v in variables:
            self.var_to_group[v] = group.id
        return group
        
    def monitor(self, row: Dict[str, float], hypothesis_errors: Dict[str, float]) -> None:
        """
        Checks for split/merge triggers based on recent hypothesis performance.
        
        If a group has consistently high errors, it is marked for splitting.
        
        Args:
            row: The current data row.
            hypothesis_errors: A map of GroupID -> Error from the current best hypothesis.
        """
        keys_to_split = []
        
        for gid, error in hypothesis_errors.items():
            if gid in self.groups:
                group = self.groups[gid]
                group.add_residual(error)
                
                # Check split condition: consistently high residual
                if group.residual_stats.median > self.split_threshold:
                    keys_to_split.append(gid)
        
        for gid in keys_to_split:
            self._split_group(gid)
            
    def _split_group(self, group_id: str) -> None:
        """
        Shatters a group back into atomic variables or smaller subgroups.
        """
        if group_id not in self.groups:
            return
            
        group = self.groups.pop(group_id)
        logger.info(f"Splitting group {group_id} (vars={group.variables}) due to high residual")
        
        # Determine how to split. Simple strategy: Atomize all.
        for var in group.variables:
            self._create_group({var})

    def get_group_id(self, variable: str) -> Optional[str]:
        """Returns the ID of the group containing the specified variable."""
        return self.var_to_group.get(variable)

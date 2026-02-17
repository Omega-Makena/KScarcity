"""
SchemaManager — Schema validation and evolution tracking.

Maintains variable metadata, validates data shape, tracks schema lineage,
and handles non-breaking evolution of feature schemas.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FieldMetadata:
    """Metadata for a single field/variable."""
    name: str
    dtype: str
    unit: Optional[str] = None
    domain: Optional[str] = None  # e.g., 'finance', 'agriculture'
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SchemaVersion:
    """A schema version with metadata."""
    version_uuid: str
    fields: List[FieldMetadata]
    created_at: str
    hash: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'version_uuid': self.version_uuid,
            'fields': [f.to_dict() for f in self.fields],
            'created_at': self.created_at,
            'hash': self.hash
        }


class SchemaManager:
    """
    Manage schema metadata and track evolution.
    
    Features:
    - Field typing and validation
    - UUID-based schema versioning
    - Auto-update on new variables
    - Schema lineage tracking
    - Migration support
    """
    
    def __init__(self, schema_history_file: Optional[str] = None):
        """
        Initialize schema manager.
        
        Args:
            schema_history_file: Path to store schema history JSON
        """
        self.schema_history_file = schema_history_file
        self.current_schema: Optional[SchemaVersion] = None
        self.schema_history: List[SchemaVersion] = []
        self.field_mapping: Dict[str, int] = {}  # field_name -> column_index
        
        logger.info("SchemaManager initialized")
    
    def infer_schema(self, data: np.ndarray, field_names: Optional[List[str]] = None) -> SchemaVersion:
        """
        Infer schema from data.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            field_names: Optional list of field names
            
        Returns:
            SchemaVersion object
        """
        n_features = data.shape[1] if data.ndim > 1 else 1
        
        # Generate field names if not provided
        if field_names is None:
            field_names = [f"feature_{i}" for i in range(n_features)]
        
        # Infer dtypes
        fields = []
        for i, name in enumerate(field_names):
            dtype_str = str(data.dtype) if data.ndim == 1 else str(data.dtype)
            fields.append(FieldMetadata(
                name=name,
                dtype=dtype_str,
                domain=None,
                description=f"Automatically inferred field {i}"
            ))
        
        # Compute hash
        schema_hash = self._compute_schema_hash(fields)
        
        # Check if this is a new schema
        if self.current_schema and self.current_schema.hash == schema_hash:
            logger.debug("Schema unchanged")
            return self.current_schema
        
        # Create new schema version
        version_uuid = hashlib.md5(schema_hash.encode()).hexdigest()[:16]
        
        schema = SchemaVersion(
            version_uuid=version_uuid,
            fields=fields,
            created_at=datetime.utcnow().isoformat(),
            hash=schema_hash
        )
        
        # Update current schema
        self.current_schema = schema
        
        # Add to history
        if not any(s.hash == schema_hash for s in self.schema_history):
            self.schema_history.append(schema)
            logger.info(f"New schema version created: {version_uuid}")
        
        # Build field mapping
        self.field_mapping = {field.name: i for i, field in enumerate(fields)}
        
        # Save history
        if self.schema_history_file:
            self._save_history()
        
        return schema
    
    def validate_data(self, data: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Validate data against current schema.
        
        Args:
            data: Data array to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.current_schema is None:
            return True, None
        
        expected_features = len(self.current_schema.fields)
        
        if data.ndim == 1:
            actual_features = 1
        else:
            actual_features = data.shape[1]
        
        if actual_features != expected_features:
            error = f"Feature count mismatch: expected {expected_features}, got {actual_features}"
            logger.warning(error)
            return False, error
        
        return True, None
    
    def get_schema_diff(self, old_schema: SchemaVersion, new_schema: SchemaVersion) -> Dict[str, Any]:
        """
        Compute difference between two schemas.
        
        Args:
            old_schema: Previous schema
            new_schema: New schema
            
        Returns:
            Dictionary describing changes
        """
        old_fields = {f.name for f in old_schema.fields}
        new_fields = {f.name for f in new_schema.fields}
        
        added = new_fields - old_fields
        removed = old_fields - new_fields
        common = old_fields & new_fields
        
        # Check for type changes
        type_changes = []
        for field_name in common:
            old_field = next(f for f in old_schema.fields if f.name == field_name)
            new_field = next(f for f in new_schema.fields if f.name == field_name)
            if old_field.dtype != new_field.dtype:
                type_changes.append(field_name)
        
        return {
            'added_fields': list(added),
            'removed_fields': list(removed),
            'type_changes': type_changes,
            'breaking': len(removed) > 0 or len(type_changes) > 0
        }
    
    def _compute_schema_hash(self, fields: List[FieldMetadata]) -> str:
        """Compute hash of schema fields."""
        field_str = json.dumps([f.to_dict() for f in fields], sort_keys=True)
        return hashlib.md5(field_str.encode()).hexdigest()
    
    def _save_history(self) -> None:
        """Save schema history to file."""
        try:
            history_dict = {
                'schemas': [s.to_dict() for s in self.schema_history],
                'current_version': self.current_schema.version_uuid if self.current_schema else None
            }
            
            with open(self.schema_history_file, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            logger.debug(f"Schema history saved to {self.schema_history_file}")
        except Exception as e:
            logger.error(f"Failed to save schema history: {e}")
    
    def load_history(self) -> None:
        """Load schema history from file."""
        if not self.schema_history_file:
            return
        
        try:
            with open(self.schema_history_file, 'r') as f:
                history_dict = json.load(f)
            
            # Reconstruct schemas
            for schema_dict in history_dict.get('schemas', []):
                fields = [FieldMetadata(**f) for f in schema_dict['fields']]
                schema = SchemaVersion(
                    version_uuid=schema_dict['version_uuid'],
                    fields=fields,
                    created_at=schema_dict['created_at'],
                    hash=schema_dict['hash']
                )
                self.schema_history.append(schema)
            
            # Set current schema
            current_version_uuid = history_dict.get('current_version')
            if current_version_uuid:
                self.current_schema = next(
                    (s for s in self.schema_history if s.version_uuid == current_version_uuid),
                    None
                )
            
            logger.info(f"Loaded {len(self.schema_history)} schema versions")
        except FileNotFoundError:
            logger.info("No existing schema history found")
        except Exception as e:
            logger.error(f"Failed to load schema history: {e}")
    
    def get_current_schema(self) -> Optional[SchemaVersion]:
        """Get current schema."""
        return self.current_schema
    
    def get_field_index(self, field_name: str) -> Optional[int]:
        """Get column index for a field name."""
        return self.field_mapping.get(field_name)

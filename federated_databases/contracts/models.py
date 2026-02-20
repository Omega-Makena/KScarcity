"""Dataset contracts for federated queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


CLASSIFICATION_ORDER = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "RESTRICTED": 2,
    "SECRET": 3,
}


@dataclass
class DataContract:
    dataset_id: str
    schema: Dict[str, str]
    classification: str = "INTERNAL"
    pii_fields: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=lambda: ["aggregate", "time_bucket"])
    approved_join_keys: List[str] = field(default_factory=list)

    def normalized(self) -> Dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "schema": dict(self.schema),
            "classification": self.classification.upper(),
            "pii_fields": sorted(set(self.pii_fields)),
            "allowed_operations": sorted({op.lower() for op in self.allowed_operations}),
            "approved_join_keys": sorted(set(self.approved_join_keys)),
        }

import enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class Role(str, enum.Enum):
    EXECUTIVE = "EXECUTIVE"
    BASKET_ADMIN = "BASKET_ADMIN"
    INSTITUTION = "INSTITUTION"

@dataclass
class User:
    id: int
    username: str
    password_hash: str
    role: Role
    basket_id: Optional[int]
    institution_id: Optional[int]

@dataclass
class Basket:
    id: int
    name: str
    description: str

@dataclass
class Institution:
    id: int
    name: str
    basket_id: int
    api_key: str

@dataclass
class OntologySchema:
    id: int
    basket_id: int
    schema_definition: str  # JSON string containing the required semantic dictionary

@dataclass
class DeltaQueueMessage:
    id: int
    institution_id: int
    basket_id: int
    payload: str  # JSON string of structural insights (Hypergraph edges)
    status: str   # 'PENDING' or 'PROCESSED'
    timestamp: float

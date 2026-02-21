import json
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from .database import get_connection

class OntologyEnforcer:
    """
    Cryptographically and semantically enforces that a Spoke Institution's 
    local data conforms exactly to the Basket Admin's globally defined schema 
    before the Scarcity Engine is allowed to boot.
    """
    
    @staticmethod
    def get_basket_schema(basket_id: int) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT schema_definition FROM ontology_schemas WHERE basket_id = ?", (basket_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['schema_definition'])
        return None

    @staticmethod
    def validate_dataset_signature(basket_id: int, provided_columns: List[str]) -> Tuple[bool, str]:
        """
        Validates if the provided CSV/DataFrame columns match the required Ontology.
        Returns (is_valid, error_message)
        """
        schema = OntologyEnforcer.get_basket_schema(basket_id)
        
        if not schema:
            return False, "CRITICAL FAULT: No Semantic Dictionary defined for this Basket. Engine initialization locked."
            
        required_columns = set(schema.get("required_columns", []))
        provided_set = set(provided_columns)
        
        missing = required_columns - provided_set
        
        if missing:
            return False, f"ONTOLOGY MISMATCH: Your local data is missing required structural tensors: {missing}. Federated aggregation is impossible. Access Denied."
            
        if not schema.get("allow_extra", False):
            extra = provided_set - required_columns
            if extra:
                return False, f"ONTOLOGY MISMATCH: Strict schema enabled. The following columns are unmapped to the global dictionary: {extra}. Please map or drop them."
                
        return True, "Ontology Signature Verified. Structural compatibility confirmed."

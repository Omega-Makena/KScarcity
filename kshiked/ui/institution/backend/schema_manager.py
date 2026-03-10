import json
import time
import pandas as pd
from typing import List, Dict, Tuple, Any
from kshiked.ui.institution.backend.database import get_connection

class SchemaManager:
    """
    Manages the creation, retrieval, and validation of custom data schemas 
    defined by Admins for their Spoke users.
    """

    @staticmethod
    def save_schema(basket_id: int, schema_name: str, fields: List[Dict[str, Any]]) -> int:
        timestamp = time.time()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO data_schemas (basket_id, schema_name, fields_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (basket_id, schema_name, json.dumps(fields), timestamp, timestamp)
            )
            return cursor.lastrowid

    @staticmethod
    def get_schemas(basket_id: int) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM data_schemas WHERE basket_id = ? ORDER BY created_at DESC", (basket_id,))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                r['fields'] = json.loads(r['fields_json'])
                results.append(r)
            return results

    @staticmethod
    def validate_dataframe(schema_fields: List[Dict[str, Any]], df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates a pandas DataFrame against a list of schema fields.
        Returns (is_valid, error_message).
        """
        if df.empty:
            return False, "Uploaded dataset is empty."

        for field in schema_fields:
            name = field['name']
            ftype = field['type']
            required = field.get('required', False)

            if required and name not in df.columns:
                return False, f"Missing required column: '{name}'"

            if name in df.columns:
                # Basic type validation
                if ftype == 'number':
                    if not pd.api.types.is_numeric_dtype(df[name]):
                        # try converting
                        try:
                            df[name] = pd.to_numeric(df[name], errors='raise')
                        except ValueError:
                            return False, f"Column '{name}' must contain only numbers."
                elif ftype == 'date':
                    try:
                        pd.to_datetime(df[name], errors='raise')
                    except Exception:
                        return False, f"Column '{name}' must contain valid dates."
                elif ftype == 'bool':
                    if not pd.api.types.is_bool_dtype(df[name]):
                        return False, f"Column '{name}' must be boolean (True/False)."
                        
        return True, "Data matches the required schema."

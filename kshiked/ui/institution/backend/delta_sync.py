import sqlite3
import json
import time
from typing import List, Dict, Any
from .database import get_connection

class DeltaSyncManager:
    """
    Manages the Asynchronous Delta Queue. This allows isolated Spoke institutions 
    to continue generating causal insights indefinitely while offline, and automatically 
    pushes the cryptographic delta payloads to the Basket Admin upon reconnection.
    """

    @staticmethod
    def queue_insight(institution_id: int, basket_id: int, payload: Dict[str, Any]) -> int:
        """Called by the Spoke's local UI to log a discovered insight to the queue."""
        payload_str = json.dumps(payload)
        timestamp = time.time()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO delta_queue (institution_id, basket_id, payload, status, timestamp) VALUES (?, ?, ?, 'PENDING', ?)",
                (institution_id, basket_id, payload_str, timestamp)
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_pending_syncs(basket_id: int) -> List[Dict[str, Any]]:
        """Called by the Basket Admin's governance UI to pull all offline insights from Spokes."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, institution_id, payload, timestamp FROM delta_queue WHERE basket_id = ? AND status = 'PENDING' ORDER BY timestamp ASC",
                (basket_id,)
            )
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "sync_id": r['id'],
                    "institution_id": r['institution_id'],
                    "payload": json.loads(r['payload']),
                    "timestamp": r['timestamp']
                })
            return results

    @staticmethod
    def mark_synced(sync_ids: List[int]):
        """Called by the Basket Admin once the FL aggregation absorbs the deltas."""
        if not sync_ids:
            return
            
        with get_connection() as conn:
            cursor = conn.cursor()
            # Batch update the processed status
            placeholders = ",".join("?" for _ in sync_ids)
            cursor.execute(
                f"UPDATE delta_queue SET status = 'PROCESSED' WHERE id IN ({placeholders})",
                sync_ids
            )
            conn.commit()

    @staticmethod
    def reject_sync(sync_id: int, message: str):
        """Called by the Basket Admin to reject an insight and request more data from the Spoke."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE delta_queue SET status = 'REJECTED' WHERE id = ?",
                (sync_id,)
            )
            # Fetch current payload to append the rejection message
            cursor.execute("SELECT payload FROM delta_queue WHERE id = ?", (sync_id,))
            row = cursor.fetchone()
            if row:
                payload = json.loads(row['payload'])
                payload['admin_message'] = message
                cursor.execute(
                    "UPDATE delta_queue SET payload = ? WHERE id = ?",
                    (json.dumps(payload), sync_id)
                )
            conn.commit()

    @staticmethod
    def get_historical_syncs(basket_id: int) -> List[Dict[str, Any]]:
        """Called by the Basket Admin's governance UI to pull historically processed insights from Spokes."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, institution_id, payload, timestamp, status FROM delta_queue WHERE basket_id = ? AND status IN ('PROCESSED', 'REJECTED') ORDER BY timestamp DESC",
                (basket_id,)
            )
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "sync_id": r['id'],
                    "institution_id": r['institution_id'],
                    "payload": json.loads(r['payload']),
                    "timestamp": r['timestamp'],
                    "status": r['status']
                })
            return results
                 
    @staticmethod
    def promote_risk(basket_id: int, title: str, description: str, composite_scores: Dict[str, float], source_sync_ids: List[int]) -> int:
        """Called by the Basket Admin to fuse multiple Spoke events into a single Promoted Risk for the Executive."""
        scores_str = json.dumps(composite_scores)
        syncs_str = json.dumps(source_sync_ids)
        timestamp = time.time()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO validated_risks (basket_id, title, description, composite_scores, source_sync_ids, status, timestamp) VALUES (?, ?, ?, ?, ?, 'PROMOTED', ?)",
                (basket_id, title, description, scores_str, syncs_str, timestamp)
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_promoted_risks(basket_id: int = None) -> List[Dict[str, Any]]:
        """Called by the Executive UI to view validated risks from a specific Basket, or globally if None."""
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if basket_id is not None:
                cursor.execute(
                    "SELECT id, basket_id, title, description, composite_scores, source_sync_ids, status, timestamp FROM validated_risks WHERE basket_id = ? ORDER BY timestamp DESC",
                    (basket_id,)
                )
            else:
                cursor.execute(
                    "SELECT id, basket_id, title, description, composite_scores, source_sync_ids, status, timestamp FROM validated_risks ORDER BY timestamp DESC"
                )
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "risk_id": r['id'],
                    "basket_id": r['basket_id'],
                    "title": r['title'],
                    "description": r['description'],
                    "composite_scores": json.loads(r['composite_scores']),
                    "source_sync_ids": json.loads(r['source_sync_ids']),
                    "status": r['status'],
                    "timestamp": r['timestamp']
                })
            return results

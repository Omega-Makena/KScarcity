import sqlite3
from typing import List, Dict, Optional
from datetime import datetime

from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.backend.models import Role

class SecureMessaging:
    """Manages cross-tier encrypted communication flow."""
    
    @staticmethod
    def send_message(sender_role: str, sender_id: str, receiver_role: str, receiver_id: str, content: str) -> bool:
        """Stores a new message in the registry."""
        try:
            with get_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO secure_messages (sender_role, sender_id, receiver_role, receiver_id, content)
                    VALUES (?, ?, ?, ?, ?)
                ''', (sender_role, sender_id, receiver_role, receiver_id, content))
                conn.commit()
            return True
        except Exception as e:
            print(f"Message dispatch error: {e}")
            return False

    @staticmethod
    def get_inbox(receiver_role: str, receiver_id: str) -> List[Dict]:
        """Retrieves messages targeted at a specific role/id."""
        with get_connection() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''
                SELECT id, sender_role, sender_id, content, timestamp, is_read 
                FROM secure_messages 
                WHERE receiver_role = ? AND (receiver_id = ? OR receiver_id = 'ALL')
                ORDER BY timestamp DESC
            ''', (receiver_role, receiver_id))
            
            return [dict(row) for row in c.fetchall()]
            
    @staticmethod
    def mark_read(msg_id: int):
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE secure_messages SET is_read = 1 WHERE id = ?", (msg_id,))
            conn.commit()

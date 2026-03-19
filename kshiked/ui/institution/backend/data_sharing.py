import time
import json
from typing import List, Dict, Any, Optional

from kshiked.ui.institution.backend.database import get_connection

class DataSharingManager:
  """
  Handles Lateral (Admin-to-Admin) data sharing requests, grants, and revocations.
  Handles Downward (Executive-to-Admin-to-Spoke) directives and acknowledgments.
  Provides immutable audit logging for all interactions.
  """

  @staticmethod
  def log_audit(actor_id: str, action_type: str, target_entity: str, details: str):
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        INSERT INTO share_audit_log (actor_id, action_type, target_entity, details, timestamp)
        VALUES (?, ?, ?, ?, ?)
      """, (actor_id, action_type, target_entity, details, time.time()))
      conn.commit()

  # ---------------------------------------------------------
  # LATERAL SHARING (Admin <-> Admin)
  # ---------------------------------------------------------

  @staticmethod
  def create_share_request(requester_basket_id: int, target_basket_id: int, reason: str, data_scope: str):
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        INSERT INTO data_share_requests (requester_basket_id, target_basket_id, reason, data_scope, created_at)
        VALUES (?, ?, ?, ?, ?)
      """, (requester_basket_id, target_basket_id, reason, data_scope, time.time()))
      req_id = c.lastrowid
      conn.commit()
      
      DataSharingManager.log_audit(
        actor_id=f"Basket_{requester_basket_id}",
        action_type="REQUEST_SHARE",
        target_entity=f"Basket_{target_basket_id}",
        details=f"Scope: {data_scope}. Reason: {reason}"
      )
      return req_id

  @staticmethod
  def get_incoming_requests(basket_id: int) -> List[Dict]:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT req.*, b.name as requester_name 
        FROM data_share_requests req
        JOIN baskets b ON req.requester_basket_id = b.id
        WHERE req.target_basket_id = ? AND req.status = 'PENDING'
        ORDER BY req.created_at DESC
      """, (basket_id,))
      return [dict(row) for row in c.fetchall()]

  @staticmethod
  def get_outgoing_requests(basket_id: int) -> List[Dict]:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT req.*, b.name as target_name 
        FROM data_share_requests req
        JOIN baskets b ON req.target_basket_id = b.id
        WHERE req.requester_basket_id = ?
        ORDER BY req.created_at DESC
      """, (basket_id,))
      return [dict(row) for row in c.fetchall()]

  @staticmethod
  def resolve_request(request_id: int, status: str, granter_basket_id: int, duration_hours: int = 24):
    """status should be 'APPROVED' or 'REJECTED'"""
    with get_connection() as conn:
      c = conn.cursor()
      now = time.time()
      c.execute("UPDATE data_share_requests SET status = ?, resolved_at = ? WHERE id = ?", (status, now, request_id))
      
      if status == 'APPROVED':
        # Fetch details to create the active share
        c.execute("SELECT requester_basket_id, data_scope FROM data_share_requests WHERE id = ?", (request_id,))
        req = c.fetchone()
        
        expires_at = now + (duration_hours * 3600)
        c.execute("""
          INSERT INTO active_data_shares (request_id, granter_basket_id, grantee_basket_id, data_scope, expires_at, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
        """, (request_id, granter_basket_id, req['requester_basket_id'], req['data_scope'], expires_at, now))
        
        DataSharingManager.log_audit(f"Basket_{granter_basket_id}", "GRANT_SHARE", f"Basket_{req['requester_basket_id']}", f"Scope: {req['data_scope']}, Duration: {duration_hours}h")
      else:
        DataSharingManager.log_audit(f"Basket_{granter_basket_id}", "REJECT_SHARE", f"Request_{request_id}", "Request Denied")
      
      conn.commit()

  @staticmethod
  def get_active_shares_granted(basket_id: int) -> List[Dict]:
    """Shares I have granted to others."""
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT ads.*, b.name as grantee_name 
        FROM active_data_shares ads
        JOIN baskets b ON ads.grantee_basket_id = b.id
        WHERE ads.granter_basket_id = ? AND ads.expires_at > ?
      """, (basket_id, time.time()))
      return [dict(row) for row in c.fetchall()]

  @staticmethod
  def get_active_shares_received(basket_id: int) -> List[Dict]:
    """Shares others have granted to me."""
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT ads.*, b.name as granter_name 
        FROM active_data_shares ads
        JOIN baskets b ON ads.granter_basket_id = b.id
        WHERE ads.grantee_basket_id = ? AND ads.expires_at > ?
      """, (basket_id, time.time()))
      return [dict(row) for row in c.fetchall()]

  @staticmethod
  def revoke_share(share_id: int, basket_id: int):
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("UPDATE active_data_shares SET expires_at = ? WHERE id = ? AND granter_basket_id = ?", (time.time(), share_id, basket_id))
      DataSharingManager.log_audit(f"Basket_{basket_id}", "REVOKE_SHARE", f"Share_{share_id}", "Early revocation")
      conn.commit()

  # ---------------------------------------------------------
  # DOWNWARD DIRECTIVES (Executive -> Admin -> Spoke)
  # ---------------------------------------------------------

  @staticmethod
  def issue_directive(sender_role: str, sender_id: str, content: str, priority: str, 
            directive_type: str = "BENCHMARK_ENFORCEMENT", 
            target_basket_id: Optional[int] = None, 
            target_institution_id: Optional[int] = None,
            requires_ack: bool = True):
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        INSERT INTO downward_directives 
        (sender_role, sender_id, target_basket_id, target_institution_id, directive_type, content, priority, requires_ack, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      """, (sender_role, sender_id, target_basket_id, target_institution_id, directive_type, content, priority, requires_ack, time.time()))
      dir_id = c.lastrowid
      
      target_str = f"Basket_{target_basket_id}" if target_basket_id else f"Spoke_{target_institution_id}" if target_institution_id else "ALL"
      DataSharingManager.log_audit(sender_id, "ISSUE_DIRECTIVE", target_str, f"Type: {directive_type}, Priority: {priority}")
      conn.commit()
      return dir_id

  @staticmethod
  def get_directives_for_basket(basket_id: int) -> List[Dict]:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT * FROM downward_directives 
        WHERE (target_basket_id = ? OR target_basket_id IS NULL) 
         AND target_institution_id IS NULL
        ORDER BY created_at DESC
      """, (basket_id,))
      directives = [dict(row) for row in c.fetchall()]
      
      for d in directives:
        c.execute("SELECT * FROM directive_acknowledgments WHERE directive_id = ? AND acknowledger_id = ?", (d['id'], f"Basket_{basket_id}"))
        d['is_acknowledged'] = c.fetchone() is not None
        
      return directives

  @staticmethod
  def get_directives_for_spoke(institution_id: int) -> List[Dict]:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        SELECT * FROM downward_directives 
        WHERE target_institution_id = ? OR (target_basket_id IS NULL AND target_institution_id IS NULL)
        ORDER BY created_at DESC
      """, (institution_id,))
      directives = [dict(row) for row in c.fetchall()]
      
      for d in directives:
        c.execute("SELECT * FROM directive_acknowledgments WHERE directive_id = ? AND acknowledger_id = ?", (d['id'], f"Spoke_{institution_id}"))
        d['is_acknowledged'] = c.fetchone() is not None
        
      return directives

  @staticmethod
  def acknowledge_directive(directive_id: int, acknowledger_id: str):
    with get_connection() as conn:
      c = conn.cursor()
      # check if already acked
      c.execute("SELECT id FROM directive_acknowledgments WHERE directive_id = ? AND acknowledger_id = ?", (directive_id, acknowledger_id))
      if c.fetchone() is None:
        c.execute("INSERT INTO directive_acknowledgments (directive_id, acknowledger_id, acknowledged_at) VALUES (?, ?, ?)", 
             (directive_id, acknowledger_id, time.time()))
        DataSharingManager.log_audit(acknowledger_id, "ACKNOWLEDGE_DIRECTIVE", f"Directive_{directive_id}", "Compliance acknowledged")
        conn.commit()


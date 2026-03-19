import sqlite3
from pathlib import Path

def migrate_db():
  db_path = Path("C:/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/ui/institution/backend/federated_registry.sqlite")
  print(f"Migrating DB at: {db_path}")
  
  try:
    with sqlite3.connect(db_path) as conn:
      cursor = conn.cursor()
      
      # Check if columns already exist
      cursor.execute("PRAGMA table_info(users)")
      columns = [info[1] for info in cursor.fetchall()]
      
      if 'totp_secret' not in columns:
        print("Adding column: totp_secret")
        # Add totp_secret (nullable, because existing user initially won't have it)
        cursor.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT;")
      else:
        print("Column totp_secret already exists")

      if 'is_2fa_enabled' not in columns:
        print("Adding column: is_2fa_enabled")
        # Add is_2fa_enabled tracking flag
        cursor.execute("ALTER TABLE users ADD COLUMN is_2fa_enabled BOOLEAN DEFAULT 0;")
      else:
        print("Column is_2fa_enabled already exists")
        
      conn.commit()
      print("Migration successful.")
  except Exception as e:
    print(f"Error during migration: {e}")

if __name__ == "__main__":
  migrate_db()

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class StateManager:
    """Manages the state database for tracking synced conversations"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.db_path = vault_path / ".claude-vault" / "state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize state database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                uuid TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_synced TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON conversations(file_path)
        """)

        conn.commit()
        conn.close()

    def get_conversation(self, uuid: str) -> Optional[Dict]:
        """
        Get conversation state by UUID

        Args:
            uuid: Conversation UUID

        Returns:
            Dictionary with conversation state or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE uuid = ?", (uuid,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "uuid": row[0],
                "file_path": row[1],
                "content_hash": row[2],
                "last_synced": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
        return None

    def save_conversation(
        self, uuid: str, file_path: str, content_hash: str, metadata: Dict = None
    ):
        """
        Save or update conversation state

        Args:
            uuid: Conversation UUID
            file_path: Path to markdown file
            content_hash: SHA-256 hash of content
            metadata: Optional metadata dictionary
        """
        from datetime import datetime

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO conversations
            (uuid, file_path, content_hash, last_synced, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                uuid,
                file_path,
                content_hash,
                datetime.now().isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()

    def find_by_path(self, file_path: str) -> Optional[Dict]:
        """
        Find conversation by file path

        Args:
            file_path: Path to search for

        Returns:
            Dictionary with conversation state or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "uuid": row[0],
                "file_path": row[1],
                "content_hash": row[2],
                "last_synced": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
        return None

    def get_all_conversations(self) -> List[Dict]:
        """Get all tracked conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations")
        rows = cursor.fetchall()
        conn.close()

        conversations = []
        for row in rows:
            conversations.append(
                {
                    "uuid": row[0],
                    "file_path": row[1],
                    "content_hash": row[2],
                    "last_synced": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                }
            )
        return conversations

    def delete_conversation(self, uuid: str):
        """Delete conversation from state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE uuid = ?", (uuid,))
        conn.commit()
        conn.close()

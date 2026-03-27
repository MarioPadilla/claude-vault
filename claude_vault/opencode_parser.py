import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from .models import Conversation, Message


class OpenCodeParser:
    """Parser for OpenCode SQLite database (opencode.db)"""

    DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"

    def parse(self, db_path: Path) -> List[Conversation]:
        """
        Parse OpenCode sessions from opencode.db

        Args:
            db_path: Path to opencode.db file

        Returns:
            List of Conversation objects
        """
        if not db_path.exists():
            raise FileNotFoundError(f"OpenCode database not found: {db_path}")

        conversations: List[Conversation] = []
        conn = sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            # Get root sessions only (skip child/sub-sessions)
            try:
                sessions = conn.execute(
                    "SELECT * FROM session WHERE parent_id IS NULL ORDER BY time_created"
                ).fetchall()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                print(f"Warning: Could not read sessions from {db_path}: {e}")
                return conversations

            for session in sessions:
                try:
                    conv = self._parse_session(conn, session)
                    if conv:
                        conversations.append(conv)
                except Exception as e:
                    print(f"Warning: Failed to parse session {session['id']}: {e}")
                    continue
        finally:
            conn.close()

        return conversations

    def _parse_session(
        self, conn: sqlite3.Connection, session: sqlite3.Row
    ) -> Optional[Conversation]:
        """Parse a single session into a Conversation"""

        session_id = session["id"]
        title = session["title"]
        directory = session["directory"]

        # Get messages ordered by creation time
        messages_rows = conn.execute(
            "SELECT * FROM message WHERE session_id = ? ORDER BY time_created",
            (session_id,),
        ).fetchall()

        messages = []
        for msg_row in messages_rows:
            msg = self._parse_message(conn, msg_row)
            if msg:
                messages.append(msg)

        if not messages:
            return None

        # Parse timestamps (milliseconds → seconds)
        created_at = datetime.fromtimestamp(session["time_created"] / 1000)
        updated_at = datetime.fromtimestamp(session["time_updated"] / 1000)

        return Conversation(
            id=session_id,
            title=title,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            tags=self._extract_tags(title, messages, directory),
        )

    def _parse_message(
        self, conn: sqlite3.Connection, msg_row: sqlite3.Row
    ) -> Optional[Message]:
        """Parse a single message and its parts"""

        msg_data = json.loads(msg_row["data"])
        role = msg_data.get("role")

        if not role:
            return None

        # Get parts for this message ordered by creation time
        parts_rows = conn.execute(
            "SELECT * FROM part WHERE message_id = ? ORDER BY time_created",
            (msg_row["id"],),
        ).fetchall()

        content_parts = []
        for part_row in parts_rows:
            try:
                part_data = json.loads(part_row["data"])
                part_text = self._extract_part_content(part_data)
                if part_text:
                    content_parts.append(part_text)
            except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
                print(f"Warning: Failed to parse part {part_row['id']}: {e}")
                continue

        if not content_parts:
            return None

        content = "\n\n".join(content_parts)

        # Ensure content is valid UTF-8
        try:
            content = content.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            pass

        # Map role to claude-vault format
        if role == "user":
            role = "human"

        return Message(
            role=role,
            content=content.strip(),
            timestamp=datetime.fromtimestamp(msg_row["time_created"] / 1000),
            uuid=msg_row["id"],
        )

    def _extract_part_content(self, part_data: dict) -> Optional[str]:
        """Extract text content from a part"""

        part_type = part_data.get("type")

        if part_type == "text":
            text = part_data.get("text", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            text = text.strip()
            return text if text else None

        if part_type == "tool":
            tool_name = part_data.get("tool", "Tool")
            state = part_data.get("state", {})
            tool_input = state.get("input", {})
            tool_output = state.get("output", "")

            # Ensure tool_output is a string
            if isinstance(tool_output, bytes):
                tool_output = tool_output.decode("utf-8", errors="replace")
            elif not isinstance(tool_output, str):
                tool_output = str(tool_output)

            lines = [f"**[Tool: {tool_name}]**"]
            if tool_input:
                try:
                    safe_input = json.dumps(tool_input, indent=2).replace("```", "~~~")
                    lines.append(f"```json\n{safe_input}\n```")
                except (TypeError, ValueError):
                    lines.append(f"```\n{str(tool_input).replace('```', '~~~')}\n```")
            if tool_output:
                # Truncate long outputs
                output_preview = (
                    tool_output[:500] + "..." if len(tool_output) > 500 else tool_output
                )
                lines.append(f"```\n{output_preview.replace('```', '~~~')}\n```")
            return "\n".join(lines)

        if part_type == "reasoning":
            text = part_data.get("text", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            text = text.strip()
            if text:
                return f"**[Reasoning]**\n\n{text.replace('```', '~~~')}"
            return None

        # Skip other part types (step-start, step-finish, patch, file, etc.)
        return None

    def _extract_tags(
        self, title: str, messages: List[Message], directory: str
    ) -> List[str]:
        """Extract tags from session"""

        tags = ["opencode-session"]

        # Extract project name from directory
        if directory:
            path_parts = Path(directory).parts
            if len(path_parts) > 2:
                project_name = path_parts[-1]
                if project_name not in ["~", "home", "tmp", "Documents", "Desktop"]:
                    tags.append(f"project:{project_name}")

        # Keyword-based tags from title and first messages
        combined = f"{title.lower()} {' '.join([m.content[:200].lower() for m in messages[:3]])}"

        keywords = {
            "python": ["python", "py", "pip", "django", "flask"],
            "javascript": ["javascript", "js", "node", "npm", "react", "typescript"],
            "debugging": ["debug", "error", "bug", "fix"],
            "api": ["api", "rest", "graphql", "endpoint"],
            "database": ["database", "sql", "postgres", "mysql"],
            "testing": ["test", "testing", "pytest", "jest"],
            "git": ["git", "commit", "branch", "merge", "rebase"],
            "config": ["config", "configuration", "settings", "env"],
        }

        for tag, patterns in keywords.items():
            if any(pattern in combined for pattern in patterns):
                tags.append(tag)

        return tags[:6]

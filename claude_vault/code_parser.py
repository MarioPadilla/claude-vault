import json
from datetime import datetime
from pathlib import Path
from typing import List

from .models import Conversation, Message


class ClaudeCodeHistoryParser:
    """Parser for Claude Code History format (JSONL from .claude folder)"""

    def parse(self, claude_path: Path) -> List[Conversation]:
        """
        Parse Claude Code History from .claude folder or specific .jsonl file

        Args:
            claude_path: Path to .claude folder or a specific .jsonl file

        Returns:
            List of Conversation objects
        """
        conversations = []

        if claude_path.is_file() and claude_path.suffix == ".jsonl":
            # Parse single JSONL file
            conv = self._parse_session_file(claude_path)
            if conv:
                conversations.append(conv)
        elif claude_path.is_dir():
            # Parse entire .claude directory
            if claude_path.name == ".claude":
                # Look in projects subdirectory
                projects_dir = claude_path / "projects"
                if projects_dir.exists():
                    for jsonl_file in projects_dir.rglob("*.jsonl"):
                        try:
                            conv = self._parse_session_file(jsonl_file)
                            if conv:
                                conversations.append(conv)
                        except Exception as e:
                            print(f"Warning: Failed to parse {jsonl_file.name}: {e}")
            else:
                # Regular directory, look for .jsonl files
                for jsonl_file in claude_path.rglob("*.jsonl"):
                    if jsonl_file.name != "history.jsonl":
                        try:
                            conv = self._parse_session_file(jsonl_file)
                            if conv:
                                conversations.append(conv)
                        except Exception as e:
                            print(f"Warning: Failed to parse {jsonl_file.name}: {e}")

        return conversations

    def _parse_session_file(self, jsonl_path: Path) -> Conversation:
        """Parse a single session JSONL file"""

        entries = []
        session_id = None

        # Read all entries from JSONL file
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)

                    # Skip file-history-snapshot entries
                    if entry.get("type") == "file-history-snapshot":
                        continue

                    # Skip meta messages
                    if entry.get("isMeta"):
                        continue

                    # Skip command execution messages
                    content = entry.get("message", {}).get("content", "")
                    if isinstance(content, str) and (
                        "<command-name>" in content
                        or "<local-command-stdout>" in content
                    ):
                        continue

                    entries.append(entry)
                    if not session_id:
                        session_id = entry.get("sessionId")

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue

        if not entries:
            return None

        # Parse messages
        messages = []
        created_at = None
        updated_at = None
        project_path = None

        for entry in entries:
            entry_type = entry.get("type")

            if entry_type in ["user", "assistant"]:
                message = self._parse_message(entry)
                if message:
                    messages.append(message)

                # Track timestamps (convert milliseconds to seconds)
                timestamp = self._parse_timestamp(entry.get("timestamp"))
                if not created_at or timestamp < created_at:
                    created_at = timestamp
                if not updated_at or timestamp > updated_at:
                    updated_at = timestamp

                # Track project path
                if not project_path:
                    project_path = entry.get("cwd", "")

        if not messages:
            return None

        # Generate title from first user message
        title = self._generate_title(messages, project_path)

        # Default timestamps
        if not created_at:
            created_at = datetime.now()
        if not updated_at:
            updated_at = created_at

        return Conversation(
            id=session_id or jsonl_path.stem,
            title=title,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            tags=self._extract_tags(title, messages, project_path),
        )

    def _parse_message(self, entry: dict) -> Message:
        """Parse a message from JSONL entry"""

        msg_data = entry.get("message", {})
        role = msg_data.get("role")

        if not role:
            return None

        # Extract content
        content = self._extract_content(msg_data)

        if not content:
            return None

        # Handle errors as special messages
        if entry.get("error"):
            error_type = entry["error"]
            content = f"⚠️ **Error: {error_type}**\n\n{content}"

        return Message(
            role=role,
            content=content.strip(),
            timestamp=self._parse_timestamp(entry.get("timestamp")),
            uuid=entry.get("uuid"),
        )

    def _extract_content(self, msg_data: dict) -> str:
        """Extract text content from message"""

        content_parts = []

        # Handle string content directly
        if isinstance(msg_data.get("content"), str):
            return msg_data["content"]

        # Handle array content
        if isinstance(msg_data.get("content"), list):
            for item in msg_data["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            content_parts.append(text)
                    elif item.get("type") == "tool_use":
                        # Format tool use
                        tool_name = item.get("name", "Tool")
                        tool_input = item.get("input", {})
                        content_parts.append(
                            f"**[Tool: {tool_name}]**\n```json\n{json.dumps(tool_input, indent=2)}\n```"
                        )
                elif isinstance(item, str):
                    content_parts.append(item)

        return "\n\n".join(content_parts)

    def _parse_timestamp(self, timestamp) -> datetime:
        """Parse timestamp - handles both ISO string and milliseconds"""
        if not timestamp:
            return datetime.now()

        try:
            # Handle milliseconds timestamp (like 1767000268465)
            if isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp / 1000)

            # Handle ISO string
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        except Exception as e:
            print(f"Warning: Could not parse timestamp {timestamp}: {e}")

        return datetime.now()

    def _generate_title(self, messages: List[Message], project_path: str = None) -> str:
        """Generate title from first user message, avoiding user-specific paths"""

        # Get first user message
        for msg in messages:
            if msg.role == "user":
                # Take first 60 chars
                title = msg.content[:60].strip().replace("\n", " ")
                if len(msg.content) > 60:
                    title += "..."

                # Add project context only if it's an actual project (not home/user dir)
                if project_path:
                    path_parts = Path(project_path).parts

                    # Skip if it's just a user directory (e.g., /Users/username or /home/user)
                    if len(path_parts) > 2:  # More than just /Users/username
                        # Use the last meaningful part (the actual project name)
                        project_name = path_parts[-1]

                        # Skip generic names
                        if project_name not in [
                            "~",
                            "home",
                            "tmp",
                            "Documents",
                            "Desktop",
                        ]:
                            title = f"[{project_name}] {title}"

                return title

        return "Code Session"

    def _extract_tags(
        self, title: str, messages: List[Message], project_path: str = None
    ) -> List[str]:
        """Extract tags from conversation"""

        tags = ["code-session"]  # Always add this tag

        # Extract from title and content
        combined = f"{title.lower()} {' '.join([m.content[:200].lower() for m in messages[:3]])}"

        keywords = {
            "python": ["python", "py", "pip", "django", "flask"],
            "javascript": ["javascript", "js", "node", "npm", "react"],
            "debugging": ["debug", "error", "bug", "fix"],
            "api": ["api", "rest", "graphql", "endpoint"],
            "database": ["database", "sql", "postgres", "mysql"],
            "testing": ["test", "testing", "pytest", "jest"],
        }

        for tag, patterns in keywords.items():
            if any(pattern in combined for pattern in patterns):
                tags.append(tag)

        return tags[:6]

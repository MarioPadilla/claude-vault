import json
import re
import uuid as uuid_module
from datetime import datetime
from pathlib import Path
from typing import List

from .models import Conversation, Message


class ClaudeExportParser:
    """Parser for Claude's conversation export format (JSON)"""

    def parse(self, export_path: Path) -> List[Conversation]:
        """
        Parse Claude export file and return list of conversations

        Args:
            export_path: Path to the conversations.json export file

        Returns:
            List of Conversation objects
        """
        # Read the JSON file
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)

        conversations = []

        # Handle both list and single conversation formats
        if isinstance(data, dict):
            data = [data]

        for conv_data in data:
            try:
                conversation = self._parse_conversation(conv_data)
                conversations.append(conversation)
            except Exception as e:
                print(
                    f"Warning: Failed to parse conversation {conv_data.get('uuid', 'unknown')}: {e}"
                )
                continue

        return conversations

    def _parse_conversation(self, conv_data: dict) -> Conversation:
        """Parse a single conversation from the export"""

        # Extract messages
        messages = []
        for msg_data in conv_data.get("chat_messages", []):
            try:
                message = self._parse_message(msg_data)
                messages.append(message)
            except Exception as e:
                print(f"Warning: Failed to parse message: {e}")
                continue

        # Create conversation object
        conversation = Conversation(
            id=conv_data["uuid"],
            title=conv_data.get("name", "Untitled Conversation"),
            messages=messages,
            created_at=self._parse_timestamp(conv_data["created_at"]),
            updated_at=self._parse_timestamp(conv_data["updated_at"]),
            tags=self._extract_tags(conv_data.get("name", "")),
        )

        return conversation

    def _parse_message(self, msg_data: dict) -> Message:
        """Parse a single message from Claude export"""

        # Extract the actual text content
        content = msg_data.get("text", "")

        # If text is empty, try to extract from content array
        if not content and "content" in msg_data:
            content_parts = []
            for content_item in msg_data["content"]:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "text"
                ):
                    content_parts.append(content_item.get("text", ""))
            content = "\n".join(content_parts)

        # Clean up leading/trailing whitespace
        content = content.strip()  # Add this line

        return Message(
            role=msg_data["sender"],
            content=content,
            timestamp=self._parse_timestamp(msg_data.get("created_at")),
            uuid=msg_data.get("uuid"),
        )

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO format timestamp from Claude export"""
        if not timestamp_str:
            return datetime.now()

        try:
            # Claude uses ISO format with Z suffix
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception as e:
            print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
            return datetime.now()

    def _extract_tags(self, title: str) -> List[str]:
        """
        Extract simple tags from conversation title
        Can be enhanced with LLM-based tagging later
        """
        tags = []

        # Common keywords to extract as tags
        keywords = [
            "code",
            "python",
            "javascript",
            "react",
            "tutorial",
            "export",
            "debug",
            "help",
            "example",
            "vault",
            "api",
            "database",
            "web",
            "design",
            "data",
        ]

        title_lower = title.lower()
        for keyword in keywords:
            if keyword in title_lower:
                tags.append(keyword)

        return tags

    def parse_conversation_from_markdown(self, post) -> Conversation:
        """
        Parse a Conversation object from a markdown file with frontmatter

        Args:
            post: frontmatter.Post object (loaded markdown with YAML frontmatter)

        Returns:
            Conversation object
        """
        from datetime import datetime

        # Extract metadata from frontmatter [3]
        title = post.get("title", "Untitled")
        conv_uuid = post.get("uuid", str(uuid_module.uuid4()))
        tags = post.get("tags", [])
        date = post.get("date", datetime.now().isoformat())
        updated = post.get("updated", date)

        # Parse dates
        created_at = datetime.fromisoformat(date.replace("Z", "+00:00"))
        date = post.get("date", datetime.now().isoformat())
        updated = post.get("updated", date)

        # Parse dates
        created_at = datetime.fromisoformat(date.replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))

        # Parse messages from content [3]
        content = post.content
        messages = []

        # Split by message headers (## 👤 You or ## 🤖 Claude) [3]
        pattern = r"## (👤 You|🤖 Claude)(?:\s*\*\([^)]+\)\*)?"

        # Split content by headers
        parts = re.split(pattern, content)

        # Process message pairs (header, content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                role_text = parts[i]
                message_content = parts[i + 1].strip()

                # Skip separator lines
                message_content = message_content.replace("---", "").strip()

                # Determine role [3][4]
                if "👤" in role_text or "You" in role_text:
                    role = "human"
                else:
                    role = "assistant"

                # Create message
                messages.append(
                    Message(
                        role=role,
                        content=message_content,
                        timestamp=None,  # Not stored in markdown
                    )
                )

        # Create and return Conversation object
        return Conversation(
            id=conv_uuid,
            title=title,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            tags=tags,
        )

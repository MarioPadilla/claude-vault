import json
import re
import uuid as uuid_module
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Conversation, Message


class ChatGPTExportParser:
    """Parser for ChatGPT's conversation export format (JSON)"""

    def parse(self, export_path: Path) -> List[Conversation]:
        """
        Parse ChatGPT export file and return list of conversations

        Args:
            export_path: Path to the conversations.json export file

        Returns:
            List of Conversation objects
        """
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)

        conversations = []

        # ChatGPT exports are always an array
        if isinstance(data, dict):
            data = [data]

        for conv_data in data:
            try:
                conversation = self._parse_conversation(conv_data)
                if conversation and conversation.messages:
                    conversations.append(conversation)
            except Exception as e:
                print(
                    f"Warning: Failed to parse conversation {conv_data.get('title', 'unknown')}: {e}"
                )
                continue

        return conversations

    def _parse_conversation(self, conv_data: dict) -> Optional[Conversation]:
        """Parse a single ChatGPT conversation from the export"""

        mapping = conv_data.get("mapping", {})
        if not mapping:
            return None

        # Build ordered messages by traversing the tree and extract metadata
        messages, metadata = self._extract_messages_and_metadata(mapping)

        if not messages:
            return None

        # Parse timestamps (ChatGPT uses Unix timestamps)
        created_at = self._parse_unix_timestamp(conv_data.get("create_time"))
        updated_at = self._parse_unix_timestamp(conv_data.get("update_time"))

        title = conv_data.get("title", "Untitled Conversation")

        # Extract conversation-level metadata
        gizmo_id = conv_data.get("gizmo_id", "")  # Custom GPT/project
        gizmo_type = conv_data.get("gizmo_type", "") or ""
        conversation_id = conv_data.get("conversation_id", "")

        # Extract all user prompts
        user_prompts = [msg.content for msg in messages if msg.role == "human"]

        # Extract ChatGPT-specific flags
        is_archived = conv_data.get("is_archived", False) or False
        is_starred = conv_data.get("is_starred", False) or False
        is_study_mode = conv_data.get("is_study_mode", False) or False
        is_do_not_remember = conv_data.get("is_do_not_remember", False) or False
        is_read_only = conv_data.get("is_read_only", False) or False
        memory_scope = conv_data.get("memory_scope", "") or ""
        voice = conv_data.get("voice", "") or ""

        return Conversation(
            id=conversation_id if conversation_id else self._generate_id_from_title(title, created_at),
            title=title,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            tags=self._extract_tags(title),
            model=metadata.get("model", ""),
            project=gizmo_id,  # Custom GPT acts as a project
            has_attachments=metadata.get("has_attachments", False),
            is_agentic=metadata.get("is_agentic", False),
            user_prompts=user_prompts,
            is_archived=is_archived,
            is_starred=is_starred,
            is_study_mode=is_study_mode,
            is_do_not_remember=is_do_not_remember,
            is_read_only=is_read_only,
            memory_scope=memory_scope,
            gizmo_type=gizmo_type,
            voice=voice,
        )

    def _extract_messages_and_metadata(self, mapping: Dict) -> tuple:
        """
        Extract ordered messages and metadata from ChatGPT's tree-structured mapping.
        Returns (messages, metadata_dict)
        """
        messages = []
        metadata = {
            "model": "",
            "has_attachments": False,
            "is_agentic": False,
        }

        # Find the root node (parent is None)
        root_id = None
        for node_id, node in mapping.items():
            if node.get("parent") is None:
                root_id = node_id
                break

        if not root_id:
            return messages, metadata

        # Traverse the tree following the first child path (main conversation)
        current_id = root_id
        while current_id:
            node = mapping.get(current_id)
            if not node:
                break

            msg_data = node.get("message")
            if msg_data:
                # Extract metadata from message
                msg_metadata = msg_data.get("metadata", {})

                # Get model info
                if msg_metadata.get("model_slug") and not metadata["model"]:
                    metadata["model"] = msg_metadata.get("model_slug", "")

                # Check for tool use (agentic)
                author = msg_data.get("author", {})
                if author.get("role") == "tool" or msg_metadata.get("is_complete"):
                    metadata["is_agentic"] = True

                # Parse the message
                message = self._parse_message(msg_data)
                if message:
                    messages.append(message)

            # Follow the first child (main conversation path)
            children = node.get("children", [])
            current_id = children[0] if children else None

        return messages, metadata

    def _extract_messages_from_mapping(self, mapping: Dict) -> List[Message]:
        """
        Extract ordered messages from ChatGPT's tree-structured mapping.
        Traverses from root to leaves following the main conversation path.
        """
        messages = []

        # Find the root node (parent is None)
        root_id = None
        for node_id, node in mapping.items():
            if node.get("parent") is None:
                root_id = node_id
                break

        if not root_id:
            return messages

        # Traverse the tree following the first child path (main conversation)
        current_id = root_id
        while current_id:
            node = mapping.get(current_id)
            if not node:
                break

            msg_data = node.get("message")
            if msg_data:
                message = self._parse_message(msg_data)
                if message:
                    messages.append(message)

            # Follow the first child (main conversation path)
            children = node.get("children", [])
            current_id = children[0] if children else None

        return messages

    def _parse_message(self, msg_data: dict) -> Optional[Message]:
        """Parse a single message from ChatGPT export"""

        # Skip hidden system messages
        metadata = msg_data.get("metadata", {})
        if metadata.get("is_visually_hidden_from_conversation"):
            return None

        author = msg_data.get("author", {})
        role = author.get("role", "")

        # Skip system messages and tool reasoning traces
        if role == "system":
            return None

        # Handle tool messages (o1/o3 reasoning) - skip them
        if role == "tool":
            return None

        # Map ChatGPT roles to our format
        if role == "user":
            normalized_role = "human"
        elif role == "assistant":
            normalized_role = "assistant"
        else:
            return None  # Skip unknown roles

        # Extract content from parts
        content_obj = msg_data.get("content", {})
        content_type = content_obj.get("content_type", "")

        # Handle different content types
        if content_type == "text":
            parts = content_obj.get("parts", [])
            content = "\n".join(str(p) for p in parts if p)
        elif content_type == "user_editable_context":
            # Skip user context/profile messages
            return None
        else:
            parts = content_obj.get("parts", [])
            content = "\n".join(str(p) for p in parts if p) if parts else ""

        content = content.strip()
        if not content:
            return None

        # Parse timestamp
        timestamp = self._parse_unix_timestamp(msg_data.get("create_time"))

        return Message(
            role=normalized_role,
            content=content,
            timestamp=timestamp,
            uuid=msg_data.get("id"),
        )

    def _parse_unix_timestamp(self, timestamp: float) -> datetime:
        """Parse Unix timestamp from ChatGPT export"""
        if not timestamp:
            return datetime.now()

        try:
            return datetime.fromtimestamp(timestamp)
        except Exception:
            return datetime.now()

    def _generate_id_from_title(self, title: str, created_at: datetime) -> str:
        """Generate a unique ID for ChatGPT conversations"""
        import hashlib

        unique_str = f"{title}_{created_at.isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _extract_tags(self, title: str) -> List[str]:
        """Extract simple tags from conversation title"""
        tags = []
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
            "css",
            "html",
            "sql",
            "ai",
            "ml",
        ]

        title_lower = title.lower()
        for keyword in keywords:
            if keyword in title_lower:
                tags.append(keyword)

        return tags

    def parse_conversation_from_markdown(self, post) -> Conversation:
        """
        Parse a Conversation object from a markdown file with frontmatter.
        Shared implementation with ClaudeExportParser.
        """
        return _parse_conversation_from_markdown_impl(post)


class ClaudeExportParser:
    """Parser for Claude's conversation export format (JSON)"""

    def __init__(self):
        self._projects_data: Dict[str, dict] = {}

    def load_projects(self, projects_path: Path) -> None:
        """
        Load projects.json to get project metadata

        Args:
            projects_path: Path to the projects.json file
        """
        if not projects_path.exists():
            return

        try:
            with open(projects_path, encoding="utf-8") as f:
                projects = json.load(f)

            if isinstance(projects, list):
                for proj in projects:
                    proj_uuid = proj.get("uuid", "")
                    if proj_uuid:
                        self._projects_data[proj_uuid] = proj
        except Exception as e:
            print(f"Warning: Could not load projects.json: {e}")

    def parse(self, export_path: Path) -> List[Conversation]:
        """
        Parse Claude export file and return list of conversations

        Args:
            export_path: Path to the conversations.json export file

        Returns:
            List of Conversation objects
        """
        # Try to load projects.json from same directory
        projects_path = export_path.parent / "projects.json"
        self.load_projects(projects_path)

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

        # Extract messages and metadata
        messages = []
        has_attachments = False
        has_artifacts = False
        is_agentic = False

        for msg_data in conv_data.get("chat_messages", []):
            try:
                message = self._parse_message(msg_data)
                messages.append(message)

                # Check for attachments
                if msg_data.get("attachments") or msg_data.get("files"):
                    has_attachments = True

                # Check for artifacts (tool use)
                if "content" in msg_data:
                    for content_item in msg_data["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "tool_use":
                                is_agentic = True
                                if content_item.get("name") == "artifacts":
                                    has_artifacts = True

            except Exception as e:
                print(f"Warning: Failed to parse message: {e}")
                continue

        # Extract all user prompts
        user_prompts = [msg.content for msg in messages if msg.role == "human"]

        # Extract project info if available
        project_data = conv_data.get("project", {})
        project_uuid = project_data.get("uuid", "") if project_data else ""
        project_name = project_data.get("name", "") if project_data else ""

        # Get additional project metadata from projects.json if available
        is_private = False
        is_starter_project = False
        prompt_template = ""

        if project_uuid and project_uuid in self._projects_data:
            proj_info = self._projects_data[project_uuid]
            is_private = proj_info.get("is_private", False) or False
            is_starter_project = proj_info.get("is_starter_project", False) or False
            prompt_template = proj_info.get("prompt_template", "") or ""
            # Use project name from projects.json if not in conversation
            if not project_name:
                project_name = proj_info.get("name", "")

        # Create conversation object
        conversation = Conversation(
            id=conv_data["uuid"],
            title=conv_data.get("name", "Untitled Conversation"),
            messages=messages,
            created_at=self._parse_timestamp(conv_data["created_at"]),
            updated_at=self._parse_timestamp(conv_data["updated_at"]),
            tags=self._extract_tags(conv_data.get("name", "")),
            project=project_name,
            has_attachments=has_attachments,
            has_artifacts=has_artifacts,
            is_agentic=is_agentic,
            user_prompts=user_prompts,
            is_private=is_private,
            is_starter_project=is_starter_project,
            prompt_template=prompt_template,
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
        Parse a Conversation object from a markdown file with frontmatter.
        Shared implementation.
        """
        return _parse_conversation_from_markdown_impl(post)


def _parse_conversation_from_markdown_impl(post) -> Conversation:
    """
    Shared implementation for parsing a Conversation from markdown with frontmatter.

    Args:
        post: frontmatter.Post object (loaded markdown with YAML frontmatter)

    Returns:
        Conversation object
    """
    # Extract metadata from frontmatter
    title = post.get("title", "Untitled")
    conv_uuid = post.get("uuid", str(uuid_module.uuid4()))
    tags = post.get("tags", [])
    date = post.get("date", datetime.now().isoformat())
    updated = post.get("updated", date)

    # Parse dates
    created_at = datetime.fromisoformat(date.replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))

    # Parse messages from content
    content = post.content
    messages = []

    # Split by message headers (## 👤 You or ## 🤖 Claude/ChatGPT)
    pattern = r"## (👤 You|🤖 Claude|🤖 ChatGPT)(?:\s*\*\([^)]+\)\*)?"

    # Split content by headers
    parts = re.split(pattern, content)

    # Process message pairs (header, content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            role_text = parts[i]
            message_content = parts[i + 1].strip()

            # Skip separator lines
            message_content = message_content.replace("---", "").strip()

            # Determine role
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

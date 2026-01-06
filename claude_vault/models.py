import hashlib
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single message in a conversation"""

    role: str  # 'human' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None
    uuid: Optional[str] = None


class Conversation(BaseModel):
    """Represents a complete conversation"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)

    # Additional metadata
    model: str = ""  # Model used (e.g., 'gpt-4', 'claude-3-opus')
    project: str = ""  # Project name if part of a project
    has_attachments: bool = False  # Whether conversation has file attachments
    has_artifacts: bool = False  # Whether conversation has artifacts (Claude)
    is_agentic: bool = False  # Whether this was an agentic/tool-use conversation

    # User prompts (all human messages consolidated)
    user_prompts: List[str] = Field(default_factory=list)

    # Claude-specific flags
    is_private: bool = False  # From projects.json
    is_starter_project: bool = False  # From projects.json
    prompt_template: str = ""  # From projects.json

    # ChatGPT-specific flags
    is_archived: bool = False
    is_starred: bool = False
    is_study_mode: bool = False
    is_do_not_remember: bool = False
    is_read_only: bool = False
    memory_scope: str = ""  # e.g., 'global_enabled'
    gizmo_type: str = ""  # Custom GPT type
    voice: str = ""  # Voice mode setting

    def content_hash(self) -> str:
        """Generate SHA-256 hash of conversation content for change detection"""
        content = f"{self.title}{''.join(m.content for m in self.messages)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_first_user_message(self) -> str:
        """Get the first user message as a preview"""
        for msg in self.messages:
            if msg.role == "human":
                return (
                    msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                )
        return ""

    def get_all_user_prompts(self) -> List[str]:
        """Get all user prompts from messages"""
        if self.user_prompts:
            return self.user_prompts
        return [msg.content for msg in self.messages if msg.role == "human"]

    def get_flags(self) -> dict:
        """Get all flag values as a dictionary"""
        return {
            "is_private": self.is_private,
            "is_starter_project": self.is_starter_project,
            "is_archived": self.is_archived,
            "is_starred": self.is_starred,
            "is_study_mode": self.is_study_mode,
            "is_do_not_remember": self.is_do_not_remember,
            "is_read_only": self.is_read_only,
        }

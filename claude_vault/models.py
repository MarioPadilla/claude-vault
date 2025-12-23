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

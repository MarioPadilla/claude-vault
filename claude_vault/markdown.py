from pathlib import Path

import frontmatter

from .models import Conversation


class MarkdownGenerator:
    """Generates Obsidian-compatible markdown files from conversations"""

    def __init__(self, template: str = None):
        self.template = template or self._default_template()

    def _default_template(self) -> str:
        """Default markdown template"""
        return """---
title: {title}
date: {date}
tags: {tags}
uuid: {uuid}
---

# {title}

{content}
"""

    def generate(self, conversation: Conversation, related_convs: list = None) -> str:
        """
        Generate markdown from conversation

        Args:
            conversation: Conversation object to convert
            related_convs: List of related Conversation objects to include in frontmatter

        Returns:
            Formatted markdown string with YAML frontmatter
        """
        # Build content from messages
        content_parts = []

        for i, msg in enumerate(conversation.messages):
            # Map Claude's sender format to readable names
            if msg.role == "human":
                role_display = "👤 You"
            elif msg.role == "assistant":
                role_display = "🤖 Claude"
            else:
                role_display = f"**{msg.role}**"

            # Format timestamp if available
            timestamp = ""
            if msg.timestamp:
                timestamp = f" *({msg.timestamp.strftime('%Y-%m-%d %H:%M')})*"

            # Add message with proper formatting
            content_parts.append(f"## {role_display}{timestamp}\n\n{msg.content}\n")

            # Add separator between messages (except last one)
            if i < len(conversation.messages) - 1:
                content_parts.append("---\n")

        content = "\n".join(content_parts)

        # Create frontmatter with metadata
        post = frontmatter.Post(content)
        post["title"] = conversation.title
        post["date"] = conversation.created_at.isoformat()
        post["updated"] = conversation.updated_at.isoformat()
        post["tags"] = conversation.tags
        post["uuid"] = conversation.id
        post["message_count"] = len(conversation.messages)
        if related_convs:
            # Create wikilinks for Obsidian
            post["related"] = [
                f"[[{r['file'].replace('.md', '')}]]" for r in related_convs
            ]
            post["related_tags"] = {r["title"]: r["common_tags"] for r in related_convs}

        return frontmatter.dumps(post)

    def save(
        self, conversation: Conversation, file_path: Path, related_convs: list = None
    ):
        """
        Generate and save markdown file

        Args:
            conversation: Conversation to save
            file_path: Path where to save the markdown file
            related_convs: List of related Conversation objects to include in frontmatter
        """
        markdown = self.generate(conversation, related_convs)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(markdown, encoding="utf-8")

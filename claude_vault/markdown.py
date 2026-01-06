from pathlib import Path
from typing import List, Optional

import frontmatter

from .models import Conversation


class MarkdownGenerator:
    """Generates Obsidian-compatible markdown files from conversations"""

    def __init__(
        self,
        template: str = None,
        source: str = "claude",
        use_hierarchy: bool = False,
    ):
        self.template = template or self._default_template()
        self.source = source
        self.use_hierarchy = use_hierarchy

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

    def generate(
        self,
        conversation: Conversation,
        related_convs: list = None,
        tag_result=None,
    ) -> str:
        """
        Generate markdown from conversation

        Args:
            conversation: Conversation object to convert
            related_convs: List of related Conversation objects to include in frontmatter
            tag_result: TagResult object with hierarchical tags (optional)

        Returns:
            Formatted markdown string with YAML frontmatter
        """
        # Build content from messages
        content_parts = []

        # Determine assistant name based on source
        assistant_name = "ChatGPT" if self.source == "chatgpt" else "Claude"

        for i, msg in enumerate(conversation.messages):
            # Map sender format to readable names
            if msg.role == "human":
                role_display = "You"
            elif msg.role == "assistant":
                role_display = f"{assistant_name}"
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
        post["uuid"] = conversation.id
        post["source"] = self.source
        post["message_count"] = len(conversation.messages)

        # Add additional metadata if present
        if conversation.model:
            post["model"] = conversation.model
        if conversation.project:
            post["project"] = conversation.project
        if conversation.has_attachments:
            post["has_attachments"] = True
        if conversation.has_artifacts:
            post["has_artifacts"] = True
        if conversation.is_agentic:
            post["is_agentic"] = True

        # Add user prompts
        if conversation.user_prompts:
            post["user_prompts"] = conversation.user_prompts

        # Add Claude-specific flags
        if conversation.is_private:
            post["is_private"] = True
        if conversation.is_starter_project:
            post["is_starter_project"] = True
        if conversation.prompt_template:
            post["prompt_template"] = conversation.prompt_template

        # Add ChatGPT-specific flags
        if conversation.is_archived:
            post["is_archived"] = True
        if conversation.is_starred:
            post["is_starred"] = True
        if conversation.is_study_mode:
            post["is_study_mode"] = True
        if conversation.is_do_not_remember:
            post["is_do_not_remember"] = True
        if conversation.is_read_only:
            post["is_read_only"] = True
        if conversation.memory_scope:
            post["memory_scope"] = conversation.memory_scope
        if conversation.gizmo_type:
            post["gizmo_type"] = conversation.gizmo_type
        if conversation.voice:
            post["voice"] = conversation.voice

        # Handle tags - either hierarchical or flat
        if self.use_hierarchy and tag_result and (
            tag_result.primary or tag_result.secondary or tag_result.tertiary
        ):
            # Hierarchical tags
            post["tags"] = tag_result.all_tags()
            post["tags_primary"] = tag_result.primary
            post["tags_secondary"] = tag_result.secondary
            post["tags_tertiary"] = tag_result.tertiary
        else:
            # Flat tags (default)
            if tag_result:
                post["tags"] = tag_result.tags
            else:
                post["tags"] = conversation.tags

        # Add related conversations
        if related_convs:
            # Create wikilinks for Obsidian
            post["related"] = [
                f"[[{r['file'].replace('.md', '')}]]" for r in related_convs
            ]
            post["related_tags"] = {r["title"]: r["common_tags"] for r in related_convs}

        return frontmatter.dumps(post)

    def save(
        self,
        conversation: Conversation,
        file_path: Path,
        related_convs: list = None,
        tag_result=None,
    ):
        """
        Generate and save markdown file

        Args:
            conversation: Conversation to save
            file_path: Path where to save the markdown file
            related_convs: List of related Conversation objects to include in frontmatter
            tag_result: TagResult object with hierarchical tags (optional)
        """
        markdown = self.generate(conversation, related_convs, tag_result)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(markdown, encoding="utf-8")

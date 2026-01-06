"""Summary report generation for Claude Vault"""

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import frontmatter
import requests


@dataclass
class ConversationMetadata:
    """Metadata extracted from a conversation for the summary report"""

    uuid: str
    title: str
    source: str  # 'claude' or 'chatgpt'
    created_at: str
    updated_at: str
    message_count: int
    file_path: str

    # Tags
    tags: List[str] = field(default_factory=list)
    tags_primary: List[str] = field(default_factory=list)
    tags_secondary: List[str] = field(default_factory=list)
    tags_tertiary: List[str] = field(default_factory=list)

    # Additional metadata (may be empty)
    model: str = ""
    project: str = ""
    has_attachments: bool = False
    has_artifacts: bool = False

    # User prompts (all human messages)
    user_prompts: List[str] = field(default_factory=list)

    # Flags - Claude specific
    is_private: bool = False
    is_starter_project: bool = False

    # Flags - ChatGPT specific
    is_archived: bool = False
    is_starred: bool = False
    is_study_mode: bool = False
    is_do_not_remember: bool = False
    is_read_only: bool = False
    memory_scope: str = ""
    gizmo_type: str = ""
    voice: str = ""

    # Summary (populated by Ollama or left as placeholder)
    summary: str = ""

    # Comment (for user to fill in)
    comment: str = ""

    def get_user_prompts_text(self, max_length: int = 500) -> str:
        """Get user prompts as a single text, truncated if needed"""
        if not self.user_prompts:
            return ""
        combined = " | ".join(p.replace("\n", " ")[:100] for p in self.user_prompts)
        if len(combined) > max_length:
            return combined[:max_length] + "..."
        return combined

    def get_active_flags(self) -> List[str]:
        """Get list of active (True) flags"""
        flags = []
        if self.is_private:
            flags.append("private")
        if self.is_starter_project:
            flags.append("starter")
        if self.is_archived:
            flags.append("archived")
        if self.is_starred:
            flags.append("starred")
        if self.is_study_mode:
            flags.append("study")
        if self.is_do_not_remember:
            flags.append("no-memory")
        if self.is_read_only:
            flags.append("read-only")
        return flags

    def get_url(self) -> str:
        """Get the URL for this conversation based on source"""
        if self.source == "chatgpt":
            return f"https://chat.openai.com/c/{self.uuid}"
        else:  # claude
            return f"https://claude.ai/chat/{self.uuid}"

    def get_title_link(self) -> str:
        """Get markdown link for title"""
        return f"[{self.title}]({self.get_url()})"


class SummaryReportGenerator:
    """Generates summary reports for synced conversations"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.conversations_dir = vault_path / "conversations"
        self.report_path = vault_path / "summary-report.md"
        self.ollama_url = "http://localhost:11434/api/generate"
        self._pending_summaries: Dict[str, str] = {}  # uuid -> status
        self._summary_results: Dict[str, str] = {}  # uuid -> summary

    def extract_metadata_from_markdown(self, md_path: Path) -> Optional[ConversationMetadata]:
        """Extract metadata from a markdown file with frontmatter"""
        try:
            post = frontmatter.load(md_path)

            # Extract user prompts from content (## You sections)
            user_prompts = post.get("user_prompts", [])
            if not user_prompts:
                # Parse from markdown content if not in frontmatter
                user_prompts = self._extract_user_prompts_from_content(post.content)

            return ConversationMetadata(
                uuid=post.get("uuid", ""),
                title=post.get("title", "Untitled"),
                source=post.get("source", "claude"),
                created_at=post.get("date", ""),
                updated_at=post.get("updated", ""),
                message_count=post.get("message_count", 0),
                file_path=str(md_path.relative_to(self.vault_path)),
                tags=post.get("tags", []),
                tags_primary=post.get("tags_primary", []),
                tags_secondary=post.get("tags_secondary", []),
                tags_tertiary=post.get("tags_tertiary", []),
                model=post.get("model", ""),
                project=post.get("project", ""),
                has_attachments=post.get("has_attachments", False),
                has_artifacts=post.get("has_artifacts", False),
                user_prompts=user_prompts,
                is_private=post.get("is_private", False),
                is_starter_project=post.get("is_starter_project", False),
                is_archived=post.get("is_archived", False),
                is_starred=post.get("is_starred", False),
                is_study_mode=post.get("is_study_mode", False),
                is_do_not_remember=post.get("is_do_not_remember", False),
                is_read_only=post.get("is_read_only", False),
                memory_scope=post.get("memory_scope", ""),
                gizmo_type=post.get("gizmo_type", ""),
                voice=post.get("voice", ""),
                summary=post.get("summary", ""),
                comment="",
            )
        except Exception as e:
            print(f"Error extracting metadata from {md_path}: {e}")
            return None

    def _extract_user_prompts_from_content(self, content: str) -> List[str]:
        """Extract user prompts from markdown content"""
        prompts = []
        # Split by message headers
        parts = re.split(r"## (?:You|Human)", content)
        for i, part in enumerate(parts[1:], 1):  # Skip first part (before first user message)
            # Get content until next header or separator
            lines = []
            for line in part.split("\n"):
                if line.startswith("## ") or line.strip() == "---":
                    break
                lines.append(line)
            prompt = "\n".join(lines).strip()
            if prompt:
                prompts.append(prompt)
        return prompts

    def get_existing_uuids(self) -> Set[str]:
        """Get UUIDs already in the summary report"""
        existing = set()

        if not self.report_path.exists():
            return existing

        try:
            content = self.report_path.read_text(encoding="utf-8")

            # Extract UUIDs from the table (they're in the URL links)
            # Pattern matches both Claude and ChatGPT URLs
            claude_pattern = r"claude\.ai/chat/([a-f0-9-]+)"
            chatgpt_pattern = r"chat\.openai\.com/c/([a-f0-9-]+)"

            for match in re.finditer(claude_pattern, content):
                existing.add(match.group(1))
            for match in re.finditer(chatgpt_pattern, content):
                existing.add(match.group(1))

        except Exception as e:
            print(f"Error reading existing report: {e}")

        return existing

    def collect_all_metadata(self) -> List[ConversationMetadata]:
        """Collect metadata from all markdown files"""
        metadata_list = []

        for md_file in self.conversations_dir.glob("*.md"):
            metadata = self.extract_metadata_from_markdown(md_file)
            if metadata:
                metadata_list.append(metadata)

        # Sort by created_at (newest first)
        metadata_list.sort(key=lambda x: x.created_at, reverse=True)

        return metadata_list

    def summarize_conversation(
        self,
        md_path: Path,
        model: str,
        prompt: str,
    ) -> str:
        """Summarize a conversation using Ollama"""
        try:
            post = frontmatter.load(md_path)
            content = post.content

            # Truncate content if too long
            if len(content) > 8000:
                content = content[:4000] + "\n\n[...truncated...]\n\n" + content[-4000:]

            full_prompt = f"""{prompt}

Conversation:
{content}

Provide a concise summary (2-3 sentences):"""

            response = requests.post(
                self.ollama_url,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    },
                },
                timeout=120,
            )

            if response.status_code == 200:
                summary = response.json()["response"].strip()
                # Clean up the summary
                summary = summary.replace("\n", " ").strip()
                return summary

        except requests.exceptions.Timeout:
            return "[Timeout - retry later]"
        except Exception as e:
            return f"[Error: {str(e)[:50]}]"

        return "[Failed to summarize]"

    def summarize_async(
        self,
        metadata: ConversationMetadata,
        model: str,
        prompt: str,
        callback=None,
    ):
        """Start async summarization for a conversation"""
        md_path = self.vault_path / metadata.file_path

        def _do_summarize():
            summary = self.summarize_conversation(md_path, model, prompt)
            self._summary_results[metadata.uuid] = summary
            self._pending_summaries[metadata.uuid] = "completed"
            if callback:
                callback(metadata.uuid, summary)

        self._pending_summaries[metadata.uuid] = "pending"
        thread = threading.Thread(target=_do_summarize)
        thread.start()

    def generate_report(
        self,
        metadata_list: List[ConversationMetadata],
        append_mode: bool = True,
    ) -> str:
        """Generate the markdown summary report table"""

        # Get existing UUIDs if appending
        existing_uuids = self.get_existing_uuids() if append_mode else set()

        # Filter to only new conversations
        new_metadata = [m for m in metadata_list if m.uuid not in existing_uuids]

        if not new_metadata and append_mode:
            return ""  # Nothing new to add

        # Build table header (added User Prompts and Flags columns)
        header = """| Title | Source | Date | Messages | Primary Tags | Secondary Tags | Tertiary Tags | Model | Project | User Prompts | Flags | Summary | Comment |
|-------|--------|------|----------|--------------|----------------|---------------|-------|---------|--------------|-------|---------|---------|"""

        # Build rows
        rows = []
        for meta in new_metadata:
            # Format tags as comma-separated
            primary = ", ".join(meta.tags_primary) if meta.tags_primary else ", ".join(meta.tags[:3]) if meta.tags else ""
            secondary = ", ".join(meta.tags_secondary) if meta.tags_secondary else ""
            tertiary = ", ".join(meta.tags_tertiary) if meta.tags_tertiary else ""

            # Get summary or placeholder
            summary = meta.summary if meta.summary else "[Pending...]"
            if meta.uuid in self._summary_results:
                summary = self._summary_results[meta.uuid]

            # Format date
            date_str = meta.created_at[:10] if meta.created_at else ""

            # Format user prompts (truncated)
            user_prompts = meta.get_user_prompts_text(max_length=200)

            # Format flags
            flags = ", ".join(meta.get_active_flags())

            # Escape pipe characters in content
            title_link = meta.get_title_link().replace("|", "\\|")
            summary = summary.replace("|", "\\|")
            user_prompts = user_prompts.replace("|", "\\|")

            row = f"| {title_link} | {meta.source} | {date_str} | {meta.message_count} | {primary} | {secondary} | {tertiary} | {meta.model} | {meta.project} | {user_prompts} | {flags} | {summary} | {meta.comment} |"
            rows.append(row)

        return header + "\n" + "\n".join(rows)

    def save_report(
        self,
        metadata_list: List[ConversationMetadata],
        append_mode: bool = True,
    ) -> int:
        """Save the summary report, returns count of new entries added"""

        existing_uuids = self.get_existing_uuids() if append_mode else set()
        new_metadata = [m for m in metadata_list if m.uuid not in existing_uuids]

        if not new_metadata:
            return 0

        table_content = self.generate_report(metadata_list, append_mode)

        if append_mode and self.report_path.exists():
            # Append to existing report
            existing_content = self.report_path.read_text(encoding="utf-8")
            # Remove header from new content (keep only rows)
            new_rows = "\n".join(table_content.split("\n")[2:])  # Skip header lines
            updated_content = existing_content.rstrip() + "\n" + new_rows
            self.report_path.write_text(updated_content, encoding="utf-8")
        else:
            # Create new report
            report_header = f"""# Conversation Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            full_content = report_header + table_content
            self.report_path.write_text(full_content, encoding="utf-8")

        return len(new_metadata)

    def update_summary_in_report(self, uuid: str, summary: str):
        """Update a specific conversation's summary in the report"""
        if not self.report_path.exists():
            return

        content = self.report_path.read_text(encoding="utf-8")

        # Find and replace the placeholder for this UUID
        # The UUID is in the URL, so we look for lines containing it
        lines = content.split("\n")
        updated_lines = []

        for line in lines:
            if uuid in line and "[Pending...]" in line:
                # Replace placeholder with actual summary
                line = line.replace("[Pending...]", summary.replace("|", "\\|"))
            updated_lines.append(line)

        self.report_path.write_text("\n".join(updated_lines), encoding="utf-8")

    def run_batch_summarization(
        self,
        metadata_list: List[ConversationMetadata],
        model: str,
        prompt: str,
        progress_callback=None,
    ) -> Dict[str, str]:
        """Run summarization for all conversations and update report"""
        results = {}

        for i, meta in enumerate(metadata_list):
            if progress_callback:
                progress_callback(i + 1, len(metadata_list), meta.title)

            md_path = self.vault_path / meta.file_path
            summary = self.summarize_conversation(md_path, model, prompt)
            results[meta.uuid] = summary

            # Update the report in place
            self.update_summary_in_report(meta.uuid, summary)

        return results

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import frontmatter

from .code_parser import ClaudeCodeHistoryParser
from .markdown import MarkdownGenerator
from .models import Conversation
from .opencode_parser import OpenCodeParser
from .parser import ClaudeExportParser
from .pii import PIIDetector
from .state import StateManager
from .tagging import OfflineTagGenerator

# Risk level ordering used to compare thresholds
_RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


class SyncEngine:
    """Main sync engine for Claude Vault"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.state = StateManager(vault_path)
        self.parser: Union[
            ClaudeExportParser, ClaudeCodeHistoryParser, OpenCodeParser
        ] = ClaudeExportParser()
        self.markdown_gen = MarkdownGenerator()
        self.conversations_dir = vault_path / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.tag_generator = OfflineTagGenerator()
        self.pii_detector = PIIDetector()

    def sync(
        self,
        export_path: Path,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        detect_pii: bool = False,
        redact_pii: bool = False,
        skip_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync conversations from Claude export to markdown files

        Args:
            export_path: Path to conversations.json export
            dry_run: If True, only simulate changes without writing files
            progress_callback: Optional callback(description, current, total) for progress
            detect_pii: Scan conversations for PII and sensitive content
            redact_pii: Replace detected PII with [REDACTED-TYPE] placeholders
            skip_sensitive: Skip conversations whose risk level meets the threshold

        Returns:
            Dictionary with sync statistics and details
        """
        results: Dict[str, Any] = {
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "recreated": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        try:
            # Parse export
            conversations = self.parser.parse(export_path)

            # Show if Ollama is available
            if self.tag_generator.is_available():
                print("[green]✓ Ollama detected - using AI tag generation[/green]")
            else:
                print(
                    "[yellow]⚠ Ollama not running - using keyword extraction[/yellow]"
                )
                print(
                    "[dim]Tip: Start Ollama with 'ollama serve' for automatic tagging[/dim]"
                )

            total = len(conversations)
            for idx, conv in enumerate(conversations, 1):
                try:
                    # Report progress
                    if progress_callback:
                        progress_callback(conv.title[:50], idx, total)

                    existing = self.state.get_conversation(conv.id)

                    # Generate content hash with encoding safety
                    try:
                        current_hash = conv.content_hash()
                    except UnicodeDecodeError as e:
                        print(
                            f"Warning: Encoding issue in conversation '{conv.title[:50]}': {e}"
                        )
                        # Try to sanitize and re-hash
                        for msg in conv.messages:
                            if isinstance(msg.content, bytes):
                                msg.content = msg.content.decode(
                                    "utf-8", errors="replace"
                                )
                        current_hash = conv.content_hash()

                    # Preserve existing tags from the markdown file.
                    # The parser only does title-keyword extraction, so
                    # conv.tags is usually empty even when a prior sync or
                    # retag already wrote AI-generated tags to the file.
                    # Without this check, every sync cycle re-invokes
                    # Ollama for conversations whose titles don't match
                    # the keyword list.
                    if (not conv.tags or len(conv.tags) < 2) and existing:
                        md_path = self.vault_path / existing["file_path"]
                        if md_path.exists():
                            saved = frontmatter.load(md_path)
                            saved_tags = saved.get("tags", [])
                            saved_summary = saved.get("summary", "")
                            if saved_tags and len(saved_tags) >= 2:
                                conv.tags = saved_tags
                                if saved_summary:
                                    conv.summary = saved_summary

                    # Generate tags/metadata if still missing
                    if not conv.tags or len(conv.tags) < 2:
                        metadata = self.tag_generator.generate_metadata(conv)
                        conv.tags = metadata["tags"]
                        conv.summary = metadata["summary"]

                    # PII / sensitive content detection
                    if detect_pii or redact_pii or skip_sensitive:
                        pii_config = self.pii_detector.config.pii
                        use_llm = pii_config.use_llm
                        pii_result = self.pii_detector.analyze(conv, use_llm=use_llm)

                        conv.pii_detected = pii_result.detected
                        conv.risk_level = pii_result.risk_level
                        conv.pii_types = pii_result.pii_types

                        # Apply PII tags to conversation
                        if pii_result.detected:
                            threshold = pii_config.risk_threshold
                            if _RISK_ORDER.get(
                                pii_result.risk_level, 0
                            ) >= _RISK_ORDER.get(threshold, 2):
                                if "sensitive" not in conv.tags:
                                    conv.tags.append("sensitive")
                            for pii_type in pii_result.pii_types:
                                tag = f"pii-{pii_type.replace('_', '-')}"
                                if tag not in conv.tags:
                                    conv.tags.append(tag)

                        # Skip sensitive conversations if requested
                        if skip_sensitive and pii_result.detected:
                            threshold = pii_config.risk_threshold
                            if _RISK_ORDER.get(
                                pii_result.risk_level, 0
                            ) >= _RISK_ORDER.get(threshold, 2):
                                results["skipped"] += 1
                                results["details"].append(
                                    {
                                        "action": "skipped",
                                        "title": conv.title,
                                        "risk_level": pii_result.risk_level,
                                        "pii_types": pii_result.pii_types,
                                    }
                                )
                                continue

                        # Redact PII from message content if requested
                        if redact_pii and pii_result.detected:
                            for msg in conv.messages:
                                msg.content = self.pii_detector.redact(msg.content)

                    # Find related conversations based on tags [3]
                    related_convs = self._find_related_by_tags(conv, conversations)

                    if not existing:
                        # New conversation
                        file_path = self._generate_path(conv)
                        action = "new"
                        results["new"] += 1

                        if not dry_run:
                            self.markdown_gen.save(conv, file_path)
                            self.state.save_conversation(
                                conv.id,
                                str(file_path.relative_to(self.vault_path)),
                                current_hash,
                                {"title": conv.title},
                            )

                        results["details"].append(
                            {
                                "action": action,
                                "title": conv.title,
                                "file_path": str(
                                    file_path.relative_to(self.vault_path)
                                ),
                            }
                        )

                    else:
                        # Conversation exists in database
                        file_path = self.vault_path / existing["file_path"]

                        # Check if file exists
                        if not file_path.exists():
                            # File deleted - recreate it [3]
                            file_path = self._generate_path(conv)
                            action = "recreated"
                            results["recreated"] += 1

                            if not dry_run:
                                self.markdown_gen.save(conv, file_path, related_convs)
                                self.state.save_conversation(
                                    conv.id,
                                    str(file_path.relative_to(self.vault_path)),
                                    current_hash,
                                    {"title": conv.title},
                                )
                                print(f"[yellow]⚠ Recreated: {file_path.name}[/yellow]")

                            results["details"].append(
                                {
                                    "action": action,
                                    "title": conv.title,
                                    "file_path": str(
                                        file_path.relative_to(self.vault_path)
                                    ),
                                }
                            )

                        elif existing["content_hash"] != current_hash:
                            # File exists but content changed [3]
                            action = "updated"
                            results["updated"] += 1

                            if not dry_run:
                                self.markdown_gen.save(conv, file_path, related_convs)
                                self.state.save_conversation(
                                    conv.id,
                                    str(file_path.relative_to(self.vault_path)),
                                    current_hash,
                                    {"title": conv.title},
                                )

                            results["details"].append(
                                {
                                    "action": action,
                                    "title": conv.title,
                                    "file_path": str(
                                        file_path.relative_to(self.vault_path)
                                    ),
                                }
                            )

                        else:
                            # File exists and unchanged
                            results["unchanged"] += 1

                except Exception as e:
                    print(f"Error processing conversation {conv.title}: {e}")
                    results["errors"] += 1
                    results["details"].append(
                        {"action": "error", "title": conv.title, "error": str(e)}
                    )

        except Exception as e:
            print(f"Error during sync: {e}")
            results["errors"] += 1

        return results

    def _generate_path(self, conversation) -> Path:
        """
        Generate file path for conversation

        Args:
            conversation: Conversation object

        Returns:
            Path object for the markdown file
        """
        date_str = conversation.created_at.strftime("%Y-%m-%d")

        # Create safe filename from title
        safe_title = re.sub(r"[^\w\s-]", "", conversation.title)
        safe_title = re.sub(r"[-\s]+", "-", safe_title)
        safe_title = safe_title[:50]  # Limit length

        filename = f"{date_str}-{safe_title}.md"
        return self.conversations_dir / filename

    def _find_moved_file(self, uuid: str) -> Optional[Path]:
        """
        Find a file that was moved/renamed by searching for UUID in frontmatter

        Args:
            uuid: Conversation UUID to search for

        Returns:
            Path to file if found, None otherwise
        """

        for md_file in self.conversations_dir.rglob("*.md"):
            try:
                post = frontmatter.load(md_file)
                if post.get("uuid") == uuid:
                    return md_file
            except Exception:
                continue

        return None

    def _find_related_by_tags(
        self, conversation: Conversation, all_conversations: list[Conversation]
    ) -> list[Dict]:
        """Find conversations with similar tags"""

        related: List[Dict] = []
        conv_tags = set(conversation.tags)

        if not conv_tags:
            return related

        for other_conv in all_conversations:
            if other_conv.id == conversation.id:
                continue

            other_tags = set(other_conv.tags)
            common_tags = conv_tags.intersection(other_tags)

            # At least 2 common tags = related [3]
            if len(common_tags) >= 2:
                related.append(
                    {
                        "id": other_conv.id,
                        "title": other_conv.title,
                        "common_tags": list(common_tags),
                        "file": self._generate_path(other_conv).name,
                    }
                )

        # Return top 5 most related
        related.sort(key=lambda x: len(x["common_tags"]), reverse=True)
        return related[:5]

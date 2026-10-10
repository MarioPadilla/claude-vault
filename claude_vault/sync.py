import hashlib
import json
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

    def _pii_fingerprint(
        self, detect_pii: bool, redact_pii: bool, skip_sensitive: bool
    ) -> str:
        """Identify the PII settings a conversation is processed under.

        Empty when no PII handling is requested. Stored next to the content hash
        so an unchanged conversation is only skipped when it was already
        processed under the same settings, by the same version of the detector.
        """
        if not (detect_pii or redact_pii or skip_sensitive):
            return ""
        pii_config = self.pii_detector.config.pii
        settings = [
            detect_pii,
            redact_pii,
            skip_sensitive,
            pii_config.use_llm,
            pii_config.risk_threshold,
            self.pii_detector.version,
        ]
        return hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]

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
        pii_fingerprint = self._pii_fingerprint(detect_pii, redact_pii, skip_sensitive)

        results: Dict[str, Any] = {
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "recreated": 0,
            "reanalyzed": 0,
            "skipped": 0,
            # Files of skipped conversations that are already in the vault
            "skipped_on_disk": [],
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

                    # Fast path: skip an unchanged conversation that's already on
                    # disk. Source exports carry no tags, so the len(conv.tags) < 2
                    # check below would otherwise re-run the LLM every sync only to
                    # discard the result. With PII handling on, it only applies when
                    # the conversation was already processed under the same settings.
                    if (
                        existing
                        and existing.get("content_hash") == current_hash
                        and (
                            not pii_fingerprint
                            or existing["metadata"].get("pii_fingerprint")
                            == pii_fingerprint
                        )
                    ):
                        existing_path = self.vault_path / existing["file_path"]
                        if existing_path.exists():
                            if pii_fingerprint and existing["metadata"].get(
                                "skipped_sensitive"
                            ):
                                # Skipped before under these settings; the file is
                                # still there, so keep saying so
                                self._report_skipped(results, conv, existing_path)
                            else:
                                results["unchanged"] += 1
                            continue

                    # Generate tags/metadata if missing
                    if not conv.tags or len(conv.tags) < 2:
                        metadata = self.tag_generator.generate_metadata(conv)
                        conv.tags = metadata["tags"]
                        conv.summary = metadata["summary"]

                    # Fingerprint recorded for this conversation; stays empty unless
                    # PII analysis ran to completion, so it is retried next sync
                    stored_fingerprint = ""

                    # PII / sensitive content detection
                    if detect_pii or redact_pii or skip_sensitive:
                        pii_config = self.pii_detector.config.pii
                        use_llm = pii_config.use_llm
                        pii_result = self.pii_detector.analyze(conv, use_llm=use_llm)

                        # A silent LLM fallback (Ollama down) is regex-only, so it
                        # must not be recorded as done
                        if pii_result.llm_used or not use_llm:
                            stored_fingerprint = pii_fingerprint

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
                                on_disk = self._existing_file(existing)
                                self._report_skipped(
                                    results,
                                    conv,
                                    on_disk,
                                    pii_result.risk_level,
                                    pii_result.pii_types,
                                )
                                if existing and not dry_run:
                                    # Remember the skip so it is not re-analysed on
                                    # every sync. The file is left as it is.
                                    self.state.save_conversation(
                                        conv.id,
                                        existing["file_path"],
                                        current_hash,
                                        {
                                            "title": conv.title,
                                            "pii_fingerprint": stored_fingerprint,
                                            "skipped_sensitive": True,
                                        },
                                    )
                                continue

                        # Redact PII from everything written to disk if requested:
                        # message content, plus the title (frontmatter, heading and
                        # filename) and the summary
                        if redact_pii and pii_result.detected:
                            for msg in conv.messages:
                                msg.content = self.pii_detector.redact(msg.content)
                            conv.title = self.pii_detector.redact(conv.title)
                            if conv.summary:
                                conv.summary = self.pii_detector.redact(conv.summary)

                    state_metadata = {
                        "title": conv.title,
                        "pii_fingerprint": stored_fingerprint,
                    }

                    # Find related conversations based on tags [3]
                    related_convs = self._find_related_by_tags(
                        conv, conversations, redact=redact_pii
                    )

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
                                state_metadata,
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
                                    state_metadata,
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

                        else:
                            # File exists. Hash changed, or a PII flag bypassed
                            # the fast path.
                            if existing.get("content_hash") == current_hash:
                                # Unchanged content re-analyzed under a PII flag.
                                action = "reanalyzed"
                                results["reanalyzed"] += 1
                            else:
                                # Real content change.
                                action = "updated"
                                results["updated"] += 1

                            if not dry_run:
                                self.markdown_gen.save(conv, file_path, related_convs)
                                self.state.save_conversation(
                                    conv.id,
                                    str(file_path.relative_to(self.vault_path)),
                                    current_hash,
                                    state_metadata,
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

    def _generate_path(self, conversation, title: Optional[str] = None) -> Path:
        """
        Generate file path for conversation

        Args:
            conversation: Conversation object
            title: Title to build the filename from, if not conversation.title

        Returns:
            Path object for the markdown file
        """
        date_str = conversation.created_at.strftime("%Y-%m-%d")

        # Create safe filename from title
        safe_title = re.sub(r"[^\w\s-]", "", title or conversation.title)
        safe_title = re.sub(r"[-\s]+", "-", safe_title)
        safe_title = safe_title[:50]  # Limit length

        filename = f"{date_str}-{safe_title}.md"
        return self.conversations_dir / filename

    def _existing_file(self, existing: Optional[Dict]) -> Optional[Path]:
        """The vault file already written for a conversation, if there is one"""
        if not existing:
            return None
        recorded: Path = self.vault_path / existing["file_path"]
        if recorded.exists():
            return recorded
        return self._find_moved_file(existing["uuid"])

    def _report_skipped(
        self,
        results: Dict[str, Any],
        conv: Conversation,
        on_disk: Optional[Path],
        risk_level: str = "",
        pii_types: Optional[List[str]] = None,
    ) -> None:
        """Count a skipped conversation, and say if its file is still in the vault.

        skip_sensitive keeps new sensitive conversations out; it does not touch a
        file that was written earlier, which may have been edited. Reporting it
        lets the user decide instead of believing the vault is clean.
        """
        results["skipped"] += 1
        detail: Dict[str, Any] = {
            "action": "skipped",
            "title": conv.title,
            "risk_level": risk_level,
            "pii_types": pii_types or [],
        }
        if on_disk:
            relative = str(on_disk.relative_to(self.vault_path))
            detail["file_path"] = relative
            results["skipped_on_disk"].append(relative)
        results["details"].append(detail)

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
        self,
        conversation: Conversation,
        all_conversations: list[Conversation],
        redact: bool = False,
    ) -> list[Dict]:
        """Find conversations with similar tags

        With redact on, the related titles and filenames are the redacted ones,
        which is what those conversations are written under. Without this the
        result depended on processing order: conversations already handled had
        redacted titles in the shared list and the rest did not.
        """

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
                other_title = other_conv.title
                if redact:
                    other_title = self.pii_detector.redact(other_title)
                related.append(
                    {
                        "id": other_conv.id,
                        "title": other_title,
                        "common_tags": list(common_tags),
                        "file": self._generate_path(other_conv, other_title).name,
                    }
                )

        # Return top 5 most related
        related.sort(key=lambda x: len(x["common_tags"]), reverse=True)
        return related[:5]

import re
from pathlib import Path
from typing import Dict

import frontmatter

from .markdown import MarkdownGenerator
from .models import Conversation
from .parser import ClaudeExportParser
from .state import StateManager
from .tagging import OfflineTagGenerator


class SyncEngine:
    """Main sync engine for Claude Vault"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.state = StateManager(vault_path)
        self.parser = ClaudeExportParser()
        self.markdown_gen = MarkdownGenerator()
        self.conversations_dir = vault_path / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.tag_generator = OfflineTagGenerator()

    def sync(self, export_path: Path) -> Dict:
        """
        Sync conversations from Claude export to markdown files

        Args:
            export_path: Path to conversations.json export

        Returns:
            Dictionary with sync statistics
        """
        results = {"new": 0, "updated": 0, "unchanged": 0, "recreated": 0, "errors": 0}

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

            for conv in conversations:
                try:
                    existing = self.state.get_conversation(conv.id)
                    current_hash = conv.content_hash()

                    # Generate tags if missing or insufficient
                    if not conv.tags or len(conv.tags) < 2:
                        conv.tags = self.tag_generator.generate_tags(conv)

                    # Find related conversations based on tags [3]
                    related_convs = self._find_related_by_tags(conv, conversations)

                    if not existing:
                        # New conversation
                        file_path = self._generate_path(conv)
                        self.markdown_gen.save(conv, file_path)
                        self.state.save_conversation(
                            conv.id,
                            str(file_path.relative_to(self.vault_path)),
                            current_hash,
                            {"title": conv.title},
                        )
                        results["new"] += 1

                    else:
                        # Conversation exists in database
                        file_path = self.vault_path / existing["file_path"]

                        # Check if file exists
                        if not file_path.exists():
                            # File deleted - recreate it [3]
                            file_path = self._generate_path(conv)
                            self.markdown_gen.save(conv, file_path, related_convs)
                            self.state.save_conversation(
                                conv.id,
                                str(file_path.relative_to(self.vault_path)),
                                current_hash,
                                {"title": conv.title},
                            )
                            results["recreated"] += 1
                            print(f"[yellow]⚠ Recreated: {file_path.name}[/yellow]")

                        elif existing["content_hash"] != current_hash:
                            # File exists but content changed [3]
                            self.markdown_gen.save(conv, file_path, related_convs)
                            self.state.save_conversation(
                                conv.id,
                                str(file_path.relative_to(self.vault_path)),
                                current_hash,
                                {"title": conv.title},
                            )
                            results["updated"] += 1

                        else:
                            # File exists and unchanged
                            results["unchanged"] += 1

                except Exception as e:
                    print(f"Error processing conversation {conv.title}: {e}")
                    results["errors"] += 1

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

    def _find_moved_file(self, uuid: str) -> Path:
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

        related = []
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

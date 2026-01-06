import re
from pathlib import Path
from typing import Dict

import frontmatter

from .markdown import MarkdownGenerator
from .models import Conversation
from .parser import ChatGPTExportParser, ClaudeExportParser
from .state import StateManager
from .tagging import OfflineTagGenerator, TagResult


class SyncEngine:
    """Main sync engine for Claude Vault"""

    def __init__(
        self,
        vault_path: Path,
        source: str = "claude",
        tag_model: str = "llama3.2:3b",
        tag_mode: str = "quick",
        use_hierarchy: bool = False,
    ):
        self.vault_path = vault_path
        self.source = source
        self.tag_mode = tag_mode
        self.use_hierarchy = use_hierarchy
        self.state = StateManager(vault_path)

        # Select appropriate parser based on source
        if source == "chatgpt":
            self.parser = ChatGPTExportParser()
        else:
            self.parser = ClaudeExportParser()

        self.markdown_gen = MarkdownGenerator(source=source, use_hierarchy=use_hierarchy)
        self.conversations_dir = vault_path / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.tag_generator = OfflineTagGenerator(model=tag_model)

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

            # Show if Ollama is available (use ASCII-safe characters)
            if self.tag_generator.is_available():
                print("* Ollama detected - using AI tag generation")
            else:
                print("! Ollama not running - using keyword extraction")
                print("  Tip: Start Ollama with 'ollama serve' for automatic tagging")

            for conv in conversations:
                try:
                    existing = self.state.get_conversation(conv.id)
                    current_hash = conv.content_hash()

                    # Generate tags if missing or insufficient
                    tag_result = None
                    if not conv.tags or len(conv.tags) < 2:
                        tag_result = self.tag_generator.generate_tags(
                            conv,
                            mode=self.tag_mode,
                            use_hierarchy=self.use_hierarchy,
                        )
                        conv.tags = tag_result.all_tags()

                    # Find related conversations based on tags
                    related_convs = self._find_related_by_tags(conv, conversations)

                    if not existing:
                        # New conversation
                        file_path = self._generate_path(conv)
                        self.markdown_gen.save(conv, file_path, related_convs, tag_result)
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
                            # File deleted - recreate it
                            file_path = self._generate_path(conv)
                            self.markdown_gen.save(conv, file_path, related_convs, tag_result)
                            self.state.save_conversation(
                                conv.id,
                                str(file_path.relative_to(self.vault_path)),
                                current_hash,
                                {"title": conv.title},
                            )
                            results["recreated"] += 1
                            print(f"! Recreated: {file_path.name}")

                        elif existing["content_hash"] != current_hash:
                            # File exists but content changed
                            self.markdown_gen.save(conv, file_path, related_convs, tag_result)
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

            # At least 2 common tags = related
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

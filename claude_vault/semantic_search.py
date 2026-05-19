"""Semantic search engine for conversations"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, cast

import frontmatter
import numpy as np
from rich.console import Console
from rich.progress import Progress

from claude_vault.embeddings import (
    ConversationChunker,
    EmbeddingGenerator,
    cosine_similarity,
)
from claude_vault.state import StateManager

console = Console()


@dataclass
class SearchResult:
    """Search result with relevance score"""

    conversation_uuid: str
    title: str
    file_path: str
    chunk_text: str
    score: float
    rank: int
    tags: List[str]
    metadata: Dict


class SemanticSearchEngine:
    """Semantic search using embeddings"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.state = StateManager(vault_path)
        self.generator = EmbeddingGenerator()
        self.chunker = ConversationChunker()

    def is_available(self) -> bool:
        """Check if semantic search is available"""
        return self.generator.is_available()

    def search(
        self, query: str, limit: int = 10, threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Perform semantic search

        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        if not self.is_available():
            console.print(
                "[yellow]⚠ Ollama not available. Cannot perform semantic search.[/yellow]"
            )
            return []

        # Generate query embedding
        console.print("[dim]Generating query embedding...[/dim]")
        query_embedding = self.generator.generate_embedding(query)

        if not query_embedding:
            console.print("[red]✗ Failed to generate query embedding[/red]")
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)

        # Get all embeddings from database
        all_embeddings = self.state.get_all_embeddings()

        if not all_embeddings:
            console.print(
                "[yellow]⚠ No embeddings found. Generate embeddings first with semantic search.[/yellow]"
            )
            return []

        # Calculate similarities
        similarities = []
        for emb in all_embeddings:
            chunk_embedding = emb["embedding"]
            score = cosine_similarity(query_vector, chunk_embedding)

            if score >= threshold:
                similarities.append(
                    {
                        "conversation_uuid": emb["conversation_uuid"],
                        "chunk_index": emb["chunk_index"],
                        "chunk_text": emb["chunk_text"],
                        "file_path": emb["file_path"],
                        "score": float(score),
                        "metadata": emb["metadata"],
                    }
                )

        # Group by conversation and aggregate scores
        conversation_scores: DefaultDict[str, Dict[str, Any]] = defaultdict(
            lambda: {"chunks": [], "max_score": 0.0}
        )
        for sim in similarities:
            conv_id = sim["conversation_uuid"]
            conversation_scores[conv_id]["chunks"].append(sim)
            conversation_scores[conv_id]["max_score"] = max(
                conversation_scores[conv_id]["max_score"], sim["score"]
            )
            conversation_scores[conv_id]["file_path"] = sim["file_path"]
            conversation_scores[conv_id]["metadata"] = sim["metadata"]

        # Rank conversations by max chunk score
        ranked = sorted(
            conversation_scores.items(),
            key=lambda x: x[1]["max_score"],
            reverse=True,
        )[:limit]

        # Format results
        results = []
        for rank, (conv_id, data) in enumerate(ranked, 1):
            # Get best matching chunk
            best_chunk = max(data["chunks"], key=lambda x: x["score"])

            # Load frontmatter to get title and tags
            file_path = Path(data["file_path"])
            if file_path.exists():
                post = frontmatter.load(file_path)
                title = cast(str, post.get("title", file_path.stem))
                tags = cast(List[str], post.get("tags", []))
            else:
                title = conv_id
                tags = []

            results.append(
                SearchResult(
                    conversation_uuid=conv_id,
                    title=title,
                    file_path=str(file_path),
                    chunk_text=best_chunk["chunk_text"],
                    score=best_chunk["score"],
                    rank=rank,
                    tags=tags,
                    metadata=data["metadata"],
                )
            )

        return results

    def ensure_embeddings_exist(self):
        """Generate embeddings for conversations that don't have them"""
        conversations_dir = self.vault_path / "conversations"
        if not conversations_dir.exists():
            console.print("[yellow]⚠ No conversations directory found[/yellow]")
            return

        # Get all conversation files
        md_files = list(conversations_dir.glob("*.md"))

        # Get conversations without embeddings
        all_convs = self.state.get_all_conversations()
        conv_with_embeddings = set()

        for conv in all_convs:
            embeddings = self.state.get_embeddings_for_conversation(conv["uuid"])
            if embeddings:
                conv_with_embeddings.add(conv["uuid"])

        # Find conversations that need embeddings
        needs_embedding = []
        for md_file in md_files:
            post = frontmatter.load(md_file)
            uuid = post.get("uuid")
            if uuid and uuid not in conv_with_embeddings:
                needs_embedding.append((uuid, md_file, post))

        if not needs_embedding:
            console.print("[green]✓ All conversations have embeddings[/green]")
            return

        console.print(
            f"[yellow]Generating embeddings for {len(needs_embedding)} conversations...[/yellow]"
        )

        with Progress() as progress:
            task = progress.add_task("Embedding...", total=len(needs_embedding))

            for uuid, _md_file, post in needs_embedding:
                self._generate_embeddings_for_file(uuid, post)
                progress.advance(task)

        console.print("[green]✓ Embeddings generated successfully![/green]")

    def _generate_embeddings_for_file(self, uuid: str, post: frontmatter.Post):
        """Generate embeddings for a single conversation file"""
        from claude_vault.models import Conversation, Message

        # Build conversation object
        title = cast(str, post.get("title", ""))
        content = post.content

        # Parse messages from markdown
        messages = []
        current_role = None
        current_content: List[str] = []

        for line in content.split("\n"):
            if line.startswith("## 👤 You"):
                if current_role and current_content:
                    messages.append(
                        Message(
                            role=current_role,
                            content="\n".join(current_content).strip(),
                        )
                    )
                current_role = "human"
                current_content = []
            elif line.startswith("## 🤖 Claude"):
                if current_role and current_content:
                    messages.append(
                        Message(
                            role=current_role,
                            content="\n".join(current_content).strip(),
                        )
                    )
                current_role = "assistant"
                current_content = []
            elif line.strip() == "---":
                continue
            else:
                current_content.append(line)

        # Add last message
        if current_role and current_content:
            messages.append(
                Message(role=current_role, content="\n".join(current_content).strip())
            )

        # Create conversation object
        conversation = Conversation(
            id=uuid,
            title=title,
            messages=messages,
            created_at=post.get("date"),  # type: ignore[arg-type]
            updated_at=post.get("updated", post.get("date")),  # type: ignore[arg-type]
            tags=cast(List[str], post.get("tags", [])),
            summary=post.get("summary") or None,  # type: ignore[arg-type]
        )

        # Chunk conversation
        chunks = self.chunker.chunk_conversation(conversation)

        # Generate embeddings for each chunk
        for chunk in chunks:
            embedding = self.generator.generate_embedding(chunk["text"])
            if embedding:
                embedding_array = np.array(embedding, dtype=np.float32)
                self.state.save_embedding(
                    conversation_uuid=uuid,
                    chunk_index=chunk["chunk_index"],
                    chunk_text=chunk["text"][:500],  # Store preview
                    embedding=embedding_array,
                    model=self.generator.model,
                )

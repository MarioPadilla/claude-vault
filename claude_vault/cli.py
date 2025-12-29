import json
from pathlib import Path
from typing import List, Optional

import frontmatter
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .code_parser import ClaudeCodeHistoryParser
from .parser import ClaudeExportParser
from .state import StateManager
from .sync import SyncEngine
from .tagging import OfflineTagGenerator

app = typer.Typer(help="Claude Vault - Sync Claude conversations to Obsidian")
console = Console()


# Helper function for search
def find_matches_with_context(
    content: str, keyword: str, context_chars: int = 100
) -> List[str]:
    """Find keyword matches with surrounding context"""
    matches = []
    content_lower = content.lower()
    keyword_lower = keyword.lower()

    index = 0
    while True:
        index = content_lower.find(keyword_lower, index)
        if index == -1:
            break

        # Extract context around match
        start = max(0, index - context_chars)
        end = min(len(content), index + len(keyword) + context_chars)
        context = content[start:end]

        matches.append(context)
        index += len(keyword)

    return matches


@app.command()
def init(vault_path: Optional[Path] = None):
    """Initialize Claude Vault in the specified directory"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"

    if config_dir.exists():
        console.print(
            "[yellow]⚠ Claude Vault already initialized in this directory[/yellow]"
        )
        return

    config_dir.mkdir(parents=True, exist_ok=True)

    # Create default config
    config_file = config_dir / "config.json"
    default_config = {
        "naming_pattern": "{date}-{title}",
        "folder_structure": "flat",
        "template": "default",
        "version": "0.1.0",
    }

    config_file.write_text(json.dumps(default_config, indent=2))

    # Create conversations directory
    (vault_path / "conversations").mkdir(exist_ok=True)

    console.print(f"[green]✓ Claude Vault initialized at {vault_path}[/green]")
    console.print(f"[dim]Config stored in {config_dir}[/dim]")
    console.print("\n[blue]Next steps:[/blue]")
    console.print("  1. Export your Claude conversations (conversations.json)")
    console.print("  2. Run: claude-vault sync path/to/conversations.json")


@app.command()
def sync(
    export_path: Path,
    vault_path: Optional[Path] = None,
    source: str = typer.Option("auto", help="Source type: auto, web, code"),
):
    """Sync Claude conversations to markdown files"""

    vault_path = vault_path or Path.cwd()
    if not export_path.exists():
        console.print(f"[red]✗ Error: Export file not found: {export_path}[/red]")
        raise typer.Exit(1)

    config_dir = vault_path / ".claude-vault"
    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        console.print("[yellow]Run 'claude-vault init' first[/yellow]")
        raise typer.Exit(1)

    # Detect format
    if source == "auto":
        if export_path.is_dir() and export_path.name == ".claude":
            source = "code"
        elif export_path.suffix == ".jsonl":
            source = "code"
        else:
            source = "web"

    console.print(f"[blue]📦 Syncing conversations from {export_path.name}...[/blue]\n")

    # Use appropriate parser
    if source == "code":
        parser = ClaudeCodeHistoryParser()
    else:
        parser = ClaudeExportParser()

    engine = SyncEngine(vault_path)
    engine.parser = parser

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing conversations...", total=None)
        result = engine.sync(export_path)
        progress.update(task, completed=100)

    # Display results
    console.print("\n[green]✓ Sync complete![/green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Status", style="dim")
    table.add_column("Count", justify="right")

    table.add_row("New conversations", f"[green]{result['new']}[/green]")
    table.add_row("Updated", f"[yellow]{result['updated']}[/yellow]")
    table.add_row("Recreated", f"[yellow]{result['recreated']}[/yellow]")
    table.add_row("Unchanged", f"[dim]{result['unchanged']}[/dim]")
    if result["errors"] > 0:
        table.add_row("Errors", f"[red]{result['errors']}[/red]")

    console.print(table)
    console.print(
        f"\n[dim]Conversations saved to: {vault_path / 'conversations'}[/dim]"
    )


@app.command()
def status(vault_path: Optional[Path] = None):
    """Show Claude Vault status and statistics"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"
    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    state = StateManager(vault_path)
    conversations = state.get_all_conversations()

    console.print("\n[blue]📊 Claude Vault Status[/blue]\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Vault Location", str(vault_path))
    table.add_row("Conversations Tracked", str(len(conversations)))
    table.add_row("Storage", str(config_dir))

    if conversations:
        latest = max(conversations, key=lambda x: x["last_synced"])
        table.add_row("Last Sync", latest["last_synced"][:19])

    console.print(table)
    console.print()


@app.command()
def verify(
    vault_path: Optional[Path] = None,
    cleanup: bool = typer.Option(
        False, "--cleanup", help="Remove orphaned database entries"
    ),
):
    """Verify integrity of tracked conversations and optionally clean up mismatches"""
    vault_path = vault_path or Path.cwd()
    state = StateManager(vault_path)
    conversations = state.get_all_conversations()

    console.print(f"[blue]🔍 Verifying {len(conversations)} conversations...[/blue]\n")

    missing = []
    for conv in conversations:
        file_path = vault_path / conv["file_path"]
        if not file_path.exists():
            missing.append(conv)

    if missing:
        console.print(f"[yellow]⚠ Found {len(missing)} missing files:[/yellow]")
        for conv in missing:
            console.print(f"  - {conv['file_path']}")

        if cleanup:
            console.print(
                f"\n[yellow]Cleaning up {len(missing)} orphaned entries...[/yellow]"
            )

            for conv in missing:
                state.delete_conversation(conv["uuid"])
                console.print(f"  ✓ Removed: {conv['file_path']}")

            console.print(
                f"\n[green]✓ Cleaned up {len(missing)} orphaned database entries[/green]"
            )
        else:
            console.print(
                "\n[dim]Tip: Run with --cleanup flag to remove these entries from database[/dim]"
            )
            console.print("[dim]Command: claude-vault verify --cleanup[/dim]")
    else:
        console.print("[green]✓ All conversations verified successfully![/green]")


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Search term"),
    vault_path: Optional[Path] = None,
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    show_related: bool = typer.Option(True, help="Show related conversations"),
):
    """Search across all conversations"""

    vault_path = vault_path or Path.cwd()
    conversations_dir = vault_path / "conversations"
    results = []

    # Search through all markdown files
    for md_file in conversations_dir.glob("*.md"):
        try:
            post = frontmatter.load(md_file)

            # Check if keyword appears in content
            if (
                keyword.lower() in post.content.lower()
                or keyword.lower() in post.get("title", "").lower()
            ):
                # Optional tag filtering
                if tag and tag not in post.get("tags", []):
                    continue

                # Find matches with context
                matches = find_matches_with_context(post.content, keyword)

                results.append(
                    {
                        "file": md_file.name,
                        "title": post.get("title", ""),
                        "tags": post.get("tags", []),
                        "related": post.get("related", []),
                        "related_tags": post.get("related_tags", {}),
                        "matches": matches,
                        "match_count": len(matches),
                        "path": md_file,
                    }
                )
        except Exception:
            continue

    # Display results
    if results:
        console.print(f"\n[green]Found in {len(results)} conversations:[/green]\n")
        for i, result in enumerate(results, 1):
            console.print(
                f"{i}. [{result['file']}] {result['title']} ({result['match_count']} matches)"
            )
            console.print(
                f"   Tags: {', '.join(result['tags']) if result['tags'] else '[dim]no tags[/dim]'}"
            )

            # Show related conversations, if enabled, with common tags
            if show_related and result["related"]:
                console.print("   [yellow]Related:[/yellow]")
                for rel_conv in result["related"][:3]:
                    # Clean wikilink format
                    clean_name = rel_conv.replace("[[", "").replace("]]", "")

                    # Show common tags if available
                    if result["related_tags"] and clean_name in result["related_tags"]:
                        common_tags = ", ".join(result["related_tags"][clean_name])
                        console.print(
                            f"      • {clean_name} [dim](common: {common_tags})[/dim]"
                        )
                    else:
                        console.print(f"      • {clean_name}")

            for match in result["matches"][:2]:  # Show first 2 matches
                console.print(f"   [dim]...{match}...[/dim]")
            console.print()

        # Display with file opening
        console.print("\n[blue]Open result?[/blue]")
        choice = typer.prompt("Enter number (or 'q' to quit)")

        if choice.isdigit() and 1 <= int(choice) <= len(results):
            selected = results[int(choice) - 1]
            # Open in default editor or show full content
            typer.launch(str(vault_path / "conversations" / selected["file"]))
        else:
            print("Exiting without opening any files.")
    else:
        console.print("[yellow]No matches found[/yellow]")


@app.command()
def retag(
    vault_path: Optional[Path] = None,
    force: bool = typer.Option(False, help="Regenerate all tags, even existing ones"),
):
    """Regenerate tags for conversations using AI"""

    vault_path = vault_path or Path.cwd()
    tag_gen = OfflineTagGenerator()
    parser = ClaudeExportParser()

    if not tag_gen.is_available():
        console.print("[red]✗ Ollama not running[/red]")
        console.print("Start Ollama with: ollama serve")
        console.print(f"Model needed: ollama pull {tag_gen.model}")
        raise typer.Exit(1)

    conversations_dir = vault_path / "conversations"
    updated = 0

    console.print("[blue]Regenerating tags...[/blue]\n")

    for md_file in conversations_dir.glob("*.md"):
        try:
            post = frontmatter.load(md_file)

            # Skip if has good tags and not forcing
            if not force and post.get("tags") and len(post.get("tags", [])) >= 3:
                continue

            # Parse conversation from file
            conv = parser.parse_conversation_from_markdown(post)

            # Generate new tags
            new_tags = tag_gen.generate_tags(conv)
            post["tags"] = new_tags

            # Save updated file
            md_file.write_text(frontmatter.dumps(post))
            updated += 1

            console.print(f"✓ {md_file.name}: {', '.join(new_tags)}")

        except Exception as e:
            console.print(f"✗ {md_file.name}: {e}")

    console.print(f"\n[green]Updated {updated} conversations[/green]")


if __name__ == "__main__":
    app()

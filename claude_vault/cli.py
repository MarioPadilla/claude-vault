import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import frontmatter
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .code_parser import ClaudeCodeHistoryParser
from .config import get_config_path, load_config
from .opencode_parser import OpenCodeParser
from .parser import ClaudeExportParser
from .semantic_search import SemanticSearchEngine
from .state import StateManager
from .sync import SyncEngine
from .tagging import OfflineTagGenerator
from .watcher import WatchManager

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
    export_path: Path = typer.Argument(None, help="Path to export file or database"),
    vault_path: Optional[Path] = None,
    source: str = typer.Option("auto", help="Source type: auto, web, code, opencode"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying them"
    ),
    detect_pii: bool = typer.Option(
        False,
        "--detect-pii",
        help="Scan conversations for PII and sensitive content (regex + optional Ollama LLM)",
    ),
    redact_pii: bool = typer.Option(
        False,
        "--redact-pii",
        help="Replace detected PII with [REDACTED-TYPE] placeholders before writing",
    ),
    skip_sensitive: bool = typer.Option(
        False,
        "--skip-sensitive",
        help="Skip syncing conversations flagged as sensitive (medium risk or higher)",
    ),
):
    """Sync Claude conversations to markdown files"""

    vault_path = vault_path or Path.cwd()

    # Resolve default path for opencode source
    if source == "opencode" and export_path is None:
        export_path = OpenCodeParser.DEFAULT_DB_PATH

    if export_path is None:
        console.print(
            "[red]✗ Error: No export path provided. Use: claude-vault sync <path>[/red]"
        )
        raise typer.Exit(1)

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
        elif export_path.suffix == ".db" or export_path.name == "opencode.db":
            source = "opencode"
        else:
            source = "web"

    if dry_run:
        console.print(
            f"[yellow]🔍 DRY RUN: Previewing changes from {export_path.name}...[/yellow]\n"
        )
    else:
        console.print(
            f"[blue]📦 Syncing conversations from {export_path.name}...[/blue]\n"
        )

    # Use appropriate parser
    parser: Union[ClaudeCodeHistoryParser, ClaudeExportParser, OpenCodeParser]
    if source == "code":
        parser = ClaudeCodeHistoryParser()
    elif source == "opencode":
        parser = OpenCodeParser()
    else:
        parser = ClaudeExportParser()

    engine = SyncEngine(vault_path)
    engine.parser = parser

    # PII detection notices
    if detect_pii or redact_pii or skip_sensitive:
        console.print("[blue]🔒 PII detection enabled[/blue]")
        if redact_pii:
            console.print("[dim]  • PII will be redacted in stored files[/dim]")
        if skip_sensitive:
            console.print(
                "[dim]  • Conversations with medium/high risk will be skipped[/dim]"
            )

    from rich.progress import BarColumn, MofNCompleteColumn, TimeRemainingColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing conversations...", total=100)

        def update_progress(description: str, current: int, total: int):
            progress.update(
                task, description=f"[cyan]{description}", completed=current, total=total
            )

        result = engine.sync(
            export_path,
            dry_run=dry_run,
            progress_callback=update_progress,
            detect_pii=detect_pii,
            redact_pii=redact_pii,
            skip_sensitive=skip_sensitive,
        )

    # Display results
    if dry_run:
        console.print(
            "\n[yellow]🔍 DRY RUN: Preview of changes (no files modified)[/yellow]\n"
        )
    else:
        console.print("\n[green]✓ Sync complete![/green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Status", style="dim")
    table.add_column("Count", justify="right")

    table.add_row("New conversations", f"[green]{result['new']}[/green]")
    table.add_row("Updated", f"[yellow]{result['updated']}[/yellow]")
    table.add_row("Recreated", f"[yellow]{result['recreated']}[/yellow]")
    table.add_row("Unchanged", f"[dim]{result['unchanged']}[/dim]")
    if result.get("skipped", 0) > 0:
        table.add_row("Skipped (sensitive)", f"[magenta]{result['skipped']}[/magenta]")
    if result["errors"] > 0:
        table.add_row("Errors", f"[red]{result['errors']}[/red]")

    console.print(table)

    # Show details for dry-run or if there are changes
    if dry_run and (
        result["new"] > 0 or result["updated"] > 0 or result["recreated"] > 0
    ):
        console.print("\n[blue]Details:[/blue]")
        for detail in result.get("details", [])[:10]:  # Show first 10
            action = detail.get("action", "unknown")
            title = detail.get("title", "Unknown")
            file_path = detail.get("file_path", "")

            action_color = {
                "new": "green",
                "updated": "yellow",
                "recreated": "yellow",
                "skipped": "magenta",
                "error": "red",
            }.get(action, "white")

            console.print(
                f"  [{action_color}]{action.upper()}[/{action_color}]: {title}"
            )
            if file_path:
                console.print(f"    [dim]→ {file_path}[/dim]")

        if len(result.get("details", [])) > 10:
            console.print(f"\n[dim]... and {len(result['details']) - 10} more[/dim]")

        console.print(
            "\n[yellow]💡 Run without --dry-run to apply these changes[/yellow]"
        )
    elif not dry_run:
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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview cleanup without applying it"
    ),
):
    """Verify integrity of tracked conversations and optionally clean up mismatches"""
    vault_path = vault_path or Path.cwd()
    state = StateManager(vault_path)
    conversations = state.get_all_conversations()

    if dry_run and cleanup:
        console.print(
            f"[yellow]🔍 DRY RUN: Verifying {len(conversations)} conversations...[/yellow]\n"
        )
    else:
        console.print(
            f"[blue]🔍 Verifying {len(conversations)} conversations...[/blue]\n"
        )

    from rich.progress import BarColumn, MofNCompleteColumn

    missing = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Checking files...", total=len(conversations))

        for conv in conversations:
            file_path = vault_path / conv["file_path"]
            if not file_path.exists():
                missing.append(conv)
            progress.advance(task)

    if missing:
        console.print(f"\n[yellow]⚠ Found {len(missing)} missing files:[/yellow]")
        for conv in missing[:10]:  # Show first 10
            console.print(f"  - {conv['file_path']}")

        if len(missing) > 10:
            console.print(f"  [dim]... and {len(missing) - 10} more[/dim]")

        if cleanup:
            if dry_run:
                console.print(
                    f"\n[yellow]🔍 DRY RUN: Would clean up {len(missing)} orphaned entries[/yellow]"
                )
                console.print("\n[blue]Entries that would be removed:[/blue]")
                for conv in missing[:10]:
                    console.print(
                        f"  - {conv['file_path']} (UUID: {conv['uuid'][:8]}...)"
                    )

                if len(missing) > 10:
                    console.print(f"  [dim]... and {len(missing) - 10} more[/dim]")

                console.print(
                    "\n[yellow]💡 Run without --dry-run to apply cleanup[/yellow]"
                )
            else:
                console.print(
                    f"\n[yellow]Cleaning up {len(missing)} orphaned entries...[/yellow]"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Removing entries...", total=len(missing)
                    )

                    for conv in missing:
                        state.delete_conversation(conv["uuid"])
                        console.print(f"  ✓ Removed: {conv['file_path']}")
                        progress.advance(task)

                console.print(
                    f"\n[green]✓ Cleaned up {len(missing)} orphaned database entries[/green]"
                )
        else:
            console.print(
                "\n[dim]Tip: Run with --cleanup flag to remove these entries from database[/dim]"
            )
            console.print("[dim]Command: claude-vault verify --cleanup[/dim]")
            console.print(
                "[dim]Preview with: claude-vault verify --cleanup --dry-run[/dim]"
            )
    else:
        console.print("\n[green]✓ All conversations verified successfully![/green]")


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Search term"),
    vault_path: Optional[Path] = None,
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    show_related: bool = typer.Option(True, help="Show related conversations"),
    mode: str = typer.Option(
        "auto",
        help="Search mode: auto (semantic if available), semantic, keyword, or hybrid",
    ),
    threshold: float = typer.Option(
        0.5, help="Minimum similarity score for semantic search (0.0-1.0)"
    ),
):
    """Search across all conversations with semantic understanding"""

    vault_path = vault_path or Path.cwd()
    conversations_dir = vault_path / "conversations"

    # Determine search mode
    semantic_engine = SemanticSearchEngine(vault_path)

    if mode == "auto":
        # Use semantic if available, fallback to keyword
        if semantic_engine.is_available():
            mode = "semantic"
        else:
            console.print(
                "[yellow]⚠ Ollama not available. Using keyword search.[/yellow]"
            )
            console.print(
                "[dim]Tip: Start Ollama with 'ollama serve' and 'ollama pull nomic-embed-text' for semantic search[/dim]\n"
            )
            mode = "keyword"

    # Perform search based on mode
    if mode == "semantic":
        # Ensure embeddings exist
        semantic_engine.ensure_embeddings_exist()

        # Perform semantic search
        semantic_results = semantic_engine.search(
            keyword, limit=10, threshold=threshold
        )

        if semantic_results:
            console.print(
                f"\n[green]Found in {len(semantic_results)} conversations (semantic search):[/green]\n"
            )
            for result in semantic_results:
                console.print(
                    f"{result.rank}. [{Path(result.file_path).name}] {result.title} (score: {result.score:.2f})"
                )
                console.print(
                    f"   Tags: {', '.join(result.tags) if result.tags else '[dim]no tags[/dim]'}"
                )
                # Show matching chunk preview
                preview = result.chunk_text[:200].replace("\n", " ")
                console.print(f"   [dim]{preview}...[/dim]")
                console.print()

            # Display with file opening
            console.print("\n[blue]Open result?[/blue]")
            choice = typer.prompt("Enter number (or 'q' to quit)")

            if choice.isdigit() and 1 <= int(choice) <= len(semantic_results):
                selected = semantic_results[int(choice) - 1]
                typer.launch(selected.file_path)
        else:
            console.print("[yellow]No matches found[/yellow]")
            console.print(
                f"[dim]Try lowering the threshold (current: {threshold})[/dim]"
            )
        return

    # Keyword search (original implementation)
    results: List[Dict[str, Any]] = []

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
        for i, kw_result in enumerate(results, 1):
            console.print(
                f"{i}. [{kw_result['file']}] {kw_result['title']} ({kw_result['match_count']} matches)"
            )
            console.print(
                f"   Tags: {', '.join(kw_result['tags']) if kw_result['tags'] else '[dim]no tags[/dim]'}"
            )

            # Show related conversations, if enabled, with common tags
            if show_related and kw_result["related"]:
                console.print("   [yellow]Related:[/yellow]")
                for rel_conv in kw_result["related"][:3]:
                    # Clean wikilink format
                    clean_name = rel_conv.replace("[[", "").replace("]]", "")

                    # Show common tags if available
                    if (
                        kw_result["related_tags"]
                        and clean_name in kw_result["related_tags"]
                    ):
                        common_tags = ", ".join(kw_result["related_tags"][clean_name])
                        console.print(
                            f"      • {clean_name} [dim](common: {common_tags})[/dim]"
                        )
                    else:
                        console.print(f"      • {clean_name}")

            for match in kw_result["matches"][:2]:  # Show first 2 matches
                console.print(f"   [dim]...{match}...[/dim]")
            console.print()

        # Display with file opening
        console.print("\n[blue]Open result?[/blue]")
        choice = typer.prompt("Enter number (or 'q' to quit)")

        if choice.isdigit() and 1 <= int(choice) <= len(results):
            kw_selected: Dict[str, Any] = results[int(choice) - 1]
            # Open in default editor or show full content
            typer.launch(str(vault_path / "conversations" / kw_selected["file"]))
        else:
            print("Exiting without opening any files.")
    else:
        console.print("[yellow]No matches found[/yellow]")


@app.command()
def retag(
    vault_path: Optional[Path] = None,
    force: bool = typer.Option(False, help="Regenerate all tags, even existing ones"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying them"
    ),
):
    """Regenerate tags for conversations using AI"""

    vault_path = vault_path or Path.cwd()
    tag_gen = OfflineTagGenerator()
    parser = ClaudeExportParser()

    # Check config
    config = tag_gen.config
    console.print(f"[dim]Using model: {config.ollama.model}[/dim]")

    if not tag_gen.is_available():
        console.print("[red]✗ Ollama not running[/red]")
        console.print("Start Ollama with: ollama serve")
        console.print(f"Model needed: ollama pull {config.ollama.model}")
        raise typer.Exit(1)

    conversations_dir = vault_path / "conversations"
    updated = 0
    details = []

    if dry_run:
        console.print("[yellow]🔍 DRY RUN: Previewing tag regeneration...[/yellow]\n")
    else:
        console.print("[blue]Regenerating tags...[/blue]\n")

    # Collect files to process
    files_to_process = []
    for md_file in conversations_dir.glob("*.md"):
        try:
            post = frontmatter.load(md_file)
            # Skip if has good tags and not forcing
            if not force and post.get("tags") and len(post.get("tags", [])) >= 3:
                continue
            files_to_process.append((md_file, post))
        except Exception:
            continue

    from rich.progress import BarColumn, MofNCompleteColumn, TimeRemainingColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing conversations...", total=len(files_to_process)
        )

        for md_file, post in files_to_process:
            try:
                progress.update(
                    task, description=f"[cyan]Processing {md_file.name[:40]}..."
                )

                # Parse conversation from file
                conv = parser.parse_conversation_from_markdown(post)

                # Generate new metadata
                metadata = tag_gen.generate_metadata(conv)

                if dry_run:
                    # Just collect details
                    details.append(
                        {
                            "file": md_file.name,
                            "old_tags": post.get("tags", []),
                            "new_tags": metadata["tags"],
                            "summary": metadata.get("summary"),
                        }
                    )
                    updated += 1
                else:
                    # Actually update the file
                    post["tags"] = metadata["tags"]
                    if metadata["summary"]:
                        post["summary"] = metadata["summary"]

                    # Save updated file
                    md_file.write_text(frontmatter.dumps(post))
                    updated += 1

                    summary_preview = (
                        f" (Summary: {metadata['summary'][:30]}...)"
                        if metadata["summary"]
                        else ""
                    )
                    console.print(
                        f"✓ {md_file.name}: {', '.join(metadata['tags'])}{summary_preview}"
                    )

                progress.advance(task)

            except Exception as e:
                console.print(f"✗ {md_file.name}: {e}")
                progress.advance(task)

    if dry_run:
        console.print(
            f"\n[yellow]🔍 DRY RUN: Would update {updated} conversations[/yellow]\n"
        )

        if details:
            console.print("[blue]Preview of changes (first 10):[/blue]")
            for detail in details[:10]:
                console.print(f"\n  [cyan]{detail['file']}[/cyan]")
                console.print(
                    f"    Old tags: {', '.join(detail['old_tags']) if detail['old_tags'] else '[dim]none[/dim]'}"
                )
                console.print(
                    f"    New tags: [green]{', '.join(detail['new_tags'])}[/green]"
                )
                if detail.get("summary"):
                    console.print(
                        f"    Summary: [dim]{detail['summary'][:60]}...[/dim]"
                    )

            if len(details) > 10:
                console.print(f"\n[dim]... and {len(details) - 10} more[/dim]")

        console.print(
            "\n[yellow]💡 Run without --dry-run to apply these changes[/yellow]"
        )
    else:
        console.print(f"\n[green]Updated {updated} conversations[/green]")


@app.command()
def config():
    """Manage Global Config for Claude Vault"""
    config = load_config()
    console.print(f"[blue]Configuration Path:[/blue] {get_config_path()}\n")

    console.print("[green]Current Settings:[/green]")
    console.print(json.dumps(config.model_dump(), indent=2))

    if typer.confirm("Do you want to edit the configuration?"):
        typer.launch(str(get_config_path()))


@app.command()
def watch(vault_path: Optional[Path] = None):
    """Start watching for conversation changes"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"

    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        console.print("[yellow]Run 'claude-vault init' first[/yellow]")
        raise typer.Exit(1)

    watch_manager = WatchManager(vault_path)

    try:
        watch_manager.start()
    except RuntimeError as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        console.print("[dim]Check watch status with: claude-vault watch-status[/dim]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")


@app.command()
def watch_add(
    path: Path = typer.Argument(..., help="Path to watch"),
    source: str = typer.Option("auto", help="Source type: auto, web, code, opencode"),
    vault_path: Optional[Path] = None,
):
    """Add a path to watch list"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"

    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    # Expand user path
    path = path.expanduser()

    # Auto-detect source type
    if source == "auto":
        if path.suffix == ".json":
            source = "web"
        elif path.suffix == ".jsonl" or ".claude" in str(path):
            source = "code"
        elif path.suffix == ".db" or path.name == "opencode.db":
            source = "opencode"
        else:
            source = "web"  # Default

    # Add to state
    state = StateManager(vault_path)
    state.add_watch_path(str(path), source)

    console.print(f"[green]✓ Added {path} to watch list ({source} exports)[/green]")
    console.print("[dim]Start watching with: claude-vault watch[/dim]")


@app.command()
def watch_remove(
    path: Path = typer.Argument(..., help="Path to remove from watch list"),
    vault_path: Optional[Path] = None,
):
    """Remove a path from watch list"""

    vault_path = vault_path or Path.cwd()
    path = path.expanduser()

    state = StateManager(vault_path)
    state.remove_watch_path(str(path))

    console.print(f"[green]✓ Removed {path} from watch list[/green]")


@app.command()
def watch_status(vault_path: Optional[Path] = None):
    """Show watch status and statistics"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"

    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    watch_manager = WatchManager(vault_path)
    status = watch_manager.get_status()

    console.print("\n[blue]📊 Watch Status[/blue]\n")

    # Status table
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    if status["is_running"]:
        table.add_row("Status", "[green]Running[/green]")
        table.add_row("PID", str(status["pid"]))
    else:
        table.add_row("Status", "[yellow]Stopped[/yellow]")

    if status["last_started"]:
        table.add_row("Last Started", status["last_started"][:19])
    if status["last_stopped"]:
        table.add_row("Last Stopped", status["last_stopped"][:19])

    table.add_row("Total Syncs", str(status["total_syncs"]))
    table.add_row("Total Errors", str(status["total_errors"]))

    console.print(table)

    # Watch paths
    if status["watch_paths"]:
        console.print("\n[blue]Watch Paths:[/blue]")
        for wp in status["watch_paths"]:
            last_sync = (
                f"Last sync: {wp['last_sync'][:19]}"
                if wp["last_sync"]
                else "Never synced"
            )
            console.print(f"  • {wp['path']} ({wp['source_type']}) - {last_sync}")
    else:
        console.print("\n[yellow]No watch paths configured[/yellow]")
        console.print("[dim]Add paths with: claude-vault watch-add <path>[/dim]")

    console.print()


@app.command()
def watch_stop(vault_path: Optional[Path] = None):
    """Stop the watch service"""

    vault_path = vault_path or Path.cwd()
    config_dir = vault_path / ".claude-vault"

    if not config_dir.exists():
        console.print("[red]✗ Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    watch_manager = WatchManager(vault_path)
    status = watch_manager.get_status()

    if not status["is_running"]:
        console.print("[yellow]⚠ Watch mode is not running[/yellow]")
        return

    # Send signal to stop the process
    import os
    import signal

    try:
        os.kill(status["pid"], signal.SIGTERM)
        console.print("[green]✓ Sent stop signal to watch process[/green]")
    except ProcessLookupError:
        console.print(
            "[yellow]⚠ Watch process not found (may have already stopped)[/yellow]"
        )
        # Clean up stale state
        state = StateManager(vault_path)
        watch_state = state.get_watch_state()
        watch_state["is_running"] = False
        state.save_watch_state(watch_state)
    except PermissionError:
        console.print("[red]✗ Permission denied to stop watch process[/red]")


if __name__ == "__main__":
    app()

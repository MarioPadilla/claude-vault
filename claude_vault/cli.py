import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

import frontmatter
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import ConfigManager, AppConfig, SummarizationConfig
from .parser import ChatGPTExportParser, ClaudeExportParser
from .state import StateManager
from .summary import SummaryReportGenerator
from .sync import SyncEngine
from .tagging import OfflineTagGenerator, TagResult


class SourceType(str, Enum):
    """Source type for conversation imports"""

    claude = "claude"
    chatgpt = "chatgpt"


class TagMode(str, Enum):
    """Tag analysis mode"""

    quick = "quick"  # Title + first/last message (default)
    full = "full"  # Full content analysis


app = typer.Typer(help="Claude Vault - Sync Claude/ChatGPT conversations to Obsidian")
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


def prompt_for_model(tag_gen: OfflineTagGenerator, config_mgr: ConfigManager) -> str:
    """Prompt user to select an Ollama model"""
    models = tag_gen.list_models()
    current_default = config_mgr.get("tagging.model", "llama3.2:3b")

    if not models:
        console.print("[yellow]No Ollama models found. Using default.[/yellow]")
        return current_default

    console.print("\n[blue]Available Ollama models:[/blue]")
    for i, model in enumerate(models, 1):
        default_marker = " [green](current default)[/green]" if model == current_default else ""
        console.print(f"  {i}. {model}{default_marker}")

    console.print(f"  0. Keep current default ({current_default})")

    choice = typer.prompt("\nSelect model number", default="0")

    if choice == "0" or not choice.isdigit():
        return current_default

    idx = int(choice) - 1
    if 0 <= idx < len(models):
        selected_model = models[idx]

        # Ask if should be made default
        make_default = typer.confirm(
            f"Make '{selected_model}' the default model?", default=False
        )
        if make_default:
            config_mgr.update_tagging(model=selected_model)
            console.print(f"[green]Default model updated to: {selected_model}[/green]")

        return selected_model

    return current_default


def run_summary_flow(vault_path: Path, config_mgr: ConfigManager) -> None:
    """Run the post-sync summary report generation flow"""

    config = config_mgr.load()
    sum_config = config.summarization

    # Ask if user wants to generate summary report
    console.print("\n[blue]Summary Report Options[/blue]")
    generate_report = typer.confirm(
        "Generate summary report?",
        default=sum_config.enabled,
    )

    if not generate_report:
        return

    # Ask if user wants Ollama summarization
    include_summaries = typer.confirm(
        "Include AI-generated summaries for each conversation? (can take a while)",
        default=sum_config.include_ollama_summary,
    )

    # Get model and prompt if summarization is requested
    effective_model = sum_config.model
    effective_prompt = sum_config.prompt

    if include_summaries:
        # Check if Ollama is available
        tag_gen = OfflineTagGenerator()
        if not tag_gen.is_available():
            console.print("[red]* Ollama not running. Summaries will be skipped.[/red]")
            include_summaries = False
        else:
            # Show current model, allow change
            console.print(f"\n[dim]Current summarization model: {sum_config.model}[/dim]")
            change_model = typer.confirm("Change model?", default=False)
            if change_model:
                effective_model = prompt_for_model(tag_gen, config_mgr)

            # Show current prompt, allow edit
            console.print(f"\n[dim]Current summarization prompt:[/dim]")
            console.print(f"[dim]{sum_config.prompt}[/dim]")
            change_prompt = typer.confirm("Edit prompt?", default=False)
            if change_prompt:
                effective_prompt = typer.prompt(
                    "Summarization prompt",
                    default=sum_config.prompt,
                )

            # Ask if should save as defaults
            save_defaults = typer.confirm("Save these settings as defaults?", default=False)
            if save_defaults:
                config_mgr.update_summarization(
                    enabled=True,
                    include_ollama_summary=include_summaries,
                    model=effective_model,
                    prompt=effective_prompt,
                )
                console.print("[green]* Settings saved as defaults[/green]")

    # Generate the summary report
    console.print("\n[blue]Generating summary report...[/blue]")

    summary_gen = SummaryReportGenerator(vault_path)
    all_metadata = summary_gen.collect_all_metadata()

    # Save report with placeholders first
    new_count = summary_gen.save_report(all_metadata, append_mode=True)

    if new_count == 0:
        console.print("[yellow]No new conversations to add to report.[/yellow]")
        return

    console.print(f"[green]* Added {new_count} conversations to summary report[/green]")

    # Run summarization if requested
    if include_summaries:
        # Get only new metadata (not already summarized)
        existing_uuids = summary_gen.get_existing_uuids()
        # We need to identify which are new - those with [Pending...] in the report
        new_metadata = [m for m in all_metadata if m.uuid not in existing_uuids or not m.summary]

        if new_metadata:
            console.print(f"\n[blue]Summarizing {len(new_metadata)} conversations...[/blue]")

            with Progress(
                SpinnerColumn(spinner_name="line"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Summarizing...",
                    total=len(new_metadata),
                )

                def update_progress(i, total, title):
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Summarizing: {title[:40]}...",
                    )

                summary_gen.run_batch_summarization(
                    new_metadata,
                    effective_model,
                    effective_prompt,
                    progress_callback=update_progress,
                )

            console.print("[green]* Summarization complete![/green]")

    console.print(f"\n[dim]Report saved to: {vault_path / 'summary-report.md'}[/dim]")


@app.command()
def init(vault_path: Optional[Path] = typer.Option(None, "--vault-path")):
    """Initialize Claude Vault in the specified directory"""

    vault_path = vault_path or Path.cwd()
    config_mgr = ConfigManager(vault_path)

    if config_mgr.ensure_initialized():
        console.print(
            "[yellow]! Claude Vault already initialized in this directory[/yellow]"
        )
        return

    # Create config directory and save default config
    config_mgr.save(AppConfig())

    # Create conversations directory
    (vault_path / "conversations").mkdir(exist_ok=True)

    console.print(f"[green]* Claude Vault initialized at {vault_path}[/green]")
    console.print(f"[dim]Config stored in {vault_path / '.claude-vault'}[/dim]")
    console.print("\n[blue]Next steps:[/blue]")
    console.print("  1. Export your Claude/ChatGPT conversations (conversations.json)")
    console.print("  2. Run: claude-vault sync path/to/conversations.json")
    console.print("  3. Use --source chatgpt for ChatGPT exports")


@app.command()
def sync(
    export_path: Path,
    vault_path: Optional[Path] = typer.Option(None, "--vault-path"),
    source: SourceType = typer.Option(
        SourceType.claude,
        "--source",
        "-s",
        help="Source format: 'claude' for Claude exports, 'chatgpt' for ChatGPT exports",
    ),
    tag_mode: Optional[TagMode] = typer.Option(
        None,
        "--tag-mode",
        "-t",
        help="Tag analysis mode: 'quick' (default) or 'full' for comprehensive analysis",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use for tagging (overrides default)",
    ),
    hierarchy: Optional[bool] = typer.Option(
        None,
        "--hierarchy",
        "-H",
        help="Generate primary/secondary/tertiary tag hierarchy",
    ),
    select_model: bool = typer.Option(
        False,
        "--select-model",
        help="Interactively select Ollama model before sync",
    ),
):
    """Sync Claude or ChatGPT conversations to markdown files"""

    vault_path = vault_path or Path.cwd()
    if not export_path.exists():
        console.print(f"[red]* Error: Export file not found: {export_path}[/red]")
        raise typer.Exit(1)

    config_mgr = ConfigManager(vault_path)
    if not config_mgr.ensure_initialized():
        console.print("[red]* Error: Claude Vault not initialized[/red]")
        console.print("[yellow]Run 'claude-vault init' first[/yellow]")
        raise typer.Exit(1)

    # Load configuration
    config = config_mgr.load()
    tagging_config = config.tagging

    # Determine effective settings (CLI args override config)
    effective_model = model or tagging_config.model
    effective_mode = tag_mode.value if tag_mode else tagging_config.mode
    effective_hierarchy = hierarchy if hierarchy is not None else tagging_config.use_hierarchy

    # Interactive model selection if requested
    tag_gen = OfflineTagGenerator(model=effective_model)
    if select_model and tag_gen.is_available():
        effective_model = prompt_for_model(tag_gen, config_mgr)

    source_name = "ChatGPT" if source == SourceType.chatgpt else "Claude"
    console.print(
        f"[blue]Syncing {source_name} conversations from {export_path.name}...[/blue]"
    )
    console.print(f"[dim]Model: {effective_model} | Mode: {effective_mode} | Hierarchy: {effective_hierarchy}[/dim]\n")

    engine = SyncEngine(
        vault_path,
        source=source.value,
        tag_model=effective_model,
        tag_mode=effective_mode,
        use_hierarchy=effective_hierarchy,
    )

    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing conversations...", total=None)
        result = engine.sync(export_path)
        progress.update(task, completed=100)

    # Display results
    console.print("\n[green]* Sync complete![/green]\n")

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

    # Run summary report flow
    run_summary_flow(vault_path, config_mgr)


@app.command()
def status(vault_path: Optional[Path] = typer.Option(None, "--vault-path")):
    """Show Claude Vault status and statistics"""

    vault_path = vault_path or Path.cwd()
    config_mgr = ConfigManager(vault_path)

    if not config_mgr.ensure_initialized():
        console.print("[red]* Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    state = StateManager(vault_path)
    conversations = state.get_all_conversations()
    config = config_mgr.load()

    console.print("\n[blue]Claude Vault Status[/blue]\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Vault Location", str(vault_path))
    table.add_row("Conversations Tracked", str(len(conversations)))
    table.add_row("Storage", str(vault_path / ".claude-vault"))

    if conversations:
        latest = max(conversations, key=lambda x: x["last_synced"])
        table.add_row("Last Sync", latest["last_synced"][:19])

    # Show tagging configuration
    table.add_row("", "")  # Separator
    table.add_row("[bold]Tagging Config[/bold]", "")
    table.add_row("  Default Model", config.tagging.model)
    table.add_row("  Default Mode", config.tagging.mode)
    table.add_row("  Use Hierarchy", str(config.tagging.use_hierarchy))

    # Show summarization configuration
    table.add_row("", "")  # Separator
    table.add_row("[bold]Summarization Config[/bold]", "")
    table.add_row("  Enabled", str(config.summarization.enabled))
    table.add_row("  Include Summaries", str(config.summarization.include_ollama_summary))
    table.add_row("  Model", config.summarization.model)
    table.add_row("  Prompt", config.summarization.prompt[:50] + "..." if len(config.summarization.prompt) > 50 else config.summarization.prompt)

    console.print(table)
    console.print()


@app.command()
def config(
    vault_path: Optional[Path] = typer.Option(None, "--vault-path"),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_model: Optional[str] = typer.Option(None, "--set-model", help="Set default Ollama model"),
    set_mode: Optional[TagMode] = typer.Option(None, "--set-mode", help="Set default tag mode"),
    set_hierarchy: Optional[bool] = typer.Option(None, "--set-hierarchy", help="Set hierarchy default"),
    select_model: bool = typer.Option(False, "--select-model", help="Interactively select model"),
):
    """View or modify Claude Vault configuration"""

    vault_path = vault_path or Path.cwd()
    config_mgr = ConfigManager(vault_path)

    if not config_mgr.ensure_initialized():
        console.print("[red]* Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    # Handle interactive model selection
    if select_model:
        tag_gen = OfflineTagGenerator()
        if not tag_gen.is_available():
            console.print("[red]* Ollama not running[/red]")
            raise typer.Exit(1)

        selected = prompt_for_model(tag_gen, config_mgr)
        console.print(f"Selected model: {selected}")
        return

    # Handle setting values
    if set_model or set_mode or set_hierarchy is not None:
        if set_model:
            config_mgr.update_tagging(model=set_model)
            console.print(f"[green]* Default model set to: {set_model}[/green]")
        if set_mode:
            config_mgr.update_tagging(mode=set_mode.value)
            console.print(f"[green]* Default mode set to: {set_mode.value}[/green]")
        if set_hierarchy is not None:
            config_mgr.update_tagging(use_hierarchy=set_hierarchy)
            console.print(f"[green]* Hierarchy default set to: {set_hierarchy}[/green]")
        return

    # Default: show configuration
    cfg = config_mgr.load()
    console.print("\n[blue]Current Configuration[/blue]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Naming Pattern", cfg.naming_pattern)
    table.add_row("Folder Structure", cfg.folder_structure)
    table.add_row("Template", cfg.template)
    table.add_row("Version", cfg.version)
    table.add_row("", "")
    table.add_row("[bold]Tagging[/bold]", "")
    table.add_row("  Model", cfg.tagging.model)
    table.add_row("  Mode", cfg.tagging.mode)
    table.add_row("  Use Hierarchy", str(cfg.tagging.use_hierarchy))
    table.add_row("", "")
    table.add_row("[bold]Summarization[/bold]", "")
    table.add_row("  Enabled", str(cfg.summarization.enabled))
    table.add_row("  Include Summaries", str(cfg.summarization.include_ollama_summary))
    table.add_row("  Model", cfg.summarization.model)
    table.add_row("  Prompt", cfg.summarization.prompt[:50] + "..." if len(cfg.summarization.prompt) > 50 else cfg.summarization.prompt)

    console.print(table)
    console.print("\n[dim]Use --set-model, --set-mode, --set-hierarchy to modify[/dim]")
    console.print("[dim]Use --select-model for interactive model selection[/dim]")


@app.command()
def verify(
    vault_path: Optional[Path] = typer.Option(None, "--vault-path"),
    cleanup: bool = typer.Option(
        False, "--cleanup", help="Remove orphaned database entries"
    ),
):
    """Verify integrity of tracked conversations and optionally clean up mismatches"""
    vault_path = vault_path or Path.cwd()
    state = StateManager(vault_path)
    conversations = state.get_all_conversations()

    console.print(f"[blue]Verifying {len(conversations)} conversations...[/blue]\n")

    missing = []
    for conv in conversations:
        file_path = vault_path / conv["file_path"]
        if not file_path.exists():
            missing.append(conv)

    if missing:
        console.print(f"[yellow]! Found {len(missing)} missing files:[/yellow]")
        for conv in missing:
            console.print(f"  - {conv['file_path']}")

        if cleanup:
            console.print(
                f"\n[yellow]Cleaning up {len(missing)} orphaned entries...[/yellow]"
            )

            for conv in missing:
                state.delete_conversation(conv["uuid"])
                console.print(f"  * Removed: {conv['file_path']}")

            console.print(
                f"\n[green]* Cleaned up {len(missing)} orphaned database entries[/green]"
            )
        else:
            console.print(
                "\n[dim]Tip: Run with --cleanup flag to remove these entries from database[/dim]"
            )
            console.print("[dim]Command: claude-vault verify --cleanup[/dim]")
    else:
        console.print("[green]* All conversations verified successfully![/green]")


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Search term"),
    vault_path: Optional[Path] = typer.Option(None, "--vault-path"),
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
                # Optional tag filtering - check all tag types
                if tag:
                    all_tags = (
                        post.get("tags", [])
                        + post.get("tags_primary", [])
                        + post.get("tags_secondary", [])
                        + post.get("tags_tertiary", [])
                    )
                    if tag not in all_tags:
                        continue

                # Find matches with context
                matches = find_matches_with_context(post.content, keyword)

                results.append(
                    {
                        "file": md_file.name,
                        "title": post.get("title", ""),
                        "tags": post.get("tags", []),
                        "tags_primary": post.get("tags_primary", []),
                        "tags_secondary": post.get("tags_secondary", []),
                        "tags_tertiary": post.get("tags_tertiary", []),
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

            # Show tags (flat or hierarchical)
            if result["tags_primary"]:
                console.print(f"   Primary: {', '.join(result['tags_primary'])}")
                if result["tags_secondary"]:
                    console.print(f"   Secondary: {', '.join(result['tags_secondary'])}")
                if result["tags_tertiary"]:
                    console.print(f"   Tertiary: {', '.join(result['tags_tertiary'])}")
            else:
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
                            f"      - {clean_name} [dim](common: {common_tags})[/dim]"
                        )
                    else:
                        console.print(f"      - {clean_name}")

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
    vault_path: Optional[Path] = typer.Option(None, "--vault-path"),
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate all tags, even existing ones"),
    tag_mode: Optional[TagMode] = typer.Option(
        None,
        "--tag-mode",
        "-t",
        help="Tag analysis mode: 'quick' or 'full'",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use",
    ),
    hierarchy: Optional[bool] = typer.Option(
        None,
        "--hierarchy",
        "-H",
        help="Generate primary/secondary/tertiary tag hierarchy",
    ),
    select_model: bool = typer.Option(
        False,
        "--select-model",
        help="Interactively select Ollama model",
    ),
):
    """Regenerate tags for conversations using AI"""

    vault_path = vault_path or Path.cwd()
    config_mgr = ConfigManager(vault_path)

    if not config_mgr.ensure_initialized():
        console.print("[red]* Error: Claude Vault not initialized[/red]")
        raise typer.Exit(1)

    # Load configuration
    config = config_mgr.load()
    tagging_config = config.tagging

    # Determine effective settings
    effective_model = model or tagging_config.model
    effective_mode = tag_mode.value if tag_mode else tagging_config.mode
    effective_hierarchy = hierarchy if hierarchy is not None else tagging_config.use_hierarchy

    tag_gen = OfflineTagGenerator(model=effective_model)

    if not tag_gen.is_available():
        console.print("[red]* Ollama not running[/red]")
        console.print("Start Ollama with: ollama serve")
        console.print(f"Model needed: ollama pull {effective_model}")
        raise typer.Exit(1)

    # Interactive model selection if requested
    if select_model:
        effective_model = prompt_for_model(tag_gen, config_mgr)
        tag_gen = OfflineTagGenerator(model=effective_model)

    parser = ClaudeExportParser()  # Works for both formats when reading markdown
    conversations_dir = vault_path / "conversations"
    updated = 0

    console.print("[blue]Regenerating tags...[/blue]")
    console.print(f"[dim]Model: {effective_model} | Mode: {effective_mode} | Hierarchy: {effective_hierarchy}[/dim]\n")

    for md_file in conversations_dir.glob("*.md"):
        try:
            post = frontmatter.load(md_file)

            # Skip if has good tags and not forcing
            if not force and post.get("tags") and len(post.get("tags", [])) >= 3:
                continue

            # Parse conversation from file
            conv = parser.parse_conversation_from_markdown(post)

            # Generate new tags
            tag_result = tag_gen.generate_tags(
                conv,
                mode=effective_mode,
                use_hierarchy=effective_hierarchy,
            )

            # Update frontmatter based on hierarchy setting
            if effective_hierarchy and (tag_result.primary or tag_result.secondary or tag_result.tertiary):
                post["tags"] = tag_result.all_tags()
                post["tags_primary"] = tag_result.primary
                post["tags_secondary"] = tag_result.secondary
                post["tags_tertiary"] = tag_result.tertiary
            else:
                post["tags"] = tag_result.tags
                # Remove hierarchy keys if they exist
                for key in ["tags_primary", "tags_secondary", "tags_tertiary"]:
                    if key in post.keys():
                        del post[key]

            # Save updated file
            md_file.write_text(frontmatter.dumps(post), encoding="utf-8")
            updated += 1

            # Display tags
            if effective_hierarchy:
                console.print(f"* {md_file.name}:")
                console.print(f"    Primary: {', '.join(tag_result.primary)}")
                console.print(f"    Secondary: {', '.join(tag_result.secondary)}")
                console.print(f"    Tertiary: {', '.join(tag_result.tertiary)}")
            else:
                console.print(f"* {md_file.name}: {', '.join(tag_result.tags)}")

        except Exception as e:
            console.print(f"x {md_file.name}: {e}")

    console.print(f"\n[green]Updated {updated} conversations[/green]")


@app.command()
def extract_prompts(
    export_path: Path = typer.Argument(..., help="Path to conversations.json export file"),
    source: SourceType = typer.Option(
        SourceType.claude, "--source", "-s", help="Source: claude or chatgpt"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (default: prompts-export.json)"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, csv, or txt"
    ),
    include_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Include conversation metadata"
    ),
):
    """
    Extract all user prompts from Claude or ChatGPT export files.

    This is a standalone command that works directly on export files
    without requiring vault initialization.
    """
    if not export_path.exists():
        console.print(f"[red]Export file not found: {export_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\nExtracting user prompts from {source.value} export...")

    # Parse conversations
    if source == SourceType.chatgpt:
        parser = ChatGPTExportParser()
    else:
        parser = ClaudeExportParser()

    try:
        conversations = parser.parse(export_path)
    except Exception as e:
        console.print(f"[red]Error parsing export: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(conversations)} conversations")

    # Extract prompts
    all_prompts = []
    total_prompts = 0

    for conv in conversations:
        user_prompts = conv.get_all_user_prompts()
        total_prompts += len(user_prompts)

        if include_metadata:
            # Get active flags
            flags = conv.get_flags()
            active_flags = [k for k, v in flags.items() if v]

            prompt_data = {
                "conversation_id": conv.id,
                "title": conv.title,
                "source": source.value,
                "created_at": conv.created_at.isoformat(),
                "model": conv.model,
                "project": conv.project,
                "flags": active_flags,
                "prompt_count": len(user_prompts),
                "prompts": user_prompts,
            }
        else:
            prompt_data = {
                "conversation_id": conv.id,
                "title": conv.title,
                "prompts": user_prompts,
            }

        all_prompts.append(prompt_data)

    # Determine output path
    if output is None:
        output = export_path.parent / f"prompts-export.{format}"

    # Write output
    if format == "json":
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        import csv
        with open(output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if include_metadata:
                writer.writerow(["conversation_id", "title", "source", "created_at", "model", "project", "flags", "prompt_index", "prompt"])
                for conv_data in all_prompts:
                    for idx, prompt in enumerate(conv_data["prompts"], 1):
                        writer.writerow([
                            conv_data["conversation_id"],
                            conv_data["title"],
                            conv_data["source"],
                            conv_data["created_at"],
                            conv_data["model"],
                            conv_data["project"],
                            "|".join(conv_data["flags"]),
                            idx,
                            prompt.replace("\n", "\\n")[:1000],  # Truncate long prompts
                        ])
            else:
                writer.writerow(["conversation_id", "title", "prompt_index", "prompt"])
                for conv_data in all_prompts:
                    for idx, prompt in enumerate(conv_data["prompts"], 1):
                        writer.writerow([
                            conv_data["conversation_id"],
                            conv_data["title"],
                            idx,
                            prompt.replace("\n", "\\n")[:1000],
                        ])
    elif format == "txt":
        with open(output, "w", encoding="utf-8") as f:
            for conv_data in all_prompts:
                f.write(f"=== {conv_data['title']} ===\n")
                f.write(f"ID: {conv_data['conversation_id']}\n")
                if include_metadata:
                    f.write(f"Source: {conv_data.get('source', 'unknown')}\n")
                    f.write(f"Date: {conv_data.get('created_at', 'unknown')}\n")
                    if conv_data.get('flags'):
                        f.write(f"Flags: {', '.join(conv_data['flags'])}\n")
                f.write(f"\n")
                for idx, prompt in enumerate(conv_data["prompts"], 1):
                    f.write(f"[Prompt {idx}]\n{prompt}\n\n")
                f.write("\n" + "-" * 80 + "\n\n")
    else:
        console.print(f"[red]Unknown format: {format}. Use json, csv, or txt.[/red]")
        raise typer.Exit(1)

    # Display summary
    console.print(f"\n[green]* Extraction complete![/green]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Conversations processed", str(len(conversations)))
    table.add_row("Total user prompts", str(total_prompts))
    table.add_row("Output format", format.upper())
    table.add_row("Output file", str(output))

    console.print(table)


if __name__ == "__main__":
    app()

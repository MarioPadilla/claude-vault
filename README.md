# Claude Vault

Transform your Claude and ChatGPT conversations into a searchable, organized knowledge base in Obsidian.

## What is Claude Vault?

Claude Vault is a command-line tool that syncs your AI conversations into beautifully formatted Markdown files that integrate seamlessly with Obsidian and other note-taking tools.

## Features

- **Multi-Platform Import**: Import from both Claude and ChatGPT exports
- **Local-first**: Your conversations, your vault, your control
- **Simple CLI**: Easy to use, powerful features
- **Bulk Historical Import**: Import your entire conversation history at once
- **Obsidian-native**: Full frontmatter, tags, and metadata support
- **AI-Powered Tagging**: Automatic tag generation using local LLMs (Ollama) - no API costs
- **Hierarchical Tags**: Optional primary/secondary/tertiary tag categorization
- **Full Content Analysis**: Deep analysis mode for comprehensive tagging
- **Summary Reports**: Generate markdown table reports of all conversations with metadata
- **AI Summarization**: Optional per-conversation summaries via Ollama
- **Rich Metadata Extraction**: Model, project, attachments, artifacts, agentic mode
- **Configurable Defaults**: Persistent configuration for model, mode, and preferences
- **Bi-directional sync**: Rename and move files freely - they stay in sync
- **Smart updates**: Only syncs what's changed
- **UUID tracking**: Maintains file relationships even after renaming
- **Cross-Conversation Search**: Search across all conversations with context
- **Smart Relationship Detection**: Automatically finds and links related conversations

---

## Release Notes

### v0.4.0 (2025-12-29)

#### New Features

- **User Prompts Extraction** - All user prompts now captured as metadata
  - Every human message stored in `user_prompts` field
  - Displayed in summary report's new "User Prompts" column
  - Searchable and exportable

- **Standalone Prompt Extractor** - New `extract-prompts` command
  - Works directly on export files (no vault required)
  - Supports JSON, CSV, and TXT output formats
  - Includes all metadata and flags
  - Example: `claude-vault extract-prompts conversations.json -f csv`

- **Comprehensive Flag Tracking** - All conversation flags now extracted
  - **Claude flags**: `is_private`, `is_starter_project`, `prompt_template`
  - **ChatGPT flags**: `is_archived`, `is_starred`, `is_study_mode`, `is_do_not_remember`, `is_read_only`, `memory_scope`, `gizmo_type`, `voice`
  - Active flags shown in summary report's new "Flags" column

- **Project Metadata Integration** - Claude projects.json now parsed
  - Automatically loads project metadata from same directory
  - Extracts project privacy settings and templates

- **Enhanced Summary Report** - Two new columns added
  - "User Prompts" - Consolidated view of all user messages (truncated)
  - "Flags" - Active flags for each conversation

#### CLI Reference

```bash
# Extract all prompts to JSON
claude-vault extract-prompts conversations.json

# Extract ChatGPT prompts to CSV
claude-vault extract-prompts chatgpt.json -s chatgpt -f csv

# Extract without metadata (prompts only)
claude-vault extract-prompts conversations.json --no-metadata

# Custom output path
claude-vault extract-prompts conversations.json -o my-prompts.json
```

---

### v0.3.1 (2025-12-28)

#### Bug Fixes

- **Windows Console Encoding Fix** - Fixed `UnicodeEncodeError` on Windows
  - Rich progress spinners now use ASCII-compatible characters
  - Resolves `'charmap' codec can't encode character` errors on Windows terminals with cp1252 encoding

---

### v0.3.0 (2025-12-28)

#### New Features

- **Summary Report Generation** - Create a markdown table summarizing all conversations
  - Automatically prompted after each sync
  - Appends new conversations to existing report (no duplicates)
  - Saved to `summary-report.md` in vault root

- **AI-Powered Conversation Summarization** - Optional per-conversation summaries
  - Uses Ollama to generate 2-3 sentence summaries
  - Configurable summarization prompt
  - Reports generate with `[Pending...]` placeholders, updated when summaries complete
  - Can skip summarization and just generate the report

- **Rich Metadata Extraction** - Enhanced metadata from exports
  - Model used (e.g., `gpt-4`, `claude-3-opus`)
  - Project name (for ChatGPT Custom GPTs or Claude Projects)
  - Has attachments/artifacts flags
  - Agentic/tool-use detection

- **Summary Report Columns**:
  - Title (hyperlink to original conversation)
  - Source (claude/chatgpt)
  - Date
  - Message count
  - Primary/Secondary/Tertiary tags
  - Model
  - Project
  - AI-generated summary (optional)
  - Comment (for user notes)

#### Configuration

New `summarization` config section:
```json
{
  "summarization": {
    "enabled": false,
    "include_ollama_summary": false,
    "model": "llama3.2:3b",
    "prompt": "Summarize this conversation in 2-3 sentences..."
  }
}
```

---

### v0.2.0 (2025-12-28)

#### New Features

- **ChatGPT Import Support** - Import ChatGPT conversation exports alongside Claude
  - Use `--source chatgpt` or `-s chatgpt` flag with sync command
  - Handles ChatGPT's tree-structured JSON format
  - Properly displays "ChatGPT" as assistant name in markdown

- **Full Content Analysis Mode** - Comprehensive tagging based on entire conversation
  - Use `--tag-mode full` or `-t full` for deep analysis
  - Analyzes all messages (up to 8000 chars) vs quick mode (title + first/last message)
  - Better tag accuracy for long, complex conversations

- **Hierarchical Tag System** - Organize tags by importance
  - Use `--hierarchy` or `-H` flag to enable
  - Generates `tags_primary`, `tags_secondary`, `tags_tertiary` in frontmatter
  - Primary: Core topics (2-3 tags)
  - Secondary: Supporting topics (2-3 tags)
  - Tertiary: Minor/peripheral topics (1-2 tags)

- **Configuration Management** - Persistent settings via config file
  - New `config` command to view and modify settings
  - Settings stored in `.claude-vault/config.json`
  - Configure default Ollama model, tag mode, hierarchy preference

- **Interactive Model Selection** - Choose Ollama model at runtime
  - Use `--select-model` flag on sync, retag, or config commands
  - Lists all available Ollama models
  - Option to save selection as new default

#### Improvements

- Enhanced CLI with consistent `--vault-path` option across all commands
- Better Windows console compatibility (ASCII-safe output)
- Improved error handling and user feedback
- Search now supports hierarchical tag filtering

#### Breaking Changes

- None - fully backward compatible with v0.1.0

---

### v0.1.0 (2025-12-21)

#### Initial Release

- Claude conversation import from JSON export
- Markdown generation with YAML frontmatter
- AI-powered tagging via Ollama
- Bi-directional sync with UUID tracking
- Cross-conversation search
- Related conversation detection
- Vault integrity verification

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Ollama** (optional but recommended for AI tagging)

### Install Claude Vault

```bash
# Clone or download the project
git clone https://github.com/MarioPadilla/claude-vault.git
cd claude-vault

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
claude-vault --help
```

### Install Ollama (Optional for AI tagging)

```bash
# On macOS
brew install ollama

# Start ollama
ollama serve

# Pull a model (any of these work)
ollama pull llama3.2:3b      # Fast, good quality
ollama pull deepseek-r1:8b   # Better reasoning
ollama pull mistral:7b       # Balanced
```

---

## Quick Start

### 1. Export Your Conversations

**For Claude:**
1. Go to [claude.ai](https://claude.ai)
2. Click profile → Settings → Export conversations
3. Downloads `conversations.json`

**For ChatGPT:**
1. Go to [chat.openai.com](https://chat.openai.com)
2. Settings → Data controls → Export data
3. Extract `conversations.json` from the downloaded zip

### 2. Initialize Vault

```bash
cd ~/Documents/ObsidianVault
claude-vault init
```

### 3. Sync Conversations

```bash
# Import Claude conversations (default)
claude-vault sync ~/Downloads/conversations.json

# Import ChatGPT conversations
claude-vault sync ~/Downloads/chatgpt/conversations.json --source chatgpt
```

### 4. Check Status

```bash
claude-vault status
```

---

## CLI Reference

### `init`

Initialize Claude Vault in the specified directory.

```bash
claude-vault init [--vault-path PATH]
```

### `sync`

Sync conversations to markdown files.

```bash
claude-vault sync EXPORT_PATH [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--vault-path` | | Path to vault directory |
| `--source` | `-s` | Source format: `claude` (default) or `chatgpt` |
| `--tag-mode` | `-t` | Tag analysis: `quick` (default) or `full` |
| `--model` | `-m` | Ollama model to use (overrides config) |
| `--hierarchy` | `-H` | Generate primary/secondary/tertiary tags |
| `--select-model` | | Interactively select Ollama model |

**Examples:**
```bash
# Basic Claude import
claude-vault sync conversations.json

# ChatGPT import with full analysis and hierarchy
claude-vault sync chatgpt.json -s chatgpt -t full -H

# Use specific model
claude-vault sync conversations.json -m deepseek-r1:8b

# Interactive model selection
claude-vault sync conversations.json --select-model
```

### `config`

View or modify configuration.

```bash
claude-vault config [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--set-model TEXT` | Set default Ollama model |
| `--set-mode [quick\|full]` | Set default tag analysis mode |
| `--set-hierarchy` | Enable/disable hierarchy by default |
| `--select-model` | Interactively select and save default model |

**Examples:**
```bash
# View current config
claude-vault config

# Set defaults
claude-vault config --set-mode full
claude-vault config --set-hierarchy
claude-vault config --set-model deepseek-r1:8b

# Interactive model selection
claude-vault config --select-model
```

### `status`

Show vault status and statistics.

```bash
claude-vault status [--vault-path PATH]
```

### `search`

Search across all conversations.

```bash
claude-vault search KEYWORD [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--tag TEXT` | Filter by tag (searches all hierarchy levels) |
| `--show-related` | Show related conversations (default: true) |

**Examples:**
```bash
claude-vault search "python"
claude-vault search "API" --tag code
claude-vault search "debugging" --no-show-related
```

### `retag`

Regenerate tags for conversations using AI.

```bash
claude-vault retag [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Regenerate all tags, even existing ones |
| `--tag-mode` | `-t` | Tag analysis: `quick` or `full` |
| `--model` | `-m` | Ollama model to use |
| `--hierarchy` | `-H` | Generate hierarchical tags |
| `--select-model` | | Interactively select model |

**Examples:**
```bash
# Retag missing/insufficient tags
claude-vault retag

# Force retag all with full analysis and hierarchy
claude-vault retag -f -t full -H

# Use specific model
claude-vault retag -m mistral:7b
```

### `verify`

Verify vault integrity and clean up orphaned entries.

```bash
claude-vault verify [--cleanup]
```

### `extract-prompts`

Extract all user prompts from export files (standalone, no vault required).

```bash
claude-vault extract-prompts EXPORT_PATH [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--source` | `-s` | Source format: `claude` (default) or `chatgpt` |
| `--output` | `-o` | Output file path (default: `prompts-export.{format}`) |
| `--format` | `-f` | Output format: `json` (default), `csv`, or `txt` |
| `--metadata/--no-metadata` | | Include conversation metadata (default: yes) |

**Examples:**
```bash
# Extract Claude prompts to JSON
claude-vault extract-prompts conversations.json

# Extract ChatGPT prompts to CSV
claude-vault extract-prompts chatgpt.json -s chatgpt -f csv

# Extract to plain text
claude-vault extract-prompts conversations.json -f txt

# Prompts only (no metadata)
claude-vault extract-prompts conversations.json --no-metadata

# Custom output path
claude-vault extract-prompts conversations.json -o ~/exports/my-prompts.json
```

**Output Formats:**

- **JSON**: Structured data with full metadata, flags, and all prompts per conversation
- **CSV**: Flat format with one row per prompt, includes conversation context
- **TXT**: Human-readable format with clear separators between conversations

---

## Configuration File

Configuration is stored in `.claude-vault/config.json`:

```json
{
  "naming_pattern": "{date}-{title}",
  "folder_structure": "flat",
  "template": "default",
  "version": "0.2.0",
  "tagging": {
    "model": "llama3.2:3b",
    "mode": "quick",
    "use_hierarchy": false
  }
}
```

---

## Output Format

### Standard Tags (default)

```yaml
---
title: Python API Tutorial
date: '2025-12-28T10:30:00'
tags:
- python
- api
- tutorial
source: claude
uuid: abc123
---
```

### Hierarchical Tags (with `--hierarchy`)

```yaml
---
title: Python API Tutorial
date: '2025-12-28T10:30:00'
tags:
- python
- api
- rest
- tutorial
- http
tags_primary:
- python
- api
tags_secondary:
- rest
- tutorial
tags_tertiary:
- http
source: chatgpt
uuid: def456
---
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not running" | Start with `ollama serve` |
| "Module not found" | Reinstall with `pip install -e .` |
| "Not initialized" | Run `claude-vault init` first |
| "Model not found" | Run `ollama pull <model-name>` |
| Unicode errors on Windows | Fixed in v0.2.0 |

---

## License

Claude Vault is available under a **dual-license model**:

### Open Source License (AGPL-3.0)

**Free for:**
- Personal use
- Open source projects
- Educational purposes
- Research and academic use
- Non-commercial applications

**Requirements under AGPL-3.0:**
- Must disclose source code of any modifications
- Must keep the same license (AGPL-3.0)
- Must provide source code to users (including SaaS/network users)

### Commercial License

**Required for:**
- Proprietary/closed-source applications
- Commercial SaaS products
- Enterprise deployments where source code disclosure is not desired

**Contact:** GitHub for commercial licensing inquiries.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Getting Help

```bash
claude-vault --help
claude-vault [COMMAND] --help
```

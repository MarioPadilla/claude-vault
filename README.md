# Claude Vault

Transform your Claude conversations into a searchable, organized knowledge base in Obsidian.

## Quick Start

```bash
pip install -e .
claude-vault init          # run inside your Obsidian vault folder
claude-vault sync ~/Downloads/conversations.json
```

That's it. Your conversations are now in `conversations/` as Markdown files.

## Installation

**Requirements:** Python 3.8+, optional: [Ollama](https://ollama.ai) for AI tagging and semantic search.

```bash
git clone https://github.com/MarioPadilla/claude-vault.git
cd claude-vault
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

**Optional — Ollama setup for AI features:**

```bash
ollama serve
ollama pull llama3.2:3b        # for tagging & summarization
ollama pull nomic-embed-text   # for semantic search
```

## Features

- **Local-first** — everything stays on your machine, no external APIs required
- **Three import formats** — Claude Web exports (`.json`), Claude Code history (`.jsonl`), and OpenCode sessions (`.db`) auto-detected
- **AI tagging & summarization** — local LLM (Ollama) generates tags and summaries; falls back to keyword extraction
- **Semantic search** — find conversations by concept, not just exact words
- **Smart relationship detection** — automatically links related conversations via shared tags
- **Bi-directional sync** — rename or move files freely; UUID tracking keeps everything in sync
- **Watch mode** — auto-syncs when your export files change
- **PII protection** — detect, redact, or skip conversations containing personal or sensitive data
- **Dry-run mode** — preview all changes before writing anything

## Usage

### Export your conversations

1. Go to [claude.ai](https://claude.ai) → Settings → Export data
2. This downloads `conversations.json`
3. For Claude Code history, point directly at `~/.claude`

### Sync

```bash
# Web export (auto-detected)
claude-vault sync ~/Downloads/conversations.json

# Claude Code history
claude-vault sync ~/.claude

# OpenCode (auto-detected from .db extension)
claude-vault sync ~/.local/share/opencode/opencode.db

# OpenCode (uses default path)
claude-vault sync --source opencode

# Preview changes without writing
claude-vault sync conversations.json --dry-run
```

### Search

```bash
# Semantic search (requires Ollama + nomic-embed-text)
claude-vault search "async error handling"

# Keyword search
claude-vault search "python" --mode keyword

# Filter by tag
claude-vault search "API" --tag debugging
```

### Watch mode

```bash
claude-vault watch-add ~/Downloads --source web   # register a path
claude-vault watch                                # start (foreground)
claude-vault watch-status
claude-vault watch-stop
```

### PII & sensitive content protection

Scan conversations for personal or confidential data before they are written to disk.

**Detected patterns:** emails, phone numbers, SSNs, credit card numbers, API keys, IP addresses, credential contexts (`password:`, `token:`, etc.). If Ollama is running, an LLM pass also classifies broader sensitive content.

**Risk levels:** `high` (SSN / credit card / API key), `medium` (email / phone), `low` (IP / credential context).

```bash
# Tag conversations that contain PII (adds pii-* tags + frontmatter fields)
claude-vault sync conversations.json --detect-pii

# Redact PII before writing — stored files contain [REDACTED-EMAIL] etc.
claude-vault sync conversations.json --detect-pii --redact-pii

# Skip conversations at medium risk or above entirely
claude-vault sync conversations.json --detect-pii --skip-sensitive
```

To make these behaviours permanent, set them in `~/.claude-vault/config.json`:

```json
{
  "pii": {
    "enabled": false,
    "redact": false,
    "skip_sensitive": false,
    "use_llm": true,
    "risk_threshold": "medium"
  }
}
```

### Other commands

```bash
claude-vault status              # vault statistics
claude-vault retag               # regenerate AI tags (requires Ollama)
claude-vault retag --force       # regenerate even existing tags
claude-vault verify              # check file/database consistency
claude-vault verify --cleanup    # remove orphaned database entries
claude-vault config              # view current configuration
```

Use `claude-vault [command] --help` for full options on any command.

## Configuration

Global config lives at `~/.claude-vault/config.json`. Run `claude-vault config` to view and edit it.

Key settings:

| Section | Key | Default | Purpose |
| ------- | --- | ------- | ------- |
| `ollama` | `model` | `llama3.2:3b` | Model used for tagging |
| `ollama` | `url` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `embeddings` | `model` | `nomic-embed-text` | Model used for semantic search |
| `pii` | `risk_threshold` | `medium` | Minimum level to tag as `sensitive` |
| `pii` | `use_llm` | `true` | Enable LLM-based classification |
| root | `custom_keywords` | `null` | Extra tag → keyword mappings for fallback tagging |

## Troubleshooting

| Error | Fix |
| ----- | --- |
| "Ollama not running" | Run `ollama serve` |
| Semantic search returns nothing | Run `ollama pull nomic-embed-text` |
| "Module not found" | Run `pip install -e .` |
| "Not initialized" | Run `claude-vault init` first |

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and pull request guidelines.

## License

Claude Vault is dual-licensed:

**AGPL-3.0 (free)** — for personal use, open-source projects, education, and non-commercial work. Modifications must be shared under the same license.

**Commercial license** — required for proprietary applications, SaaS products, or any deployment where you cannot comply with AGPL-3.0 copyleft terms. Contact via GitHub for pricing.

| Use case | License |
| -------- | ------- |
| Personal / open-source / education | AGPL-3.0 ✅ |
| Internal tool (source shared with employees) | AGPL-3.0 ✅ |
| Commercial SaaS or closed-source product | Commercial 💼 |
| Enterprise deployment (no source disclosure) | Commercial 💼 |

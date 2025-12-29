# Claude Vault

Transform your Claude conversations into a searchable, organized knowledge base in Obsidian.

## What is Claude Vault?

Claude Vault is a command-line tool that syncs your Claude AI conversations into beautifully formatted Markdown files that integrate seamlessly with Obsidian and other note-taking tools.

## Features

- ✅ **Local-first**: Your conversations, your vault, your control
- ✅ **Simple CLI**: Easy to use, powerful features
- ✅ **Bulk Historical Import**: Import your entire Claude conversation history at once
- ✅ **Obsidian-native**: Full frontmatter, tags, and metadata support
- ✅ **AI-Powered Tagging**: Automatic tag generation using local LLMs (Ollama) - no API costs
- ✅ **Bi-directional sync**: Rename and move files freely - they stay in sync
- ✅ **Smart updates**: Only syncs what's changed
- ✅ **UUID tracking**: Maintains file relationships even after renaming
- ✅ **Cross-Conversation Search**: Search across all conversations with context and navigate to related ones
- ✅ **Smart Relationship Detection**: Automatically finds and links related conversations via common tags

## Key Features of the Code Parser

The JSONL parser handles:
- ✅ **Session grouping** - Groups messages by `sessionId`
- ✅ **Tool results** - Shows before/after state for tools
- ✅ **Timestamps** - Preserves message timing
- ✅ **Summary as title** - Uses the summary line as conversation title
- ✅ **Code-specific tags** - Adds 'code-session' tag to differentiate from web chats

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

### Test
```bash
python tests/test_parser.py
```

### Install Ollama (Optional for AI tagging)
```bash
# On macOS
brew install ollama

# Start ollama
ollama serve

# Pull a balanced model (quality/speed)
ollama pull llama3.2:3b
```

## Quick Usage

### 1. Export Your Claude Conversations

1. Go to [claude.ai](https://claude.ai)
2. Click profile → Settings
3. Export conversations (downloads `conversations.json`)

### 2. Initialize Vault

```bash
# Navigate to your Obsidian vault
cd ~/Documents/ObsidianVault

# Initialize Claude Vault
claude-vault init
```

### 3. Sync Conversations
```bash
# Import all conversations
claude-vault sync ~/Downloads/conversations.json
```

## Supported Formats

Claude Vault supports multiple Claude conversation sources:

- **Claude Web Conversations** (.json) - From claude.ai exports
- **Claude Code History** (.jsonl) - From Claude Code IDE integration

Both formats are automatically detected based on file extension, or you can specify with `--source`:

```bash
# Auto-detect format
claude-vault sync conversations.json
claude-vault sync code-history.jsonl
# Sync from entire .claude folder
claude-vault sync ~/.claude

# Explicit source
claude-vault sync export.json --source web
claude-vault sync export.jsonl --source code

### 4. Check Status
```bash
claude-vault status
```

## Common Commands
```bash
# Search conversations
claude-vault search "python"

# Search with tag filter
claude-vault search "API" --tag code

# Regenerate tags with AI
claude-vault retag

# Verify vault integrity
claude-vault verify

# Clean up orphaned entries
claude-vault verify --cleanup
```

## Troubleshooting

**"Ollama not running":** Start with `ollama serve`

**"Module not found:"** Reinstall with `pip install -e .`

**"Not initialized:"** Run `claude-vault init` first

## License

Claude Vault is available under a **dual-license model**:

### 🆓 Open Source License (AGPL-3.0)

**Free for:**
- ✅ Personal use
- ✅ Open source projects
- ✅ Educational purposes
- ✅ Research and academic use
- ✅ Non-commercial applications

**Requirements under AGPL-3.0:**
- Must disclose source code of any modifications
- Must keep the same license (AGPL-3.0)
- Must provide source code to users (including SaaS/network users)
- Any derivative work must also be licensed under AGPL-3.0

**Perfect for:** Developers, hobbyists, students, and open-source contributors who want to freely use and modify Claude Vault.

---

### 💼 Commercial License

**Required for:**
- ❌ Proprietary/closed-source applications
- ❌ Commercial SaaS products
- ❌ Enterprise deployments where source code disclosure is not desired
- ❌ Products that cannot comply with AGPL-3.0 copyleft terms

**Benefits of Commercial License:**
- ✅ Use Claude Vault in proprietary applications
- ✅ No obligation to disclose your source code
- ✅ Freedom from AGPL-3.0 copyleft requirements
- ✅ Priority support (optional)
- ✅ Custom modifications and consulting (optional)

**Pricing:** Contact me for a quote based on your use case.

📧 **Contact:** Github
📝 **Subject:** Claude Vault Commercial License Inquiry

---

### ❓ Which License Do I Need?

| Use Case | License Needed |
|----------|---------------|
| Building an open-source tool | AGPL-3.0 (Free) ✅ |
| Learning/experimenting | AGPL-3.0 (Free) ✅ |
| Contributing to Claude Vault | AGPL-3.0 (Free) ✅ |
| Internal company tool (source shared with employees) | AGPL-3.0 (Free) ✅ |
| Commercial SaaS product | Commercial 💼 |
| Closed-source application | Commercial 💼 |
| Selling a product that includes Claude Vault | Commercial 💼 |
| Enterprise deployment (no source disclosure) | Commercial 💼 |

**Still unsure?** Contact me

---

**Note:** By using, modifying, or distributing Claude Vault without obtaining a commercial license, you agree to comply with the AGPL-3.0 terms.

### CLI Commands

#### `init`

Initialize Claude Vault in the specified directory.

```bash
claude-vault init [VAULT_PATH]
```

**Example:**
```bash
claude-vault init
claude-vault init ~/Documents/my-vault
```

#### `sync`

Sync Claude conversations to markdown files.

```bash
claude-vault sync [EXPORT_PATH]
```

**Examples:**
```bash
claude-vault sync ~/Downloads/conversations.json
```

#### `status`

Show Claude Vault status and statistics.

```bash
claude-vault status
```

#### `verify`

Verify integrity of tracked conversations and optionally clean up mismatches.

```bash
claude-vault verify
```

**Examples:**
```bash
claude-vault verify
claude-vault verify --cleanup
```

#### `search`

Search across all conversations.

```bash
claude-vault search KEYWORD [OPTIONS]
```
**Arguments:**
- `KEYWORD` - Search term **(required)**

**Options:**
- `--tag TEXT` - Filter by tag
- `--show-related / --no-show-related` - Show related conversations (default: enabled)

**Examples:**
```bash
# Basic search
claude-vault search "python"

# Search with tag filter
claude-vault search "machine learning" --tag "ai"

# Search without showing related conversations
claude-vault search "debugging" --no-show-related
```

#### `retag`

Regenerate tags for conversations using AI.

```bash
claude-vault retag [OPTIONS]
```

**Options:**
- `--force` - Regenerate all tags, even existing ones

**Examples:**
```bash
# Tag conversations without tags
claude-vault retag

# Force regenerate all tags
claude-vault retag --force
```

**Requirements:** Requires Ollama to be running with `llama3.2:3b` model installed.

### Getting Help

Get help for any command:

```bash
claude-vault --help
claude-vault [COMMAND] --help
```

**Examples:**
```bash
claude-vault sync --help
claude-vault search --help
```

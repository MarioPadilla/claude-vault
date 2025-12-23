---
date: '2025-12-22T08:20:19.408343+00:00'
message_count: 2
tags:
- export
title: Exporting conversations and file formats
updated: '2025-12-22T08:20:58.702986+00:00'
uuid: e30de7aa-6664-4cf3-8242-105db3f6308f
---

## 👤 You *(2025-12-22 08:20)*

I would like to understand the export file that gets generated from our conversation.
I am trying to create a Claude Vault, could you give me some code examples and other kind of formats in your answer so I can have multiple examples in the generated export file.
Whare can i find the option and what formats can i export to? (This is just a test)

---

## 🤖 Claude *(2025-12-22 08:20)*

I'll help you understand the export feature and create various examples for your Claude Vault!

## Finding the Export Option

To export a conversation in Claude:
1. Look for the **three-dot menu** (⋮) at the top-right of the conversation
2. Click on it and select **"Export conversation"**
3. You'll typically see options for different formats

## Available Export Formats

The common export formats are:
- **Markdown (.md)** - Plain text with formatting
- **JSON** - Structured data format
- **Text (.txt)** - Simple plain text

Now let me create various examples for your vault:

## Code Examples

### Python Example
```
Viewing artifacts created via the Analysis Tool web feature preview isn’t yet supported on mobile.
```

### JavaScript/React Component
```
Viewing artifacts created via the Analysis Tool web feature preview isn’t yet supported on mobile.
```

## Structured Data Example (JSON)

Here's what a typical export structure might look like:

```json
{
  "conversation_id": "conv_abc123xyz",
  "title": "Understanding Claude Exports",
  "created_at": "2024-12-22T10:30:00Z",
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {
      "role": "user",
      "content": "I would like to understand the export file...",
      "timestamp": "2024-12-22T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "I'll help you understand...",
      "timestamp": "2024-12-22T10:30:15Z",
      "artifacts": [
        {
          "type": "code",
          "language": "python",
          "title": "Data Processor"
        }
      ]
    }
  ]
}
```

## Markdown Format Example

When exported as Markdown, conversations typically look like:

```markdown
# Understanding Claude Exports

**Date:** December 22, 2024
**Model:** Claude Sonnet 4

---

## User
I would like to understand the export file...

## Assistant
I'll help you understand the export feature...

### Code: Data Processor (Python)
```python
import json
# code here...
```

## Understanding the Export Structure

The exported file will contain:

1. **Metadata** - Conversation title, date, model used
2. **Message History** - Complete back-and-forth exchanges
3. **Artifacts** - Any code, documents, or interactive content created
4. **Formatting** - Preserved markdown, code blocks, and structure

**Tips for Your Claude Vault:**
- Use descriptive titles for easy searching
- Export regularly to avoid losing valuable conversations
- Consider organizing exports by project or topic
- JSON format is best for programmatic access
- Markdown is best for human readability

The export will capture everything in our conversation, including these code examples, the React component, and all the formatting!
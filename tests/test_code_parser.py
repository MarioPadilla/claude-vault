from pathlib import Path

from claude_vault.code_parser import ClaudeCodeHistoryParser
from claude_vault.markdown import MarkdownGenerator


def test_code_history_parsing():
    """Test with your actual Claude Code History"""

    claude_history_file = Path("./code-history.jsonl")

    if not claude_history_file.exists():
        print("⚠️  code-history.jsonl file not found.")
        return

    # Parse the code-history.jsonl file
    parser = ClaudeCodeHistoryParser()
    conversations = parser.parse(claude_history_file)

    print(f"\n✓ Found {len(conversations)} code sessions\n")

    # Show details of sessions
    if conversations:
        for i, conv in enumerate(conversations[:3], 1):  # Show first 3
            print(f"\n{i}. {conv.title}")
            print(f"   Session ID: {conv.id}")
            print(f"   Messages: {len(conv.messages)}")
            print(f"   Created: {conv.created_at}")
            print(f"   Tags: {conv.tags}")

        # Generate markdown for first conversation
        conv = conversations[0]
        md_gen = MarkdownGenerator()
        markdown = md_gen.generate(conv)

        # Save to test file
        output_path = Path("test_code_output.md")
        md_gen.save(conv, output_path)
        print(f"\n✓ Markdown saved to: {output_path}")
        print("\nFirst 500 characters of output:")
        print("-" * 50)
        print(markdown[:500])
        print("-" * 50)


if __name__ == "__main__":
    test_code_history_parsing()

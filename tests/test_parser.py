from pathlib import Path

from claude_vault.markdown import MarkdownGenerator
from claude_vault.parser import ClaudeExportParser


def test_basic_parsing():
    """Test with your actual conversations.json file"""

    # Update this path to your actual export file
    export_path = Path("conversations.json")

    if not export_path.exists():
        print("⚠️  Place your conversations.json in the project root to run this test")
        return

    # Parse the export
    parser = ClaudeExportParser()
    conversations = parser.parse(export_path)

    print(f"\n✓ Found {len(conversations)} conversations\n")

    # Show details of first conversation
    if conversations:
        conv = conversations[0]
        print(f"Title: {conv.title}")
        print(f"Messages: {len(conv.messages)}")
        print(f"Created: {conv.created_at}")
        print(f"Tags: {conv.tags}")
        print(f"UUID: {conv.id}")

        # Generate markdown
        md_gen = MarkdownGenerator()
        markdown = md_gen.generate(conv)

        # Save to test file
        output_path = Path("test_output.md")
        md_gen.save(conv, output_path)
        print(f"\n✓ Markdown saved to: {output_path}")
        print("\nFirst 500 characters of output:")
        print("-" * 50)
        print(markdown[:500])
        print("-" * 50)


if __name__ == "__main__":
    test_basic_parsing()

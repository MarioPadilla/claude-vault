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


def test_markdown_writes_scan_marker_only_when_scanned():
    from datetime import datetime

    from claude_vault.markdown import MarkdownGenerator
    from claude_vault.models import Conversation, Message

    def make(**extra):
        return Conversation(
            id="c1",
            title="T",
            messages=[Message(role="human", content="hi")],
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            **extra,
        )

    generator = MarkdownGenerator()

    plain = generator.generate(make())
    assert "pii_scanned" not in plain

    scanned = generator.generate(
        make(pii_scanned=True, pii_detector="abc123", pii_llm_used=True)
    )
    assert "pii_scanned: true" in scanned
    assert "pii_detector: abc123" in scanned
    assert "pii_llm: true" in scanned
    assert "pii_detected" not in scanned


def test_find_file_problem_classifies_files(tmp_path):
    from datetime import datetime

    from claude_vault.markdown import MarkdownGenerator, find_file_problem
    from claude_vault.models import Conversation, Message

    conv = Conversation(
        id="abc-123",
        title="Excel",
        messages=[
            Message(role="human", content="question"),
            Message(role="assistant", content="answer"),
            Message(role="user", content="thanks"),
        ],
        created_at=datetime(2025, 6, 3),
        updated_at=datetime(2025, 6, 3),
    )
    full = MarkdownGenerator().generate(conv)

    def check(text):
        path = tmp_path / "x.md"
        path.write_text(text, encoding="utf-8")
        return find_file_problem(path)

    assert check(full) is None
    assert check("") == "is empty"
    assert "holds 2 of 3 messages" in check(full[: full.index("answer")])
    assert "holds 1 of 3 messages" in check(full[: full.index("🤖 Claude")])
    assert "no conversation text" in check(full[: full.index("\n---\n", 5) + 5])
    assert "no uuid" in check("---\ntitle: x\n---\n\n## 👤 You\n\nhi\n")


def test_find_file_problem_accepts_older_files_without_message_count(tmp_path):
    from claude_vault.markdown import find_file_problem

    path = tmp_path / "old.md"
    path.write_text(
        "---\nuuid: abc\ntitle: T\n---\n\n## 👤 You\n\nhi\n", encoding="utf-8"
    )

    assert find_file_problem(path) is None


def test_find_file_problem_counts_headings_with_timestamps(tmp_path):
    from claude_vault.markdown import find_file_problem

    path = tmp_path / "ts.md"
    path.write_text(
        "---\nuuid: abc\nmessage_count: 2\n---\n\n"
        "## 👤 You *(2025-06-03 10:15)*\n\nhi\n\n---\n\n"
        "## 🤖 Claude *(2025-06-03 10:16)*\n\nhello\n",
        encoding="utf-8",
    )

    assert find_file_problem(path) is None

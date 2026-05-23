import json
from pathlib import Path

from claude_vault.code_parser import ClaudeCodeHistoryParser
from claude_vault.markdown import MarkdownGenerator


def _write_session_jsonl(path: Path, session_id: str, user_text: str) -> None:
    """Minimal valid JSONL with a sessionId header line and a user message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "permission-mode", "sessionId": session_id}) + "\n")
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-05-08T12:00:00.000Z",
                    "cwd": "/tmp/proj",
                    "message": {"role": "user", "content": user_text},
                }
            )
            + "\n"
        )


def test_parser_skips_subagent_jsonl_files(tmp_path):
    """agent-*.jsonl are subagent transcripts that share the parent's
    sessionId. Treating them as separate conversations causes UUID
    collisions; the parser must skip them."""
    claude_dir = tmp_path / ".claude"
    projects = claude_dir / "projects" / "myproj"
    projects.mkdir(parents=True)

    parent_sid = "11111111-1111-1111-1111-111111111111"
    _write_session_jsonl(projects / f"{parent_sid}.jsonl", parent_sid, "Parent task")
    _write_session_jsonl(
        projects / "agent-abc123.jsonl", parent_sid, "Subagent dispatch"
    )
    _write_session_jsonl(
        projects / "agent-def456.jsonl", parent_sid, "Another subagent"
    )

    convs = ClaudeCodeHistoryParser().parse(claude_dir)

    assert len(convs) == 1
    assert convs[0].id == parent_sid
    assert "Parent task" in convs[0].messages[0].content


def test_parser_skips_internal_jsonl_when_passed_as_file(tmp_path):
    """If a user points sync directly at an internal JSONL (agent-*.jsonl or
    history.jsonl), the parser must still skip it — the file branch of
    parse() must apply the same _is_internal_jsonl() check as the directory
    walk, otherwise the UUID-collision/clobbering protection is bypassed."""
    parent_sid = "33333333-3333-3333-3333-333333333333"

    agent_file = tmp_path / "agent-deadbeef.jsonl"
    _write_session_jsonl(agent_file, parent_sid, "Subagent stamped with parent sid")
    assert ClaudeCodeHistoryParser().parse(agent_file) == []

    history_file = tmp_path / "history.jsonl"
    _write_session_jsonl(history_file, "any-sid", "Prompt history line")
    assert ClaudeCodeHistoryParser().parse(history_file) == []


def test_parser_skips_history_jsonl(tmp_path):
    """history.jsonl is the top-level prompt-history index, not a session."""
    claude_dir = tmp_path / "data"
    claude_dir.mkdir()
    _write_session_jsonl(claude_dir / "history.jsonl", "ignored-sid", "Should skip")
    _write_session_jsonl(
        claude_dir / "real-session.jsonl",
        "22222222-2222-2222-2222-222222222222",
        "Real session",
    )

    convs = ClaudeCodeHistoryParser().parse(claude_dir)

    assert len(convs) == 1
    assert "Real session" in convs[0].messages[0].content


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

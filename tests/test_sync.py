from datetime import datetime
from unittest.mock import patch

from claude_vault.models import Conversation, Message
from claude_vault.sync import SyncEngine


def _make_conv(content: str, uuid: str, title: str = "Test") -> Conversation:
    return Conversation(
        id=uuid,
        title=title,
        messages=[Message(role="human", content=content)],
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def test_sync_skips_llm_when_content_unchanged(tmp_path):
    """Re-syncing an unchanged conversation must not call generate_metadata.

    The content hash is stable across runs, so when a conversation's source
    content has not changed, there is no reason to pay the LLM cost again —
    the markdown would not be rewritten either way.
    """
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    # Simulate real parser behavior: each call returns a fresh Conversation
    # object with no tags populated (source exports don't carry tags).
    def fresh_convs(*_args, **_kwargs):
        return [_make_conv("Stable content", uuid="conv-stable-001", title="Stable")]

    engine = SyncEngine(tmp_path)

    with patch.object(engine.parser, "parse", side_effect=fresh_convs):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ) as mock_meta:
            first = engine.sync(export_file)
            assert first["new"] == 1
            assert mock_meta.call_count == 1

            second = engine.sync(export_file)
            assert second["unchanged"] == 1
            assert second["new"] == 0
            assert second["updated"] == 0
            # Key assertion: LLM NOT called again on the unchanged re-run.
            assert mock_meta.call_count == 1

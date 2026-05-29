from datetime import datetime
from unittest.mock import patch

from claude_vault.models import Conversation, Message
from claude_vault.pii import PIIScanResult
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


def test_sync_with_detect_pii_bypasses_fast_path_on_unchanged(tmp_path):
    """--detect-pii must run analysis even on unchanged conversations.

    If a user first syncs without PII flags and later re-runs
    `sync --detect-pii`, unchanged conversations should be analysed so
    PII tags can be retrofitted into the markdown. The fast-path skip
    is only safe when none of the PII flags are in play.
    """
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    def fresh_convs(*_args, **_kwargs):
        return [_make_conv("email alice@example.com", uuid="conv-pii-001", title="Pii")]

    engine = SyncEngine(tmp_path)
    clean_result = PIIScanResult(detected=False, risk_level="none")

    with patch.object(engine.parser, "parse", side_effect=fresh_convs):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=clean_result
            ) as mock_analyze:
                # First sync: plain, no PII flags. Writes state, no analyze.
                first = engine.sync(export_file)
                assert first["new"] == 1
                assert mock_analyze.call_count == 0

                # Second sync with --detect-pii: must analyse even though
                # content hash matches.
                second = engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 1
                assert second["unchanged"] == 0


def test_sync_reruns_llm_when_content_changed(tmp_path):
    """Changed content must re-run the LLM and count as "updated".

    The fast path only skips when the content hash matches. When the same
    conversation (same id) comes back with different message content, the
    hash differs, so the skip must NOT fire: the LLM re-runs and the result
    is an update, never a skip. This guards the branch the fast path
    refactored from elif/else into a single else.
    """
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    engine = SyncEngine(tmp_path)

    with patch.object(
        engine.tag_generator,
        "generate_metadata",
        return_value={"tags": ["t1", "t2"], "summary": "s"},
    ) as mock_meta:
        # First sync: original content.
        with patch.object(
            engine.parser,
            "parse",
            side_effect=lambda *_a, **_k: [
                _make_conv("Original content", uuid="conv-mutates-001")
            ],
        ):
            first = engine.sync(export_file)
            assert first["new"] == 1
            assert mock_meta.call_count == 1

        # Second sync: same id, different content -> hash changes.
        with patch.object(
            engine.parser,
            "parse",
            side_effect=lambda *_a, **_k: [
                _make_conv("Edited content", uuid="conv-mutates-001")
            ],
        ):
            second = engine.sync(export_file)
            assert second["updated"] == 1
            assert second["unchanged"] == 0
            assert second["new"] == 0
            # LLM re-runs because the content hash no longer matches.
            assert mock_meta.call_count == 2

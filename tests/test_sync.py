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
                # the content hash matches, and the run must count as
                # "reanalyzed" (not "updated") since the source is unchanged.
                second = engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 1
                assert second["unchanged"] == 0
                assert second["reanalyzed"] == 1
                assert second["updated"] == 0


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
            assert second["reanalyzed"] == 0
            # LLM re-runs because the content hash no longer matches.
            assert mock_meta.call_count == 2


def test_sync_redact_pii_keeps_content_hash_stable(tmp_path):
    """--redact-pii must not poison the stored content hash.

    The hash is computed over the source content before redaction runs, and
    the parser yields a fresh conversation from the raw export each sync, so
    a later plain sync still sees the conversation as unchanged — no
    oscillation between reanalyzed/updated/unchanged across runs.
    """
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    def fresh_convs(*_args, **_kwargs):
        return [_make_conv("email alice@example.com", uuid="conv-redact-001")]

    engine = SyncEngine(tmp_path)
    detected = PIIScanResult(detected=True, risk_level="low", pii_types=["email"])

    with patch.object(engine.parser, "parse", side_effect=fresh_convs):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(engine.pii_detector, "analyze", return_value=detected):
                with patch.object(
                    engine.pii_detector, "redact", side_effect=lambda text: "[REDACTED]"
                ) as mock_redact:
                    # Plain first sync stores the source content hash.
                    assert engine.sync(export_file)["new"] == 1

                    # --redact-pii rewrites the markdown for unchanged source,
                    # counted as reanalyzed (not updated).
                    second = engine.sync(export_file, redact_pii=True)
                    assert mock_redact.call_count == 1
                    assert second["reanalyzed"] == 1
                    assert second["updated"] == 0

                    # A later plain sync still sees it as unchanged: redaction
                    # never touched the stored source hash.
                    third = engine.sync(export_file)
                    assert third["unchanged"] == 1
                    assert third["reanalyzed"] == 0
                    assert third["updated"] == 0


def _pii_engine(tmp_path):
    """Engine over an empty vault, with a parser yielding one stable conversation"""
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    engine = SyncEngine(tmp_path)
    return engine, export_file


def _analysed(llm_used=True):
    return PIIScanResult(detected=False, risk_level="none", llm_used=llm_used)


def test_sync_skips_unchanged_conversation_already_processed_under_same_pii(
    tmp_path,
):
    """Same content and same PII settings as last time: nothing to redo.

    Regression test for the fast path being bypassed whenever a PII flag was set.
    """
    engine, export_file = _pii_engine(tmp_path)

    with patch.object(
        engine.parser,
        "parse",
        side_effect=lambda *_a, **_k: [_make_conv("hello", uuid="conv-fp-001")],
    ):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed()
            ) as mock_analyze:
                first = engine.sync(export_file, detect_pii=True)
                assert first["new"] == 1
                assert mock_analyze.call_count == 1

                second = engine.sync(export_file, detect_pii=True)
                assert second["unchanged"] == 1
                assert second["reanalyzed"] == 0
                assert mock_analyze.call_count == 1


def test_sync_reanalyzes_when_pii_settings_change(tmp_path):
    """A different operation must not be skipped as already done.

    Skipping would silently no-op a newly requested flag, e.g. retrofitting
    --redact-pii onto a conversation that was only scanned before.
    """
    engine, export_file = _pii_engine(tmp_path)

    with patch.object(
        engine.parser,
        "parse",
        side_effect=lambda *_a, **_k: [_make_conv("hello", uuid="conv-fp-002")],
    ):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed()
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 1

                # detect -> redact: different settings, must run again
                second = engine.sync(export_file, redact_pii=True)
                assert second["reanalyzed"] == 1
                assert mock_analyze.call_count == 2

                # and is then skipped under the new settings
                third = engine.sync(export_file, redact_pii=True)
                assert third["unchanged"] == 1
                assert mock_analyze.call_count == 2


def test_sync_reanalyzes_when_pii_config_changes(tmp_path):
    """Changing risk_threshold or use_llm changes the outcome, so it re-runs"""
    engine, export_file = _pii_engine(tmp_path)

    with patch.object(
        engine.parser,
        "parse",
        side_effect=lambda *_a, **_k: [_make_conv("hello", uuid="conv-fp-003")],
    ):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed()
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 1

                engine.pii_detector.config.pii.risk_threshold = "high"
                engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 2

                engine.pii_detector.config.pii.use_llm = False
                engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 3


def test_sync_does_not_record_pii_done_when_llm_fell_back(tmp_path):
    """Ollama being down degrades analysis to regex-only, so retry next sync"""
    engine, export_file = _pii_engine(tmp_path)
    engine.pii_detector.config.pii.use_llm = True

    with patch.object(
        engine.parser,
        "parse",
        side_effect=lambda *_a, **_k: [_make_conv("hello", uuid="conv-fp-004")],
    ):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed(llm_used=False)
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                second = engine.sync(export_file, detect_pii=True)
                assert second["unchanged"] == 0
                assert second["reanalyzed"] == 1
                assert mock_analyze.call_count == 2

            # Once the LLM answers, it is recorded and the next run skips
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed(llm_used=True)
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                third = engine.sync(export_file, detect_pii=True)
                assert third["unchanged"] == 1
                assert mock_analyze.call_count == 1


def test_sync_records_pii_done_without_llm_when_llm_disabled(tmp_path):
    """With use_llm off there is no LLM to wait for, regex-only is complete"""
    engine, export_file = _pii_engine(tmp_path)
    engine.pii_detector.config.pii.use_llm = False

    with patch.object(
        engine.parser,
        "parse",
        side_effect=lambda *_a, **_k: [_make_conv("hello", uuid="conv-fp-005")],
    ):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed(llm_used=False)
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                second = engine.sync(export_file, detect_pii=True)
                assert second["unchanged"] == 1
                assert mock_analyze.call_count == 1


def test_sync_pii_fingerprint_is_empty_without_pii_handling(tmp_path):
    engine, _ = _pii_engine(tmp_path)

    assert engine._pii_fingerprint(False, False, False) == ""
    assert engine._pii_fingerprint(True, False, False) != ""
    assert engine._pii_fingerprint(True, False, False) != engine._pii_fingerprint(
        False, True, False
    )

from datetime import datetime
from pathlib import Path
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
                    assert mock_redact.called
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


def _redaction_engine(tmp_path):
    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()
    engine = SyncEngine(tmp_path)
    engine.pii_detector.config.pii.use_llm = False
    return engine, export_file


def _titled_conv(title, uuid, body="please deploy"):
    return Conversation(
        id=uuid,
        title=title,
        messages=[Message(role="human", content=body)],
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def test_redact_pii_also_redacts_title_summary_and_filename(tmp_path):
    """Everything written to disk must be redacted, not just message bodies.

    Regression test: the title went into the frontmatter, the heading and the
    filename, and the LLM summary into the frontmatter, all unredacted.
    """
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Ping alice@example.com about the deploy", "conv-title-001")

    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={
                "tags": ["t1", "t2"],
                "summary": "Asked alice@example.com to deploy",
            },
        ):
            engine.sync(export_file, redact_pii=True)

    files = list((tmp_path / "conversations").glob("*.md"))
    assert len(files) == 1
    assert "alice" not in files[0].name
    assert "example" not in files[0].name
    text = files[0].read_text(encoding="utf-8")
    assert "alice@example.com" not in text
    assert "[REDACTED-EMAIL]" in text


def test_detect_pii_flags_pii_that_is_only_in_the_title(tmp_path):
    """A secret in the title alone must be detected, not just ones in messages"""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Rotate key AKIAIOSFODNN7EXAMPLE", "conv-title-002")

    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            engine.sync(export_file, detect_pii=True)

    text = next((tmp_path / "conversations").glob("*.md")).read_text(encoding="utf-8")
    assert "pii_detected: true" in text


def test_redact_pii_hides_related_titles_regardless_of_order(tmp_path):
    """related_tags/wikilinks name other conversations by their written title.

    The shared conversation list is mutated as each one is processed, so without
    redacting there too, an earlier file leaked a later conversation's real title
    while a later file saw the already-redacted one.
    """
    engine, export_file = _redaction_engine(tmp_path)
    convs = [
        _titled_conv("Ping alice@example.com", "conv-rel-001"),
        _titled_conv("Notes for bob@example.com", "conv-rel-002"),
    ]
    for conv in convs:
        conv.tags = ["shared-a", "shared-b"]

    with patch.object(engine.parser, "parse", return_value=convs):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["shared-a", "shared-b"], "summary": "s"},
        ):
            engine.sync(export_file, redact_pii=True)

    for md_file in (tmp_path / "conversations").glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        assert "alice@example.com" not in text
        assert "bob@example.com" not in text
        assert "alice" not in md_file.name and "bob" not in md_file.name


def _skip_engine(tmp_path):
    engine, export_file = _redaction_engine(tmp_path)
    engine.pii_detector.config.pii.risk_threshold = "medium"
    return engine, export_file


def _flagged():
    return PIIScanResult(
        detected=True, risk_level="high", pii_types=["email"], llm_used=True
    )


def _sync_with_analysis(engine, export_file, conv, result, **flags):
    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=result
            ) as mock_analyze:
                return engine.sync(export_file, **flags), mock_analyze


def test_skip_sensitive_reports_but_keeps_a_file_written_before_the_skip(tmp_path):
    """skip_sensitive keeps new sensitive text out; it must not claim the vault is
    clean while an earlier file with that text is still there."""
    engine, export_file = _skip_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-skip-001", body="mail alice@example.com")

    # Written by a plain sync, before skip_sensitive was turned on
    first, _ = _sync_with_analysis(engine, export_file, conv, _flagged())
    written = list((tmp_path / "conversations").glob("*.md"))
    assert first["new"] == 1 and len(written) == 1
    before = written[0].read_text(encoding="utf-8")

    second, _ = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )

    assert second["skipped"] == 1
    # Shown to the user with the OS separators, so compare as paths
    assert [Path(p) for p in second["skipped_on_disk"]] == [
        Path("conversations") / written[0].name
    ]
    assert written[0].exists()
    assert written[0].read_text(encoding="utf-8") == before


def test_skip_sensitive_of_a_new_conversation_reports_no_file(tmp_path):
    engine, export_file = _skip_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-skip-002", body="mail alice@example.com")

    result, _ = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )

    assert result["skipped"] == 1
    assert result["skipped_on_disk"] == []
    assert list((tmp_path / "conversations").glob("*.md")) == []


def test_skip_sensitive_is_remembered_so_it_is_not_reanalysed_every_sync(tmp_path):
    """A skipped conversation wrote no state, so every sync (and every watch event)
    ran full analysis, including the LLM, on it again."""
    engine, export_file = _skip_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-skip-003", body="mail alice@example.com")
    _sync_with_analysis(engine, export_file, conv, _flagged())

    _, first_analyze = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )
    second, second_analyze = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )

    assert first_analyze.call_count == 1
    assert second_analyze.call_count == 0
    # Still reported as skipped, with its file, not quietly counted as unchanged
    assert second["skipped"] == 1
    assert second["unchanged"] == 0
    assert len(second["skipped_on_disk"]) == 1


def test_skip_sensitive_reanalyses_when_settings_change(tmp_path):
    """The remembered skip applies to the settings it was made under"""
    engine, export_file = _skip_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-skip-004", body="mail alice@example.com")
    _sync_with_analysis(engine, export_file, conv, _flagged())
    _sync_with_analysis(engine, export_file, conv, _flagged(), skip_sensitive=True)

    engine.pii_detector.config.pii.risk_threshold = "high"
    _, analyze = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )

    assert analyze.call_count == 1


def test_skip_sensitive_dry_run_does_not_record_the_skip(tmp_path):
    engine, export_file = _skip_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-skip-005", body="mail alice@example.com")
    _sync_with_analysis(engine, export_file, conv, _flagged())

    _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True, dry_run=True
    )
    _, analyze = _sync_with_analysis(
        engine, export_file, conv, _flagged(), skip_sensitive=True
    )

    assert analyze.call_count == 1


def test_sync_rescans_when_the_pii_detector_changed(tmp_path):
    """An unchanged conversation scanned by an older detector must be scanned again.

    Otherwise a detector upgrade (new secret formats, fewer false positives) never
    reaches the conversations already in the vault, and the settings fingerprint
    still matches.
    """
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-version-001")

    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            with patch.object(
                engine.pii_detector, "analyze", return_value=_analysed()
            ) as mock_analyze:
                engine.sync(export_file, detect_pii=True)
                engine.sync(export_file, detect_pii=True)
                assert mock_analyze.call_count == 1  # same detector: skipped

                with patch.object(
                    type(engine.pii_detector),
                    "version",
                    new_callable=lambda: property(lambda self: "newdetector"),
                ):
                    second = engine.sync(export_file, detect_pii=True)

    assert mock_analyze.call_count == 2
    assert second["reanalyzed"] == 1
    assert second["unchanged"] == 0


def _read_frontmatter(tmp_path):
    import frontmatter

    files = list((tmp_path / "conversations").glob("*.md"))
    assert len(files) == 1
    return frontmatter.load(files[0])


def test_scanned_clean_conversation_is_marked_as_scanned(tmp_path):
    """'Scanned and clean' and 'never scanned' used to look identical on disk, since
    only a positive detection wrote anything. A reader (such as an MCP server)
    cannot treat 'unknown' as safe, so it has to be able to tell them apart."""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-marker-001")

    _sync_with_analysis(
        engine, export_file, conv, _analysed(llm_used=True), detect_pii=True
    )

    post = _read_frontmatter(tmp_path)
    assert post["pii_scanned"] is True
    assert post["pii_detector"] == engine.pii_detector.version
    assert post["pii_llm"] is True
    assert "pii_detected" not in post.metadata


def test_unscanned_conversation_has_no_scan_marker(tmp_path):
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-marker-002")

    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            engine.sync(export_file)

    post = _read_frontmatter(tmp_path)
    assert "pii_scanned" not in post.metadata
    assert "pii_detector" not in post.metadata


def test_scan_marker_records_whether_the_llm_took_part(tmp_path):
    """A regex-only scan (Ollama down) is weaker, and the reader should know"""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-marker-003")

    _sync_with_analysis(
        engine, export_file, conv, _analysed(llm_used=False), detect_pii=True
    )

    assert _read_frontmatter(tmp_path)["pii_llm"] is False


def test_scan_marker_is_written_next_to_a_detection(tmp_path):
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv(
        "Deploy notes", "conv-marker-004", body="mail alice@example.com"
    )

    _sync_with_analysis(engine, export_file, conv, _flagged(), detect_pii=True)

    post = _read_frontmatter(tmp_path)
    assert post["pii_scanned"] is True
    assert post["pii_detected"] is True
    assert post["risk_level"] == "high"


def test_conversation_scanned_by_an_older_detector_is_scanned_again(tmp_path):
    """Files written before the marker existed have none, and DETECTOR_REVISION was
    bumped for that, so the next PII-enabled sync adds it"""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Deploy notes", "conv-marker-005")
    _sync_with_analysis(engine, export_file, conv, _analysed(), detect_pii=True)
    engine.state.save_conversation(
        conv.id,
        next((tmp_path / "conversations").glob("*.md"))
        .relative_to(tmp_path)
        .as_posix(),
        conv.content_hash(),
        {"title": conv.title, "pii_fingerprint": "from-an-older-detector"},
    )

    second, _ = _sync_with_analysis(
        engine, export_file, conv, _analysed(), detect_pii=True
    )

    assert second["reanalyzed"] == 1
    assert _read_frontmatter(tmp_path)["pii_scanned"] is True


def _store_chunk(engine, vault, conv_id, text):
    """Put a chunk in the search index the way semantic search does"""
    import numpy as np

    engine.state.save_embedding(
        conv_id, 0, text, np.array([0.1, 0.2, 0.3], dtype=np.float32), "test-model"
    )


def _index_chunks(engine):
    return [e["chunk_text"] for e in engine.state.get_all_embeddings()]


def test_redacting_a_conversation_also_clears_its_search_index_entries(tmp_path):
    """The embeddings table keeps a copy of each chunk's text and search shows it.
    Re-syncing with redaction rewrote the file but left the original, unredacted
    text in state.db, so search handed it back."""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Notes", "conv-idx-001", body="my email is alice@example.com")

    _sync_with_analysis(engine, export_file, conv, _analysed())
    _store_chunk(engine, tmp_path, "conv-idx-001", "my email is alice@example.com")
    assert any("alice@example.com" in c for c in _index_chunks(engine))

    medium = PIIScanResult(
        detected=True, risk_level="medium", pii_types=["email"], llm_used=True
    )
    _sync_with_analysis(engine, export_file, conv, medium, redact_pii=True)

    assert _index_chunks(engine) == []


def test_new_content_clears_stale_search_index_entries(tmp_path):
    engine, export_file = _redaction_engine(tmp_path)
    original = _titled_conv("Notes", "conv-idx-002", body="first version")
    _sync_with_analysis(engine, export_file, original, _analysed())
    _store_chunk(engine, tmp_path, "conv-idx-002", "first version")

    changed = _titled_conv("Notes", "conv-idx-002", body="second version")
    _sync_with_analysis(engine, export_file, changed, _analysed())

    assert _index_chunks(engine) == []


def test_unchanged_conversation_keeps_its_search_index_entries(tmp_path):
    """Dropping the index on every sync would force every embedding to be rebuilt"""
    engine, export_file = _redaction_engine(tmp_path)
    conv = _titled_conv("Notes", "conv-idx-003", body="stable text")
    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(
            engine.tag_generator,
            "generate_metadata",
            return_value={"tags": ["t1", "t2"], "summary": "s"},
        ):
            engine.sync(export_file)
            _store_chunk(engine, tmp_path, "conv-idx-003", "stable text")
            result = engine.sync(export_file)

    assert result["unchanged"] == 1
    assert _index_chunks(engine) == ["stable text"]


def test_dry_run_does_not_touch_the_search_index(tmp_path):
    engine, export_file = _redaction_engine(tmp_path)
    original = _titled_conv("Notes", "conv-idx-004", body="first version")
    _sync_with_analysis(engine, export_file, original, _analysed())
    _store_chunk(engine, tmp_path, "conv-idx-004", "first version")

    changed = _titled_conv("Notes", "conv-idx-004", body="second version")
    _sync_with_analysis(engine, export_file, changed, _analysed(), dry_run=True)

    assert _index_chunks(engine) == ["first version"]

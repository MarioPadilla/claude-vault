import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from claude_vault.models import Conversation, Message
from claude_vault.pii import PIIDetector, PIIScanResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    with patch("claude_vault.pii.load_config") as mock_cfg:
        mock_cfg.return_value.ollama.url = "http://localhost:11434/api/generate"
        mock_cfg.return_value.ollama.model = "llama3.2:3b"
        mock_cfg.return_value.ollama.timeout = 15
        mock_cfg.return_value.pii.use_llm = True
        mock_cfg.return_value.pii.risk_threshold = "medium"
        return PIIDetector()


def _make_conv(content: str, title: str = "Test") -> Conversation:
    return Conversation(
        title=title,
        messages=[Message(role="human", content=content)],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Regex scan – positive detections
# ---------------------------------------------------------------------------


def test_scan_detects_email(detector):
    result = detector.scan("Contact me at alice@example.com for details.")
    assert result.detected
    assert "email" in result.pii_types
    assert result.risk_level == "medium"


def test_scan_detects_phone(detector):
    result = detector.scan("Call me on +1 (555) 123-4567 anytime.")
    assert result.detected
    assert "phone" in result.pii_types
    assert result.risk_level == "medium"


def test_scan_detects_ssn(detector):
    result = detector.scan("My SSN is 123-45-6789.")
    assert result.detected
    assert "ssn" in result.pii_types
    assert result.risk_level == "high"


def test_scan_detects_credit_card(detector):
    result = detector.scan("Card number: 4111 1111 1111 1111")
    assert result.detected
    assert "credit_card" in result.pii_types
    assert result.risk_level == "high"


def test_scan_detects_api_key_openai(detector):
    result = detector.scan("Use this key: sk-abcdefghijklmnopqrstuvwxyz123456")
    assert result.detected
    assert "api_key" in result.pii_types
    assert result.risk_level == "high"


def test_scan_detects_ip_address(detector):
    result = detector.scan("Server is at 192.168.1.100.")
    assert result.detected
    assert "ip_address" in result.pii_types
    assert result.risk_level == "low"


def test_scan_detects_credential_context(detector):
    result = detector.scan("password: mysecretpassword123")
    assert result.detected
    assert "credential_context" in result.pii_types


# ---------------------------------------------------------------------------
# Regex scan – clean text
# ---------------------------------------------------------------------------


def test_scan_clean_text_returns_no_detection(detector):
    result = detector.scan("This is a completely harmless message about cooking.")
    assert not result.detected
    assert result.risk_level == "none"
    assert result.pii_types == []


# ---------------------------------------------------------------------------
# Risk level heuristic
# ---------------------------------------------------------------------------


def test_risk_level_high_beats_medium(detector):
    text = "Email alice@example.com, SSN 123-45-6789"
    result = detector.scan(text)
    assert result.risk_level == "high"


def test_risk_level_none_for_empty(detector):
    result = detector.scan("")
    assert result.risk_level == "none"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_replaces_email(detector):
    redacted = detector.redact("Send to bob@domain.org please.")
    assert "bob@domain.org" not in redacted
    assert "[REDACTED-EMAIL]" in redacted


def test_redact_replaces_ssn(detector):
    redacted = detector.redact("SSN: 987-65-4321")
    assert "987-65-4321" not in redacted
    assert "[REDACTED-SSN]" in redacted


def test_redact_clean_text_unchanged(detector):
    original = "Nothing sensitive here at all."
    assert detector.redact(original) == original


def test_redact_multiple_types(detector):
    text = "Email: test@example.com, card: 4111 1111 1111 1111"
    redacted = detector.redact(text)
    assert "test@example.com" not in redacted
    assert "[REDACTED-EMAIL]" in redacted


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


def test_classify_with_llm_sensitive(detector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps(
            {"is_sensitive": True, "risk_level": "high", "reason": "Contains secrets"}
        )
    }
    with patch("requests.post", return_value=mock_response):
        result = detector.classify_with_llm(_make_conv("My password is hunter2"))
    assert result["is_sensitive"] is True
    assert result["risk_level"] == "high"


def test_classify_with_llm_not_sensitive(detector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps(
            {"is_sensitive": False, "risk_level": "none", "reason": "Safe content"}
        )
    }
    with patch("requests.post", return_value=mock_response):
        result = detector.classify_with_llm(_make_conv("Let's talk about gardening."))
    assert result["is_sensitive"] is False
    assert result["risk_level"] == "none"


def test_classify_with_llm_fallback_on_error(detector):
    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = detector.classify_with_llm(_make_conv("Some content"))
    assert result["is_sensitive"] is False
    assert result["risk_level"] == "none"


# ---------------------------------------------------------------------------
# analyze() – LLM elevates risk
# ---------------------------------------------------------------------------


def test_analyze_llm_elevates_risk(detector):
    """LLM flags as high risk even though regex finds nothing."""
    conv = _make_conv("Top-secret internal strategy document.")

    mock_get = Mock(status_code=200)
    mock_post = Mock()
    mock_post.status_code = 200
    mock_post.json.return_value = {
        "response": json.dumps(
            {"is_sensitive": True, "risk_level": "high", "reason": "Confidential"}
        )
    }

    with patch("requests.get", return_value=mock_get):
        with patch("requests.post", return_value=mock_post):
            result = detector.analyze(conv, use_llm=True)

    assert result.risk_level == "high"
    assert result.detected


def test_analyze_no_llm_uses_regex_only(detector):
    conv = _make_conv("Phone: 555-867-5309")
    result = detector.analyze(conv, use_llm=False)
    assert result.detected
    assert "phone" in result.pii_types


# ---------------------------------------------------------------------------
# Sync integration – skip_sensitive
# ---------------------------------------------------------------------------


def test_sync_skip_sensitive(tmp_path):
    """Conversations above the risk threshold should be counted as skipped."""
    from claude_vault.sync import SyncEngine

    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")  # content doesn't matter; parser is mocked
    (tmp_path / ".claude-vault").mkdir()
    (tmp_path / "conversations").mkdir()

    conv = _make_conv("My email is secret@corp.com", title="Sensitive Convo")
    conv.id = "aaaa-bbbb-cccc-dddd"

    engine = SyncEngine(tmp_path)

    # Mock parser to return our prepared conversation
    with patch.object(engine.parser, "parse", return_value=[conv]):
        scan_result = PIIScanResult(
            detected=True,
            risk_level="medium",
            pii_types=["email"],
            matches={"email": ["secret@corp.com"]},
        )
        with patch.object(engine.pii_detector, "analyze", return_value=scan_result):
            result = engine.sync(
                export_file, detect_pii=True, skip_sensitive=True, dry_run=True
            )

    assert result["skipped"] == 1
    assert result["new"] == 0


# ---------------------------------------------------------------------------
# Sync integration – redact_pii
# ---------------------------------------------------------------------------


def test_sync_redact_pii_writes_redacted_content(tmp_path):
    """After sync with --redact-pii, the output file must not contain raw PII."""
    from claude_vault.sync import SyncEngine

    export_file = tmp_path / "conversations.json"
    export_file.write_text("[]")  # content doesn't matter; parser is mocked
    (tmp_path / ".claude-vault").mkdir()

    conv = _make_conv("Contact alice@private.io for info.", title="Redact Test")
    conv.id = "aaaa-bbbb-cccc-1111"

    engine = SyncEngine(tmp_path)

    scan_result = PIIScanResult(
        detected=True,
        risk_level="medium",
        pii_types=["email"],
        matches={"email": ["alice@private.io"]},
    )
    with patch.object(engine.parser, "parse", return_value=[conv]):
        with patch.object(engine.pii_detector, "analyze", return_value=scan_result):
            result = engine.sync(export_file, detect_pii=True, redact_pii=True)

    assert result["new"] == 1

    # Find the written markdown and verify PII is redacted
    md_files = list((tmp_path / "conversations").glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "alice@private.io" not in content
    assert "[REDACTED-EMAIL]" in content


# ---------------------------------------------------------------------------
# llm_used – distinguishes a real LLM verdict from a silent fallback
# ---------------------------------------------------------------------------


def test_classify_with_llm_reports_classified(detector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({"is_sensitive": False, "risk_level": "none"})
    }
    with patch("requests.post", return_value=mock_response):
        result = detector.classify_with_llm(_make_conv("Let's talk about gardening."))
    assert result["classified"] is True


def test_classify_with_llm_fallback_is_not_classified(detector):
    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = detector.classify_with_llm(_make_conv("Some content"))
    assert result["classified"] is False


def test_analyze_marks_llm_used_only_when_llm_answered(detector):
    conv = _make_conv("Nothing sensitive here.")
    mock_post = Mock(status_code=200)
    mock_post.json.return_value = {
        "response": json.dumps({"is_sensitive": False, "risk_level": "none"})
    }

    with patch("requests.get", return_value=Mock(status_code=200)):
        with patch("requests.post", return_value=mock_post):
            assert detector.analyze(conv, use_llm=True).llm_used is True

        with patch("requests.post", side_effect=Exception("timeout")):
            assert detector.analyze(conv, use_llm=True).llm_used is False

    # Ollama unreachable, or LLM disabled: never used
    with patch("requests.get", side_effect=Exception("down")):
        assert detector.analyze(conv, use_llm=True).llm_used is False
    assert detector.analyze(conv, use_llm=False).llm_used is False

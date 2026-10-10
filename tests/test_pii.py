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


# ---------------------------------------------------------------------------
# Precision – coding sessions are full of strings that only look like secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "commit 99e319457dcc825fd9a5f1e5af7fe32b84e40d6c fixed it",
        "sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "session 3f2504e0-4f89-11d3-9a0c-0305e82c3301 started",
        "call getUserAuthenticationTokenFromRequestHeader() here",
        "const handleSubmitFormValidationAndRedirect = () => {}",
        "timestamp 1700000000 and id 5551234567",
        "max_tokens = 4096 and token_count: 1200",
        "order 1234567890123456 shipped",
        '{"file_path": "/Users/dev/repos/app/src/semantic_search_engine_module.py"}',
    ],
)
def test_scan_ignores_lookalikes_common_in_code(detector, text):
    result = detector.scan(text)
    assert not result.detected, result.pii_types


# ---------------------------------------------------------------------------
# Recall – secrets that previously slipped through
# ---------------------------------------------------------------------------


def _fake(prefix: str, body: str) -> str:
    """Join a provider prefix and a made-up body at runtime.

    These are fixtures, not credentials. Built in two parts so no complete
    token-shaped literal sits in the source for secret scanners to flag.
    """
    return prefix + body


@pytest.mark.parametrize(
    "text, expected_type",
    [
        (
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "credential_context",
        ),
        ("AKIAIOSFODNN7EXAMPLE", "api_key"),
        (
            _fake("xoxb-", "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"),
            "api_key",
        ),
        (_fake("sk_live_", "51H8abcdefghijklmnopqrstuv"), "api_key"),
        ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH", "api_key"),
        ("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789", "api_key"),
        ("Authorization: Bearer abc123def456ghi789xyz", "api_key"),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "jwt",
        ),
        ("postgres://admin:hunter2@db.internal:5432/prod", "connection_string"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA", "private_key"),
        ("key is Zk3Jq9XvB2mN7pLw4RtY8uHc1VsD6fGa5E", "secret_token"),
    ],
)
def test_scan_detects_common_secret_formats(detector, text, expected_type):
    result = detector.scan(text)
    assert expected_type in result.pii_types, result.pii_types
    assert result.detected


def test_scan_connection_string_password_is_not_reported_as_email(detector):
    result = detector.scan("postgres://admin:hunter2@db.internal:5432/prod")
    assert "email" not in result.pii_types


def test_scan_credit_card_requires_valid_checksum(detector):
    assert "credit_card" in detector.scan("card 4111 1111 1111 1111").pii_types
    assert "credit_card" not in detector.scan("id 1234 5678 9012 3456").pii_types


def test_scan_phone_requires_formatting(detector):
    assert "phone" in detector.scan("call 555-867-5309").pii_types
    assert "phone" in detector.scan("call +1 (555) 123-4567").pii_types
    assert "phone" not in detector.scan("id 5558675309").pii_types


def test_scan_reports_the_full_matched_secret(detector):
    """matches feeds redaction, so it must hold the whole value, not a group"""
    result = detector.scan("token ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789 here")
    assert result.matches["api_key"] == ["ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"]


# ---------------------------------------------------------------------------
# Redaction of the new types
# ---------------------------------------------------------------------------


def test_redact_removes_secrets_and_keeps_lookalikes(detector):
    sha = "99e319457dcc825fd9a5f1e5af7fe32b84e40d6c"
    text = (
        f"commit {sha}\n"
        "curl -H 'Authorization: Bearer abc123def456ghi789xyz'\n"
        "DATABASE_URL=postgres://admin:hunter2@db.internal/prod\n"
        "AKIAIOSFODNN7EXAMPLE"
    )
    redacted = detector.redact(text)

    assert sha in redacted
    assert "abc123def456ghi789xyz" not in redacted
    assert "hunter2" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted


def test_redact_removes_whole_private_key_block(detector):
    text = (
        "before\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\nabcdef==\n"
        "-----END PRIVATE KEY-----\nafter"
    )
    redacted = detector.redact(text)

    assert "MIIEvQIBADANBgkq" not in redacted
    assert "[REDACTED-PRIVATE-KEY]" in redacted
    assert redacted.startswith("before") and redacted.endswith("after")


def test_redact_removes_truncated_private_key_to_end_of_text(detector):
    redacted = detector.redact("x\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq")
    assert "MIIEvQIBADANBgkq" not in redacted


# ---------------------------------------------------------------------------
# Regressions found in review of the detector rewrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "a-" * 100_000,
        "token-" * 30_000,
        "a." * 100_000,
        "a" * 200_000,
        "password=" * 20_000,
        "://" * 60_000,
    ],
)
def test_scan_and_redact_run_in_linear_time_on_adversarial_input(detector, text):
    """Unbounded repetition in a pattern backtracks quadratically (60s+ at 200k)"""
    import time

    start = time.perf_counter()
    detector.scan(text)
    detector.redact(text)
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize(
    "text",
    [
        "password: hunter2",
        "token: short1",
        "password=abc123",
        '{"password": "hunter2hunter2"}',
        '"api_key": "abcdefgh12345678"',
        "Authorization: Basic dXNlcjpzdXBlcnNlY3JldHBhc3N3b3Jk",
        "redis://:s3cretpass@cache:6379/0",
    ],
)
def test_scan_detects_short_quoted_and_basic_auth_credentials(detector, text):
    assert detector.scan(text).detected


def test_redact_removes_whole_quoted_passphrase(detector):
    redacted = detector.redact('password="correct horse battery"')
    assert "horse" not in redacted and "battery" not in redacted


@pytest.mark.parametrize(
    "text",
    [
        "max_tokens = 4096 and token_count: 1200",
        "input tokens: 12345678 output tokens: 99",
    ],
)
def test_scan_ignores_numeric_token_counts(detector, text):
    assert not detector.scan(text).detected

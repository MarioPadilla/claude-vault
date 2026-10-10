import json

import pytest

from claude_vault import config as config_module
from claude_vault.config import load_config

PII_ON = {"pii": {"enabled": True, "redact": True, "skip_sensitive": True}}


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_FILE", path)
    return path


def test_load_config_missing_file_returns_defaults_silently(config_file, recwarn):
    config = load_config()

    assert config.pii.enabled is False
    assert len(recwarn) == 0


def test_load_config_reads_plain_utf8(config_file):
    config_file.write_text(json.dumps(PII_ON), encoding="utf-8")

    config = load_config()

    assert config.pii.enabled and config.pii.redact and config.pii.skip_sensitive


def test_load_config_accepts_utf8_bom(config_file):
    """Windows editors add a BOM; it used to silently reset every setting"""
    config_file.write_bytes(b"\xef\xbb\xbf" + json.dumps(PII_ON).encode("utf-8"))

    config = load_config()

    assert config.pii.enabled and config.pii.redact and config.pii.skip_sensitive


def test_load_config_reads_non_ascii_content(config_file):
    config_file.write_text(
        json.dumps({**PII_ON, "custom_keywords": {"café": ["ā", "日本語"]}}),
        encoding="utf-8",
    )

    config = load_config()

    assert config.pii.enabled
    assert config.custom_keywords == {"café": ["ā", "日本語"]}


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"pii": {"enabled": "maybe"}}'],
    ids=["invalid_json", "invalid_value"],
)
def test_load_config_warns_loudly_when_falling_back_to_defaults(config_file, content):
    """A broken config must not silently switch PII handling off"""
    config_file.write_text(content, encoding="utf-8")

    with pytest.warns(UserWarning, match="PII handling is OFF"):
        config = load_config()

    assert config.pii.enabled is False

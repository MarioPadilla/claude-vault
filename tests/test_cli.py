import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from claude_vault.cli import app
from claude_vault.config import Config

runner = CliRunner()


def test_init_command(tmp_path):
    """Test that init command creates necessary files"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Claude Vault initialized" in result.stdout

        # Check files were created
        config_dir = Path(".claude-vault")
        assert config_dir.exists()
        assert (config_dir / "config.json").exists()
        assert Path("conversations").exists()

        # Verify config content
        with open(config_dir / "config.json") as f:
            config = json.load(f)
            assert config["version"] == "0.1.0"


def test_init_already_initialized(tmp_path):
    """Test init command when already initialized"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Run once
        runner.invoke(app, ["init"])

        # Run again
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "already initialized" in result.stdout


def test_config_command(tmp_path):
    """Test config command output"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Initialize first
        runner.invoke(app, ["init"])

        # Run config command
        result = runner.invoke(app, ["config"], input="n\n")  # Say no to editing

        assert result.exit_code == 0
        assert "Current Settings" in result.stdout
        assert "llama3.2:3b" in result.stdout  # Default model


@patch("claude_vault.cli.OfflineTagGenerator")
@patch("claude_vault.cli.ClaudeExportParser")
def test_retag_command_no_ollama(mock_parser, mock_tag_gen_cls, tmp_path):
    """Test retag fails when Ollama is not available"""
    # Mock instance
    mock_tag_gen = MagicMock()
    mock_tag_gen_cls.return_value = mock_tag_gen

    # Mock config on the instance
    mock_tag_gen.config = Config()

    # Mock is_available to return False
    mock_tag_gen.is_available.return_value = False

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["retag"])

        assert result.exit_code == 1
        assert "Ollama not running" in result.stdout


def test_sync_no_file():
    """Test sync command with missing file"""
    result = runner.invoke(app, ["sync", "nonexistent.json"])
    assert result.exit_code == 1
    assert "Export file not found" in result.stdout


def test_sync_opencode_source_no_path(tmp_path):
    """Test sync with --source opencode resolves default DB path"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["sync", "--source", "opencode"])
        # Default path does not exist in CI; expect a file-not-found error
        assert result.exit_code == 1
        assert "opencode.db" in result.stdout


def test_sync_auto_detects_db_file(tmp_path):
    """Test that .db files are auto-detected as opencode source"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        # Create a valid but empty OpenCode SQLite database
        fake_db = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(fake_db))
        conn.executescript("""
            CREATE TABLE session (
                id text PRIMARY KEY, project_id text NOT NULL, parent_id text,
                slug text NOT NULL, directory text NOT NULL, title text NOT NULL,
                version text NOT NULL, share_url text, summary_additions integer,
                summary_deletions integer, summary_files integer, summary_diffs text,
                revert text, permission text, time_created integer NOT NULL,
                time_updated integer NOT NULL, time_compacting integer,
                time_archived integer, workspace_id text
            );
            CREATE TABLE message (
                id text PRIMARY KEY, session_id text NOT NULL,
                time_created integer NOT NULL, time_updated integer NOT NULL,
                data text NOT NULL
            );
            CREATE TABLE part (
                id text PRIMARY KEY, message_id text NOT NULL, session_id text NOT NULL,
                time_created integer NOT NULL, time_updated integer NOT NULL,
                data text NOT NULL
            );
        """)
        conn.close()
        result = runner.invoke(app, ["sync", str(fake_db), "--dry-run"])
        # Should detect as opencode, parse 0 sessions, and exit cleanly
        assert result.exit_code == 0
        assert result.output  # progress/result table was printed


def test_sync_auto_detects_projects_subdir(tmp_path):
    """A renamed .claude directory is auto-detected when it contains projects/<hash>/*.jsonl.

    Covers the common case of syncing from a copy of ~/.claude that lives
    in a differently-named directory (e.g. after copying from another
    machine).
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        source_dir = Path("claude-remote")
        (source_dir / "projects" / "abc123").mkdir(parents=True)
        (source_dir / "projects" / "abc123" / "session.jsonl").touch()
        result = runner.invoke(app, ["sync", str(source_dir), "--dry-run"])
        assert result.exit_code == 0


def test_sync_auto_detects_loose_jsonl_dir(tmp_path):
    """A directory containing .jsonl files is auto-detected as code source."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        source_dir = Path("claude-jsonl-dump")
        source_dir.mkdir()
        (source_dir / "session1.jsonl").touch()
        result = runner.invoke(app, ["sync", str(source_dir), "--dry-run"])
        assert result.exit_code == 0


def _run_sync_with_pii_config(tmp_path, pii_config, extra_args=()):
    """Run `sync` against a stubbed engine and return the kwargs it received"""
    config = Config()
    config.pii = pii_config

    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        export_file = Path("conversations.json")
        export_file.write_text("[]")

        with patch("claude_vault.cli.load_config", return_value=config):
            with patch("claude_vault.cli.SyncEngine") as mock_engine_cls:
                mock_engine_cls.return_value.sync.return_value = {
                    "new": 0,
                    "updated": 0,
                    "recreated": 0,
                    "unchanged": 0,
                    "errors": 0,
                    "details": [],
                }
                result = runner.invoke(app, ["sync", str(export_file), *extra_args])

        assert result.exit_code == 0, result.stdout
        return mock_engine_cls.return_value.sync.call_args.kwargs


def test_sync_applies_pii_settings_from_config(tmp_path):
    """Regression test: config.pii must apply without any CLI flags"""
    from claude_vault.config import PIIConfig

    kwargs = _run_sync_with_pii_config(
        tmp_path, PIIConfig(enabled=True, redact=True, skip_sensitive=True)
    )

    assert kwargs["detect_pii"] is True
    assert kwargs["redact_pii"] is True
    assert kwargs["skip_sensitive"] is True


def test_sync_pii_defaults_off_without_config_or_flags(tmp_path):
    """No config and no flags must leave PII handling off"""
    from claude_vault.config import PIIConfig

    kwargs = _run_sync_with_pii_config(tmp_path, PIIConfig())

    assert kwargs["detect_pii"] is False
    assert kwargs["redact_pii"] is False
    assert kwargs["skip_sensitive"] is False


def test_sync_cli_flag_enables_pii_when_config_is_off(tmp_path):
    """An explicit flag still enables PII for a single run"""
    from claude_vault.config import PIIConfig

    kwargs = _run_sync_with_pii_config(
        tmp_path, PIIConfig(), extra_args=["--redact-pii"]
    )

    assert kwargs["redact_pii"] is True
    assert kwargs["detect_pii"] is False


@pytest.mark.parametrize("recorded_pid", [0, -1, None])
def test_watch_stop_does_not_signal_an_invalid_recorded_pid(tmp_path, recorded_pid):
    """The pid comes from the state DB. os.kill(0, ...) signals the process group
    and os.kill(-1, ...) every process the user can signal."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])

        with patch("claude_vault.cli.WatchManager") as mock_manager:
            mock_manager.return_value.get_status.return_value = {
                "is_running": True,
                "pid": recorded_pid,
            }
            with patch("os.kill") as mock_kill:
                result = runner.invoke(app, ["watch-stop"])

        assert result.exit_code == 0
        mock_kill.assert_not_called()
        assert "No valid watch process id" in result.stdout


def test_init_mentions_pii_scanning(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "pasted secrets" in result.stdout
        assert "--detect-pii" in result.stdout


def _sync_export(tmp_path, *extra_args):
    """Sync a one-conversation export in a fresh vault, with config.pii untouched"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        export_file = Path("conversations.json")
        export_file.write_text(
            json.dumps(
                [
                    {
                        "uuid": "c1",
                        "name": "Chat",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z",
                        "chat_messages": [
                            {
                                "uuid": "m1",
                                "sender": "human",
                                "text": "hello",
                                "created_at": "2024-01-01T00:00:00Z",
                            },
                            {
                                "uuid": "m2",
                                "sender": "assistant",
                                "text": "hi",
                                "created_at": "2024-01-01T00:00:01Z",
                            },
                        ],
                    }
                ]
            )
        )
        with patch("claude_vault.cli.load_config", return_value=Config()):
            return runner.invoke(app, ["sync", str(export_file), *extra_args])


def test_sync_says_when_files_were_written_without_a_pii_scan(tmp_path):
    result = _sync_export(tmp_path)

    assert result.exit_code == 0
    assert "written without a PII scan" in result.stdout
    assert "rotate it" in result.stdout


def test_sync_does_not_warn_when_pii_scanning_is_on(tmp_path):
    result = _sync_export(tmp_path, "--detect-pii")

    assert result.exit_code == 0
    assert "written without a PII scan" not in result.stdout


def test_sync_dry_run_does_not_warn_about_unscanned_files(tmp_path):
    result = _sync_export(tmp_path, "--dry-run")

    assert result.exit_code == 0
    assert "written without a PII scan" not in result.stdout


# ---------------------------------------------------------------------------
# verify – files that exist but were cut short
# ---------------------------------------------------------------------------


def _synced_vault(tmp_path, messages=4):
    """Init a vault in the cwd and sync one conversation; returns its markdown file"""
    runner.invoke(app, ["init"])
    export_file = Path("conversations.json")
    export_file.write_text(
        json.dumps(
            [
                {
                    "uuid": "c1",
                    "name": "Excel Spreadsheet Connection",
                    "created_at": "2025-06-03T00:00:00Z",
                    "updated_at": "2025-06-03T00:00:00Z",
                    "chat_messages": [
                        {
                            "uuid": f"m{i}",
                            "sender": "human" if i % 2 == 0 else "assistant",
                            "text": f"message number {i} with some text",
                            "created_at": "2025-06-03T00:00:00Z",
                        }
                        for i in range(messages)
                    ],
                }
            ]
        )
    )
    with patch("claude_vault.cli.load_config", return_value=Config()):
        runner.invoke(app, ["sync", str(export_file)])
    return next(Path("conversations").glob("*.md"))


def test_verify_passes_an_intact_vault(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _synced_vault(tmp_path)

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 0
        assert "verified successfully" in result.stdout


def test_verify_detects_a_truncated_file(tmp_path):
    """A failed write leaves the file in place but cut short. verify only checked
    that it exists, so it reported success on a destroyed collection."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        md_file = _synced_vault(tmp_path)
        md_file.write_bytes(md_file.read_bytes()[:220])

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "damaged" in result.stdout
        assert "verified successfully" not in result.stdout
        assert md_file.name in result.stdout


def test_verify_detects_an_empty_file(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        md_file = _synced_vault(tmp_path)
        md_file.write_text("")

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "is empty" in result.stdout


def test_verify_cleanup_does_not_remove_damaged_files(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        md_file = _synced_vault(tmp_path)
        md_file.write_text("")

        result = runner.invoke(app, ["verify", "--cleanup"])

        assert result.exit_code == 1
        assert md_file.exists()
        assert "does not touch them" in result.stdout


def test_verify_still_cleans_up_missing_files(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        md_file = _synced_vault(tmp_path)
        md_file.unlink()

        result = runner.invoke(app, ["verify", "--cleanup"])

        assert result.exit_code == 0
        assert "Cleaned up 1 orphaned" in result.stdout

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from claude_vault.opencode_parser import OpenCodeParser


@pytest.fixture
def test_db(tmp_path):
    """Create a minimal test OpenCode SQLite database"""
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db_path))

    # Create tables matching OpenCode schema
    conn.executescript("""
        CREATE TABLE project (
            id text PRIMARY KEY,
            worktree text NOT NULL,
            vcs text,
            name text,
            icon_url text,
            icon_color text,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            time_initialized integer,
            sandboxes text NOT NULL,
            commands text
        );
        CREATE TABLE session (
            id text PRIMARY KEY,
            project_id text NOT NULL,
            parent_id text,
            slug text NOT NULL,
            directory text NOT NULL,
            title text NOT NULL,
            version text NOT NULL,
            share_url text,
            summary_additions integer,
            summary_deletions integer,
            summary_files integer,
            summary_diffs text,
            revert text,
            permission text,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            time_compacting integer,
            time_archived integer,
            workspace_id text
        );
        CREATE TABLE message (
            id text PRIMARY KEY,
            session_id text NOT NULL,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            data text NOT NULL
        );
        CREATE TABLE part (
            id text PRIMARY KEY,
            message_id text NOT NULL,
            session_id text NOT NULL,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            data text NOT NULL
        );
    """)

    # Insert test project
    conn.execute(
        "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "proj_test",
            "/Users/dev/test-project",
            "git",
            "test-project",
            None,
            None,
            1700000000000,
            1700000000000,
            None,
            "[]",
            None,
        ),
    )

    # Insert root session
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ses_test123",
            "proj_test",
            None,  # parent_id (root session)
            "test-session",
            "/Users/dev/test-project",
            "Test Session Title",
            "1.0",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1700000100000,
            1700000300000,
            None,
            None,
            None,
        ),
    )

    # Insert child session (should be skipped)
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ses_child456",
            "proj_test",
            "ses_test123",  # parent_id (child session)
            "child-session",
            "/Users/dev/test-project",
            "Child Session",
            "1.0",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1700000150000,
            1700000150000,
            None,
            None,
            None,
        ),
    )

    # Insert messages
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            "msg_user1",
            "ses_test123",
            1700000110000,
            1700000110000,
            json.dumps({"role": "user", "time": {"created": 1700000110000}}),
        ),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            "msg_asst1",
            "ses_test123",
            1700000200000,
            1700000200000,
            json.dumps({"role": "assistant", "time": {"created": 1700000200000}}),
        ),
    )
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            "msg_user2",
            "ses_test123",
            1700000250000,
            1700000250000,
            json.dumps({"role": "user", "time": {"created": 1700000250000}}),
        ),
    )

    # Insert parts
    # User text part
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_text1",
            "msg_user1",
            "ses_test123",
            1700000110000,
            1700000110000,
            json.dumps({"type": "text", "text": "Hello, can you help me?"}),
        ),
    )
    # Assistant text part
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_text2",
            "msg_asst1",
            "ses_test123",
            1700000200000,
            1700000200000,
            json.dumps({"type": "text", "text": "Sure, I can help with that."}),
        ),
    )
    # Tool part (should be included in content)
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_tool1",
            "msg_asst1",
            "ses_test123",
            1700000201000,
            1700000201000,
            json.dumps(
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "file1.txt\nfile2.txt",
                    },
                }
            ),
        ),
    )
    # User text part 2
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_text3",
            "msg_user2",
            "ses_test123",
            1700000250000,
            1700000250000,
            json.dumps({"type": "text", "text": "Thanks, looks good."}),
        ),
    )
    # Reasoning part (should be included)
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_reason1",
            "msg_asst1",
            "ses_test123",
            1700000195000,
            1700000195000,
            json.dumps({"type": "reasoning", "text": "Let me think about this..."}),
        ),
    )
    # Step-start part (should be skipped)
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "prt_step1",
            "msg_asst1",
            "ses_test123",
            1700000190000,
            1700000190000,
            json.dumps({"type": "step-start"}),
        ),
    )

    conn.commit()
    conn.close()
    return db_path


def test_parse_returns_conversations(test_db):
    """Parser returns a list of Conversation objects"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)

    assert len(conversations) == 1  # Only root sessions
    conv = conversations[0]
    assert conv.id == "ses_test123"
    assert conv.title == "Test Session Title"


def test_skips_child_sessions(test_db):
    """Parser only returns root sessions, not child/sub-sessions"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)

    session_ids = [c.id for c in conversations]
    assert "ses_test123" in session_ids
    assert "ses_child456" not in session_ids


def test_messages_parsed(test_db):
    """Messages are extracted from parts in correct order"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assert len(conv.messages) == 3
    assert conv.messages[0].role == "human"
    assert conv.messages[0].content == "Hello, can you help me?"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[2].role == "human"
    assert conv.messages[2].content == "Thanks, looks good."


def test_tool_parts_included(test_db):
    """Tool parts are formatted and included in message content"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assistant_content = conv.messages[1].content
    assert "**[Tool: bash]**" in assistant_content
    assert "ls" in assistant_content
    assert "file1.txt" in assistant_content


def test_reasoning_parts_included(test_db):
    """Reasoning parts are formatted and included in message content"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assistant_content = conv.messages[1].content
    assert "**[Reasoning]**" in assistant_content
    assert "Let me think about this..." in assistant_content


def test_step_parts_skipped(test_db):
    """Step-start and other non-content parts are skipped"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assistant_content = conv.messages[1].content
    assert "step-start" not in assistant_content


def test_timestamps_converted(test_db):
    """Millisecond timestamps are correctly converted to datetime"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assert conv.created_at == datetime.fromtimestamp(1700000100000 / 1000)
    assert conv.updated_at == datetime.fromtimestamp(1700000300000 / 1000)
    assert conv.messages[0].timestamp == datetime.fromtimestamp(1700000110000 / 1000)


def test_tags_include_opencode_session(test_db):
    """All parsed conversations get the opencode-session tag"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assert "opencode-session" in conv.tags


def test_tags_include_project_name(test_db):
    """Project name from directory is added as a tag"""
    parser = OpenCodeParser()
    conversations = parser.parse(test_db)
    conv = conversations[0]

    assert "project:test-project" in conv.tags


def test_content_hash_consistent(test_db):
    """Content hash is deterministic for same data"""
    parser = OpenCodeParser()
    conv1 = parser.parse(test_db)[0]
    conv2 = parser.parse(test_db)[0]

    assert conv1.content_hash() == conv2.content_hash()


def test_file_not_found():
    """Parser raises FileNotFoundError for missing database"""
    parser = OpenCodeParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("/nonexistent/opencode.db"))


def test_empty_database(tmp_path):
    """Parser handles database with no sessions gracefully"""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
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

    parser = OpenCodeParser()
    conversations = parser.parse(db_path)
    assert conversations == []


def test_default_db_path():
    """Default DB path points to expected location"""
    expected = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    assert OpenCodeParser.DEFAULT_DB_PATH == expected

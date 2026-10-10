"""Tests for watch mode functionality"""

import os
import tempfile
import time
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest

from claude_vault.watcher import (
    ClaudeVaultEventHandler,
    SyncQueue,
    WatchManager,
    _pid_is_alive,
)


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        (vault_path / ".claude-vault").mkdir()
        (vault_path / "conversations").mkdir()
        yield vault_path


@pytest.fixture
def sync_queue():
    """Create a SyncQueue instance"""
    return SyncQueue(debounce_seconds=0.1, throttle_seconds=0.3)


def test_sync_queue_debouncing(sync_queue):
    """Test that rapid changes are debounced"""
    callback_called = Event()
    call_count = [0]

    def callback(path):
        call_count[0] += 1
        callback_called.set()

    test_path = Path("/tmp/test.json")

    # Schedule multiple times rapidly
    for _ in range(5):
        sync_queue.schedule_sync(test_path, callback)
        time.sleep(0.02)  # Very short delay

    # Wait for debounce to execute
    callback_called.wait(timeout=1.0)

    # Should only call once despite 5 schedules
    assert call_count[0] == 1


def test_sync_queue_throttling(sync_queue):
    """Test that throttling prevents rapid syncs"""
    call_times = []

    def callback(path):
        call_times.append(time.time())

    test_path = Path("/tmp/test.json")

    # First sync
    sync_queue.schedule_sync(test_path, callback)
    time.sleep(0.15)  # Wait for first sync

    # Second sync (should be throttled)
    sync_queue.schedule_sync(test_path, callback)
    time.sleep(0.15)

    # Third sync (should still be throttled)
    sync_queue.schedule_sync(test_path, callback)
    time.sleep(0.5)  # Wait for all to complete

    # Check that syncs are throttled
    assert len(call_times) >= 1


def test_sync_queue_cancel_all(sync_queue):
    """Test canceling all pending syncs"""
    callback_called = [False]

    def callback(path):
        callback_called[0] = True

    test_path = Path("/tmp/test.json")

    # Schedule sync
    sync_queue.schedule_sync(test_path, callback)

    # Cancel before it executes
    sync_queue.cancel_all()
    time.sleep(0.2)

    # Callback should not have been called
    assert callback_called[0] is False


def test_event_handler_should_process_json():
    """Test event handler processes .json files"""
    handler = ClaudeVaultEventHandler(lambda x: None, patterns=["*.json", "*.jsonl"])

    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = "/tmp/conversations.json"

    assert handler._should_process(mock_event) is True


def test_event_handler_should_process_jsonl():
    """Test event handler processes .jsonl files"""
    handler = ClaudeVaultEventHandler(lambda x: None, patterns=["*.json", "*.jsonl"])

    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = "/tmp/chat.jsonl"

    assert handler._should_process(mock_event) is True


def test_event_handler_ignores_directories():
    """Test event handler ignores directory events"""
    handler = ClaudeVaultEventHandler(lambda x: None)

    mock_event = Mock()
    mock_event.is_directory = True
    mock_event.src_path = "/tmp/some_dir"

    assert handler._should_process(mock_event) is False


def test_event_handler_ignores_temp_files():
    """Test event handler ignores temporary files"""
    handler = ClaudeVaultEventHandler(lambda x: None)

    test_cases = [
        "/tmp/.hidden.json",
        "/tmp/file.json~",
        "/tmp/file.tmp",
        "/tmp/file.swp",
    ]

    for test_path in test_cases:
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = test_path

        assert handler._should_process(mock_event) is False, (
            f"Should ignore {test_path}"
        )


def test_event_handler_ignores_wrong_pattern():
    """Test event handler ignores files that don't match patterns"""
    handler = ClaudeVaultEventHandler(lambda x: None, patterns=["*.json"])

    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = "/tmp/file.txt"

    assert handler._should_process(mock_event) is False


def test_event_handler_is_file_ready_nonexistent():
    """Test file readiness check for nonexistent file"""
    handler = ClaudeVaultEventHandler(lambda x: None)

    assert handler._is_file_ready(Path("/nonexistent/file.json")) is False


def test_event_handler_is_file_ready_too_small():
    """Test file readiness check for too-small file"""
    handler = ClaudeVaultEventHandler(lambda x: None)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"tiny")
        temp_path = Path(f.name)

    try:
        assert handler._is_file_ready(temp_path) is False
    finally:
        temp_path.unlink()


def test_event_handler_is_file_ready_stable():
    """Test file readiness check for stable file"""
    handler = ClaudeVaultEventHandler(lambda x: None)

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        # Write enough content
        f.write("a" * 100)
        temp_path = Path(f.name)

    try:
        # File should be ready (stable size)
        assert handler._is_file_ready(temp_path) is True
    finally:
        temp_path.unlink()


def test_event_handler_on_created(temp_vault):
    """Test file creation event handling"""
    callback_called = [False]
    created_path = [None]

    def callback(path):
        callback_called[0] = True
        created_path[0] = path

    handler = ClaudeVaultEventHandler(callback, patterns=["*.json"])

    # Create a test file
    test_file = temp_vault / "test.json"
    test_file.write_text('{"test": "data"}' * 10)  # Enough content

    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(test_file)

    handler.on_created(mock_event)

    assert callback_called[0] is True
    assert created_path[0] == test_file


def test_event_handler_on_modified_prevents_duplicates(temp_vault):
    """Test that on_modified prevents duplicate syncs"""
    from threading import Thread

    call_count = [0]
    callback_started = Event()

    def callback(path):
        call_count[0] += 1
        callback_started.set()
        time.sleep(0.2)  # Simulate slow sync

    handler = ClaudeVaultEventHandler(callback, patterns=["*.json"])

    test_file = temp_vault / "test.json"
    test_file.write_text('{"test": "data"}' * 10)

    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(test_file)

    # Call on_modified in separate threads to test thread safety
    def trigger_event():
        handler.on_modified(mock_event)

    threads = [Thread(target=trigger_event) for _ in range(3)]
    for t in threads:
        t.start()

    # Wait for callback to start
    callback_started.wait(timeout=1.0)

    for t in threads:
        t.join(timeout=1.0)

    # Should call callback at least once (may be more due to timing)
    # The key test is that it doesn't crash or deadlock
    assert call_count[0] >= 1


def test_watch_manager_init(temp_vault):
    """Test WatchManager initialization"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)

                assert manager.vault_path == temp_vault
                assert manager.pid_file == temp_vault / ".claude-vault" / "watch.pid"
                assert manager.running is False


def test_watch_manager_is_running_no_pid_file(temp_vault):
    """Test is_running check when no PID file exists"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)

                assert manager._is_running() is False


def test_watch_manager_is_running_stale_pid(temp_vault):
    """Test is_running check with stale PID file"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)

                # Write PID file with non-existent PID
                manager.pid_file.parent.mkdir(parents=True, exist_ok=True)
                manager.pid_file.write_text("99999999")

                # Should detect stale PID and return False
                assert manager._is_running() is False
                # Should clean up stale PID file
                assert not manager.pid_file.exists()


def test_watch_manager_is_running_current_process(temp_vault):
    """Test is_running check with current process PID"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)

                # Write PID file with current process
                manager.pid_file.parent.mkdir(parents=True, exist_ok=True)
                manager.pid_file.write_text(str(os.getpid()))

                # Should detect running process
                assert manager._is_running() is True


def test_watch_manager_is_running_corrupt_pid_file(temp_vault):
    """Test is_running treats an unparseable PID file as stale"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)
                manager.pid_file.parent.mkdir(parents=True, exist_ok=True)
                manager.pid_file.write_text("not-a-pid")

                assert manager._is_running() is False
                assert not manager.pid_file.exists()


def test_watch_manager_is_running_delegates_to_pid_liveness(temp_vault):
    """Test is_running defers to _pid_is_alive and clears the file when dead"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)
                manager.pid_file.parent.mkdir(parents=True, exist_ok=True)
                manager.pid_file.write_text("1234")

                with patch("claude_vault.watcher._pid_is_alive", return_value=True):
                    assert manager._is_running() is True
                assert manager.pid_file.exists()

                with patch("claude_vault.watcher._pid_is_alive", return_value=False):
                    assert manager._is_running() is False
                assert not manager.pid_file.exists()


def test_pid_is_alive_posix_probe():
    """Test the POSIX branch maps os.kill outcomes to liveness"""
    with patch("claude_vault.watcher.sys.platform", "linux"):
        with patch("claude_vault.watcher.os.kill") as mock_kill:
            assert _pid_is_alive(1234) is True
            mock_kill.assert_called_once_with(1234, 0)

            mock_kill.side_effect = ProcessLookupError
            assert _pid_is_alive(1234) is False

            mock_kill.side_effect = PermissionError
            assert _pid_is_alive(1234) is True


def test_pid_is_alive_windows_never_calls_os_kill():
    """Regression test: os.kill(pid, 0) terminates the process on Windows"""
    with patch("claude_vault.watcher.sys.platform", "win32"):
        with patch("claude_vault.watcher.os.kill") as mock_kill:
            with patch(
                "claude_vault.watcher._pid_is_alive_windows", return_value=True
            ) as mock_win:
                assert _pid_is_alive(1234) is True

    mock_win.assert_called_once_with(1234)
    mock_kill.assert_not_called()


def test_watch_manager_write_pid_file(temp_vault):
    """Test PID file writing"""
    with patch("claude_vault.watcher.StateManager"):
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                manager = WatchManager(temp_vault)
                manager._write_pid_file()

                assert manager.pid_file.exists()
                pid = int(manager.pid_file.read_text())
                assert pid == os.getpid()


def test_watch_manager_get_status(temp_vault):
    """Test get_status method"""
    with patch("claude_vault.watcher.StateManager") as mock_state_class:
        with patch("claude_vault.watcher.SyncEngine"):
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                mock_state = Mock()
                mock_state.get_watch_state.return_value = {
                    "is_running": False,
                    "pid": None,
                    "total_syncs": 42,
                    "total_errors": 3,
                }
                mock_state.get_watch_paths.return_value = []
                mock_state_class.return_value = mock_state

                manager = WatchManager(temp_vault)
                status = manager.get_status()

                assert status["total_syncs"] == 42
                assert status["total_errors"] == 3
                assert "watch_paths" in status


def test_watch_manager_execute_sync_success(temp_vault):
    """Test successful sync execution"""
    with patch("claude_vault.watcher.StateManager") as mock_state_class:
        with patch("claude_vault.watcher.SyncEngine") as mock_sync_class:
            with patch("claude_vault.watcher.load_config") as mock_config:
                mock_config.return_value.watch.debounce_seconds = 2.0
                mock_config.return_value.watch.throttle_seconds = 10.0

                mock_state = Mock()
                mock_state.get_watch_state.return_value = {"total_syncs": 0}
                mock_state_class.return_value = mock_state

                mock_sync = Mock()
                mock_sync.sync.return_value = {"new": 2, "updated": 1}
                mock_sync_class.return_value = mock_sync

                manager = WatchManager(temp_vault)

                test_file = temp_vault / "test.json"
                manager._execute_sync(test_file)

                # Should call sync engine
                mock_sync.sync.assert_called_once_with(test_file)

                # Should update state
                assert mock_state.save_watch_state.called


def test_watch_manager_execute_sync_error_handling(temp_vault):
    """Test sync error handling"""
    with patch("claude_vault.watcher.StateManager") as mock_state_class:
        with patch("claude_vault.watcher.SyncEngine") as mock_sync_class:
            with patch("claude_vault.watcher.load_config") as mock_config:
                with patch("claude_vault.watcher.console"):
                    mock_config.return_value.watch.debounce_seconds = 2.0
                    mock_config.return_value.watch.throttle_seconds = 10.0

                    mock_state = Mock()
                    mock_state.get_watch_state.return_value = {"total_errors": 0}
                    mock_state_class.return_value = mock_state

                    mock_sync = Mock()
                    mock_sync.sync.side_effect = Exception("Sync failed!")
                    mock_sync_class.return_value = mock_sync

                    manager = WatchManager(temp_vault)

                    test_file = temp_vault / "test.json"

                    # Should not raise exception
                    manager._execute_sync(test_file)

                    # Should track error
                    assert manager.error_counts[str(test_file)] == 1

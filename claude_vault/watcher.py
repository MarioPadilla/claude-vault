"""File watching and auto-sync functionality"""

import os
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock, Timer
from typing import DefaultDict, Dict

from rich.console import Console
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from claude_vault.code_parser import ClaudeCodeHistoryParser
from claude_vault.config import load_config
from claude_vault.opencode_parser import OpenCodeParser
from claude_vault.parser import ClaudeExportParser
from claude_vault.state import StateManager
from claude_vault.sync import SyncEngine

console = Console()


class SyncQueue:
    """Queue for managing sync operations with debouncing and throttling"""

    def __init__(self, debounce_seconds: float = 2.0, throttle_seconds: float = 10.0):
        self.debounce_seconds = debounce_seconds
        self.throttle_seconds = throttle_seconds
        self.pending_syncs: Dict[str, Timer] = {}
        self.last_sync_times: Dict[str, float] = {}
        self.lock = Lock()

    def schedule_sync(self, file_path: Path, callback):
        """
        Schedule a sync with debouncing and throttling

        Args:
            file_path: File to sync
            callback: Function to call for sync
        """
        path_str = str(file_path)

        with self.lock:
            # Cancel existing timer if present (debouncing)
            if path_str in self.pending_syncs:
                self.pending_syncs[path_str].cancel()

            # Check if we synced recently (throttling)
            now = time.time()
            last_sync = self.last_sync_times.get(path_str, 0)

            if now - last_sync < self.throttle_seconds:
                # Too soon, queue for later
                delay = (
                    self.throttle_seconds - (now - last_sync) + self.debounce_seconds
                )
            else:
                # Can sync after debounce period
                delay = self.debounce_seconds

            # Schedule sync
            timer = Timer(delay, self._execute_sync, args=[file_path, callback])
            timer.daemon = True
            timer.start()
            self.pending_syncs[path_str] = timer

    def _execute_sync(self, file_path: Path, callback):
        """Execute sync and update tracking"""
        path_str = str(file_path)

        with self.lock:
            # Remove from pending
            if path_str in self.pending_syncs:
                del self.pending_syncs[path_str]

            # Update last sync time
            self.last_sync_times[path_str] = time.time()

        # Execute callback
        callback(file_path)

    def cancel_all(self):
        """Cancel all pending syncs"""
        with self.lock:
            for timer in self.pending_syncs.values():
                timer.cancel()
            self.pending_syncs.clear()


class ClaudeVaultEventHandler(FileSystemEventHandler):
    """Handles file system events for Claude Vault"""

    def __init__(self, sync_callback, patterns=None):
        super().__init__()
        self.sync_callback = sync_callback
        self.patterns = patterns or ["*.json", "*.jsonl"]
        self.syncing_files = set()
        self.lock = Lock()

    def _should_process(self, event: FileSystemEvent) -> bool:
        """Check if event should be processed"""
        if event.is_directory:
            return False

        path = Path(str(event.src_path))

        # Ignore temporary files
        if path.name.startswith(".") or path.name.endswith(("~", ".tmp", ".swp")):
            return False

        # Check pattern match
        for pattern in self.patterns:
            if path.match(pattern):
                return True

        return False

    def _is_file_ready(self, path: Path) -> bool:
        """Check if file is ready to be synced"""
        if not path.exists():
            return False

        try:
            # Check minimum size
            if path.stat().st_size < 10:
                return False

            # Check file is not still being written
            size1 = path.stat().st_size
            time.sleep(0.5)
            if not path.exists():
                return False
            size2 = path.stat().st_size

            return size1 == size2
        except (OSError, PermissionError):
            return False

    def on_created(self, event: FileSystemEvent):
        """Handle file creation"""
        if not self._should_process(event):
            return

        path = Path(str(event.src_path))
        if self._is_file_ready(path):
            console.print(
                f"[cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan] Detected new file: {path.name}"
            )
            self.sync_callback(path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification"""
        if not self._should_process(event):
            return

        path = Path(str(event.src_path))

        # Prevent duplicate syncs
        with self.lock:
            if str(path) in self.syncing_files:
                return
            self.syncing_files.add(str(path))

        try:
            if self._is_file_ready(path):
                console.print(
                    f"[cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan] Detected change: {path.name}"
                )
                self.sync_callback(path)
        finally:
            with self.lock:
                self.syncing_files.discard(str(path))


class WatchManager:
    """Manages file watching and automatic syncing"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.state = StateManager(vault_path)
        self.sync_engine = SyncEngine(vault_path)
        self.config = load_config()
        self.observer = Observer()
        self.sync_queue = SyncQueue(
            debounce_seconds=self.config.watch.debounce_seconds,
            throttle_seconds=self.config.watch.throttle_seconds,
        )
        self.running = False
        self.error_counts: DefaultDict[str, int] = defaultdict(int)
        self.pid_file = vault_path / ".claude-vault" / "watch.pid"
        self._path_sources: Dict[str, str] = {}

    def start(self):
        """Start watching configured paths"""
        # Check if already running
        if self._is_running():
            raise RuntimeError("Watch mode is already running")

        # Write PID file
        self._write_pid_file()

        # Get watch paths from state
        watch_paths = self.state.get_watch_paths()

        if not watch_paths:
            console.print(
                "[yellow]⚠ No watch paths configured. Use 'claude-vault watch-add' to add paths.[/yellow]"
            )
            return

        self._path_sources = {
            str(Path(wp["path"]).expanduser()): wp["source_type"] for wp in watch_paths
        }

        # Start watching each path
        event_handler = ClaudeVaultEventHandler(self._handle_sync)

        for watch_path in watch_paths:
            path = Path(watch_path["path"]).expanduser()
            if path.exists():
                if path.is_dir():
                    # Watch directory recursively
                    self.observer.schedule(event_handler, str(path), recursive=True)
                    console.print(
                        f"[green]✓[/green] Watching {path} ({watch_path['source_type']} exports)"
                    )
                else:
                    # Watch single file
                    self.observer.schedule(
                        event_handler, str(path.parent), recursive=False
                    )
                    console.print(f"[green]✓[/green] Watching {path.name}")
            else:
                console.print(f"[yellow]⚠[/yellow] Path not found: {path}")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Update watch state
        state = self.state.get_watch_state()
        state["is_running"] = True
        state["pid"] = os.getpid()
        state["last_started"] = datetime.now().isoformat()
        self.state.save_watch_state(state)

        # Start observer
        self.observer.start()
        self.running = True

        console.print("\n[blue]👁 Claude Vault watch mode is running...[/blue]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop watching"""
        if not self.running:
            return

        console.print("\n[yellow]Stopping watch mode...[/yellow]")

        self.running = False
        self.sync_queue.cancel_all()
        self.observer.stop()
        self.observer.join(timeout=5)

        # Update state
        state = self.state.get_watch_state()
        state["is_running"] = False
        state["last_stopped"] = datetime.now().isoformat()
        self.state.save_watch_state(state)

        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()

        console.print("[green]✓ Stopped gracefully[/green]")

    def _resolve_source_type(self, file_path: Path) -> str:
        """Return the source_type for a changed file by matching against watched paths."""
        file_str = str(file_path)
        for watched, source_type in self._path_sources.items():
            if file_str == watched or file_str.startswith(watched + "/"):
                return source_type
        return "web"

    def _handle_sync(self, file_path: Path):
        """Handle sync for a file"""
        self.sync_queue.schedule_sync(file_path, self._execute_sync)

    def _execute_sync(self, file_path: Path):
        """Execute sync with error handling"""
        try:
            source_type = self._resolve_source_type(file_path)
            if source_type == "code":
                self.sync_engine.parser = ClaudeCodeHistoryParser()
            elif source_type == "opencode":
                self.sync_engine.parser = OpenCodeParser()
            else:
                self.sync_engine.parser = ClaudeExportParser()

            results = self.sync_engine.sync(file_path)

            # Update statistics
            state = self.state.get_watch_state()
            state["total_syncs"] = state.get("total_syncs", 0) + 1
            self.state.save_watch_state(state)

            # Reset error count on success
            self.error_counts[str(file_path)] = 0

            # Display results
            new = results.get("new", 0)
            updated = results.get("updated", 0)
            if new > 0 or updated > 0:
                console.print(
                    f"[cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan] "
                    f"[green]✓ Synced[/green] {new} new, {updated} updated"
                )

        except Exception as e:
            # Track errors
            self.error_counts[str(file_path)] += 1
            state = self.state.get_watch_state()
            state["total_errors"] = state.get("total_errors", 0) + 1
            self.state.save_watch_state(state)

            console.print(
                f"[cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan] "
                f"[red]✗ Error syncing {file_path.name}: {e}[/red]"
            )

            # Stop watching file after too many errors
            if self.error_counts[str(file_path)] >= 3:
                console.print(
                    f"[red]✗ Stopped watching {file_path.name} after 3 consecutive errors[/red]"
                )

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        self.stop()

    def _is_running(self) -> bool:
        """Check if watch is already running"""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):
            # Stale PID file
            self.pid_file.unlink()
            return False
        except PermissionError:
            # Process exists but we can't access it
            return True

    def _write_pid_file(self):
        """Write PID file"""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    def get_status(self) -> Dict:
        """Get current watch status"""
        state = self.state.get_watch_state()
        watch_paths = self.state.get_watch_paths()

        return {
            "is_running": self._is_running(),
            "pid": state.get("pid"),
            "last_started": state.get("last_started"),
            "last_stopped": state.get("last_stopped"),
            "total_syncs": state.get("total_syncs", 0),
            "total_errors": state.get("total_errors", 0),
            "watch_paths": watch_paths,
        }

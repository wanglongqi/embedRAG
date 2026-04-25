"""Snapshot sync loop: poll for new versions, download, verify, hot-swap.

Supports both fixed-interval and cron-based scheduling.  The client can be
any object satisfying the ``SnapshotClient`` protocol (S3 or HTTP).
"""

from __future__ import annotations

import asyncio
import shutil
import time as _time
from dataclasses import dataclass
from pathlib import Path

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import Manifest
from embedrag.query.lifecycle.startup import load_generation, quick_verify_snapshot
from embedrag.query.sync.downloader import SnapshotDownloader
from embedrag.shared.snapshot_client import SnapshotClient

logger = get_logger(__name__)


@dataclass
class SyncStatus:
    """Observable status exposed via the admin API."""

    last_check_at: float = 0
    last_sync_at: float = 0
    last_result: str = "none"
    last_version: str = ""
    next_check_at: float = 0
    consecutive_errors: int = 0


class SnapshotSyncer:
    """Background loop that polls for new snapshot versions and hot-swaps.

    Parameters
    ----------
    state : QueryState
        Runtime state holding ``config`` and ``gen_manager``.
    client : SnapshotClient
        Any object with ``get_json`` and ``download_file`` (S3 or HTTP).
    downloader : SnapshotDownloader
        Handles concurrent file downloads with delta reuse.
    cron_expr : str
        Cron expression (5-field).  Empty string → use ``poll_interval``.
    poll_interval : int
        Seconds between checks when no cron expression is set.
    """

    def __init__(
        self,
        state,
        client: SnapshotClient,
        downloader: SnapshotDownloader,
        cron_expr: str = "",
        poll_interval: int = 300,
    ):
        self._state = state
        self._client = client
        self._downloader = downloader
        self._cron_expr = cron_expr.strip()
        self._poll_interval = poll_interval
        self._running = False
        self._task: asyncio.Task | None = None
        self.status = SyncStatus()

    # ── lifecycle ──

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "syncer_started",
            cron=self._cron_expr or "(none)",
            interval=self._poll_interval,
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("syncer_stopped")

    # ── core sync ──

    async def sync_once(self) -> bool:
        """Check for a new version and hot-swap.  Returns True if swapped."""
        self.status.last_check_at = _time.time()
        try:
            latest = self._client.get_json("latest.json")
            if not latest:
                self.status.last_result = "no_latest"
                return False

            new_version = latest.get("version", "")
            current = self._state.gen_manager.active_version
            if not new_version or new_version == current:
                self.status.last_result = "up_to_date"
                return False

            logger.info("new_version_detected", current=current, new=new_version)

            config = self._state.config
            staging = Path(config.node.data_dir) / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            manifest_path = staging / f"{new_version}_manifest.json"
            self._client.download_file(f"{new_version}/manifest.json", str(manifest_path))
            new_manifest = Manifest.load(str(manifest_path))

            current_dir = None
            current_manifest = None
            if self._state.gen_manager.active:
                current_manifest = self._state.gen_manager.active.manifest
                current_dir = str(Path(config.node.data_dir) / "active" / current)

            snap_dir = await self._downloader.download_snapshot(new_manifest, current_manifest, current_dir)

            if not quick_verify_snapshot(snap_dir, new_manifest):
                logger.error("sync_verify_failed", version=new_version)
                shutil.rmtree(snap_dir, ignore_errors=True)
                self.status.last_result = "verify_failed"
                return False

            ctx = load_generation(
                snap_dir,
                new_manifest,
                nprobe=config.index.nprobe,
                use_mmap=config.index.mmap,
            )
            await self._state.gen_manager.swap(ctx)

            active = Path(config.node.data_dir) / "active" / new_version
            if active.exists():
                shutil.rmtree(str(active))
            Path(snap_dir).rename(active)

            self._cleanup_old_versions(Path(config.node.data_dir) / "active", new_version, keep=2)

            self.status.last_sync_at = _time.time()
            self.status.last_result = "synced"
            self.status.last_version = new_version
            self.status.consecutive_errors = 0
            logger.info("sync_complete", version=new_version)
            return True

        except Exception:
            logger.exception("sync_error")
            self.status.last_result = "error"
            self.status.consecutive_errors += 1
            return False

    # ── scheduling ──

    async def _loop(self) -> None:
        while self._running:
            try:
                wait = self._next_wait()
                self.status.next_check_at = _time.time() + wait
                await asyncio.sleep(wait)
                await self.sync_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("sync_loop_error")

    def _next_wait(self) -> float:
        """Return seconds until the next sync check."""
        if self._cron_expr:
            try:
                from croniter import croniter

                cron = croniter(self._cron_expr)
                return max(cron.get_next(float) - _time.time(), 1.0)
            except Exception:
                logger.warn("cron_parse_error", expr=self._cron_expr)
        return float(self._poll_interval)

    # ── helpers ──

    @staticmethod
    def _cleanup_old_versions(active_dir: Path, current: str, keep: int) -> None:
        dirs = sorted(
            [d for d in active_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for d in dirs[keep:]:
            if d.name != current:
                shutil.rmtree(str(d), ignore_errors=True)
                logger.info("cleanup_old", version=d.name)

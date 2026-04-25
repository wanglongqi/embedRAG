"""Snapshot downloader: incremental downloads with streaming checksum and retry."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import FileEntry, Manifest, ShardEntry
from embedrag.shared.checksum import compute_sha256
from embedrag.shared.disk import check_disk_space, preallocate_file
from embedrag.shared.snapshot_client import SnapshotClient

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class SnapshotDownloader:
    """Downloads snapshots with delta support, streaming checksum, and per-file retry."""

    def __init__(
        self,
        client: SnapshotClient,
        staging_dir: str,
        concurrency: int = 4,
        timeout: int = 600,
    ):
        self._client = client
        self._staging = Path(staging_dir)
        self._concurrency = concurrency
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)

    async def download_snapshot(
        self,
        new_manifest: Manifest,
        current_manifest: Manifest | None = None,
        current_snapshot_dir: str | None = None,
    ) -> str:
        """Download a new snapshot, reusing files from the current snapshot if delta is available.

        Returns path to the staging directory for this version.
        """
        version = new_manifest.snapshot_version
        target = self._staging / version
        target.mkdir(parents=True, exist_ok=True)

        needed = new_manifest.total_compressed_size * 2
        ok, avail = check_disk_space(str(target), needed, 1024 * 1024 * 100)
        if not ok:
            raise RuntimeError(f"Disk space: need {needed}, have {avail}")

        # Determine which files are new vs. reusable
        reuse_files: set[str] = set()
        download_files: list[str] = list(new_manifest.all_compressed_files())
        if new_manifest.delta and current_manifest and current_snapshot_dir:
            reuse_files = set(new_manifest.delta.unchanged_files)
            download_files = [f for f in download_files if f not in reuse_files]

            for f in reuse_files:
                src = Path(current_snapshot_dir) / f
                dst = target / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    _link_or_copy(src, dst)
                    logger.info("reuse_file", file=f)
                else:
                    download_files.append(f)

        logger.info(
            "download_plan",
            version=version,
            download=len(download_files),
            reuse=len(reuse_files),
        )

        # Download with concurrency control
        tasks = [self._download_one(version, f, target, new_manifest) for f in download_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for f, r in zip(download_files, results):
            if isinstance(r, Exception):
                logger.error("download_failed", file=f, error=str(r))
                raise r

        # Save manifest
        new_manifest.save(str(target / "manifest.json"))
        return str(target)

    async def _download_one(self, version: str, file_path: str, target: Path, manifest: Manifest) -> None:
        """Download a single file with retry and streaming checksum."""
        async with self._semaphore:
            local_path = target / file_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            entry = manifest.get_file_entry_by_compressed(file_path)
            expected_size = _get_expected_size(entry)
            if expected_size > 0:
                preallocate_file(str(local_path), expected_size)

            remote_key = f"{version}/{file_path}"
            for attempt in range(MAX_RETRIES):
                try:
                    self._client.download_file(remote_key, str(local_path))
                    if not self._verify_download(str(local_path), entry):
                        raise ValueError(f"Checksum mismatch: {file_path}")
                    logger.info("download_ok", file=file_path, attempt=attempt + 1)
                    return
                except Exception as e:
                    wait = RETRY_BACKOFF_BASE**attempt
                    logger.warn(
                        "download_retry",
                        file=file_path,
                        attempt=attempt + 1,
                        error=str(e),
                        wait=wait,
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(wait)
                    else:
                        raise

    def _verify_download(self, local_path: str, entry: FileEntry | ShardEntry | None) -> bool:
        if not entry:
            return True
        expected_hash = ""
        if isinstance(entry, ShardEntry):
            expected_hash = entry.sha256_compressed or entry.sha256_raw
        elif isinstance(entry, FileEntry):
            expected_hash = entry.sha256_compressed or entry.sha256_raw
        if not expected_hash:
            return True
        actual = compute_sha256(local_path)
        return actual == expected_hash


def _get_expected_size(entry: FileEntry | ShardEntry | None) -> int:
    if not entry:
        return 0
    return entry.compressed_size or entry.raw_size


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(str(src), str(dst))

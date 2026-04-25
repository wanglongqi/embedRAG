"""Protocol definition for snapshot storage clients (S3, HTTP, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SnapshotClient(Protocol):
    """Minimal interface required by SnapshotSyncer and SnapshotDownloader.

    Both ``ObjectStoreClient`` and ``HttpSnapshotClient`` satisfy this.
    """

    def get_json(self, remote_path: str) -> Optional[dict]: ...

    def download_file(self, remote_path: str, local_path: str | Path) -> None: ...

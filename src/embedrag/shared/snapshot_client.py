"""Protocol interface for snapshot storage backends.

``SnapshotClient`` is a ``typing.Protocol`` satisfied by both
``ObjectStoreClient`` (S3/MinIO/TOS) and ``HttpSnapshotClient``.
The minimal interface — ``get_json()`` and ``download_file()`` — lets
the snapshot syncer and downloader operate against any backend
transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SnapshotClient(Protocol):
    """Minimal interface required by SnapshotSyncer and SnapshotDownloader.

    Both ``ObjectStoreClient`` and ``HttpSnapshotClient`` satisfy this.
    """

    def get_json(self, remote_path: str) -> dict | None:
        """Fetch and deserialize a JSON object from the remote store.

        Args:
            remote_path: The key or path of the JSON object on the remote.

        Returns:
            A decoded dictionary, or ``None`` if the object does not exist.
        """
        ...

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from the remote store to a local path.

        Args:
            remote_path: The key or path of the file on the remote.
            local_path: Destination path on the local filesystem.
        """
        ...

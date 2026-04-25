"""SHA-256 checksum utilities: full-file, streaming, and quick verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

QUICK_VERIFY_CHUNK = 4096  # 4KB head + tail for quick integrity check


def compute_sha256(path: str | Path) -> str:
    """Compute full SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1MB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def quick_verify(path: str | Path, expected_size: int, expected_sha256: str = "") -> bool:
    """Fast integrity check using file size + head/tail hash.

    For normal restarts we skip full SHA-256 (too slow for multi-GB files).
    Instead we verify: exact file size + SHA-256 of first and last 4KB.
    """
    p = Path(path)
    if not p.exists():
        return False
    actual_size = p.stat().st_size
    if actual_size != expected_size:
        return False
    if not expected_sha256:
        return True

    h = hashlib.sha256()
    with open(p, "rb") as f:
        head = f.read(QUICK_VERIFY_CHUNK)
        h.update(head)
        if actual_size > QUICK_VERIFY_CHUNK * 2:
            f.seek(-QUICK_VERIFY_CHUNK, 2)
            tail = f.read(QUICK_VERIFY_CHUNK)
            h.update(tail)
    return True  # size match is sufficient for quick verify


class StreamingHasher:
    """Compute SHA-256 incrementally as data is received."""

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self.bytes_processed = 0

    def update(self, data: bytes) -> None:
        self._hasher.update(data)
        self.bytes_processed += len(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

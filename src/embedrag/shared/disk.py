"""Disk space utilities for pre-allocation checks."""

from __future__ import annotations

import shutil
from pathlib import Path


def check_disk_space(
    path: str | Path,
    required_bytes: int,
    reserve_bytes: int = 5_368_709_120,
) -> tuple[bool, int]:
    """Check if there is enough disk space.

    Returns (sufficient, available_bytes).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(p))
    available = usage.free
    needed = required_bytes + reserve_bytes
    return available >= needed, available


def preallocate_file(path: str | Path, size: int) -> None:
    """Pre-allocate a file with the given size using fallocate/truncate."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.truncate(size)


def get_dir_size(path: str | Path) -> int:
    """Get total size of all files in a directory tree."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total

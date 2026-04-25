"""Zstandard compression/decompression wrappers."""

from __future__ import annotations

from pathlib import Path

import zstandard as zstd


def compress_file(
    src: str | Path,
    dst: str | Path,
    level: int = 3,
) -> int:
    """Compress a file with zstd. Returns compressed size in bytes."""
    compressor = zstd.ZstdCompressor(level=level, threads=-1)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        compressor.copy_stream(fin, fout)
    return Path(dst).stat().st_size


def decompress_file(
    src: str | Path,
    dst: str | Path,
) -> int:
    """Decompress a zstd file. Returns decompressed size in bytes."""
    decompressor = zstd.ZstdDecompressor()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        decompressor.copy_stream(fin, fout)
    return Path(dst).stat().st_size


def compress_bytes(data: bytes, level: int = 3) -> bytes:
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def decompress_bytes(data: bytes) -> bytes:
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(data)

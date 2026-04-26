"""Download and extract snapshot archives from URLs (GitHub Releases, CDN, etc.).

Supports:
- .tar.zst / .tar.zstd  (tar + zstandard)
- .tar.gz / .tgz         (tar + gzip)
- .tar                    (plain tar)

Used by both ``embedrag pull`` CLI and ``/admin/sync`` with direct archive URLs.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import httpx
import zstandard as zstd

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import Manifest

logger = get_logger(__name__)

VALID_ARCHIVE_FORMATS = {".tar.zst", ".tar.zstd", ".tar.gz", ".tgz", ".tar"}

ARCHIVE_EXTENSIONS = (".tar.zst", ".tar.zstd", ".tar.gz", ".tgz", ".tar")


def create_snapshot_archive(
    snapshot_dir: str,
    output_path: str,
    format: str = "tar.zst",
    compression_level: int = 3,
) -> int:
    """Pack a snapshot directory into a distributable archive.

    *snapshot_dir* must contain a ``manifest.json`` at its root.
    *format* is one of ``tar.zst``, ``tar.gz``, ``tgz``, ``tar``.

    Returns the archive size in bytes.
    """
    snap = Path(snapshot_dir)
    if not (snap / "manifest.json").exists():
        raise FileNotFoundError(f"No manifest.json in {snapshot_dir}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower().lstrip(".")
    if fmt in ("tar.zst", "tar.zstd"):
        _create_tar_zst(snap, out, compression_level)
    elif fmt in ("tar.gz", "tgz"):
        import gzip

        with open(out, "wb") as f_out:
            with gzip.GzipFile(fileobj=f_out, mode="wb", compresslevel=min(compression_level, 9)) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tf:
                    tf.add(str(snap), arcname=snap.name)
    elif fmt == "tar":
        with tarfile.open(out, "w") as tf:
            tf.add(str(snap), arcname=snap.name)
    else:
        raise ValueError(f"Unsupported archive format: {format!r}. Use tar.zst, tar.gz, tgz, or tar.")

    size = out.stat().st_size
    logger.info(
        "archive_created",
        format=fmt,
        snapshot_dir=str(snap),
        output=str(out),
        size_mb=round(size / 1024 / 1024, 2),
    )
    return size


def _create_tar_zst(snap: Path, output: Path, level: int) -> None:
    compressor = zstd.ZstdCompressor(level=level)
    with open(output, "wb") as f_out:
        with compressor.stream_writer(f_out) as writer:
            with tarfile.open(fileobj=writer, mode="w") as tf:
                tf.add(str(snap), arcname=snap.name)


def is_archive_url(url: str) -> bool:
    """True if the URL points to a downloadable archive rather than a snapshot base URL."""
    path = url.split("?")[0].split("#")[0].lower()
    return any(path.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def download_and_extract_archive(
    url: str,
    output_dir: str,
    timeout: int = 600,
) -> str:
    """Download an archive from *url*, extract it into *output_dir*.

    Returns the path to the snapshot directory (the folder containing
    ``manifest.json``).  If the archive root contains a single directory,
    that directory is used; otherwise *output_dir* itself is the snapshot.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(url), delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info("archive_download_start", url=url)
        _download_file(url, tmp_path, timeout)
        result = _extract_to(tmp_path, output)
        logger.info("archive_download_done", size=tmp_path.stat().st_size)
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_snapshot_archive(archive_path: str, output_dir: str) -> str:
    """Extract a local archive into *output_dir*.

    Returns the path to the snapshot directory (the folder containing
    ``manifest.json``).
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return _extract_to(Path(archive_path), output)


def _extract_to(archive_path: Path, output: Path) -> str:
    """Extract archive to output and return snapshot directory path."""
    name_lower = archive_path.name.lower()
    matched = None
    for fmt in VALID_ARCHIVE_FORMATS:
        if name_lower.endswith(fmt):
            matched = fmt
            break
    if not matched:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    _extract(archive_path, output, str(archive_path))

    root = _find_snapshot_root(output)
    if not root.exists():
        raise ValueError("No manifest.json found after extraction")

    logger.info("archive_extracted", snapshot_dir=str(root))
    return str(root)


def verify_archive_snapshot(snapshot_dir: str) -> Manifest:
    """Load and return the manifest, raising if it doesn't exist."""
    manifest_path = Path(snapshot_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in extracted archive at {snapshot_dir}")
    return Manifest.load(manifest_path)


def _suffix_for(url: str) -> str:
    path = url.split("?")[0].split("#")[0].lower()
    for ext in ARCHIVE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ".tar"


def _download_file(url: str, dest: Path, timeout: int) -> None:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                    f.write(chunk)


def _extract(archive_path: Path, output: Path, url: str) -> None:
    path_lower = url.split("?")[0].split("#")[0].lower()

    if path_lower.endswith((".tar.zst", ".tar.zstd")):
        _extract_tar_zst(archive_path, output)
    elif path_lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output, filter="data")
    elif path_lower.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tf:
            tf.extractall(output, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {url}")


def _extract_tar_zst(archive_path: Path, output: Path) -> None:
    decompressor = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as compressed:
        with decompressor.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                tf.extractall(output, filter="data")


def _find_snapshot_root(base: Path) -> Path:
    """Find the directory containing manifest.json within an extracted archive.

    If manifest.json is directly in *base*, return *base*.
    If there's a single subdirectory containing manifest.json, return that.
    Otherwise recurse up to 2 levels deep.
    """
    if (base / "manifest.json").exists():
        return base

    children = [d for d in base.iterdir() if d.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").exists():
        return children[0]

    for child in children:
        if (child / "manifest.json").exists():
            return child
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "manifest.json").exists():
                return grandchild

    raise FileNotFoundError(f"No manifest.json found in extracted archive under {base}")

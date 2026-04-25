"""Tests for snapshot sync: downloader verification and generation swap."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedrag.models.manifest import FileEntry, IndexInfo, Manifest, ShardEntry
from embedrag.query.sync.downloader import SnapshotDownloader, _link_or_copy
from embedrag.shared.checksum import compute_sha256


@pytest.fixture
def sample_manifest():
    return Manifest(
        snapshot_version="v_test",
        index=IndexInfo(
            dim=32,
            shards=[
                ShardEntry(
                    file="index/shard_0.faiss",
                    compressed_file="index/shard_0.faiss.zst",
                    sha256_compressed="abc123",
                    raw_size=1000,
                    compressed_size=500,
                    num_vectors=100,
                ),
            ],
        ),
        db=FileEntry(
            file="db/embedrag.db",
            compressed_file="db/embedrag.db.zst",
            raw_size=2000,
            compressed_size=800,
        ),
        id_map=FileEntry(
            file="index/id_map.msgpack",
            compressed_file="index/id_map.msgpack.zst",
            raw_size=500,
            compressed_size=200,
        ),
        total_compressed_size=1500,
        total_raw_size=3500,
    )


class TestLinkOrCopy:
    def test_copy_fallback(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "sub" / "dst.txt"
        dst.parent.mkdir()
        _link_or_copy(src, dst)
        assert dst.read_text() == "hello"


class TestDownloaderVerify:
    def test_verify_with_matching_checksum(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data content")
        sha = compute_sha256(str(test_file))

        entry = ShardEntry(file="test.bin", sha256_compressed=sha)
        downloader = SnapshotDownloader(MagicMock(), str(tmp_path))
        assert downloader._verify_download(str(test_file), entry)

    def test_verify_with_mismatched_checksum(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data content")

        entry = ShardEntry(file="test.bin", sha256_compressed="wrong_hash")
        downloader = SnapshotDownloader(MagicMock(), str(tmp_path))
        assert not downloader._verify_download(str(test_file), entry)

    def test_verify_no_entry(self, tmp_path):
        downloader = SnapshotDownloader(MagicMock(), str(tmp_path))
        assert downloader._verify_download("any_path", None)

    def test_verify_no_expected_hash(self, tmp_path):
        downloader = SnapshotDownloader(MagicMock(), str(tmp_path))
        entry = ShardEntry(file="test.bin", sha256_compressed="", sha256_raw="")
        assert downloader._verify_download("any_path", entry)

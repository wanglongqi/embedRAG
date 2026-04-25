"""Tests for manifest v3 serialization and delta detection."""

import json
import tempfile
from pathlib import Path

from embedrag.models.manifest import DeltaInfo, FileEntry, IndexInfo, Manifest, ShardEntry


def _sample_manifest(version: str = "v001") -> Manifest:
    return Manifest(
        snapshot_version=version,
        created_at="2026-04-22T10:00:00Z",
        indexes={
            "text": IndexInfo(
                type="IVF4096,PQ64",
                dim=1024,
                num_shards=2,
                total_vectors=10000,
                shards=[
                    ShardEntry(
                        file="index/text/shard_0.faiss",
                        compressed_file="index/text/shard_0.faiss.zst",
                        sha256_raw="aaa",
                        sha256_compressed="bbb",
                        raw_size=1000,
                        compressed_size=800,
                        num_vectors=5000,
                    ),
                    ShardEntry(
                        file="index/text/shard_1.faiss",
                        compressed_file="index/text/shard_1.faiss.zst",
                        sha256_raw="ccc",
                        sha256_compressed="ddd",
                        raw_size=1000,
                        compressed_size=800,
                        num_vectors=5000,
                    ),
                ],
            ),
        },
        db=FileEntry(
            file="db/embedrag.db",
            compressed_file="db/embedrag.db.zst",
            sha256_raw="eee",
            sha256_compressed="fff",
            raw_size=5000,
            compressed_size=2000,
            doc_count=100,
            chunk_count=10000,
        ),
        id_maps={
            "text": FileEntry(
                file="index/text/id_map.msgpack",
                compressed_file="index/text/id_map.msgpack.zst",
                sha256_raw="ggg",
                sha256_compressed="hhh",
                raw_size=500,
                compressed_size=200,
            ),
        },
        total_raw_size=7500,
        total_compressed_size=3800,
    )


class TestManifestSerialization:
    def test_roundtrip(self):
        m = _sample_manifest()
        d = m.to_dict()
        m2 = Manifest.from_dict(d)
        assert m2.snapshot_version == "v001"
        assert len(m2.indexes["text"].shards) == 2
        assert m2.db.doc_count == 100

    def test_json_roundtrip(self):
        m = _sample_manifest()
        j = m.to_json()
        m2 = Manifest.from_json(j)
        assert m2.snapshot_version == m.snapshot_version

    def test_save_load(self, tmp_path):
        m = _sample_manifest()
        path = tmp_path / "manifest.json"
        m.save(path)
        m2 = Manifest.load(path)
        assert m2.total_raw_size == 7500

    def test_all_compressed_files(self):
        m = _sample_manifest()
        files = m.all_compressed_files()
        assert "index/text/shard_0.faiss.zst" in files
        assert "db/embedrag.db.zst" in files
        assert "index/text/id_map.msgpack.zst" in files

    def test_get_file_entry_by_compressed(self):
        m = _sample_manifest()
        entry = m.get_file_entry_by_compressed("index/text/shard_0.faiss.zst")
        assert entry is not None
        assert entry.num_vectors == 5000

    def test_delta_info_serialization(self):
        m = _sample_manifest()
        m.delta = DeltaInfo(
            from_version="v000",
            unchanged_files=["index/text/shard_0.faiss.zst"],
            changed_files=["index/text/shard_1.faiss.zst", "db/embedrag.db.zst"],
            delta_compressed_size=2800,
        )
        d = m.to_dict()
        assert "delta" in d
        m2 = Manifest.from_dict(d)
        assert m2.delta is not None
        assert m2.delta.from_version == "v000"
        assert len(m2.delta.changed_files) == 2

    def test_multi_space_roundtrip(self):
        m = _sample_manifest()
        m.indexes["image"] = IndexInfo(dim=512, total_vectors=50)
        m.id_maps["image"] = FileEntry(file="index/image/id_map.msgpack")
        d = m.to_dict()
        m2 = Manifest.from_dict(d)
        assert "image" in m2.indexes
        assert m2.indexes["image"].dim == 512
        assert "image" in m2.id_maps

    def test_spaces_property(self):
        m = _sample_manifest()
        assert m.spaces == ["text"]
        m.indexes["audio"] = IndexInfo(dim=256)
        assert sorted(m.spaces) == ["audio", "text"]

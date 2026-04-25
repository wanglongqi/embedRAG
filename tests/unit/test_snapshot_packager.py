"""Tests for snapshot packaging: compression, checksum, delta detection."""

import os
from pathlib import Path

import faiss
import msgpack
import numpy as np

from embedrag.models.manifest import FileEntry, IndexInfo, Manifest, ShardEntry
from embedrag.shared.checksum import compute_sha256
from embedrag.writer.snapshot import SnapshotPackager


def _create_test_files(tmp_path: Path) -> tuple[str, dict[str, IndexInfo], dict[str, str], str]:
    """Create minimal test shard + db + id_map files.

    Returns (build_dir, space_index_infos, space_id_map_paths, db_path).
    """
    build_dir = str(tmp_path / "build")
    index_dir = Path(build_dir) / "index" / "text"
    index_dir.mkdir(parents=True)
    db_dir = Path(build_dir) / "db"
    db_dir.mkdir(parents=True)

    dim = 64
    n = 100
    vecs = np.random.randn(n, dim).astype(np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    shard_path = index_dir / "shard_0.faiss"
    faiss.write_index(index, str(shard_path))

    id_map = {i: f"chunk_{i}" for i in range(n)}
    id_map_path = str(index_dir / "id_map.msgpack")
    with open(id_map_path, "wb") as f:
        msgpack.pack(id_map, f)

    import sqlite3
    db_path = str(db_dir / "embedrag.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chunks (id TEXT)")
    for i in range(n):
        conn.execute("INSERT INTO chunks VALUES (?)", (f"chunk_{i}",))
    conn.commit()
    conn.close()

    space_index_infos = {
        "text": IndexInfo(
            type="Flat",
            dim=dim,
            num_shards=1,
            total_vectors=n,
            shards=[ShardEntry(file="index/text/shard_0.faiss", raw_size=shard_path.stat().st_size, num_vectors=n)],
        ),
    }
    space_id_map_paths = {"text": id_map_path}
    return build_dir, space_index_infos, space_id_map_paths, db_path


class TestSnapshotPackager:
    def test_compression_creates_zst_files(self, tmp_path):
        build_dir, space_index_infos, space_id_map_paths, db_path = _create_test_files(tmp_path)
        output_dir = str(tmp_path / "snapshot")

        packager = SnapshotPackager(compression_level=3)
        manifest = packager.package(
            build_dir=build_dir,
            output_dir=output_dir,
            space_index_infos=space_index_infos,
            space_id_map_paths=space_id_map_paths,
            db_path=db_path,
            doc_count=10,
            chunk_count=100,
            version="v001",
        )

        assert manifest.snapshot_version == "v001"
        assert manifest.db.doc_count == 10
        assert manifest.db.chunk_count == 100
        assert manifest.total_compressed_size > 0
        assert manifest.total_compressed_size <= manifest.total_raw_size

        for shard in manifest.indexes["text"].shards:
            assert Path(output_dir, shard.compressed_file).exists()
            assert shard.sha256_compressed != ""

        assert Path(output_dir, manifest.db.compressed_file).exists()
        assert Path(output_dir, manifest.id_maps["text"].compressed_file).exists()

    def test_manifest_checksums_valid(self, tmp_path):
        build_dir, space_index_infos, space_id_map_paths, db_path = _create_test_files(tmp_path)
        output_dir = str(tmp_path / "snapshot")

        packager = SnapshotPackager()
        manifest = packager.package(
            build_dir=build_dir,
            output_dir=output_dir,
            space_index_infos=space_index_infos,
            space_id_map_paths=space_id_map_paths,
            db_path=db_path,
            doc_count=10,
            chunk_count=100,
            version="v001",
        )

        for shard in manifest.indexes["text"].shards:
            actual = compute_sha256(Path(output_dir) / shard.compressed_file)
            assert actual == shard.sha256_compressed

    def test_delta_detection(self, tmp_path):
        """Delta detection was removed for multi-space; verify packaging still works twice."""
        build_dir, space_index_infos, space_id_map_paths, db_path = _create_test_files(tmp_path)

        out1 = str(tmp_path / "snap_v001")
        packager = SnapshotPackager()
        m1 = packager.package(
            build_dir=build_dir, output_dir=out1,
            space_index_infos=space_index_infos, space_id_map_paths=space_id_map_paths,
            db_path=db_path, doc_count=10, chunk_count=100, version="v001",
        )

        out2 = str(tmp_path / "snap_v002")
        m2 = packager.package(
            build_dir=build_dir, output_dir=out2,
            space_index_infos=space_index_infos, space_id_map_paths=space_id_map_paths,
            db_path=db_path, doc_count=10, chunk_count=100, version="v002",
            previous_manifest=m1,
        )

        assert m2.snapshot_version == "v002"
        assert m2.previous_version == "v001"
        assert m2.total_compressed_size > 0

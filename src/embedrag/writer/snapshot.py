"""Snapshot packager: compresses files, detects deltas, builds manifest v2."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import (
    FileEntry,
    IndexInfo,
    Manifest,
    ShardEntry,
)
from embedrag.shared.checksum import compute_sha256
from embedrag.shared.compression import compress_file
from embedrag.shared.object_store import ObjectStoreClient

logger = get_logger(__name__)


class SnapshotPackager:
    """Package raw build output into a compressed, checksummed snapshot."""

    def __init__(self, compression_level: int = 3):
        self._compression_level = compression_level

    def package(
        self,
        build_dir: str,
        output_dir: str,
        space_index_infos: dict[str, IndexInfo],
        space_id_map_paths: dict[str, str],
        db_path: str,
        doc_count: int,
        chunk_count: int,
        version: str,
        previous_manifest: Manifest | None = None,
    ) -> Manifest:
        """Create a complete snapshot with compressed files and manifest.

        Supports multiple named embedding spaces.
        """
        t0 = time.monotonic()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "db").mkdir(exist_ok=True)

        all_indexes: dict[str, IndexInfo] = {}
        all_id_maps: dict[str, FileEntry] = {}
        total_raw = 0
        total_compressed = 0

        for space, index_info in space_index_infos.items():
            (out / "index" / space).mkdir(parents=True, exist_ok=True)

            shard_entries: list[ShardEntry] = []
            for shard in index_info.shards:
                raw_path = Path(build_dir) / shard.file
                compressed_file = shard.file + ".zst"
                compressed_path = out / compressed_file
                compressed_path.parent.mkdir(parents=True, exist_ok=True)

                compressed_size = compress_file(raw_path, compressed_path, self._compression_level)
                shard_entries.append(
                    ShardEntry(
                        file=shard.file,
                        compressed_file=compressed_file,
                        sha256_raw=compute_sha256(raw_path),
                        sha256_compressed=compute_sha256(compressed_path),
                        raw_size=raw_path.stat().st_size,
                        compressed_size=compressed_size,
                        num_vectors=shard.num_vectors,
                    )
                )
                total_raw += raw_path.stat().st_size
                total_compressed += compressed_size

            all_indexes[space] = IndexInfo(
                type=index_info.type,
                dim=index_info.dim,
                metric=index_info.metric,
                nprobe_default=index_info.nprobe_default,
                num_shards=len(shard_entries),
                total_vectors=index_info.total_vectors,
                shards=shard_entries,
            )

            id_map_raw = space_id_map_paths[space]
            id_map_file = f"index/{space}/id_map.msgpack"
            id_map_compressed_file = f"index/{space}/id_map.msgpack.zst"
            id_map_compressed_size = compress_file(
                id_map_raw, out / id_map_compressed_file, self._compression_level
            )
            all_id_maps[space] = FileEntry(
                file=id_map_file,
                compressed_file=id_map_compressed_file,
                sha256_raw=compute_sha256(id_map_raw),
                sha256_compressed=compute_sha256(out / id_map_compressed_file),
                raw_size=Path(id_map_raw).stat().st_size,
                compressed_size=id_map_compressed_size,
            )
            total_raw += Path(id_map_raw).stat().st_size
            total_compressed += id_map_compressed_size

        db_compressed_file = "db/embedrag.db.zst"
        db_compressed_size = compress_file(
            db_path, out / db_compressed_file, self._compression_level
        )
        db_entry = FileEntry(
            file="db/embedrag.db",
            compressed_file=db_compressed_file,
            sha256_raw=compute_sha256(db_path),
            sha256_compressed=compute_sha256(out / db_compressed_file),
            raw_size=Path(db_path).stat().st_size,
            compressed_size=db_compressed_size,
            doc_count=doc_count,
            chunk_count=chunk_count,
        )
        total_raw += db_entry.raw_size
        total_compressed += db_entry.compressed_size

        manifest = Manifest(
            manifest_version=3,
            snapshot_version=version,
            created_at=datetime.now(UTC).isoformat(),
            previous_version=previous_manifest.snapshot_version if previous_manifest else "",
            schema_version=3,
            indexes=all_indexes,
            db=db_entry,
            id_maps=all_id_maps,
            total_raw_size=total_raw,
            total_compressed_size=total_compressed,
        )

        manifest.save(out / "manifest.json")
        elapsed = time.monotonic() - t0
        logger.info(
            "snapshot_packaged",
            version=version,
            spaces=list(space_index_infos.keys()),
            raw_mb=round(total_raw / 1024 / 1024, 1),
            compressed_mb=round(total_compressed / 1024 / 1024, 1),
            ratio=round(total_compressed / total_raw, 2) if total_raw else 0,
            elapsed_s=round(elapsed, 1),
        )
        return manifest

    # TODO: Re-implement multi-space delta detection in a future iteration.


class SnapshotPublisher:
    """Upload a packaged snapshot to object storage."""

    def __init__(self, client: ObjectStoreClient):
        self._client = client

    def publish(self, snapshot_dir: str, manifest: Manifest) -> float:
        """Upload all snapshot files and update latest.json.

        Returns upload time in seconds.
        """
        t0 = time.monotonic()
        version = manifest.snapshot_version
        base = Path(snapshot_dir)

        self._client.upload_file(
            base / "manifest.json",
            f"{version}/manifest.json",
        )

        for idx_info in manifest.indexes.values():
            for shard in idx_info.shards:
                remote = f"{version}/{shard.compressed_file}"
                self._client.upload_file(base / shard.compressed_file, remote)

        self._client.upload_file(
            base / manifest.db.compressed_file,
            f"{version}/{manifest.db.compressed_file}",
        )

        for idm in manifest.id_maps.values():
            self._client.upload_file(
                base / idm.compressed_file,
                f"{version}/{idm.compressed_file}",
            )

        self._client.put_json(
            "latest.json",
            {
                "version": version,
                "published_at": datetime.now(UTC).isoformat(),
            },
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "snapshot_published",
            version=version,
            elapsed_s=round(elapsed, 1),
        )
        return elapsed

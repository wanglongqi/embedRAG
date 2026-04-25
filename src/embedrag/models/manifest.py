"""Manifest v3: self-describing snapshot metadata with per-space indexes and checksums."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ShardEntry:
    file: str
    compressed_file: str = ""
    sha256_raw: str = ""
    sha256_compressed: str = ""
    raw_size: int = 0
    compressed_size: int = 0
    num_vectors: int = 0


@dataclass
class IndexInfo:
    type: str = "IVF4096,PQ64"
    dim: int = 1024
    metric: str = "IP"
    nprobe_default: int = 32
    num_shards: int = 4
    total_vectors: int = 0
    shards: list[ShardEntry] = field(default_factory=list)


@dataclass
class FileEntry:
    """A single file in the snapshot (db or id_map)."""
    file: str
    compressed_file: str = ""
    sha256_raw: str = ""
    sha256_compressed: str = ""
    raw_size: int = 0
    compressed_size: int = 0
    doc_count: int = 0
    chunk_count: int = 0


@dataclass
class DeltaInfo:
    from_version: str = ""
    unchanged_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    delta_compressed_size: int = 0


@dataclass
class Manifest:
    manifest_version: int = 3
    snapshot_version: str = ""
    created_at: str = ""
    previous_version: str = ""
    schema_version: int = 3

    indexes: dict[str, IndexInfo] = field(default_factory=lambda: {"text": IndexInfo()})
    db: FileEntry = field(default_factory=lambda: FileEntry(file="db/embedrag.db"))
    id_maps: dict[str, FileEntry] = field(
        default_factory=lambda: {"text": FileEntry(file="index/text/id_map.msgpack")}
    )

    total_raw_size: int = 0
    total_compressed_size: int = 0

    delta: Optional[DeltaInfo] = None

    @property
    def spaces(self) -> list[str]:
        return list(self.indexes.keys())

    def all_compressed_files(self) -> list[str]:
        """Return list of all compressed file paths in the snapshot."""
        files: list[str] = []
        for idx_info in self.indexes.values():
            for shard in idx_info.shards:
                files.append(shard.compressed_file or shard.file)
        files.append(self.db.compressed_file or self.db.file)
        for idm in self.id_maps.values():
            files.append(idm.compressed_file or idm.file)
        return files

    def get_file_entry_by_compressed(self, compressed_path: str) -> Optional[FileEntry | ShardEntry]:
        for idx_info in self.indexes.values():
            for shard in idx_info.shards:
                if (shard.compressed_file or shard.file) == compressed_path:
                    return shard
        if (self.db.compressed_file or self.db.file) == compressed_path:
            return self.db
        for idm in self.id_maps.values():
            if (idm.compressed_file or idm.file) == compressed_path:
                return idm
        return None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, raw: str) -> "Manifest":
        return cls.from_dict(json.loads(raw))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        def _idx_to_dict(idx: IndexInfo) -> dict:
            return {
                "type": idx.type,
                "dim": idx.dim,
                "metric": idx.metric,
                "nprobe_default": idx.nprobe_default,
                "num_shards": idx.num_shards,
                "total_vectors": idx.total_vectors,
                "shards": [
                    {
                        "file": s.file,
                        "compressed_file": s.compressed_file,
                        "sha256_raw": s.sha256_raw,
                        "sha256_compressed": s.sha256_compressed,
                        "raw_size": s.raw_size,
                        "compressed_size": s.compressed_size,
                        "num_vectors": s.num_vectors,
                    }
                    for s in idx.shards
                ],
            }

        def _fe_to_dict(fe: FileEntry) -> dict:
            return {
                "file": fe.file,
                "compressed_file": fe.compressed_file,
                "sha256_raw": fe.sha256_raw,
                "sha256_compressed": fe.sha256_compressed,
                "raw_size": fe.raw_size,
                "compressed_size": fe.compressed_size,
            }

        d: dict = {
            "manifest_version": self.manifest_version,
            "snapshot_version": self.snapshot_version,
            "created_at": self.created_at,
            "previous_version": self.previous_version,
            "schema_version": self.schema_version,
            "indexes": {space: _idx_to_dict(idx) for space, idx in self.indexes.items()},
            "db": {
                "file": self.db.file,
                "compressed_file": self.db.compressed_file,
                "sha256_raw": self.db.sha256_raw,
                "sha256_compressed": self.db.sha256_compressed,
                "raw_size": self.db.raw_size,
                "compressed_size": self.db.compressed_size,
                "doc_count": self.db.doc_count,
                "chunk_count": self.db.chunk_count,
            },
            "id_maps": {space: _fe_to_dict(fe) for space, fe in self.id_maps.items()},
            "total_raw_size": self.total_raw_size,
            "total_compressed_size": self.total_compressed_size,
        }
        if self.delta:
            d["delta"] = {
                "from_version": self.delta.from_version,
                "unchanged_files": self.delta.unchanged_files,
                "changed_files": self.delta.changed_files,
                "delta_compressed_size": self.delta.delta_compressed_size,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        def _parse_index(d: dict) -> IndexInfo:
            shards = [ShardEntry(**s) for s in d.get("shards", [])]
            return IndexInfo(
                type=d.get("type", "IVF4096,PQ64"),
                dim=d.get("dim", 1024),
                metric=d.get("metric", "IP"),
                nprobe_default=d.get("nprobe_default", 32),
                num_shards=d.get("num_shards", 4),
                total_vectors=d.get("total_vectors", 0),
                shards=shards,
            )

        def _parse_file_entry(d: dict, default_file: str = "") -> FileEntry:
            return FileEntry(
                file=d.get("file", default_file),
                compressed_file=d.get("compressed_file", ""),
                sha256_raw=d.get("sha256_raw", ""),
                sha256_compressed=d.get("sha256_compressed", ""),
                raw_size=d.get("raw_size", 0),
                compressed_size=d.get("compressed_size", 0),
                doc_count=d.get("doc_count", 0),
                chunk_count=d.get("chunk_count", 0),
            )

        indexes = {
            space: _parse_index(d)
            for space, d in data.get("indexes", {}).items()
        } or {"text": IndexInfo()}

        id_maps = {
            space: _parse_file_entry(d, f"index/{space}/id_map.msgpack")
            for space, d in data.get("id_maps", {}).items()
        } or {"text": FileEntry(file="index/text/id_map.msgpack")}

        db_data = data.get("db", {})
        db = _parse_file_entry(db_data, "db/embedrag.db")

        delta_data = data.get("delta")
        delta = None
        if delta_data:
            delta = DeltaInfo(
                from_version=delta_data.get("from_version", ""),
                unchanged_files=delta_data.get("unchanged_files", []),
                changed_files=delta_data.get("changed_files", []),
                delta_compressed_size=delta_data.get("delta_compressed_size", 0),
            )

        return cls(
            manifest_version=data.get("manifest_version", 3),
            snapshot_version=data.get("snapshot_version", ""),
            created_at=data.get("created_at", ""),
            previous_version=data.get("previous_version", ""),
            schema_version=data.get("schema_version", 3),
            indexes=indexes,
            db=db,
            id_maps=id_maps,
            total_raw_size=data.get("total_raw_size", 0),
            total_compressed_size=data.get("total_compressed_size", 0),
            delta=delta,
        )

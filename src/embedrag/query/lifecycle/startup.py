"""Startup precheck and snapshot loading for the query node."""

from __future__ import annotations

import resource
from pathlib import Path

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import Manifest
from embedrag.query.index.generation import GenerationContext
from embedrag.query.index.hotfix import HotfixBuffer
from embedrag.query.index.id_mapping import IDMapper
from embedrag.query.index.shard import ShardWorker
from embedrag.query.retrieval.dense import ShardManager
from embedrag.query.storage import QueryDocStore, ReadOnlySQLitePool
from embedrag.shared.checksum import quick_verify
from embedrag.shared.compression import decompress_file

logger = get_logger(__name__)


def load_generation(
    snapshot_dir: str,
    manifest: Manifest,
    nprobe: int = 32,
    use_mmap: bool = True,
    pool_size: int = 8,
    memory_budget_mb: int = 1200,
) -> GenerationContext:
    """Load a snapshot into a GenerationContext.

    Decompresses files if needed, loads FAISS shards (per space) with staged
    memory checks, and opens read-only SQLite.
    """
    snap = Path(snapshot_dir)
    logger.info("load_generation_start", version=manifest.snapshot_version, dir=snapshot_dir)

    # Decompress DB
    db_compressed = snap / manifest.db.compressed_file
    db_raw = snap / manifest.db.file
    if db_compressed.exists() and not db_raw.exists():
        db_raw.parent.mkdir(parents=True, exist_ok=True)
        decompress_file(str(db_compressed), str(db_raw))

    budget_bytes = memory_budget_mb * 1024 * 1024
    all_shard_managers: dict[str, ShardManager] = {}
    all_hotfix_buffers: dict[str, HotfixBuffer] = {}
    total_vectors = 0

    for space, idx_info in manifest.indexes.items():
        # Decompress FAISS shards for this space
        for shard_info in idx_info.shards:
            compressed = snap / shard_info.compressed_file
            raw = snap / shard_info.file
            if compressed.exists() and not raw.exists():
                raw.parent.mkdir(parents=True, exist_ok=True)
                decompress_file(str(compressed), str(raw))

        # Decompress id_map for this space
        idm = manifest.id_maps.get(space)
        if idm:
            idmap_compressed = snap / idm.compressed_file
            idmap_raw = snap / idm.file
            if idmap_compressed.exists() and not idmap_raw.exists():
                idmap_raw.parent.mkdir(parents=True, exist_ok=True)
                decompress_file(str(idmap_compressed), str(idmap_raw))
        else:
            idmap_raw = snap / f"index/{space}/id_map.msgpack"

        # Load FAISS shards with memory check
        shard_workers: list[ShardWorker] = []
        for shard_info in idx_info.shards:
            shard_path = str(snap / shard_info.file)
            worker = ShardWorker(shard_path, nprobe=nprobe, use_mmap=use_mmap)
            shard_workers.append(worker)

            rss = _get_rss_bytes()
            if rss > budget_bytes:
                for w in shard_workers:
                    w.shutdown()
                raise MemoryError(
                    f"RSS {rss // 1024 // 1024}MB exceeds budget {memory_budget_mb}MB "
                    f"after loading {len(shard_workers)} shards for space '{space}'"
                )

        shard_sizes = [s.num_vectors for s in idx_info.shards]
        id_mapper = IDMapper.load(str(idmap_raw), shard_sizes)
        sm = ShardManager(shard_workers, id_mapper)
        all_shard_managers[space] = sm
        all_hotfix_buffers[space] = HotfixBuffer(dim=idx_info.dim)
        total_vectors += sm.total_vectors

    # Open read-only SQLite and verify schema version
    import sqlite3

    from embedrag.writer.schema import check_schema_version

    check_conn = sqlite3.connect(f"file:{db_raw}?mode=ro", uri=True)
    try:
        check_schema_version(check_conn)
    finally:
        check_conn.close()

    db_pool = ReadOnlySQLitePool(str(db_raw), pool_size=pool_size)
    doc_store = QueryDocStore(db_pool)

    logger.info(
        "load_generation_done",
        version=manifest.snapshot_version,
        spaces=list(all_shard_managers.keys()),
        total_vectors=total_vectors,
    )
    return GenerationContext(
        version=manifest.snapshot_version,
        shard_managers=all_shard_managers,
        db_pool=db_pool,
        doc_store=doc_store,
        hotfix_buffers=all_hotfix_buffers,
        manifest=manifest,
    )


def quick_verify_snapshot(snapshot_dir: str, manifest: Manifest) -> bool:
    """Quick integrity check using file size + head/tail hash."""
    snap = Path(snapshot_dir)
    for idx_info in manifest.indexes.values():
        for shard in idx_info.shards:
            f = snap / (shard.compressed_file or shard.file)
            expected_size = shard.compressed_size or shard.raw_size
            if not quick_verify(f, expected_size):
                logger.warn("quick_verify_fail", file=str(f))
                return False

    db_f = snap / (manifest.db.compressed_file or manifest.db.file)
    if not quick_verify(db_f, manifest.db.compressed_size or manifest.db.raw_size):
        logger.warn("quick_verify_fail", file=str(db_f))
        return False

    return True


def _get_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss  # bytes on macOS, KB on Linux

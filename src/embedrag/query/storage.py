"""Read-only SQLite connection pool and document store for the query node."""

from __future__ import annotations

import json
import queue
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from embedrag.logging_setup import get_logger
from embedrag.models.api import ChunkResult

logger = get_logger(__name__)


class ReadOnlySQLitePool:
    """Read-only connection pool. No locks, no WAL, pure parallel reads."""

    def __init__(self, db_path: str, pool_size: int = 8):
        self._db_path = db_path
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA cache_size=-65536")  # 64MB
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.row_factory = sqlite3.Row
            self._pool.put(conn)
        self._pool_size = pool_size
        logger.info("ro_pool_init", db=db_path, size=pool_size)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


class QueryDocStore:
    """Document/chunk store with LRU caching for the query node."""

    def __init__(self, pool: ReadOnlySQLitePool, cache_size: int = 10_000):
        self._pool = pool
        self._cache_size = cache_size
        self._get_chunk_cached = lru_cache(maxsize=cache_size)(self._get_chunk_uncached)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkResult]:
        """Fetch chunks by IDs, using cache for hot chunks."""
        results = []
        for cid in chunk_ids:
            row = self._get_chunk_cached(cid)
            if row:
                results.append(row)
        return results

    def _get_chunk_uncached(self, chunk_id: str) -> ChunkResult | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT c.chunk_id, c.doc_id, c.text, c.level, c.level_type, "
                "c.metadata_json, d.title "
                "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.doc_id "
                "WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                return None
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            metadata["title"] = row["title"] or ""
            return ChunkResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                score=0.0,
                level=row["level"],
                level_type=row["level_type"],
                metadata=metadata,
            )

    def get_parent_chunk_text(self, chunk_id: str) -> str | None:
        """Get the parent chunk's text for context expansion."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT p.text FROM chunks c JOIN chunks p ON c.parent_chunk_id = p.chunk_id WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            return row["text"] if row else None

    def get_ancestors(self, chunk_id: str, max_depth: int = 2) -> list[ChunkResult]:
        """Get ancestor chunks via the closure table."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT c.chunk_id, c.doc_id, c.text, c.level, c.level_type, "
                "c.metadata_json, cl.depth "
                "FROM chunk_closure cl "
                "JOIN chunks c ON cl.ancestor_id = c.chunk_id "
                "WHERE cl.descendant_id = ? AND cl.depth > 0 AND cl.depth <= ? "
                "ORDER BY cl.depth ASC",
                (chunk_id, max_depth),
            ).fetchall()
            results = []
            for row in rows:
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                results.append(
                    ChunkResult(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        score=0.0,
                        level=row["level"],
                        level_type=row["level_type"],
                        metadata=metadata,
                    )
                )
            return results

    def get_neighbors(self, chunk_id: str, before: int = 2, after: int = 2) -> dict:
        """Get neighboring chunks (by seq_in_parent under the same parent).

        Returns {"before": [...], "current": {...}, "after": [...]}.
        """
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT doc_id, parent_chunk_id, seq_in_parent FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                return {"before": [], "current": None, "after": []}

            doc_id = row["doc_id"]
            parent_id = row["parent_chunk_id"]
            seq = row["seq_in_parent"]

            if parent_id:
                condition = "parent_chunk_id = ? AND doc_id = ?"
                params_base = [parent_id, doc_id]
            else:
                condition = "parent_chunk_id IS NULL AND doc_id = ?"
                params_base = [doc_id]

            before_rows = conn.execute(
                f"SELECT chunk_id, text, seq_in_parent, level_type FROM chunks "
                f"WHERE {condition} AND seq_in_parent < ? "
                f"ORDER BY seq_in_parent DESC LIMIT ?",
                params_base + [seq, before],
            ).fetchall()

            after_rows = conn.execute(
                f"SELECT chunk_id, text, seq_in_parent, level_type FROM chunks "
                f"WHERE {condition} AND seq_in_parent > ? "
                f"ORDER BY seq_in_parent ASC LIMIT ?",
                params_base + [seq, after],
            ).fetchall()

            current = self._get_chunk_cached(chunk_id)

            return {
                "before": [
                    {
                        "chunk_id": r["chunk_id"],
                        "text": r["text"],
                        "seq": r["seq_in_parent"],
                        "level_type": r["level_type"],
                    }
                    for r in reversed(before_rows)
                ],
                "current": {
                    "chunk_id": chunk_id,
                    "text": current.text if current else "",
                    "seq": seq,
                }
                if current
                else None,
                "after": [
                    {
                        "chunk_id": r["chunk_id"],
                        "text": r["text"],
                        "seq": r["seq_in_parent"],
                        "level_type": r["level_type"],
                    }
                    for r in after_rows
                ],
            }

    def clear_cache(self) -> None:
        self._get_chunk_cached.cache_clear()

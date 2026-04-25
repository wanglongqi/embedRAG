"""SQLite WAL-mode read/write connection pool for the writer node."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import numpy as np

from embedrag.logging_setup import get_logger
from embedrag.models.chunk import ChunkNode, Document
from embedrag.writer.schema import initialize_schema

logger = get_logger(__name__)


def _embed_to_blob(embedding: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes()


def _blob_to_embed(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class WriterSQLitePool:
    """Read/write split connection pool with WAL mode for the writer node."""

    def __init__(
        self,
        db_path: str,
        max_readers: int = 4,
        wal_autocheckpoint: int = 1000,
        cache_size_mb: int = 64,
    ):
        self._db_path = db_path
        self._cache_size_mb = cache_size_mb
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._writer = self._create_conn(readonly=False)
        self._writer.execute(f"PRAGMA wal_autocheckpoint={wal_autocheckpoint}")
        self._write_lock = asyncio.Lock()

        self._readers: asyncio.Queue[sqlite3.Connection] = asyncio.Queue(maxsize=max_readers)
        for _ in range(max_readers):
            self._readers.put_nowait(self._create_conn(readonly=True))
        initialize_schema(self._writer)
        logger.info("writer_pool_init", db=db_path, readers=max_readers)

    def _create_conn(self, readonly: bool) -> sqlite3.Connection:
        mode = "ro" if readonly else "rwc"
        uri = f"file:{self._db_path}?mode={mode}"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA cache_size=-{self._cache_size_mb * 1024}")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    @asynccontextmanager
    async def read_conn(self) -> AsyncIterator[sqlite3.Connection]:
        conn = await self._readers.get()
        try:
            yield conn
        finally:
            self._readers.put_nowait(conn)

    @asynccontextmanager
    async def write_conn(self) -> AsyncIterator[sqlite3.Connection]:
        async with self._write_lock:
            yield self._writer

    @contextmanager
    def write_conn_sync(self) -> Iterator[sqlite3.Connection]:
        yield self._writer

    def checkpoint(self) -> None:
        self._writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self.checkpoint()
        self._writer.close()
        while not self._readers.empty():
            try:
                conn = self._readers.get_nowait()
                conn.close()
            except asyncio.QueueEmpty:
                break

    # ── Document operations ──

    async def insert_document(self, doc: Document) -> None:
        async with self.write_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id, title, source, doc_type, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc.doc_id, doc.title, doc.source, doc.doc_type,
                 json.dumps(doc.metadata)),
            )
            conn.commit()

    async def insert_documents_batch(self, docs: list[Document]) -> None:
        async with self.write_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO documents
                   (doc_id, title, source, doc_type, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                [(d.doc_id, d.title, d.source, d.doc_type,
                  json.dumps(d.metadata)) for d in docs],
            )
            conn.commit()

    async def get_document(self, doc_id: str) -> Optional[Document]:
        async with self.read_conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if not row:
                return None
            return Document(
                doc_id=row["doc_id"],
                title=row["title"],
                source=row["source"],
                doc_type=row["doc_type"],
                metadata=json.loads(row["metadata_json"]),
            )

    # ── Chunk operations ──

    async def insert_chunks_batch(self, chunks: list[ChunkNode], space: str = "text") -> None:
        async with self.write_conn() as conn:
            chunk_rows = []
            emb_rows = []
            for c in chunks:
                chunk_rows.append((
                    c.chunk_id, c.doc_id, c.parent_chunk_id,
                    c.level, c.level_type, c.seq_in_parent,
                    c.text, json.dumps(c.metadata),
                ))
                if c.embedding:
                    emb_rows.append((c.chunk_id, space, _embed_to_blob(c.embedding)))
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, doc_id, parent_chunk_id, level, level_type,
                    seq_in_parent, text, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                chunk_rows,
            )
            if emb_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, space, embedding) "
                    "VALUES (?, ?, ?)",
                    emb_rows,
                )
            conn.commit()

    async def cleanup_before_upsert(self, doc_ids: list[str]) -> int:
        """Remove stale FTS and closure rows for docs about to be re-ingested.

        Called before insert_chunks_batch on re-ingest so that FTS5
        (which doesn't participate in CASCADE) stays consistent.
        Returns the number of stale chunk rows cleaned.
        """
        if not doc_ids:
            return 0
        async with self.write_conn() as conn:
            placeholders = ",".join("?" * len(doc_ids))
            chunk_ids = [
                r[0] for r in conn.execute(
                    f"SELECT chunk_id FROM chunks WHERE doc_id IN ({placeholders})",
                    doc_ids,
                ).fetchall()
            ]
            if not chunk_ids:
                return 0
            cp = ",".join("?" * len(chunk_ids))
            conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({cp})", chunk_ids,
            )
            conn.execute(
                f"DELETE FROM chunk_closure WHERE descendant_id IN ({cp})", chunk_ids,
            )
            conn.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({cp})", chunk_ids,
            )
            conn.commit()
        return len(chunk_ids)

    async def insert_fts_batch(self, chunks: list[ChunkNode], doc_titles: dict[str, str]) -> None:
        from embedrag.text.normalize import normalize_for_fts

        async with self.write_conn() as conn:
            rows = []
            for c in chunks:
                title = doc_titles.get(c.doc_id, "")
                tags = c.metadata.get("tags", "")
                if isinstance(tags, list):
                    tags = " ".join(tags)
                rows.append((
                    c.chunk_id, c.text, normalize_for_fts(c.text),
                    title, normalize_for_fts(title), tags,
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO chunks_fts "
                "(chunk_id, text, text_norm, title, title_norm, tags) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    async def insert_closure_batch(self, relations: list[tuple[str, str, int]]) -> None:
        """Insert closure table entries: (ancestor_id, descendant_id, depth)."""
        async with self.write_conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_closure (ancestor_id, descendant_id, depth) VALUES (?, ?, ?)",
                relations,
            )
            conn.commit()

    async def get_all_chunks_with_embeddings(self, space: str = "text") -> list[tuple[str, np.ndarray]]:
        """Read all (chunk_id, embedding) pairs for a given space."""
        async with self.read_conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings WHERE space = ?",
                (space,),
            ).fetchall()
            return [(r["chunk_id"], _blob_to_embed(r["embedding"])) for r in rows]

    async def get_embedding_spaces(self) -> list[str]:
        """Return all distinct embedding space names."""
        async with self.read_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT space FROM chunk_embeddings ORDER BY space"
            ).fetchall()
            return [r[0] for r in rows]

    async def get_chunk_count(self) -> int:
        async with self.read_conn() as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
            return row[0]

    async def get_doc_count(self) -> int:
        async with self.read_conn() as conn:
            row = conn.execute("SELECT count(*) FROM documents").fetchone()
            return row[0]

    async def delete_document(self, doc_id: str) -> int:
        """Delete a document and all its chunks. Returns chunks deleted."""
        async with self.write_conn() as conn:
            chunk_ids = [
                r[0] for r in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
            ]
            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                conn.execute(
                    f"DELETE FROM chunk_closure WHERE descendant_id IN ({placeholders})",
                    chunk_ids,
                )
                conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
                conn.execute(
                    f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return len(chunk_ids)

    def export_query_db(self, output_path: str) -> tuple[int, int]:
        """Export a lean read-only SQLite for query nodes (no embedding column)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(output_path)
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.execute("PRAGMA synchronous=FULL")

        dst.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                doc_type TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                parent_chunk_id TEXT,
                level INTEGER NOT NULL DEFAULT 0,
                level_type TEXT NOT NULL DEFAULT 'chunk',
                seq_in_parent INTEGER NOT NULL DEFAULT 0,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(level);
            CREATE TABLE IF NOT EXISTS chunk_closure (
                ancestor_id TEXT NOT NULL,
                descendant_id TEXT NOT NULL,
                depth INTEGER NOT NULL,
                PRIMARY KEY (ancestor_id, descendant_id)
            );
            CREATE INDEX IF NOT EXISTS idx_closure_desc ON chunk_closure(descendant_id, depth);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED, text, text_norm, title, title_norm, tags,
                tokenize='trigram case_sensitive 0'
            );
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                description TEXT NOT NULL DEFAULT ''
            );
        """)
        dst.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (3, "exported by writer"),
        )

        src = self._writer
        for row in src.execute("SELECT * FROM documents"):
            dst.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                tuple(row),
            )

        for row in src.execute(
            "SELECT chunk_id, doc_id, parent_chunk_id, level, level_type, "
            "seq_in_parent, text, metadata_json, created_at FROM chunks"
        ):
            dst.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", tuple(row))

        for row in src.execute("SELECT * FROM chunk_closure"):
            dst.execute("INSERT INTO chunk_closure VALUES (?,?,?)", tuple(row))

        for row in src.execute(
            "SELECT chunk_id, text, text_norm, title, title_norm, tags FROM chunks_fts"
        ):
            dst.execute("INSERT INTO chunks_fts VALUES (?,?,?,?,?,?)", tuple(row))

        doc_count = dst.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunk_count = dst.execute("SELECT count(*) FROM chunks").fetchone()[0]

        dst.commit()
        dst.execute("VACUUM")
        dst.close()
        return doc_count, chunk_count

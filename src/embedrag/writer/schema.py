"""DDL definitions and schema migration for the writer SQLite database.

Only the current schema version (v3) is defined here. There is no
backward-compatibility migration chain -- if a database is outdated, the
writer node auto-migrates it on startup; the query node refuses to start
and tells the operator to run ``embedrag migrate <db_path>`` first.
"""

from __future__ import annotations

import json
import sqlite3

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

CURRENT_SCHEMA_VERSION = 3

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    doc_type TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_chunk_id TEXT REFERENCES chunks(chunk_id),
    level INTEGER NOT NULL DEFAULT 0,
    level_type TEXT NOT NULL DEFAULT 'chunk',
    seq_in_parent INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(level);

CREATE TABLE IF NOT EXISTS chunk_closure (
    ancestor_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    descendant_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    depth INTEGER NOT NULL,
    PRIMARY KEY (ancestor_id, descendant_id)
);

CREATE INDEX IF NOT EXISTS idx_closure_desc ON chunk_closure(descendant_id, depth);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    text_norm,
    title,
    title_norm,
    tags,
    tokenize='trigram case_sensitive 0'
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    space TEXT NOT NULL DEFAULT 'text',
    embedding BLOB NOT NULL,
    PRIMARY KEY (chunk_id, space)
);
CREATE INDEX IF NOT EXISTS idx_chunk_emb_space ON chunk_embeddings(space);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT NOT NULL DEFAULT ''
);
"""


class SchemaVersionError(RuntimeError):
    """Raised when the DB schema version doesn't match CURRENT_SCHEMA_VERSION."""


def get_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def check_schema_version(conn: sqlite3.Connection) -> None:
    """Raise SchemaVersionError if the DB is not at CURRENT_SCHEMA_VERSION.

    Used by the query node to refuse loading outdated snapshots.
    """
    current = get_schema_version(conn)
    if current < CURRENT_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Database schema v{current} is outdated (need v{CURRENT_SCHEMA_VERSION}). "
            f"Run: embedrag migrate <path-to-embedrag.db>"
        )


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create all tables from scratch if needed, or migrate from any older version.

    Called by the writer node on startup (auto-migrate) and by ``embedrag migrate``.
    """
    current = get_schema_version(conn)
    if current >= CURRENT_SCHEMA_VERSION:
        logger.info("schema_up_to_date", version=current)
        return

    logger.info("schema_migrate", from_version=current, to_version=CURRENT_SCHEMA_VERSION)

    if current == 0:
        _fresh_install(conn)
    else:
        _migrate_to_v3(conn, current)

    logger.info("schema_initialized", version=CURRENT_SCHEMA_VERSION)


def _fresh_install(conn: sqlite3.Connection) -> None:
    """Brand-new database: create everything at v3."""
    conn.executescript(SCHEMA_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
        (CURRENT_SCHEMA_VERSION, "Schema v3: FTS normalized, multi-space embeddings"),
    )
    conn.commit()


def _migrate_to_v3(conn: sqlite3.Connection, current: int) -> None:
    """Migrate any pre-v3 database to v3.

    Handles v1 (old FTS without text_norm) and v2 (FTS with text_norm but
    no chunk_embeddings) in a single pass.
    """
    if current < 2:
        _rebuild_fts_with_norm(conn)

    if current < 3:
        _add_chunk_embeddings(conn)

    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
        (CURRENT_SCHEMA_VERSION, "Schema v3: FTS normalized, multi-space embeddings"),
    )
    conn.commit()


def _rebuild_fts_with_norm(conn: sqlite3.Connection) -> None:
    """Drop old FTS table and rebuild with text_norm/title_norm columns."""
    from embedrag.text.normalize import normalize_for_fts

    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "  chunk_id UNINDEXED, text, text_norm, title, title_norm, tags,"
        "  tokenize='trigram case_sensitive 0'"
        ")"
    )

    rows = conn.execute(
        "SELECT c.chunk_id, c.text, COALESCE(d.title, '') AS title, "
        "c.metadata_json "
        "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.doc_id"
    ).fetchall()

    if rows:
        fts_rows = []
        for r in rows:
            text = r[1] or ""
            title = r[2] or ""
            meta = json.loads(r[3]) if r[3] else {}
            tags = meta.get("tags", "")
            if isinstance(tags, list):
                tags = " ".join(tags)
            fts_rows.append(
                (
                    r[0],
                    text,
                    normalize_for_fts(text),
                    title,
                    normalize_for_fts(title),
                    tags,
                )
            )
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, text, text_norm, title, title_norm, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            fts_rows,
        )
    logger.info("fts_rebuilt_with_norm", rows=len(rows) if rows else 0)


def _add_chunk_embeddings(conn: sqlite3.Connection) -> None:
    """Add chunk_embeddings table, migrating data from old embedding column if present."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunk_embeddings ("
        "  chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,"
        "  space TEXT NOT NULL DEFAULT 'text',"
        "  embedding BLOB NOT NULL,"
        "  PRIMARY KEY (chunk_id, space)"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_emb_space ON chunk_embeddings(space)")

    has_embedding_col = any(
        r[1] == "embedding" for r in conn.execute("PRAGMA table_info(chunks)").fetchall()
    )
    migrated = 0
    if has_embedding_col:
        rows = conn.execute(
            "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, space, embedding) "
                "VALUES (?, 'text', ?)",
                [(r[0], r[1]) for r in rows],
            )
            migrated = len(rows)
        try:
            conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
        except sqlite3.OperationalError:
            conn.execute("UPDATE chunks SET embedding = NULL")

    logger.info("chunk_embeddings_added", migrated=migrated)

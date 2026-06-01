"""Vector + item acquisition for clustering.

Supports several co-equal sources so the tool works standalone or against
the embedRAG vector store:

- Files: ``.jsonl`` / ``.csv`` of ``{id, text, [embedding]}``, or a ``.npy``
  matrix of precomputed vectors.
- Passed-in python objects: a list of texts, or a numpy/array of vectors.
- A writer SQLite DB: exact vectors from ``chunk_embeddings`` + text from
  ``chunks`` (with optional filters).
- A loaded query-node generation: vectors reconstructed from the FAISS index
  (exact for Flat/IVF-Flat, approximate for IVF,PQ).

When no vectors are available, callers fall back to embedding the text via an
embedding service, or to a local TF-IDF representation (no service needed).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from embedrag.cluster.types import Item
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


def load_items_from_file(
    path: str,
    text_field: str = "text",
    id_field: str = "id",
    embedding_field: str = "embedding",
) -> tuple[list[Item], np.ndarray | None]:
    """Load items (and optional inline embeddings) from a .jsonl or .csv file.

    Returns ``(items, vectors_or_None)``. ``vectors`` is returned only when
    every row carries an embedding under ``embedding_field``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows = _read_jsonl(p)
    elif suffix == ".json":
        rows = _read_json_array(p)
    elif suffix == ".csv":
        rows = _read_csv(p)
    else:
        raise ValueError(f"Unsupported input file type: {suffix} (use .jsonl, .json, or .csv)")

    items: list[Item] = []
    vectors: list[list[float]] = []
    have_all_vectors = True
    for i, row in enumerate(rows):
        rid = str(row.get(id_field, i))
        text = str(row.get(text_field, "") or "")
        items.append(Item(id=rid, text=text))
        emb = row.get(embedding_field)
        if emb is None:
            have_all_vectors = False
        elif have_all_vectors:
            if isinstance(emb, str):
                emb = json.loads(emb)
            vectors.append([float(x) for x in emb])

    if have_all_vectors and vectors:
        arr = np.asarray(vectors, dtype=np.float32)
        logger.info("cluster_source_file", path=str(p), items=len(items), with_embeddings=True, dim=arr.shape[1])
        return items, arr

    logger.info("cluster_source_file", path=str(p), items=len(items), with_embeddings=False)
    return items, None


def _read_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json_array(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # allow {"items": [...]} or {"documents": [...]}
        for key in ("items", "documents", "data", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError("JSON object has no list under items/documents/data/rows")
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list or an object containing a list")
    return data


def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_vectors_npy(path: str, items: list[Item] | None = None) -> tuple[list[Item], np.ndarray]:
    """Load a ``.npy`` matrix of vectors; synthesize ids if no items given."""
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array in {path}, got shape {arr.shape}")
    if items is None:
        items = [Item(id=str(i)) for i in range(arr.shape[0])]
    elif len(items) != arr.shape[0]:
        raise ValueError(f"items ({len(items)}) and vectors ({arr.shape[0]}) length mismatch")
    return items, arr


def items_from_texts(texts: list[str], ids: list[str] | None = None) -> list[Item]:
    """Build items from a list of texts."""
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    return [Item(id=str(i), text=str(t or "")) for i, t in zip(ids, texts)]


def read_writer_db(
    db_path: str,
    space: str = "text",
    filters: dict | None = None,
    limit: int | None = None,
) -> tuple[list[Item], np.ndarray]:
    """Read exact vectors + text from a writer DB's ``chunk_embeddings`` table.

    ``filters`` may contain ``doc_type`` and/or ``doc_id`` to restrict the set.
    Raises if the DB has no populated ``chunk_embeddings`` table.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "chunk_embeddings"):
            raise ValueError(
                f"{db_path} has no chunk_embeddings table (a query-node export drops it). "
                "Use a writer DB, or cluster a loaded generation via FAISS reconstruction."
            )
        where = ["ce.space = ?"]
        params: list = [space]
        filters = filters or {}
        if filters.get("doc_type"):
            where.append("d.doc_type = ?")
            params.append(filters["doc_type"])
        if filters.get("doc_id"):
            where.append("c.doc_id = ?")
            params.append(filters["doc_id"])
        sql = (
            "SELECT c.chunk_id AS id, c.text AS text, ce.embedding AS embedding "
            "FROM chunk_embeddings ce "
            "JOIN chunks c ON c.chunk_id = ce.chunk_id "
            "LEFT JOIN documents d ON c.doc_id = d.doc_id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY c.chunk_id"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No vectors found in {db_path} for space '{space}' with the given filters")

    items = [Item(id=r["id"], text=r["text"] or "") for r in rows]
    vectors = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    logger.info("cluster_source_writer_db", db=db_path, space=space, items=len(items), dim=vectors.shape[1])
    return items, vectors


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def tfidf_vectors(texts: list[str], max_features: int = 4096) -> np.ndarray:
    """Build a dense TF-IDF representation for the no-embedding-service path.

    Uses char n-grams so it works for CJK text without word segmentation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    has_cjk = any(_contains_cjk(t) for t in texts)
    if has_cjk:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=max_features)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words="english")
    matrix = vec.fit_transform(texts)
    return matrix.toarray().astype(np.float32)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

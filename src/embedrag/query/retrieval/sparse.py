"""Sparse retrieval via SQLite FTS5 trigram index.

This module implements a hybrid keyword search strategy designed for both
space-delimited languages (e.g., English) and scriptio-continua languages
(e.g., Chinese, Japanese). It uses a tiered approach combining SQLite's
FTS5 trigram index for fast BM25-ranked matches and a LIKE-based fallback
for short terms and bigrams.

Tiered retrieval strategy:
    1. **FTS5 MATCH** (primary, fast, BM25-ranked):
       Uses trigram-based indexing. For scriptio-continua segments, the query
       is decomposed into sliding windows to handle punctuation breaks.
    2. **LIKE fallback** (secondary, slower, length-ranked):
       Activated for short segments (< 3 characters) and bigrams extracted
       from long segments to bridge punctuation boundaries.

Requires schema v3 which includes the `text_norm` column in the FTS table.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from embedrag.logging_setup import get_logger
from embedrag.query.storage import ReadOnlySQLitePool
from embedrag.text.normalize import normalize_query

logger = get_logger(__name__)

_FTS5_SPECIAL = re.compile(r'["\(\)\*\+\^{}:]')

# Scriptio-continua scripts: no inter-word spaces, need sliding-window
# decomposition for FTS5 trigram to match across punctuation boundaries.
_SCRIPTIO_CONTINUA = re.compile(
    r"[\u0e01-\u0e5b"  # Thai
    r"\u0e81-\u0edf"  # Lao
    r"\u1000-\u109f"  # Myanmar (Burmese)
    r"\u1780-\u17ff"  # Khmer
    r"\u0f00-\u0fff"  # Tibetan
    r"\u2e80-\u9fff"  # CJK Radicals, Kangxi, CJK Unified
    r"\u3040-\u30ff"  # Hiragana + Katakana
    r"\u31f0-\u31ff"  # Katakana Ext
    r"\uac00-\ud7af"  # Hangul Syllables (Korean)
    r"\u1100-\u11ff"  # Hangul Jamo
    r"\u3130-\u318f"  # Hangul Compat Jamo
    r"\ua960-\ua97f"  # Hangul Jamo Ext-A
    r"\ud7b0-\ud7ff"  # Hangul Jamo Ext-B
    r"\uf900-\ufaff"  # CJK Compat Ideographs
    r"\U00020000-\U0002fa1f]"  # CJK Ext B-F + Compat Supplement
)

TRIGRAM_MIN_LEN = 3
MAX_LIKE_TERMS = 6


@dataclass
class SparseResult:
    """A single hit from the sparse keyword search.

    Attributes:
        chunk_id (str): The unique identifier of the retrieved chunk.
        score (float): The relevance score. For FTS matches, this is the
            negative BM25 rank. For LIKE matches, it's a length-based heuristic.
    """

    chunk_id: str
    score: float


class SparseRetriever:
    """Keyword search via SQLite FTS5 trigram index with optional metadata filters.

    This retriever handles the complexities of multilingual keyword search
    by splitting queries into FTS-eligible segments and short fallback segments.
    """

    def __init__(self, pool: ReadOnlySQLitePool):
        """Initialize the SparseRetriever.

        Args:
            pool (ReadOnlySQLitePool): The connection pool to the query node's
                read-only SQLite database.
        """
        self._pool = pool

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list[SparseResult], float]:
        """Search for chunks using a combination of FTS5 and LIKE fallback.

        Args:
            query_text (str): The raw keyword query string.
            top_k (int): The maximum number of results to return.
            filters (dict, optional): Metadata filters to apply (e.g., `doc_type`, `doc_id`).

        Returns:
            tuple[list[SparseResult], float]: A tuple containing:
                - list[SparseResult]: The merged and ranked search results.
                - float: The elapsed time in milliseconds.
        """
        if not query_text or not query_text.strip():
            return [], 0.0

        t0 = time.monotonic()
        fts_segs, short_segs = self._split_segments(query_text)

        with self._pool.connection() as conn:
            results: list[SparseResult] = []
            try:
                if fts_segs:
                    fts_query = self._segments_to_fts(fts_segs)
                    if filters:
                        results = self._search_with_filters(conn, fts_query, top_k, filters)
                    else:
                        results = self._search_simple(conn, fts_query, top_k)

                if short_segs:
                    like_results = self._search_like_fallback(conn, short_segs, top_k, filters)
                    seen = {r.chunk_id for r in results}
                    for r in like_results:
                        if r.chunk_id not in seen:
                            results.append(r)
                            seen.add(r.chunk_id)
            except Exception:
                logger.warn(
                    "sparse_query_error",
                    fts_segs=fts_segs[:3],
                    short_segs=short_segs[:3],
                    exc_info=True,
                )

        elapsed = (time.monotonic() - t0) * 1000
        return results, elapsed

    def _search_simple(self, conn, fts_query: str, top_k: int) -> list[SparseResult]:
        """Execute a simple FTS5 MATCH query."""
        rows = conn.execute(
            "SELECT chunk_id, rank FROM chunks_fts WHERE text_norm MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        return [SparseResult(chunk_id=r["chunk_id"], score=-r["rank"]) for r in rows]

    def _search_with_filters(
        self,
        conn,
        fts_query: str,
        top_k: int,
        filters: dict,
    ) -> list[SparseResult]:
        """Execute an FTS5 MATCH query joined with the chunks table for filtering."""
        where_clauses: list[str] = []
        params: list = [fts_query]

        if "doc_type" in filters:
            where_clauses.append("c.metadata_json LIKE ?")
            params.append(f'%"doc_type": "{filters["doc_type"]}"%')
        if "doc_id" in filters:
            where_clauses.append("c.doc_id = ?")
            params.append(filters["doc_id"])

        filter_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(top_k)

        rows = conn.execute(
            f"SELECT f.chunk_id, f.rank "
            f"FROM chunks_fts f "
            f"JOIN chunks c ON f.chunk_id = c.chunk_id "
            f"WHERE f.text_norm MATCH ? {filter_sql} "
            f"ORDER BY f.rank "
            f"LIMIT ?",
            params,
        ).fetchall()
        return [SparseResult(chunk_id=r["chunk_id"], score=-r["rank"]) for r in rows]

    def _search_like_fallback(
        self,
        conn,
        terms: list[str],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SparseResult]:
        """LIKE-based fallback for terms shorter than the trigram minimum.

        Queries the FTS5 content backing table's `c2` column (text_norm).

        Args:
            conn: SQLite connection.
            terms (list[str]): List of short terms or bigrams.
            top_k (int): Result limit.
            filters (dict, optional): Metadata filters.

        Returns:
            list[SparseResult]: Ranked hits.
        """
        if not terms:
            return []

        if len(terms) > MAX_LIKE_TERMS:
            logger.warning(
                "like_terms_capped",
                original=len(terms),
                cap=MAX_LIKE_TERMS,
            )
            terms = terms[:MAX_LIKE_TERMS]

        where_parts = ["fc.c2 LIKE ?" for _ in terms]
        params: list = [f"%{t}%" for t in terms]

        text_clause = " OR ".join(where_parts)
        filter_clauses: list[str] = []
        if filters:
            if "doc_type" in filters:
                filter_clauses.append("c.metadata_json LIKE ?")
                params.append(f'%"doc_type": "{filters["doc_type"]}"%')
            if "doc_id" in filters:
                filter_clauses.append("c.doc_id = ?")
                params.append(filters["doc_id"])
        filter_sql = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""
        params.append(top_k)

        rows = conn.execute(
            f"SELECT fc.c0 AS chunk_id, length(fc.c2) AS tlen "
            f"FROM chunks_fts_content fc "
            f"JOIN chunks c ON fc.c0 = c.chunk_id "
            f"WHERE ({text_clause}) {filter_sql} "
            f"ORDER BY tlen ASC "
            f"LIMIT ?",
            params,
        ).fetchall()
        return [SparseResult(chunk_id=r[0], score=1.0 / max(r[1], 1)) for r in rows]

    def _split_segments(self, query_text: str) -> tuple[list[str], list[str]]:
        """Split query into FTS-eligible segments and short segments.

        Normalization (NFKC + casefold + trad->simp) is applied. For
        scriptio-continua segments, 2-char bigrams are extracted for fallback.

        Args:
            query_text (str): The raw input query.

        Returns:
            tuple[list[str], list[str]]: (fts_eligible_segments, short_fallback_segments).
        """
        text = normalize_query(query_text.strip())
        text = _FTS5_SPECIAL.sub(" ", text)

        fts_segs: list[str] = []
        short_segs: list[str] = []

        segments = text.split()
        if not segments:
            merged = text.replace(" ", "")
            if merged:
                segments = [merged]

        for seg in segments:
            if len(seg) >= TRIGRAM_MIN_LEN:
                fts_segs.append(seg)
                if _SCRIPTIO_CONTINUA.search(seg):
                    bigrams: list[str] = []
                    for i in range(len(seg) - 1):
                        a, b = seg[i], seg[i + 1]
                        if _SCRIPTIO_CONTINUA.match(a) and _SCRIPTIO_CONTINUA.match(b):
                            bigrams.append(a + b)
                    if bigrams:
                        remaining = MAX_LIKE_TERMS - len(short_segs)
                        if remaining <= 0:
                            pass
                        elif len(bigrams) <= remaining:
                            short_segs.extend(bigrams)
                        else:
                            head = remaining // 2 or 1
                            tail = remaining - head
                            short_segs.extend(bigrams[:head])
                            if tail:
                                short_segs.extend(bigrams[-tail:])
            elif seg:
                short_segs.append(seg)

        if short_segs:
            short_segs = list(dict.fromkeys(short_segs))

        return fts_segs, short_segs

    @staticmethod
    def _segments_to_fts(segments: list[str]) -> str:
        """Build an FTS5 MATCH expression from normalized segments.

        For CJK segments, it emits the full phrase AND overlapping 3-char
        sliding windows to improve recall across punctuation boundaries.

        Args:
            segments (list[str]): List of normalized query segments.

        Returns:
            str: An FTS5 MATCH expression string.
        """
        terms: list[str] = []
        for seg in segments:
            seg = seg.replace('"', '""')
            terms.append(f'"{seg}"')
            if len(seg) <= TRIGRAM_MIN_LEN:
                continue
            sc_count = sum(1 for c in seg if _SCRIPTIO_CONTINUA.match(c))
            if sc_count < 2:
                continue
            for i in range(len(seg) - TRIGRAM_MIN_LEN + 1):
                window = seg[i : i + TRIGRAM_MIN_LEN]
                if any(_SCRIPTIO_CONTINUA.match(c) for c in window):
                    terms.append(f'"{window}"')
        return " OR ".join(terms) if terms else '""'

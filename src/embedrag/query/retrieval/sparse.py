"""Sparse retrieval via SQLite FTS5 trigram index.

Tiered retrieval strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **FTS5 MATCH** (primary, fast, BM25-ranked):
   Uses ``trigram case_sensitive 0``.  For scriptio-continua segments (CJK,
   Korean, Japanese kana, Thai, Lao, Myanmar, Khmer, Tibetan) the query is
   decomposed into the full phrase + overlapping 3-char sliding windows so
   partial matches surface even when punctuation breaks contiguous substrings.
   For space-delimited languages (Latin, Cyrillic, etc.) only the full phrase
   is emitted — FTS5 trigram handles them natively.

2. **LIKE fallback** (secondary, slower, length-ranked):
   Activated for any segment shorter than 3 characters (e.g. "仁", "MD", "X")
   and for 2-char bigrams extracted from scriptio-continua segments to bridge
   punctuation boundaries.  Queries the FTS content backing table's ``c2``
   column (``text_norm``, NFKC + casefold + diacritic-stripped + trad-to-simp).
   Total LIKE terms are capped at ``MAX_LIKE_TERMS`` (default 6) with boundary
   selection (first + last bigrams) to bound scan cost on large tables.

Both paths run and results are merged (FTS results first, LIKE de-duplicated).

Requires schema v3 (``text_norm`` column in FTS table).
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
    chunk_id: str
    score: float


class SparseRetriever:
    """Keyword search via SQLite FTS5 trigram index with optional metadata filters.

    Strategy per query segment:
      - len >= 3  -> FTS5 MATCH on text_norm (fast, BM25-ranked)
      - len < 3   -> LIKE fallback on FTS content table's text_norm (c2)

    Both paths run and results are merged (FTS results first, LIKE deduped).
    """

    def __init__(self, pool: ReadOnlySQLitePool):
        self._pool = pool

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list[SparseResult], float]:
        """Search using FTS5 trigram + LIKE fallback for short terms.

        Returns (results, elapsed_ms).
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

    # ── FTS path (text_norm MATCH) ──

    def _search_simple(self, conn, fts_query: str, top_k: int) -> list[SparseResult]:
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

    # ── LIKE fallback on FTS content table's text_norm (c2) ──

    def _search_like_fallback(
        self,
        conn,
        terms: list[str],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SparseResult]:
        """LIKE-based fallback for terms shorter than the trigram minimum.

        Queries the FTS5 content backing table's ``c2`` column (text_norm)
        so that "礼" matches text originally containing "禮".

        FTS5 content table layout:
          c0=chunk_id, c1=text, c2=text_norm, c3=title, c4=title_norm, c5=tags
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

    # ── Query building ──

    def _split_segments(self, query_text: str) -> tuple[list[str], list[str]]:
        """Split query into FTS-eligible segments (>=3 chars) and short segments.

        All segments are normalized (NFKC + casefold + trad->simp).

        For scriptio-continua segments, 2-char bigrams are emitted for the LIKE
        fallback path.  To avoid O(n) LIKE scans on long queries, only boundary
        bigrams are kept (first half + last half of the cap) so that both the
        start and end of the query contribute to recall.  Total bigrams across
        all segments are capped at ``MAX_LIKE_TERMS``.
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

        For CJK-heavy segments, emit the full phrase AND overlapping 3-char
        windows so that partial matches surface even when punctuation breaks the
        contiguous substring.  For non-CJK segments (Latin, Cyrillic, etc.), emit
        only the full phrase -- FTS5 trigram already handles those correctly and
        sliding windows would create excessive false positives.

        "谁栽到地下死了" -> "谁栽到地下死了" OR "谁栽到" OR "栽到地" OR ...
        "machine"         -> "machine"
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

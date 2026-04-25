"""Hierarchy-aware context expansion: fetch parent/ancestor text for matched chunks."""

from __future__ import annotations

from embedrag.models.api import ChunkResult
from embedrag.query.storage import QueryDocStore


class HierarchyExpander:
    """Expands search results with parent context from the chunk hierarchy."""

    def __init__(self, doc_store: QueryDocStore):
        self._store = doc_store

    def expand(self, chunks: list[ChunkResult], depth: int = 1) -> list[ChunkResult]:
        """Add ancestor context to each chunk result.

        For depth=1, uses the fast parent_chunk_id lookup.
        For depth>1, uses the closure table to fetch multiple ancestor levels,
        concatenating their text from outermost to innermost.
        """
        if depth <= 0:
            return chunks

        for chunk in chunks:
            if depth == 1:
                parent_text = self._store.get_parent_chunk_text(chunk.chunk_id)
                if parent_text:
                    chunk.parent_text = parent_text
            else:
                ancestors = self._store.get_ancestors(chunk.chunk_id, max_depth=depth)
                if ancestors:
                    parts = [f"[{a.level_type}] {a.text}" for a in reversed(ancestors)]
                    chunk.parent_text = "\n---\n".join(parts)

        return chunks

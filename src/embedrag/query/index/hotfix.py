"""In-memory hotfix buffer for emergency writes on the query node.

Provides a small FAISS ``IndexFlatIP`` buffer that supports add, delete
(mark), and search operations without rebuilding the main index. Useful
for urgent document corrections between snapshot releases. Cleared
automatically when the next snapshot generation is loaded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import faiss
import numpy as np

from embedrag.logging_setup import get_logger
from embedrag.models.api import ChunkResult

logger = get_logger(__name__)


@dataclass
class HotfixChunkData:
    """Data stored per chunk in the hotfix buffer.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        doc_id: Identifier of the parent document.
        text: Raw text content of the chunk.
        metadata: Arbitrary key-value metadata attached to the chunk.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class HotfixBuffer:
    """Small in-memory buffer for emergency writes on the query node.

    Supports add, delete (mark), and search operations.
    Cleared when the next snapshot is loaded.
    """

    def __init__(self, dim: int = 1024, max_size: int = 10_000):
        self._dim = dim
        self._max_size = max_size
        self._index = faiss.IndexFlatIP(dim)
        self._pos_to_id: dict[int, str] = {}
        self._chunks: dict[str, HotfixChunkData] = {}
        self._deleted_ids: set[str] = set()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of vectors currently in the hotfix buffer."""
        return self._index.ntotal

    @property
    def deleted_count(self) -> int:
        """Number of chunk IDs marked as deleted."""
        return len(self._deleted_ids)

    def add(self, chunk_id: str, vector: np.ndarray, chunk_data: HotfixChunkData) -> None:
        """Add a chunk to the hotfix buffer.

        Args:
            chunk_id: Unique identifier for the chunk.
            vector: The embedding vector (will be reshaped to 1D float32).
            chunk_data: Associated metadata and text for the chunk.

        Note:
            If the buffer is full, the chunk is silently ignored and a
            warning is logged.
        """
        with self._lock:
            if self._index.ntotal >= self._max_size:
                logger.warn("hotfix_buffer_full", max_size=self._max_size)
                return
            pos = self._index.ntotal
            vec = vector.reshape(1, -1).astype(np.float32)
            self._index.add(vec)
            self._pos_to_id[pos] = chunk_id
            self._chunks[chunk_id] = chunk_data

    def delete(self, chunk_id: str) -> None:
        """Mark a chunk ID as deleted so it is excluded from search results.

        The chunk is not actually removed from the index; it is tracked in
        a deletion set and filtered at query time.

        Args:
            chunk_id: The chunk identifier to mark as deleted.
        """
        with self._lock:
            self._deleted_ids.add(chunk_id)

    def is_deleted(self, chunk_id: str) -> bool:
        """Check whether a chunk ID has been marked as deleted.

        Args:
            chunk_id: The chunk identifier to check.

        Returns:
            ``True`` if the chunk was previously deleted via ``delete()``.
        """
        return chunk_id in self._deleted_ids

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search the hotfix buffer. Returns [(chunk_id, score)]."""
        if self._index.ntotal == 0:
            return []
        q = query.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q, k)
        results = []
        for score, pos in zip(distances[0], indices[0]):
            if pos >= 0 and pos in self._pos_to_id:
                cid = self._pos_to_id[pos]
                if cid not in self._deleted_ids:
                    results.append((cid, float(score)))
        return results

    def get_chunk(self, chunk_id: str) -> ChunkResult | None:
        """Retrieve a chunk's data by its ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            A ``ChunkResult`` populated with the stored data, or ``None``
            if the chunk ID is not in the buffer.
        """
        data = self._chunks.get(chunk_id)
        if not data:
            return None
        return ChunkResult(
            chunk_id=data.chunk_id,
            doc_id=data.doc_id,
            text=data.text,
            score=0.0,
            metadata=data.metadata,
        )

    def clear(self) -> None:
        """Reset the hotfix buffer, removing all chunks and deletion marks."""
        with self._lock:
            self._index.reset()
            self._pos_to_id.clear()
            self._chunks.clear()
            self._deleted_ids.clear()
        logger.info("hotfix_buffer_cleared")

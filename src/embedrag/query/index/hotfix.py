"""Hotfix buffer: small in-memory emergency write buffer for the query node."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np

from embedrag.logging_setup import get_logger
from embedrag.models.api import ChunkResult

logger = get_logger(__name__)


@dataclass
class HotfixChunkData:
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
        return self._index.ntotal

    @property
    def deleted_count(self) -> int:
        return len(self._deleted_ids)

    def add(self, chunk_id: str, vector: np.ndarray, chunk_data: HotfixChunkData) -> None:
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
        """Mark a chunk ID as deleted (filtered from main index results)."""
        with self._lock:
            self._deleted_ids.add(chunk_id)

    def is_deleted(self, chunk_id: str) -> bool:
        return chunk_id in self._deleted_ids

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search the hotfix buffer. Returns [(chunk_id, score)]."""
        if self._index.ntotal == 0:
            return []
        q = query.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        D, I = self._index.search(q, k)
        results = []
        for score, pos in zip(D[0], I[0]):
            if pos >= 0 and pos in self._pos_to_id:
                cid = self._pos_to_id[pos]
                if cid not in self._deleted_ids:
                    results.append((cid, float(score)))
        return results

    def get_chunk(self, chunk_id: str) -> Optional[ChunkResult]:
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
        with self._lock:
            self._index.reset()
            self._pos_to_id.clear()
            self._chunks.clear()
            self._deleted_ids.clear()
        logger.info("hotfix_buffer_cleared")

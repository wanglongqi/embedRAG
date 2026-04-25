"""FAISS shard worker: loads one shard and handles search requests."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class ShardWorker:
    """Loads a single FAISS shard and serves search requests.

    Designed to run in-process (GIL released by FAISS C++ during search)
    or in a subprocess via multiprocessing for true parallelism.
    """

    def __init__(self, shard_path: str, nprobe: int = 32, use_mmap: bool = True):
        self._path = shard_path
        self._nprobe = nprobe
        io_flag = faiss.IO_FLAG_MMAP if use_mmap else 0
        self._index = faiss.read_index(shard_path, io_flag)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = nprobe
        logger.info(
            "shard_loaded",
            path=shard_path,
            vectors=self._index.ntotal,
            mmap=use_mmap,
        )

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search this shard.

        Args:
            query_vectors: (n_queries, dim) float32 array.
            top_k: number of results per query.

        Returns:
            (distances, indices) arrays of shape (n_queries, top_k).
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            n = query_vectors.shape[0]
            return np.empty((n, 0), dtype=np.float32), np.empty((n, 0), dtype=np.int64)
        distances, indices = self._index.search(query_vectors, k)
        return distances, indices

    def shutdown(self) -> None:
        del self._index
        logger.info("shard_unloaded", path=self._path)

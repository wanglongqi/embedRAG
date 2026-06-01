"""FAISS shard worker for single-shard search.

``ShardWorker`` loads one FAISS index shard (with optional memory-mapping),
configures nprobe, and exposes a ``search()`` method that releases the GIL
during C++ FAISS execution. Designed to be used by ``ShardManager`` which
dispatches across multiple shard workers concurrently.
"""

from __future__ import annotations

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
        """Total number of vectors stored in this shard."""
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

    def reconstruct_all(self) -> np.ndarray:
        """Reconstruct all stored vectors from this shard.

        Exact for Flat / IVF,Flat indexes; approximate for product-quantized
        (IVF,PQ) indexes. Returns a ``(ntotal, dim)`` float32 array.
        """
        n = self._index.ntotal
        if n == 0:
            d = self._index.d if hasattr(self._index, "d") else 0
            return np.empty((0, d), dtype=np.float32)
        # IVF indexes need a direct map to support reconstruction by id.
        try:
            self._index.make_direct_map()
        except (AttributeError, RuntimeError):
            pass
        vectors = self._index.reconstruct_n(0, n)
        return np.asarray(vectors, dtype=np.float32)

    def shutdown(self) -> None:
        """Release the loaded FAISS index and free resources."""
        del self._index
        logger.info("shard_unloaded", path=self._path)

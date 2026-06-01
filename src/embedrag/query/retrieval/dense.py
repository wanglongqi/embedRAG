"""Dense retriever: parallel shard search with result merging.

This module provides the core vector search functionality for the query node.
It manages a pool of FAISS shard workers, dispatches queries to them in parallel,
and merges the partial results into a final ranked list.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from embedrag.logging_setup import get_logger
from embedrag.query.index.id_mapping import IDMapper
from embedrag.query.index.shard import ShardWorker

logger = get_logger(__name__)


@dataclass
class DenseResult:
    """A single hit from the dense vector search.

    Attributes:
        chunk_id (str): The unique identifier of the retrieved chunk.
        score (float): The similarity score (usually inner product/dot product) between
            the query vector and the chunk's vector. Higher is more similar.
    """

    chunk_id: str
    score: float


class ShardManager:
    """Manages multiple FAISS shard workers and dispatches parallel searches.

    The index is split into multiple shards during the build phase. This manager
    holds references to the loaded `ShardWorker` instances and uses a thread pool
    to execute searches across all shards concurrently, minimizing latency.
    """

    def __init__(self, workers: list[ShardWorker], id_mapper: IDMapper):
        """Initialize the ShardManager.

        Args:
            workers (list[ShardWorker]): A list of loaded `ShardWorker` instances, one for each index shard.
            id_mapper (IDMapper): An `IDMapper` instance used to translate FAISS internal
                integer IDs back to string `chunk_id`s.
        """
        self._workers = workers
        self._id_mapper = id_mapper
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(workers)))

    @property
    def total_vectors(self) -> int:
        """int: The total number of vectors across all managed shards."""
        return sum(w.ntotal for w in self._workers)

    @property
    def num_shards(self) -> int:
        """int: The number of active shards being managed."""
        return len(self._workers)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[DenseResult]:
        """Search all shards in parallel and merge the results.

        This method dispatches the query to all workers via a thread pool. Once all
        workers return their local top-k results, the lists are concatenated,
        sorted globally by score, and truncated to the final `top_k`.

        Args:
            query_vector (np.ndarray): A 1D or 2D float32 numpy array representing the query embedding.
                If 1D, it will be reshaped to (1, dim).
            top_k (int): The maximum number of total results to return.

        Returns:
            list[DenseResult]: A list of `DenseResult` objects, sorted by score in descending order.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        futures = []
        for shard_idx, worker in enumerate(self._workers):
            fut = self._executor.submit(self._search_one, shard_idx, worker, query_vector, top_k)
            futures.append(fut)

        all_results: list[DenseResult] = []
        for fut in futures:
            all_results.extend(fut.result())

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def _search_one(self, shard_idx: int, worker: ShardWorker, query: np.ndarray, top_k: int) -> list[DenseResult]:
        """Execute a search on a single shard worker and resolve IDs."""
        distances, indices = worker.search(query, top_k)
        results: list[DenseResult] = []
        for dist, fid in zip(distances[0], indices[0]):
            if fid < 0:
                continue
            chunk_id = self._id_mapper.resolve_single(shard_idx, int(fid))
            if chunk_id:
                results.append(DenseResult(chunk_id=chunk_id, score=float(dist)))
        return results

    def reconstruct_all(self) -> tuple[list[str], np.ndarray]:
        """Reconstruct every stored vector with its chunk id.

        Returns ``(chunk_ids, vectors)``. Exact for Flat/IVF-Flat shards,
        approximate for IVF,PQ. Vectors whose ids cannot be resolved are
        skipped.
        """
        all_ids: list[str] = []
        chunks: list[np.ndarray] = []
        for shard_idx, worker in enumerate(self._workers):
            vecs = worker.reconstruct_all()
            for local_idx in range(vecs.shape[0]):
                chunk_id = self._id_mapper.resolve_single(shard_idx, local_idx)
                if chunk_id:
                    all_ids.append(chunk_id)
                    chunks.append(vecs[local_idx])
        if not chunks:
            return [], np.empty((0, 0), dtype=np.float32)
        return all_ids, np.stack(chunks).astype(np.float32)

    def shutdown(self) -> None:
        """Shut down the thread pool and release all worker resources."""
        self._executor.shutdown(wait=False)
        for w in self._workers:
            w.shutdown()


class DenseRetriever:
    """High-level dense retrieval interface.

    Wraps the `ShardManager` to provide a clean search API, handling timing
    and the filtering of deleted chunks (hotfixes) before returning the final results.
    """

    def __init__(self, shard_manager: ShardManager):
        """Initialize the DenseRetriever.

        Args:
            shard_manager (ShardManager): The active `ShardManager` handling index shards.
        """
        self._shard_manager = shard_manager

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        deleted_ids: set[str] | None = None,
    ) -> tuple[list[DenseResult], float]:
        """Execute a dense search and filter out logically deleted chunks.

        To accommodate filtering without returning fewer results than requested,
        this method queries the underlying shards for `top_k * 2` results, filters
        out any chunk IDs present in `deleted_ids`, and then truncates to `top_k`.

        Args:
            query_vector (np.ndarray): The query embedding vector.
            top_k (int): The final number of desired results.
            deleted_ids (set[str], optional): An optional set of `chunk_id` strings that
                should be excluded from the search results (typically used for hot-swapping
                deletes before the next snapshot).

        Returns:
            tuple[list[DenseResult], float]: A tuple containing:
                - The list of filtered `DenseResult` objects.
                - The elapsed time in milliseconds for the search operation.
        """
        t0 = time.monotonic()
        raw_results = self._shard_manager.search(query_vector, top_k * 2)
        if deleted_ids is not None:
            raw_results = [r for r in raw_results if r.chunk_id not in deleted_ids]
        elapsed = (time.monotonic() - t0) * 1000
        return raw_results[:top_k], elapsed

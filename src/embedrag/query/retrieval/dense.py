"""Dense retriever: parallel shard search with result merging."""

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
    chunk_id: str
    score: float


class ShardManager:
    """Manages multiple FAISS shard workers and dispatches parallel searches."""

    def __init__(self, workers: list[ShardWorker], id_mapper: IDMapper):
        self._workers = workers
        self._id_mapper = id_mapper
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(workers)))

    @property
    def total_vectors(self) -> int:
        return sum(w.ntotal for w in self._workers)

    @property
    def num_shards(self) -> int:
        return len(self._workers)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[DenseResult]:
        """Search all shards in parallel and merge results by score."""
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
        distances, indices = worker.search(query, top_k)
        results: list[DenseResult] = []
        for dist, fid in zip(distances[0], indices[0]):
            if fid < 0:
                continue
            chunk_id = self._id_mapper.resolve_single(shard_idx, int(fid))
            if chunk_id:
                results.append(DenseResult(chunk_id=chunk_id, score=float(dist)))
        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        for w in self._workers:
            w.shutdown()


class DenseRetriever:
    """High-level dense retrieval: shard search + optional hotfix merge."""

    def __init__(self, shard_manager: ShardManager):
        self._shard_manager = shard_manager

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        deleted_ids: set[str] | None = None,
    ) -> tuple[list[DenseResult], float]:
        """Search dense index, filtering out deleted IDs.

        Returns (results, elapsed_ms).
        """
        t0 = time.monotonic()
        raw_results = self._shard_manager.search(query_vector, top_k * 2)
        if deleted_ids:
            raw_results = [r for r in raw_results if r.chunk_id not in deleted_ids]
        elapsed = (time.monotonic() - t0) * 1000
        return raw_results[:top_k], elapsed

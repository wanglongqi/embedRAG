"""Tests for dense retriever: shard search, parallel merge, filtering."""

import faiss
import msgpack
import numpy as np
import pytest

from embedrag.query.index.id_mapping import IDMapper
from embedrag.query.index.shard import ShardWorker
from embedrag.query.retrieval.dense import DenseResult, DenseRetriever, ShardManager


@pytest.fixture
def two_shard_setup(tmp_path):
    """Create two FAISS Flat shards with known vectors."""
    dim = 32
    rng = np.random.RandomState(42)
    n_per_shard = 100

    id_map = {}
    shard_paths = []
    for s in range(2):
        vecs = rng.randn(n_per_shard, dim).astype(np.float32)
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        path = str(tmp_path / f"shard_{s}.faiss")
        faiss.write_index(index, path)
        shard_paths.append(path)
        for i in range(n_per_shard):
            id_map[str(s * n_per_shard + i)] = f"chunk_{s}_{i}"

    map_path = str(tmp_path / "id_map.msgpack")
    with open(map_path, "wb") as f:
        msgpack.pack(id_map, f)

    return dim, shard_paths, map_path, n_per_shard


class TestShardWorker:
    def test_load_and_search(self, two_shard_setup):
        dim, paths, _, n = two_shard_setup
        worker = ShardWorker(paths[0], nprobe=32, use_mmap=False)
        assert worker.ntotal == n

        q = np.random.randn(1, dim).astype(np.float32)
        faiss.normalize_L2(q)
        D, I = worker.search(q, 5)
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)
        assert all(i >= 0 for i in I[0])
        worker.shutdown()


class TestIDMapper:
    def test_load_and_resolve(self, two_shard_setup):
        _, _, map_path, n = two_shard_setup
        mapper = IDMapper.load(map_path, [n, n])
        resolved = mapper.resolve(0, [0, 1, 2])
        assert resolved == ["chunk_0_0", "chunk_0_1", "chunk_0_2"]

        resolved2 = mapper.resolve(1, [0, 1])
        assert resolved2 == ["chunk_1_0", "chunk_1_1"]


class TestShardManager:
    def test_parallel_search(self, two_shard_setup):
        dim, paths, map_path, n = two_shard_setup
        workers = [ShardWorker(p, use_mmap=False) for p in paths]
        mapper = IDMapper.load(map_path, [n, n])
        mgr = ShardManager(workers, mapper)

        assert mgr.total_vectors == n * 2
        q = np.random.randn(dim).astype(np.float32)
        results = mgr.search(q, 10)
        assert len(results) == 10
        assert all(isinstance(r, DenseResult) for r in results)
        assert results[0].score >= results[-1].score
        mgr.shutdown()


class TestDenseRetriever:
    def test_with_deleted_ids(self, two_shard_setup):
        dim, paths, map_path, n = two_shard_setup
        workers = [ShardWorker(p, use_mmap=False) for p in paths]
        mapper = IDMapper.load(map_path, [n, n])
        mgr = ShardManager(workers, mapper)
        retriever = DenseRetriever(mgr)

        q = np.random.randn(dim).astype(np.float32)
        results_no_filter, _ = retriever.search(q, 10)
        top_id = results_no_filter[0].chunk_id

        results_filtered, _ = retriever.search(q, 10, deleted_ids={top_id})
        assert top_id not in [r.chunk_id for r in results_filtered]
        mgr.shutdown()

"""Tests for FAISS index builder."""

import numpy as np
import faiss
import msgpack

from embedrag.config import IndexBuildConfig
from embedrag.writer.index_builder import IndexBuilder


class TestIndexBuilder:
    def test_build_flat_small_dataset(self, tmp_path):
        config = IndexBuildConfig(num_shards=2)
        builder = IndexBuilder(config, dim=64)

        n = 500
        rng = np.random.RandomState(42)
        chunk_ids = [f"c_{i}" for i in range(n)]
        embeddings = rng.randn(n, 64).astype(np.float32)

        index_info, id_map_path = builder.build(chunk_ids, embeddings, str(tmp_path))

        assert index_info.total_vectors == n
        assert index_info.num_shards <= 2
        assert sum(s.num_vectors for s in index_info.shards) == n

        with open(id_map_path, "rb") as f:
            id_map = msgpack.unpack(f)
        assert len(id_map) == n

    def test_build_ivf_larger_dataset(self, tmp_path):
        config = IndexBuildConfig(num_shards=2, ivf_nlist=16, pq_m=8)
        builder = IndexBuilder(config, dim=64)

        n = 5000
        rng = np.random.RandomState(42)
        chunk_ids = [f"c_{i}" for i in range(n)]
        embeddings = rng.randn(n, 64).astype(np.float32)

        index_info, id_map_path = builder.build(chunk_ids, embeddings, str(tmp_path))

        assert index_info.total_vectors == n
        assert "IVF" in index_info.type

        for shard in index_info.shards:
            idx = faiss.read_index(str(tmp_path / shard.file))
            assert idx.ntotal == shard.num_vectors

    def test_shards_deterministic(self, tmp_path):
        config = IndexBuildConfig(num_shards=4)
        builder = IndexBuilder(config, dim=32)

        n = 200
        rng = np.random.RandomState(42)
        chunk_ids = [f"c_{i}" for i in range(n)]
        embeddings = rng.randn(n, 32).astype(np.float32)

        info1, _ = builder.build(chunk_ids, embeddings, str(tmp_path / "r1"))
        info2, _ = builder.build(chunk_ids, embeddings, str(tmp_path / "r2"))

        for s1, s2 in zip(info1.shards, info2.shards):
            assert s1.num_vectors == s2.num_vectors

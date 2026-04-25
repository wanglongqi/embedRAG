"""Tests for RRF fusion and hotfix buffer."""

import numpy as np

from embedrag.query.index.hotfix import HotfixBuffer, HotfixChunkData
from embedrag.query.retrieval.dense import DenseResult
from embedrag.query.retrieval.fusion import rrf_fuse
from embedrag.query.retrieval.sparse import SparseResult


class TestRRFFusion:
    def test_basic_fusion(self):
        dense = [
            DenseResult(chunk_id="a", score=0.9),
            DenseResult(chunk_id="b", score=0.8),
            DenseResult(chunk_id="c", score=0.7),
        ]
        sparse = [
            SparseResult(chunk_id="b", score=5.0),
            SparseResult(chunk_id="d", score=4.0),
            SparseResult(chunk_id="a", score=3.0),
        ]
        fused = rrf_fuse(dense, sparse, top_k=3)
        assert len(fused) == 3
        ids = [f.chunk_id for f in fused]
        # a and b appear in both lists, so they should rank higher
        assert "a" in ids
        assert "b" in ids

    def test_disjoint_results(self):
        dense = [DenseResult(chunk_id="a", score=0.9)]
        sparse = [SparseResult(chunk_id="b", score=5.0)]
        fused = rrf_fuse(dense, sparse, top_k=2)
        assert len(fused) == 2
        ids = {f.chunk_id for f in fused}
        assert ids == {"a", "b"}

    def test_empty_inputs(self):
        fused = rrf_fuse([], [], top_k=5)
        assert len(fused) == 0

    def test_weights(self):
        dense = [DenseResult(chunk_id="a", score=0.9)]
        sparse = [SparseResult(chunk_id="b", score=5.0)]
        fused_equal = rrf_fuse(dense, sparse, top_k=2, dense_weight=1.0, sparse_weight=1.0)
        assert len(fused_equal) == 2
        fused_dense = rrf_fuse(dense, sparse, top_k=2, dense_weight=2.0, sparse_weight=0.5)
        # With heavy dense weight, "a" should rank higher
        assert fused_dense[0].chunk_id == "a"


class TestHotfixBuffer:
    def test_add_and_search(self):
        buf = HotfixBuffer(dim=32, max_size=100)
        rng = np.random.RandomState(42)
        vec = rng.randn(32).astype(np.float32)

        data = HotfixChunkData(chunk_id="hf_1", doc_id="d1", text="hotfix text")
        buf.add("hf_1", vec, data)
        assert buf.size == 1

        results = buf.search(vec, 5)
        assert len(results) == 1
        assert results[0][0] == "hf_1"

    def test_delete_filtering(self):
        buf = HotfixBuffer(dim=32, max_size=100)
        rng = np.random.RandomState(42)
        vec = rng.randn(32).astype(np.float32)

        data = HotfixChunkData(chunk_id="hf_del", doc_id="d1", text="to delete")
        buf.add("hf_del", vec, data)
        buf.delete("hf_del")

        results = buf.search(vec, 5)
        assert len(results) == 0
        assert buf.is_deleted("hf_del")

    def test_clear(self):
        buf = HotfixBuffer(dim=32, max_size=100)
        rng = np.random.RandomState(42)
        for i in range(10):
            vec = rng.randn(32).astype(np.float32)
            data = HotfixChunkData(chunk_id=f"c_{i}", doc_id="d", text=f"t{i}")
            buf.add(f"c_{i}", vec, data)
        buf.delete("c_0")
        assert buf.size == 10
        assert buf.deleted_count == 1

        buf.clear()
        assert buf.size == 0
        assert buf.deleted_count == 0

    def test_get_chunk(self):
        buf = HotfixBuffer(dim=32, max_size=100)
        vec = np.random.randn(32).astype(np.float32)
        data = HotfixChunkData(chunk_id="c1", doc_id="d1", text="hello")
        buf.add("c1", vec, data)

        result = buf.get_chunk("c1")
        assert result is not None
        assert result.text == "hello"
        assert buf.get_chunk("nonexistent") is None

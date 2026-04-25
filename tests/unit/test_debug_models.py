"""Tests for debug search API models."""

from embedrag.models.api import (
    ChunkResult,
    DebugDenseHit,
    DebugFusedHit,
    DebugSearchRequest,
    DebugSearchResponse,
    DebugSparseHit,
    DebugTiming,
)


class TestDebugSearchRequest:
    def test_defaults(self):
        req = DebugSearchRequest(query_text="test query")
        assert req.query_text == "test query"
        assert req.top_k == 10
        assert req.mode == "hybrid"
        assert req.filters is None
        assert req.expand_context is True
        assert req.context_depth == 1

    def test_custom_fields(self):
        req = DebugSearchRequest(
            query_text="机器学习",
            top_k=20,
            mode="sparse",
            filters={"doc_type": "article"},
            expand_context=False,
            context_depth=2,
        )
        assert req.query_text == "机器学习"
        assert req.top_k == 20
        assert req.mode == "sparse"
        assert req.filters == {"doc_type": "article"}
        assert req.expand_context is False


class TestDebugHitModels:
    def test_dense_hit(self):
        hit = DebugDenseHit(chunk_id="c-001", score=0.85)
        assert hit.chunk_id == "c-001"
        assert hit.score == 0.85

    def test_sparse_hit(self):
        hit = DebugSparseHit(chunk_id="c-002", score=-1.5)
        assert hit.chunk_id == "c-002"
        assert hit.score == -1.5

    def test_fused_hit_defaults(self):
        hit = DebugFusedHit(
            chunk_id="c-003",
            rrf_score=0.05,
            dense_score=0.9,
            sparse_score=-1.2,
        )
        assert hit.dense_rank == -1
        assert hit.sparse_rank == -1

    def test_fused_hit_with_ranks(self):
        hit = DebugFusedHit(
            chunk_id="c-004",
            rrf_score=0.033,
            dense_score=0.75,
            sparse_score=-2.1,
            dense_rank=3,
            sparse_rank=1,
        )
        assert hit.dense_rank == 3
        assert hit.sparse_rank == 1


class TestDebugTiming:
    def test_defaults(self):
        t = DebugTiming()
        assert t.embedding_ms == 0
        assert t.dense_ms == 0
        assert t.sparse_ms == 0
        assert t.fusion_ms == 0
        assert t.fetch_ms == 0
        assert t.expand_ms == 0
        assert t.total_ms == 0

    def test_with_values(self):
        t = DebugTiming(
            embedding_ms=15.2,
            dense_ms=3.1,
            sparse_ms=1.5,
            fusion_ms=0.2,
            fetch_ms=0.8,
            expand_ms=0.3,
            total_ms=21.1,
        )
        assert t.embedding_ms == 15.2
        assert t.total_ms == 21.1


class TestDebugSearchResponse:
    def test_minimal(self):
        resp = DebugSearchResponse(query_text="test", mode="hybrid")
        assert resp.query_text == "test"
        assert resp.mode == "hybrid"
        assert resp.fts_query == ""
        assert resp.dense_results == []
        assert resp.sparse_results == []
        assert resp.fused_results == []
        assert resp.final_chunks == []
        assert resp.config_snapshot == {}

    def test_full_response(self):
        resp = DebugSearchResponse(
            query_text="RAG系统",
            mode="hybrid",
            fts_query='"RAG系统"',
            embedding_time_ms=12.5,
            dense_results=[
                DebugDenseHit(chunk_id="c-1", score=0.92),
                DebugDenseHit(chunk_id="c-2", score=0.85),
            ],
            sparse_results=[
                DebugSparseHit(chunk_id="c-1", score=-1.1),
                DebugSparseHit(chunk_id="c-3", score=-2.0),
            ],
            fused_results=[
                DebugFusedHit(
                    chunk_id="c-1",
                    rrf_score=0.05,
                    dense_score=0.92,
                    sparse_score=-1.1,
                    dense_rank=0,
                    sparse_rank=0,
                ),
            ],
            final_chunks=[
                ChunkResult(
                    chunk_id="c-1",
                    doc_id="d-1",
                    text="RAG系统介绍...",
                    score=0.05,
                ),
            ],
            timing=DebugTiming(
                embedding_ms=12.5,
                dense_ms=2.1,
                sparse_ms=1.0,
                fusion_ms=0.1,
                fetch_ms=0.5,
                expand_ms=0.2,
                total_ms=16.4,
            ),
            config_snapshot={"enable_sparse": True, "max_top_k": 100},
        )
        assert len(resp.dense_results) == 2
        assert len(resp.sparse_results) == 2
        assert len(resp.fused_results) == 1
        assert len(resp.final_chunks) == 1
        assert resp.timing.total_ms == 16.4
        assert resp.config_snapshot["enable_sparse"] is True

    def test_serialization_roundtrip(self):
        resp = DebugSearchResponse(
            query_text="test",
            mode="dense",
            dense_results=[DebugDenseHit(chunk_id="c-1", score=0.5)],
            timing=DebugTiming(dense_ms=5.0, total_ms=5.0),
        )
        data = resp.model_dump()
        restored = DebugSearchResponse(**data)
        assert restored.query_text == "test"
        assert restored.dense_results[0].chunk_id == "c-1"
        assert restored.timing.dense_ms == 5.0

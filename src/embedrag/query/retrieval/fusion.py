"""Reciprocal Rank Fusion (RRF) for merging dense and sparse results."""

from __future__ import annotations

from dataclasses import dataclass

from embedrag.query.retrieval.dense import DenseResult
from embedrag.query.retrieval.sparse import SparseResult


@dataclass
class FusedResult:
    chunk_id: str
    rrf_score: float
    dense_score: float
    sparse_score: float


def rrf_fuse(
    dense_results: list[DenseResult],
    sparse_results: list[SparseResult],
    top_k: int,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[FusedResult]:
    """Merge dense and sparse results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank_i)) for each ranking list.
    k is the smoothing constant (default 60 per the original paper).
    """
    scores: dict[str, dict] = {}

    for rank, dr in enumerate(dense_results):
        if dr.chunk_id not in scores:
            scores[dr.chunk_id] = {"rrf": 0.0, "dense": dr.score, "sparse": 0.0}
        scores[dr.chunk_id]["rrf"] += dense_weight / (k + rank + 1)
        scores[dr.chunk_id]["dense"] = max(scores[dr.chunk_id]["dense"], dr.score)

    for rank, sr in enumerate(sparse_results):
        if sr.chunk_id not in scores:
            scores[sr.chunk_id] = {"rrf": 0.0, "dense": 0.0, "sparse": sr.score}
        scores[sr.chunk_id]["rrf"] += sparse_weight / (k + rank + 1)
        scores[sr.chunk_id]["sparse"] = max(scores[sr.chunk_id]["sparse"], sr.score)

    fused = [
        FusedResult(
            chunk_id=cid,
            rrf_score=s["rrf"],
            dense_score=s["dense"],
            sparse_score=s["sparse"],
        )
        for cid, s in scores.items()
    ]
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    return fused[:top_k]

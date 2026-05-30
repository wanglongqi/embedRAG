"""Reciprocal Rank Fusion (RRF) for merging dense and sparse results.

This module provides an implementation of the Reciprocal Rank Fusion algorithm,
which is used to combine multiple ranked result lists into a single, unified
ranking without requiring score normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

from embedrag.query.retrieval.dense import DenseResult
from embedrag.query.retrieval.sparse import SparseResult


@dataclass
class FusedResult:
    """A single hit from the fused search results.

    Attributes:
        chunk_id (str): The unique identifier of the retrieved chunk.
        rrf_score (float): The calculated RRF score for this chunk.
        dense_score (float): The original score from the dense retriever.
        sparse_score (float): The original score from the sparse retriever.
    """

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

    The RRF score for a document is calculated as:
        RRFscore(d) = sum( weight / (k + rank_i(d)) )
    where `rank_i(d)` is the rank of document `d` in the i-th ranking list.

    RRF is highly effective because it does not require the underlying scores
    (e.g., dot product for dense and BM25 for sparse) to be on the same scale.

    Args:
        dense_results (list[DenseResult]): Ranked results from the dense retriever.
        sparse_results (list[SparseResult]): Ranked results from the sparse retriever.
        top_k (int): The number of final fused results to return.
        k (int, optional): The smoothing constant used in the RRF formula.
            Defaults to 60, which is the value recommended in the original RRF paper.
        dense_weight (float, optional): A multiplier for the dense ranking's
            contribution to the final score. Defaults to 1.0.
        sparse_weight (float, optional): A multiplier for the sparse ranking's
            contribution to the final score. Defaults to 1.0.

    Returns:
        list[FusedResult]: A list of `FusedResult` objects, sorted by `rrf_score`
            in descending order.
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

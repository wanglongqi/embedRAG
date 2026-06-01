"""End-to-end clustering pipeline.

Wires the stages together: normalize -> (optional) reduce -> select params /
cluster -> evaluate -> explain -> label -> visualize, returning a
``ClusterResult``. The sync core (`cluster_vectors`) needs no network; text
vectorization and LLM labeling are layered on top by callers (CLI / HTTP).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from embedrag.cluster import evaluate, explain, label, store, visualize
from embedrag.cluster.algorithms import make_backend
from embedrag.cluster.preprocess import l2_normalize, reduce_dims
from embedrag.cluster.types import ClusterResult, Item
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


def cluster_vectors(
    vectors: np.ndarray,
    items: list[Item],
    *,
    algorithm: str = "auto",
    reduce: str = "auto",
    n_components: int = 0,
    auto: bool = True,
    params: dict | None = None,
    top_keywords: int = 10,
    top_reps: int = 5,
    ground_truth: list | None = None,
    run_id: str | None = None,
    space: str = "text",
    source: str = "",
) -> ClusterResult:
    """Cluster a matrix of vectors and return a fully explained result.

    This is the pure, synchronous core (no embedding service, no LLM calls).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
    if vectors.shape[0] != len(items):
        raise ValueError(f"vectors ({vectors.shape[0]}) and items ({len(items)}) length mismatch")

    run_id = run_id or store.make_run_id()
    normalized = l2_normalize(vectors)
    clustering_vecs, reduce_used = reduce_dims(normalized, method=reduce, n_components=n_components)

    algo_name, assignment, chosen_params, sweep = evaluate.select_params(
        clustering_vecs, algorithm, overrides=params or {}, auto=auto
    )
    labels = assignment.labels

    clusters, members, _centroids, sim_matrix = explain.explain(
        normalized, labels, assignment.probabilities, items, top_keywords=top_keywords, top_reps=top_reps
    )
    label.apply_keyword_labels(clusters)

    metrics = evaluate.internal_metrics(clustering_vecs, labels)
    if ground_truth is not None:
        metrics["external"] = evaluate.external_metrics(labels, ground_truth)

    backend = make_backend(algo_name, **chosen_params)
    projection = visualize.build_projection(clustering_vecs, labels, assignment.probabilities, items)
    projection["method"] = reduce_used if reduce_used != "none" else "raw"
    panels = visualize.build_panels(backend.panels, clustering_vecs, assignment, clusters, sim_matrix, sweep)

    noise_count = int(np.sum(labels == -1))
    result = ClusterResult(
        run_id=run_id,
        algorithm=algo_name,
        params=chosen_params,
        space=space,
        created_at=datetime.now(UTC).isoformat(),
        source=source,
        n_items=len(items),
        n_clusters=len(clusters),
        noise_count=noise_count,
        clusters=clusters,
        members=members,
        metrics=metrics,
        sweep=sweep,
        projection=projection,
        viz=panels,
    )
    logger.info(
        "cluster_done",
        run_id=run_id,
        algorithm=algo_name,
        n_clusters=len(clusters),
        noise=noise_count,
        silhouette=metrics.get("silhouette"),
    )
    return result


async def apply_llm_labels(
    result: ClusterResult,
    chat_url: str,
    model: str = "",
    api_key: str = "",
    language: str = "auto",
) -> None:
    """Replace keyword labels with LLM-generated topic names (in place)."""
    await label.label_clusters_llm(result.clusters, chat_url=chat_url, model=model, api_key=api_key, language=language)
